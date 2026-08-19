"""参数调优指南自动生成脚本（零第三方依赖）。

读取 models-optimize-sheel/report.py 的输出（opt_summary.csv +
best_serve_cmd.txt）与 hardware_env.json，生成交付级《参数调优指南》草稿：

  0 摘要 → 1 目标与方法 → 2 调优结果（锁定配置/评分表/参数说明）→
  3 验收对比（默认配置 vs 锁定配置）→ 4 生产部署建议（硬件资源配置
  估算 + 部署清单）→ 5 调优过程问题记录 → 附录

机器可读部分自动填充，结论留 TODO 由测试人员补填。估算口径：
  - 权重体积 = 参数量(十亿) × 每参数字节（fp16/bf16=2, fp8/int8=1,
    int4/nf4=0.5），参数量取自 hardware_env.json 的 params_b
  - 单卡显存需求优先用锁定配置实测显存峰值（mem_pct_max × 单卡显存），
    与权重估算取大者 × 1.25 余量后向上取整到常见容量档
  - 卡数 = ceil(权重体积 / 显存利用率 / 单卡容量)
  - CPU/内存/功耗按经验公式估算；以上均为估算，需人工复核

用法示例::

    python generate_tuning_guide.py --opt-dir opt_report \\
        --env hardware_env.json \\
        --p2-before ../results/run-p2-baseline \\
        --p2-after ../results/run-p2-locked \\
        --output tuning_guide_ascend.md

依赖：Python 3.9+ 标准库；复用 generate_report.py 的格式化工具。
"""

import argparse
import csv
import json
import math
import os
import shlex

from generate_report import (TODO, fmt_num, fmt_opt, fmt_unit, load_env,
                             md_table, section, safe_float)

VRAM_TIERS = [24, 32, 40, 48, 64, 80, 96, 128]
BPP = {"fp16": 2.0, "bf16": 2.0, "half": 2.0, "float16": 2.0,
       "fp8": 1.0, "w8a8": 1.0, "int8": 1.0, "int4": 0.5,
       "nf4": 0.5, "fp4": 0.5, "float32": 4.0}

PARAM_DESC = {
    "--dtype": "权重计算精度（影响显存与吞吐）",
    "--quantization": "权重量化方式",
    "--max-model-len": "最大上下文长度",
    "--gpu-memory-utilization": "显存利用率上限（KV Cache 预算来源）",
    "--max-num-seqs": "并发序列上限（调度窗口）",
    "--max-num-batched-tokens": "单批次最大 token 数（Prefill 粒度）",
    "--enable-chunked-prefill": "分块预填充（长请求抢占调度）",
    "--kv-cache-dtype": "KV Cache 精度（fp8 省显存）",
    "--tensor-parallel-size": "张量并行卡数",
    "--pipeline-parallel-size": "流水线并行数",
    "--enable-prefix-caching": "前缀缓存（重复请求加速）",
    "--served-model-name": "服务对外模型名",
    "--trust-remote-code": "允许远端模型代码",
    "--enforce-eager": "禁用 CUDA Graph（降显存、升时延）",
}


def parse_serve_cmd(cmd):
    """解析 serve 命令为 {flag: value}，布尔 flag 值为 True。"""
    params = {}
    if not cmd:
        return params
    tokens = shlex.split(cmd)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            if "=" in tok:
                key, value = tok.split("=", 1)
                params[key] = value
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                params[tok] = tokens[i + 1]
                i += 1
            else:
                params[tok] = True
        i += 1
    return params


def load_opt_summary(opt_dir):
    path = os.path.join(opt_dir, "opt_summary.csv")
    if not os.path.exists(path):
        path = os.path.join(opt_dir, "summary.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[error] 未找到评分表（opt_summary.csv / summary.csv）：{opt_dir}")
    with open(path, "r", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("config_id")], path


def load_best_cmd(opt_dir):
    path = os.path.join(opt_dir, "best_serve_cmd.txt")
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as fh:
        cmd = fh.read().strip()
    return cmd, path


def pick_top1(rows):
    """ACCEPT 中 score 最高者（与 report.py 口径一致）。"""
    accepted = [r for r in rows if (r.get("verdict") or "").upper() == "ACCEPT"]
    accepted.sort(key=lambda r: safe_float(r, "score") or -1.0, reverse=True)
    return accepted[0] if accepted else None


def weights_gb(env, cmd_params):
    model_cfg = ((env.get("model_deployment") or {}).get("model") or {})
    params_b = model_cfg.get("params_b")
    if not params_b:
        return None
    quant = cmd_params.get("--quantization") or model_cfg.get("quantization") or ""
    bpp = BPP.get(quant.strip().lower())
    if bpp is None:
        dtype = cmd_params.get("--dtype", "").strip().lower()
        bpp = BPP.get(dtype, 2.0)
    return float(params_b) * bpp


def suggest_vram_gb(gb):
    if gb is None:
        return None
    for tier in VRAM_TIERS:
        if gb <= tier:
            return tier
    return int(math.ceil(gb))


def est_hardware(env, cmd_params, top1):
    """按锁定配置实测 + 经验公式估算硬件需求，返回 (表行, 依据说明)。"""
    acc = (env.get("hardware") or {}).get("accelerator") or {}
    model = ((env.get("model_deployment") or {}).get("model") or {})
    hbm = safe_float(acc, "hbm_capacity_gb")
    tdp = safe_float(acc, "tdp_w")
    gmu = float(cmd_params.get("--gpu-memory-utilization", "0.9") or 0.9)

    weights = weights_gb(env, cmd_params)
    mem_pct = safe_float(top1, "mem_pct_max")
    measured = (mem_pct / 100.0 * hbm) if (mem_pct and hbm) else None

    need = max(measured or 0, (weights or 0) * 1.1)
    need = need * 1.25 if need else None
    vram = suggest_vram_gb(need)

    if weights and hbm and gmu:
        min_cards = int(math.ceil(weights / gmu / hbm))
    else:
        min_cards = None
    rec_cards = (min_cards + 1) if min_cards else None

    mem_gb = max(32.0, (weights or 0) * 4.0)
    cpu_cores = max(8, (min_cards or 1) * 4)
    power_w = int(tdp * (min_cards or 1) * 1.2) if tdp else None

    basis = []
    if weights:
        basis.append(f"权重估算 {fmt_num(weights, 0)} GB（{model.get('params_b')}B × "
                     f"{BPP.get(cmd_params.get('--dtype', '').strip().lower(), '?')} B/参）")
    if measured:
        basis.append(f"锁定配置实测显存峰值 {fmt_num(mem_pct, 0)}% × "
                     f"{fmt_num(hbm, 0)} GB ≈ {fmt_num(measured, 1)} GB/卡")
    basis.append("按 ×1.25 余量取常见容量档；估算值需人工复核")

    rows = [
        ["GPU 型号", fmt_opt(acc.get("name")), "硬件环境"],
        ["最小卡数（张量并行）", fmt_num(min_cards, 0) if min_cards else TODO,
         "权重体积 ÷ 显存利用率 ÷ 单卡容量"],
        ["推荐卡数（含冗余）", fmt_num(rec_cards, 0) if rec_cards else TODO,
         "最小卡数 + 1（故障/多实例备用）"],
        ["单卡显存建议", f"{vram} GB" if vram else TODO, "；".join(basis)],
        ["CPU 核数建议", f"≥ {cpu_cores} 核", "经验：每卡 ≥4 核 + 调度开销"],
        ["内存建议", f"≥ {fmt_num(mem_gb, 0)} GB", "经验：≥ 4×权重，且 ≥ 32 GB"],
        ["存储建议", (f"NVMe ≥ {fmt_num((weights or 0) * 2, 0)} GB"
                      if weights else TODO), "模型权重 ×2 + 日志/KV 落盘余量"],
        ["互联要求", ("多卡需 "
                      f"{fmt_opt(acc.get('interconnect_type')) or '高速互联'}"
                      if (min_cards or 1) > 1 else "单卡无特殊要求"),
         "张量并行跨卡通信"],
        ["功耗/散热预算", f"≥ {power_w} W" if power_w else TODO,
         f"{fmt_opt(tdp)} W/卡 × 卡数 × 1.2 冗余"],
    ]
    return rows


def compare_p2(before_dir, after_dir):
    """验收对比：两个 run_matrix 输出目录（默认配置 vs 锁定配置）。"""
    def load(dir_path):
        path = os.path.join(dir_path, "summary.csv")
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("input_len")]
        ok = [r for r in rows if (r.get("status") or "") == "OK"]
        short = [r for r in ok if r.get("phase") == "short"]
        long = [r for r in ok if r.get("phase") == "long"]

        def median(items, key):
            vals = sorted(v for v in (safe_float(r, key) for r in items)
                          if v is not None)
            if not vals:
                return None
            n = len(vals)
            return vals[n // 2] if n % 2 else \
                (vals[n // 2 - 1] + vals[n // 2]) / 2

        def max_of(items, key):
            vals = [v for v in (safe_float(r, key) for r in items)
                    if v is not None]
            return max(vals) if vals else None

        def mean_of(items, key):
            vals = [v for v in (safe_float(r, key) for r in items)
                    if v is not None]
            return sum(vals) / len(vals) if vals else None

        return {
            "n_short": len(short), "n_long": len(long),
            "n_oom": sum(1 for r in rows if (r.get("status") or "")
                         in ("OOM", "FAILED")),
            "thr_short": median(short, "output_throughput"),
            "thr_long": median(long, "output_throughput"),
            "ttft_p99": median(short, "p99_ttft_ms"),
            "tpot_p99": median(short, "p99_tpot_ms"),
            "mem_pct_max": max_of(rows, "gpu_mem_pct_max"),
            "gpu_util": mean_of(rows, "gpu_util_mean"),
            "leak_warn": sum(1 for r in rows if r.get("gpu_leak_warning")),
        }

    before = load(before_dir) if before_dir else None
    after = load(after_dir) if after_dir else None
    return before, after


def render_acceptance(before, after, before_dir, after_dir):
    if not before or not after:
        return (["| 指标 | before | after | 变化 |", "|---|---|---|---|",
                 f"| 验收对比数据 | {fmt_opt(os.path.basename(before_dir) if before_dir else None)} | "
                 f"{fmt_opt(os.path.basename(after_dir) if after_dir else None)} | TODO |"],
                None)

    def pct(new, old):
        if old in (None, 0) or new is None:
            return "-"
        return f"{(new - old) / old * 100:+.1f}%"

    def fmt(v):
        return fmt_num(v, 1) if v is not None else "-"

    rows = []
    for desc, key, label in (
            ("↑", "thr_short", "short 组中位吞吐 (tok/s)"),
            ("↑", "thr_long", "long 组中位吞吐 (tok/s)"),
            ("↓", "ttft_p99", "TTFT p99 (ms)"),
            ("↓", "tpot_p99", "TPOT p99 (ms)"),
            ("↓", "mem_pct_max", "显存峰值 % (max)"),
            ("↑", "gpu_util", "GPU 均利用率 % (mean)")):
        b, a = before.get(key), after.get(key)
        rows.append([f"{desc} {label}", fmt(b), fmt(a), pct(a, b)])
    rows.append(["OOM/失败组数", fmt_num(before["n_oom"], 0),
                 fmt_num(after["n_oom"], 0), "-"])
    rows.append(["显存泄漏警告组数", fmt_num(before["leak_warn"], 0),
                 fmt_num(after["leak_warn"], 0),
                 "无" if after["leak_warn"] == 0 else "存在"])

    gains = []
    if before["thr_short"] and after["thr_short"]:
        gains.append(f"短文本中位吞吐 {pct(after['thr_short'], before['thr_short'])}")
    if before["ttft_p99"] and after["ttft_p99"]:
        gains.append(f"TTFT p99 {pct(after['ttft_p99'], before['ttft_p99'])}")
    if before["mem_pct_max"] and after["mem_pct_max"]:
        gains.append(f"显存峰值 {pct(after['mem_pct_max'], before['mem_pct_max'])}")
    summary = "；".join(gains) if gains else "变化幅度见上表"
    return rows, summary


def main():
    p = argparse.ArgumentParser(description="参数调优指南自动生成脚本")
    p.add_argument("--opt-dir", required=True,
                   help="report.py 输出目录（含 opt_summary.csv + best_serve_cmd.txt）")
    p.add_argument("--env", required=True, help="hardware_env.json 路径")
    p.add_argument("--p2-before", default=None,
                   help="验收 before：默认配置 run_matrix 输出目录")
    p.add_argument("--p2-after", default=None,
                   help="验收 after：锁定配置 run_matrix 输出目录")
    p.add_argument("--output", default=None,
                   help="输出路径（默认 tuning_guide_<vendor>.md）")
    args = p.parse_args()

    env = load_env(args.env)
    vendor = env.get("vendor") or ""
    rows, score_path = load_opt_summary(args.opt_dir)
    cmd, cmd_path = load_best_cmd(args.opt_dir)
    top1 = pick_top1(rows)
    params = parse_serve_cmd(cmd or "")
    before, after = compare_p2(args.p2_before, args.p2_after)
    acc_rows, gain_summary = render_acceptance(
        before, after, args.p2_before, args.p2_after)
    hw_rows = est_hardware(env, params, top1) if top1 else None

    model_name = ((env.get("model_deployment") or {}).get("model") or {}) \
        .get("name") or "模型"
    title = f"{model_name} 参数调优指南（{fmt_opt(vendor)}）"
    out = [f"# {title}", ""]

    # 0 摘要
    out.append(section("0. 摘要（一页结论）"))
    meta = env.get("report_meta") or {}
    out.append(md_table(["项", "值"], [
        ["模型", fmt_opt(model_name)],
        ["厂商/芯片", f"{fmt_opt(vendor)} / {fmt_opt(((env.get('hardware') or {}).get('accelerator') or {}).get('name'))}"],
        ["报告编号", fmt_opt(meta.get("report_id"))],
        ["测试人", fmt_opt(meta.get("tester"))],
        ["日期", fmt_opt(meta.get("date"))],
    ]))
    out.append("")
    out.append(section("0.1 结论速览（自动）", 2))
    if top1:
        top_score = safe_float(top1, "score")
        out.append(md_table(["项目", "结论"], [
            ["锁定配置", fmt_opt(top1.get("config_id"))],
            ["最高评分", fmt_num(top_score) if top_score is not None else TODO],
            ["短探针吞吐 (tok/s)", fmt_opt(top1.get("short_thr"))],
            ["长探针吞吐 (tok/s)", fmt_opt(top1.get("long_thr"))],
            ["TTFT p99 (ms)", fmt_opt(top1.get("ttft_p99_ms"))],
            ["显存峰值 (%)", fmt_opt(top1.get("mem_pct_max"))],
            ["GPU 平均利用率 (%)", fmt_opt(top1.get("gpu_util_mean"))],
            ["验收收益", gain_summary if gain_summary else TODO],
        ]))
    else:
        out.append("> 无 ACCEPT 记录，锁定配置取自收敛引擎最终输出。")
    out.append("")
    out.append(section("0.2 关键结论（人工）", 2))
    out.append("- 推荐配置是否可用：TODO（可直接投产 / 有条件 / 不推荐）")
    out.append("- 制约因素：TODO（显存 / 并发 / 长文本 / 软件栈兼容）")
    out.append("- 与默认配置相比的收益与代价：TODO")
    out.append("")

    # 1 目标与方法
    out.append(section("1. 调优目标与方法（人工简述）"))
    out.append("- 目标：TODO（在固定模型与硬件上寻找最佳 `vllm serve` 参数组合）")
    out.append("- 方法：分层收敛（Serve 参数分组独立扫描，不做全排列）+ 探针模式"
               "（每组双探针 short/long，<15 分钟快速评分）+ 决策闭环（方案书 4.2 映射表）")
    out.append("- 收敛矩阵：Round 1 显存×并发网格（9 组）→ Round 2 kv-dtype 补扫"
               "（2 组）→ Round 2.5 复活点验证（可选）→ Round 3 调度 A/B（6 组）"
               "→ 锁定；详见 `models-optimize-sheel/STEP_BY_STEP.md`")
    out.append("")

    # 2 调优结果
    out.append(section("2. 调优结果"))
    out.append(section("2.1 锁定配置（自动）", 2))
    if cmd:
        out.append("> 完整启动命令（来自 `best_serve_cmd.txt`）：")
        out.append("")
        out.append("```bash")
        out.append(cmd)
        out.append("```")
        out.append("")
        out.append(section("2.2 关键参数取值说明（自动 + 人工）", 2))
        known, other = [], []
        for key, value in params.items():
            desc = PARAM_DESC.get(key, "")
            if desc:
                known.append([key, fmt_opt(value), desc, TODO])
            else:
                other.append(key)
        if known:
            out.append(md_table(["参数", "取值", "作用", "取值依据（人工）"],
                                known))
        if other:
            out.append(f"> 未识别的参数（无内置说明）：{'、'.join(other)}，"
                       "作用与取值依据请在正文补充。")
        out.append("")
    else:
        out.append("> 收敛未完成（无 `best_serve_cmd.txt`），本节需人工填写。")
        out.append("")
    out.append(section("2.3 配置评分表（自动，按 score 降序）", 2))
    out.append(f"> 数据源：`{score_path}`（共 {len(rows)} 组）")
    out.append("")
    score_rows = [[r.get("_run") or "-", r.get("round") or "-",
                   r.get("config_id") or "-", r.get("verdict") or "-",
                   fmt_num(safe_float(r, "score"), 1) if safe_float(r, "score") is not None else "-",
                   r.get("short_thr") or "-", r.get("long_thr") or "-",
                   r.get("ttft_p99_ms") or "-", r.get("mem_pct_max") or "-",
                   r.get("gpu_util_mean") or "-",
                   (r.get("problems") or "-").replace("|", "\\|"),
                   (r.get("suggestions") or "-").replace("|", "\\|")] for r in rows]
    out.append(md_table(["run", "轮次", "配置", "verdict", "score",
                         "短吞吐", "长吞吐", "TTFT p99(ms)", "显存峰值%",
                         "GPU均%", "问题", "建议"], score_rows))
    out.append("")

    # 3 验收对比
    out.append(section("3. 验收对比（默认配置 vs 锁定配置）"))
    out.append(md_table(["指标", "before", "after", "变化"], acc_rows))
    out.append("")
    out.append("- 收益结论（人工复核）：TODO（吞吐/时延/显存的实际改善与业务意义）")
    out.append("")

    # 4 生产部署建议
    out.append(section("4. 生产部署建议"))
    out.append(section("4.1 服务器硬件资源配置建议（自动估算 + 人工复核）", 2))
    if hw_rows:
        out.append("> 基于锁定配置实测与经验公式估算，**需按实际业务负载复核**。")
        out.append("")
        out.append(md_table(["资源", "建议", "依据"], hw_rows))
    else:
        out.append("> 无锁定配置评分数据，本节需人工填写。")
    out.append("")
    out.append(section("4.2 生产环境部署建议（人工清单）", 2))
    out.append("- [ ] 部署形态：TODO（单机多卡张量并行 / 多实例负载均衡 / 容器化）")
    out.append("- [ ] 高可用：TODO（主备切换 / 多副本 + 流量分配）")
    out.append("- [ ] 健康检查与告警：TODO（`/v1/models` 探活、GPU 温度/显存/利用率阈值）")
    out.append("- [ ] 扩缩容策略：TODO（按吞吐目标与排队时延指标）")
    out.append("- [ ] 故障恢复：TODO（OOM 重启、卡故障替换、日志保留策略）")
    out.append("- [ ] 安全与合规：TODO（模型鉴权、接口白名单、数据不落盘）")
    out.append("")

    # 5 问题记录
    out.append(section("5. 调优过程问题与决策记录（自动）"))
    issues = [r for r in rows if (r.get("verdict") or "").upper() in ("REJECT", "WARN")]
    if issues:
        out.append("> 从评分表提取 REJECT/WARN 配置及其原因，作为调优过程证据。")
        out.append("")
        issue_rows = [[r.get("config_id") or "-", r.get("verdict") or "-",
                       (r.get("problems") or "-").replace("|", "\\|"),
                       (r.get("suggestions") or "-").replace("|", "\\|")]
                      for r in issues]
        out.append(md_table(["配置", "verdict", "问题", "建议"], issue_rows))
    else:
        out.append("> 无 REJECT/WARN 记录。")
    out.append("")

    # 附录
    out.append(section("附录 A 复现步骤"))
    out.append(f"> 完整操作手册见 `models-optimize-sheel/STEP_BY_STEP.md`；"
               f"收敛引擎规则与评分判据见 `probe_only.py` / `decision.py`。")
    out.append("")
    out.append(section("附录 B 原始数据索引"))
    out.append(md_table(["数据", "路径", "说明"], [
        ["配置评分表", fmt_opt(score_path), "所有配置逐组指标与判定"],
        ["锁定启动命令", fmt_opt(cmd_path), "最佳 vLLM serve 命令（交付物）"],
        ["收敛旅程回放", fmt_opt(os.path.join(args.opt_dir, "report.md")),
         "report.py 输出的完整回放（含引擎每步输出）"],
    ]))
    out.append("")

    if not args.output:
        out_name = f"tuning_guide_{vendor}.md" if vendor else "tuning_guide.md"
        args.output = out_name
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"已生成: {args.output}")


if __name__ == "__main__":
    main()
