"""参数调优指南自动生成脚本（客户版，零第三方依赖）。

读取 models-optimize-sheel/report.py 的输出（opt_summary.csv +
best_serve_cmd.txt）与 hardware_env.json，生成面向客户的
《参数调优指南》草稿——只呈现结果，不呈现调优过程：

  0 摘要 → 1 硬件信息 → 2 环境信息（含框架）→ 3 参数配置 →
  4 调优前后对比效果 → 5 提升指标与关键性能优化总结

调优过程的完整技术细节（收敛旅程回放 / 全量评分表 / REJECT 记录）保留在
models-optimize-sheel/report.py 输出的 report.md（面向技术）。

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
import os
import shlex

from generate_report import (TODO, fmt_num, fmt_opt, load_env, md_table,
                             section, safe_float)

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

ENV_ARG_MAP = {
    "tensor_parallel_size": "--tensor-parallel-size",
    "pipeline_parallel_size": "--pipeline-parallel-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "kv_cache_dtype": "--kv-cache-dtype",
    "enable_prefix_caching": "--enable-prefix-caching",
    "enable_chunked_prefill": "--enable-chunked-prefill",
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
        return fh.read().strip(), path


def pick_top1(rows):
    """ACCEPT 中 score 最高者（与 report.py 口径一致）。"""
    accepted = [r for r in rows if (r.get("verdict") or "").upper() == "ACCEPT"]
    accepted.sort(key=lambda r: safe_float(r, "score") or -1.0, reverse=True)
    return accepted[0] if accepted else None


def compare_p2(before_dir, after_dir):
    """验收对比：两个 run_matrix 输出目录（调优前默认配置 vs 调优后锁定配置）。"""
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


def pct_change(new, old):
    if old in (None, 0) or new is None:
        return None
    return (new - old) / old * 100


def render_acceptance(before, after, before_dir, after_dir):
    """返回 (对比表行, 收益摘要)。"""
    if not before or not after:
        return ([["验收对比数据",
                  fmt_opt(os.path.basename(before_dir) if before_dir else None),
                  fmt_opt(os.path.basename(after_dir) if after_dir else None),
                  TODO]], None)

    def fmt(v):
        return fmt_num(v, 1) if v is not None else "-"

    def cell(new, old, up):
        if old is None or new is None:
            return "-"
        chg = pct_change(new, old)
        if chg is None:
            return "-"
        good = (chg >= 0) if up else (chg <= 0)
        return f"{chg:+.1f}%{' ✓' if good else ''}"

    rows = [
        ["短文本中位吞吐 (tok/s)", fmt(before["thr_short"]),
         fmt(after["thr_short"]), cell(after["thr_short"],
                                        before["thr_short"], True)],
        ["长文本中位吞吐 (tok/s)", fmt(before["thr_long"]),
         fmt(after["thr_long"]), cell(after["thr_long"],
                                       before["thr_long"], True)],
        ["TTFT p99 (ms)", fmt(before["ttft_p99"]), fmt(after["ttft_p99"]),
         cell(after["ttft_p99"], before["ttft_p99"], False)],
        ["TPOT p99 (ms)", fmt(before["tpot_p99"]), fmt(after["tpot_p99"]),
         cell(after["tpot_p99"], before["tpot_p99"], False)],
        ["显存峰值 % (max)", fmt(before["mem_pct_max"]),
         fmt(after["mem_pct_max"]), cell(after["mem_pct_max"],
                                          before["mem_pct_max"], False)],
        ["GPU 平均利用率 % (mean)", fmt(before["gpu_util"]),
         fmt(after["gpu_util"]), cell(after["gpu_util"],
                                       before["gpu_util"], True)],
        ["OOM/失败组数", fmt_num(before["n_oom"], 0),
         fmt_num(after["n_oom"], 0), "-"],
        ["显存泄漏警告组数", fmt_num(before["leak_warn"], 0),
         fmt_num(after["leak_warn"], 0),
         "无" if after["leak_warn"] == 0 else "存在"],
    ]

    gains = []
    chg = pct_change(after["thr_short"], before["thr_short"])
    if chg is not None:
        gains.append(f"短文本中位吞吐 {chg:+.1f}%")
    chg = pct_change(after["ttft_p99"], before["ttft_p99"])
    if chg is not None:
        gains.append(f"TTFT p99 {chg:+.1f}%")
    chg = pct_change(after["mem_pct_max"], before["mem_pct_max"])
    if chg is not None:
        gains.append(f"显存峰值 {chg:+.1f}%")
    return rows, "；".join(gains) if gains else None


def main():
    p = argparse.ArgumentParser(description="参数调优指南自动生成脚本（客户版）")
    p.add_argument("--opt-dir", required=True,
                   help="report.py 输出目录（含 opt_summary.csv + best_serve_cmd.txt）")
    p.add_argument("--env", required=True, help="hardware_env.json 路径")
    p.add_argument("--p2-before", default=None,
                   help="调优前：默认配置 run_matrix 输出目录")
    p.add_argument("--p2-after", default=None,
                   help="调优后：锁定配置 run_matrix 输出目录")
    p.add_argument("--output", default=None,
                   help="输出路径（默认 tuning_guide_<vendor>.md）")
    args = p.parse_args()

    env = load_env(args.env)
    vendor = env.get("vendor") or ""
    rows, score_path = load_opt_summary(args.opt_dir)
    cmd, cmd_path = load_best_cmd(args.opt_dir)
    params = parse_serve_cmd(cmd or "")
    before, after = compare_p2(args.p2_before, args.p2_after)
    acc_rows, gain_summary = render_acceptance(
        before, after, args.p2_before, args.p2_after)

    model_cfg = ((env.get("model_deployment") or {}).get("model") or {})
    model_name = model_cfg.get("name") or "模型"
    acc = (env.get("hardware") or {}).get("accelerator") or {}
    server = (env.get("hardware") or {}).get("server") or {}
    stack = (env.get("hardware") or {}).get("software_stack") or {}
    meta = env.get("report_meta") or {}

    out = [f"# {model_name} 参数调优指南（{fmt_opt(vendor)}）", ""]

    # 0 摘要
    out.append(section("0. 摘要（一页结论）"))
    out.append(section("0.1 基本信息（自动）", 2))
    out.append(md_table(["项", "值"], [
        ["模型", fmt_opt(model_name)],
        ["厂商/芯片", f"{fmt_opt(vendor)} / {fmt_opt(acc.get('name'))}"],
        ["服务器", fmt_opt(server.get("model"))],
        ["报告编号", fmt_opt(meta.get("report_id"))],
        ["测试人", fmt_opt(meta.get("tester"))],
        ["日期", fmt_opt(meta.get("date"))],
    ]))
    out.append("")
    out.append(section("0.2 结论速览（自动）", 2))
    if cmd:
        out.append(md_table(["项目", "结论"], [
            ["锁定配置", fmt_opt(params.get("--max-num-seqs")) + " 并发 / "
             f"{fmt_opt(params.get('--max-model-len'))} 上下文 / "
             f"{fmt_opt(params.get('--kv-cache-dtype'))} KV Cache"],
            ["调优收益", gain_summary if gain_summary else TODO],
        ]))
    else:
        out.append("> 调优未完成（无 `best_serve_cmd.txt`），本节需人工填写。")
    out.append("")
    out.append(section("0.3 关键结论（人工）", 2))
    out.append("- 推荐配置是否可直接投产：TODO（可 / 有条件 / 不推荐）")
    out.append("- 主要收益与残余制约：TODO")
    out.append("")

    # 1 硬件信息
    out.append(section("1. 硬件信息（自动）"))
    out.append(section("1.1 服务器", 2))
    out.append(md_table(["项", "值"], [
        ["服务器型号", fmt_opt(server.get("model"))],
        ["CPU", fmt_opt(server.get("cpu"))],
        ["CPU 核数", fmt_num(server.get("cpu_cores"), 0)],
        ["内存", fmt_unit_gb(server.get("memory_gb"))],
        ["存储", fmt_opt(server.get("nvme"))],
    ]))
    out.append("")
    out.append(section("1.2 加速卡", 2))
    out.append(md_table(["项", "值"], [
        ["型号", fmt_opt(acc.get("name"))],
        ["架构", fmt_opt(acc.get("arch"))],
        ["显存容量", fmt_unit_gb(acc.get("hbm_capacity_gb"))],
        ["显存带宽", fmt_unit(acc.get("hbm_bandwidth_gbps"), "GB/s")],
        ["算力 (FP16/BF16/FP8)",
         " / ".join(filter(lambda s: s != TODO, [
             fmt_unit(acc.get("fp16_tflops"), "TFLOPS"),
             fmt_unit(acc.get("bf16_tflops"), "TFLOPS"),
             fmt_unit(acc.get("fp8_tflops"), "TFLOPS")])) or TODO],
        ["测试卡数", fmt_num(acc.get("cards_used"), 0)],
        ["单卡功耗", fmt_unit(acc.get("tdp_w"), "W")],
        ["互联", " / ".join(filter(lambda s: s != TODO, [
            fmt_opt(acc.get("interconnect_type")),
            fmt_opt(acc.get("interconnect_bandwidth"))])) or TODO],
    ]))
    out.append("")

    # 2 环境信息
    out.append(section("2. 环境信息（含框架）（自动）"))
    runtime = stack.get("runtime") or {}
    engine = stack.get("inference_engine") or {}
    out.append(md_table(["项", "值"], [
        ["操作系统", fmt_opt(server.get("os"))],
        ["内核", fmt_opt(server.get("kernel"))],
        ["容器环境", fmt_opt(server.get("docker"))],
        ["驱动版本", fmt_opt(stack.get("driver_version"))],
        ["运行时", " / ".join(filter(lambda s: s != TODO, [
            fmt_opt(runtime.get("name")),
            fmt_opt(runtime.get("version"))])) or TODO],
        ["推理引擎", " / ".join(filter(lambda s: s != TODO, [
            fmt_opt(engine.get("name")),
            fmt_opt(engine.get("version")),
            fmt_opt(engine.get("fork"))])) or TODO],
        ["计算库", fmt_opt(stack.get("compute_lib"))],
        ["容器镜像", fmt_opt(stack.get("container_image"))],
    ]))
    out.append("")

    # 3 参数配置
    out.append(section("3. 参数配置"))
    out.append(section("3.1 调优前后参数对比（自动 + 人工）", 2))
    if cmd:
        env_args = (env.get("model_deployment") or {}).get("engine_args") or {}
        rows3 = []
        for key in sorted(ENV_ARG_MAP, key=lambda k: -len(k)):
            flag = ENV_ARG_MAP[key]
            before_val = env_args.get(key)
            if before_val in (None, ""):
                before_val = TODO
            elif isinstance(before_val, bool):
                before_val = "是" if before_val else "否"
            after_val = params.get(flag)
            if after_val is None:
                after_val = "未设置"
            elif after_val is True:
                after_val = "开启"
            else:
                after_val = fmt_opt(after_val)
            if before_val == TODO and flag not in params:
                continue
            rows3.append([flag, before_val, after_val,
                          PARAM_DESC.get(flag, "")])
        rows3.sort(key=lambda r: (r[1] == r[2], r[1] == TODO))
        out.append(md_table(["参数", "调优前", "调优后", "作用"], rows3))
    else:
        out.append("> 调优未完成，本节需人工填写。")
    out.append("")
    out.append(section("3.2 调优后完整启动命令（自动）", 2))
    if cmd:
        out.append("```bash")
        out.append(cmd)
        out.append("```")
    else:
        out.append("> 调优未完成（无 `best_serve_cmd.txt`）。")
    out.append("")

    # 4 调优前后对比效果
    out.append(section("4. 调优前后对比效果"))
    out.append(section("4.1 关键指标对比（自动）", 2))
    out.append("> 数据源：调优前 `{0}`（默认配置）与调优后 `{1}`（锁定配置）"
               "的全量矩阵压测；表中 ✓ 表示该指标改善。".format(
                   fmt_opt(os.path.basename(args.p2_before))
                   if args.p2_before else TODO,
                   fmt_opt(os.path.basename(args.p2_after))
                   if args.p2_after else TODO))
    out.append("")
    out.append(md_table(["指标", "调优前", "调优后", "变化"], acc_rows))
    out.append("")
    out.append(section("4.2 收益结论（人工复核）", 2))
    if gain_summary:
        out.append(f"- 自动摘要：{gain_summary}")
    out.append("- 对业务的实际意义：TODO（时延达标 / 吞吐满足并发 / 显存余量）")
    out.append("")

    # 5 提升指标与关键性能优化总结
    out.append(section("5. 提升指标与关键性能优化总结"))
    out.append(section("5.1 关键提升指标（自动 + 人工补充）", 2))
    changed = [r for r in acc_rows if r[3] not in ("-", TODO)]
    if changed:
        out.append(md_table(["指标", "调优前", "调优后", "提升"], changed))
    else:
        out.append("> 暂无对比数据（未提供 `--p2-before/--p2-after`），"
                   "本节需人工填写。")
    out.append("")
    out.append("- 其他未量化收益：TODO（稳定性 / 长文本表现 / 部署便利性）")
    out.append("")
    out.append(section("5.2 优化要点总结（人工）", 2))
    out.append("- 核心调优点：TODO（如显存利用率、并发上限、KV Cache 精度等）")
    out.append("- 生产部署建议：TODO（卡数 / 多实例 / 监控告警阈值）")
    out.append("")

    # 数据来源
    out.append(section("附录 数据来源（内部存档）"))
    out.append("> 本报告由脚本自动生成，以下为生成依据的原始数据。")
    out.append("")
    out.append(md_table(["数据", "路径"], [
        ["调优评分表", fmt_opt(score_path)],
        ["锁定启动命令", fmt_opt(cmd_path)],
        ["调优前矩阵", fmt_opt(os.path.basename(args.p2_before)) if args.p2_before else TODO],
        ["调优后矩阵", fmt_opt(os.path.basename(args.p2_after)) if args.p2_after else TODO],
    ]))
    out.append("")

    if not args.output:
        out_name = f"tuning_guide_{vendor}.md" if vendor else "tuning_guide.md"
        args.output = out_name
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"已生成: {args.output}")


def fmt_unit_gb(value):
    return fmt_unit(value, "GB")


def fmt_unit(value, unit):
    if value in (None, ""):
        return TODO
    return f"{fmt_num(value, 0)} {unit}"


if __name__ == "__main__":
    main()