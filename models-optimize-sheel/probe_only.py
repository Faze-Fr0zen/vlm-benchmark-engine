"""Serve 参数调优探针执行器 + 收敛引擎（半自动模式，测试与服务分离）。

使用方式（配合优化手册 STEP_BY_STEP.md）：
  1. 按手册/上一轮输出，手动启动带目标 Serve 参数的 vllm serve
  2. 跑本脚本: python probe_only.py --config "mem=0.95,seqs=128,kv=auto"
  3. 脚本自动执行双探针（short 测调度上限 / long 测显存上限）+ GPU 采样
     （--gpu-csv 指向服务端旁路采样的 gpu_stats.csv），评分并追加评分表
  4. 收敛引擎自动判定：本轮进度、下一组启动参数（完整命令）、晋级结论

收敛矩阵（完整版，方案书 2.1/2.2）：
  Round 1  网格定位  9 组: mem(0.90/0.95/0.98) x seqs(64/128/256), kv=auto
  Round 2  维度补扫  2 组: top1 + kv=fp8（对照 top1+auto）
  Round 2.5 复活点  1~2 组: fp8 赢时验证大显存/高并发 REJECT 点
  Round 3  调度 A/B  6 组: chunked(off/on) x batched(4K/8K/16K)
  Round 4  验收      锁定配置 -> run_matrix.py 全量矩阵

每轮判据（decision.py 4.2 闭环 + 收敛规则）全部内置，跑完自动输出
"下一组启动命令"，无需思考。--lite 走 4090 小体量矩阵（8+4 组）。

结果: <result-dir>/<run-id>/summary.csv（每配置一行评分）。
"""

import argparse
import csv
import datetime as dt
import itertools
import json
import os
import sys

_TEST_SHEEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "models-test-sheel")
if _TEST_SHEEL not in sys.path:
    sys.path.insert(0, _TEST_SHEEL)

from run_matrix import (  # noqa: E402
    extract_metrics, probe_vllm_flags, build_command,
)
from serve_manager import config_to_flags  # noqa: E402
from decision import evaluate  # noqa: E402

RESULTS_DIR = "/vllm-workspace/results/vllm-optimize-result"
DEFAULT_SERVE_BASE = ("vllm serve /models/Qwen3-0.6B "
                      "--served-model-name qwen3-0.6b --dtype half "
                      "--max-model-len 32768")

# ---- 收敛矩阵定义 ---------------------------------------------------------
ROUND1_MEM = [0.90, 0.95, 0.98]
ROUND1_SEQS = [64, 128, 256]
ROUND2_KV = ["auto", "fp8"]
ROUND3_CHUNKED = [False, True]
ROUND3_BATCHED = [4096, 8192, 16384]
LITE_MEM = [0.90, 0.95]
LITE_SEQS = [64, 128]
LITE_BATCHED = [4096, 8192]

# 探针定义（双探针展开，供收敛引擎直接使用）
PROBE_SHORT = {"input_len": 1024, "output_len": 512, "request_rate": "inf",
               "num_prompts": 120, "max_concurrency": 64}
# long 输入 30K：30K 输入 + 1K 输出 = 31K < 默认服务 max-model-len 32K；
# 原 32K 输入 + 1K 输出会超限导致全部请求 400 失败
PROBE_LONG = {"input_len": 30000, "output_len": 1024, "request_rate": "1",
              "num_prompts": 16, "max_concurrency": 1}
PROBE_SHORT_LITE = dict(PROBE_SHORT, num_prompts=60)
PROBE_LONG_LITE = dict(PROBE_LONG, num_prompts=8)

SUMMARY_FIELDS = [
    "round", "config_id", "verdict", "score",
    "gpu_memory_utilization", "max_num_seqs", "kv_cache_dtype",
    "enable_chunked_prefill", "max_num_batched_tokens",
    "short_thr", "long_thr", "ttft_p99_ms", "mem_pct_max",
    "gpu_util_mean", "short_status", "long_status", "leak_warning",
    "problems", "suggestions", "timestamp",
]


# ---- 收敛矩阵生成 ---------------------------------------------------------

def round1_combos(lite=False):
    mems = LITE_MEM if lite else ROUND1_MEM
    seqs = LITE_SEQS if lite else ROUND1_SEQS
    return [{"gpu_memory_utilization": m, "max_num_seqs": s,
             "kv_cache_dtype": "auto"}
            for m, s in itertools.product(mems, seqs)]


def round2_combos(top1):
    return [dict(top1, kv_cache_dtype=kv) for kv in ROUND2_KV]


def round3_combos(top1, lite=False):
    batched = LITE_BATCHED if lite else ROUND3_BATCHED
    return [dict(top1, enable_chunked_prefill=ch, max_num_batched_tokens=b)
            for ch, b in itertools.product(ROUND3_CHUNKED, batched)]


def config_label(config):
    return "_".join(f"{k}={v}" for k, v in config.items())


def config_arg(config):
    return ",".join(f"{k}={v}" for k, v in config.items())


def serve_command(config, serve_base):
    return " ".join(serve_base.split() + config_to_flags(config))


# ---- 评分表读写 -----------------------------------------------------------

def load_summary(path):
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    return rows


def append_summary(path, row):
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(row)


def to_float(row, key):
    try:
        return float(row.get(key)) if row.get(key) not in ("", None) else None
    except (TypeError, ValueError):
        return None


def accept_rows(rows):
    return [r for r in rows
            if r.get("verdict") == "ACCEPT"
            and to_float(r, "score") is not None]


def top1(rows, phase_round):
    cand = [r for r in accept_rows(rows) if r.get("round") == phase_round]
    if not cand:
        return None
    cand.sort(key=lambda r: (to_float(r, "score") or -1,
                             to_float(r, "gpu_memory_utilization") or 0,
                             to_float(r, "max_num_seqs") or 0), reverse=True)
    return cand[0]


def config_from_row(row):
    cfg = {}
    for key in ("gpu_memory_utilization", "max_num_seqs", "kv_cache_dtype",
                "enable_chunked_prefill", "max_num_batched_tokens"):
        v = row.get(key)
        if v in ("", None):
            continue
        if key in ("max_num_seqs", "max_num_batched_tokens"):
            cfg[key] = int(v)
        elif key == "gpu_memory_utilization":
            cfg[key] = float(v)
        elif key == "enable_chunked_prefill":
            cfg[key] = (v == "True")
        else:
            cfg[key] = v
    return cfg


# ---- 收敛引擎 -------------------------------------------------------------

def next_action(rows, lite=False, serve_base=DEFAULT_SERVE_BASE):
    """读评分表，输出下一步动作。返回 (文本行列表, 退出码 0=继续 1=锁定期权)。"""
    lines = []

    def cid_of(round_tag, cfg):
        return f"R{round_tag}-" + config_label(cfg)

    # Round 1 是否完成
    r1_combos = round1_combos(lite)
    r1_done = {r.get("config_id") for r in rows if r.get("round") == "1"}
    pending_r1 = [c for c in r1_combos
                  if cid_of(1, c) not in r1_done]
    if pending_r1:
        cfg = pending_r1[0]
        lines.append(f"[Round 1] 已完成 {len(r1_done)}/{len(r1_combos)}，"
                     f"下一组（{len(pending_r1)} 组剩余）:")
        lines.append(f"  启动命令: {serve_command(cfg, serve_base)}")
        lines.append(f"  [判定] 启动后运行: python probe_only.py --config "
                     f"\"{config_arg(cfg)}\"")
        return lines, 0

    # Round 1 完成 -> 晋级判定
    lines.append(f"[Round 1] 完成 {len(r1_combos)} 组，收敛判定:")
    accepted = accept_rows([r for r in rows if r.get("round") == "1"])
    if not accepted:
        lines.append("  [结论] 全部 REJECT——显存/并发均不满足，按建议降档：")
        lines.append("    ① 检查服务是否正常启动（serve_logs 与 /v1/models）")
        lines.append("    ② 全部 OOM 则手工降低 max-num-seqs 到 32 重测一组验证")
        return lines, 1
    top = top1(rows, "1")
    rejected = [r for r in rows if r.get("round") == "1"
                and r.get("verdict") == "REJECT"]
    if rejected:
        mems = sorted({r.get("gpu_memory_utilization") for r in rejected})
        lines.append(f"  [显存天花板] REJECT 组合出现在 mem={mems}，"
                     f"该档以上不再尝试")
    lines.append(f"  [top1] {top['config_id']} score={top['score']}")
    top_cfg = config_from_row(top)

    # Round 2: top1 上补扫 kv-dtype
    # 完成判定放宽：auto 对照组复用 R1 top1 记录；fp8 组按 seqs 匹配
    # （显存不足时 mem 可能被手动下调，如 0.95->0.9，容忍该差异）
    r2_combos = round2_combos(top_cfg)
    r2_auto_done = top is not None
    r2_fp8_done = any(r.get("round") == "2"
                      and r.get("kv_cache_dtype") == "fp8"
                      and str(r.get("max_num_seqs")) == str(top_cfg.get("max_num_seqs"))
                      for r in rows)
    pending_r2 = [c for c in r2_combos
                  if not (c["kv_cache_dtype"] == "auto" and r2_auto_done)
                  and not (c["kv_cache_dtype"] == "fp8" and r2_fp8_done)]
    if pending_r2:
        cfg = pending_r2[0]
        lines.append(f"[Round 2] kv-dtype 补扫（{len(pending_r2)} 组剩余）:")
        lines.append(f"  启动命令: {serve_command(cfg, serve_base)}")
        lines.append(f"  [判定] 启动后运行: python probe_only.py --config "
                     f"\"{config_arg(cfg)}\"")
        return lines, 0

    # Round 2 完成 -> fp8 胜负判定 -> Round 2.5 或 Round 3
    fp8_row = next((r for r in rows
                    if r.get("round") == "2"
                    and r.get("kv_cache_dtype") == "fp8"), None)
    auto_row = next((r for r in rows
                     if r.get("round") == "2"
                     and r.get("kv_cache_dtype") == "auto"), None)
    if auto_row is None:
        auto_row = next((r for r in rows
                         if r.get("config_id") == top["config_id"]), None)
    if fp8_row and auto_row and \
            (to_float(fp8_row, "score") or -1) > (to_float(auto_row, "score") or -1):
        lines.append(f"[Round 2] fp8 胜出（{fp8_row['score']} vs {auto_row['score']}），")
        rejected = [r for r in rows if r.get("round") == "1"
                    and r.get("verdict") == "REJECT"]
        if rejected:
            lines.append("  进入 Round 2.5 复活点验证:")
            lines.append(f"  启动命令: {serve_command(dict(top_cfg, kv_cache_dtype='fp8'), serve_base)}")
            lines.append("  [判定] 以 fp8 重测 Round 1 中 REJECT 的大显存/高并发点")
            return lines, 1
        lines.append("  R1 无 REJECT 点，跳过 Round 2.5，进入 Round 3 调度 A/B")
        top_cfg = dict(top_cfg, kv_cache_dtype="fp8")

    # fp8 未胜出 -> Round 3 调度 A/B
    if not fp8_row and not auto_row:
        lines.append("  [warn] Round 2 无评分记录，跳过 fp8 判定")
    lines.append(f"[Round 3] 调度 A/B 控制变量（top1 = {top['config_id']}）:")
    r3_combos = round3_combos(top_cfg, lite)

    def _r3_key(cfg):
        return (str(cfg.get("max_num_seqs")), str(cfg.get("kv_cache_dtype")),
                str(cfg.get("enable_chunked_prefill")),
                str(cfg.get("max_num_batched_tokens")))

    r3_done_keys = {_r3_key(config_from_row(r))
                    for r in rows if r.get("round") == "3"}
    pending_r3 = [c for c in r3_combos if _r3_key(c) not in r3_done_keys]
    if pending_r3:
        cfg = pending_r3[0]
        lines.append(f"  （{len(pending_r3)} 组剩余）")
        lines.append(f"  启动命令: {serve_command(cfg, serve_base)}")
        lines.append(f"  [判定] 启动后运行: python probe_only.py --config "
                     f"\"{config_arg(cfg)}\"")
        return lines, 0

    # Round 3 完成 -> 锁定
    best = top1(rows, "3")
    best_r1 = top1(rows, "1")
    if best and config_from_row(best).get("max_num_batched_tokens") \
            == max(ROUND3_BATCHED):
        lines.append(f"  [提示] Round 3 最优 batched 触到扫描上限 "
                     f"{max(ROUND3_BATCHED)}，可手工扩展一轮验证")
    if best and (to_float(best, "score") or 0) > (to_float(best_r1, "score") or 0):
        final_cfg = config_from_row(best)
        lines.append(f"[Round 3] B 最优胜出: {best['config_id']} "
                     f"score={best['score']}（baseline {best_r1['score']}）")
    else:
        final_cfg = config_from_row(best_r1)
        lines.append(f"[Round 3] B 未超过 baseline，锁定 Round1 top1: "
                     f"{best_r1['config_id']}")
    lines.append("[锁定] 最终 serve 命令:")
    lines.append(f"  {serve_command(final_cfg, serve_base)}")
    lines.append("[验收] 用该命令启动服务后，跑全量矩阵:")
    lines.append("  python run_matrix.py --gpu-csv <服务端 gpu_stats.csv>")
    return lines, 1


# ---- 探针执行 -------------------------------------------------------------

def run_probes(args, probe_dir):
    import argparse as ap
    import subprocess
    import time as _time

    bench_args = ap.Namespace(
        model=args.model, tokenizer=args.tokenizer,
        base_url=args.base_url, vllm_cmd=args.vllm_cmd,
        timeout=args.probe_timeout)
    flags, _ = probe_vllm_flags(bench_args.vllm_cmd.split())
    os.makedirs(probe_dir, exist_ok=True)

    results = {}
    for name, probe in (("short", PROBE_SHORT_LITE if args.lite
                         else PROBE_SHORT),
                        ("long", PROBE_LONG_LITE if args.lite
                         else PROBE_LONG)):
        p_dir = os.path.join(probe_dir, name)
        os.makedirs(p_dir, exist_ok=True)
        case = {"phase": "probe", "input_len": probe["input_len"],
                "output_len": probe["output_len"],
                "request_rate": probe["request_rate"]}
        cmd = build_command(case, probe["num_prompts"],
                            probe["max_concurrency"], bench_args, flags,
                            p_dir)
        print(f"  [probe:{name}] {' '.join(cmd)}")
        t0 = _time.time()
        log = os.path.join(p_dir, f"{name}.log")
        status = "FAILED"
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=bench_args.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                status = "TIMEOUT"
            else:
                status = "OK" if proc.returncode == 0 \
                    else f"FAILED(rc={proc.returncode})"
        metrics = {"probe": name, "status": status,
                   "duration_s": round(_time.time() - t0, 1)}
        result_path = os.path.join(p_dir, "result.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as fh:
                    metrics.update(extract_metrics(json.load(fh)))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  [warn] 解析 {result_path} 失败: {exc}")
        if status != "OK":
            tail = ""
            with open(log, "r", encoding="utf-8", errors="replace") as fh:
                tail = "\n".join(fh.readlines()[-80:])
            if "out of memory" in tail.lower() or "oom" in tail.lower():
                status = "OOM"
            metrics["status"] = status
        results[name] = metrics
    return results


def gpu_window_stats(gpu_csv, start_ts, end_ts):
    """从服务端 gpu_stats.csv 按窗口聚合 GPU 指标（与 run_matrix 一致）。"""
    from run_matrix import aggregate_gpu_window
    m, _mx, mem, leak = aggregate_gpu_window(gpu_csv, start_ts, end_ts)
    return m, mem, leak


# ---- 主流程 ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Serve 参数调优探针执行器（半自动 + 收敛引擎）")
    p.add_argument("--config", required=True,
                   help="当前服务的参数，逗号分隔，如 "
                        "mem=0.95,seqs=128,kv=auto（round 自动识别）")
    p.add_argument("--round", default=None,
                   help="手动指定轮次（缺省按 config 内容自动识别: kv 变化=2，"
                        "chunked/batched=3，其余=1）")
    p.add_argument("--model", default="qwen3-0.6b", help="模型名")
    p.add_argument("--tokenizer", default="/models/Qwen3-0.6B",
                   help="tokenizer 路径")
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="vLLM 服务地址")
    p.add_argument("--vllm-cmd", default="vllm", help="vllm 可执行文件")
    p.add_argument("--serve-base", default=DEFAULT_SERVE_BASE,
                   help="基础 serve 命令（收敛引擎输出完整启动命令时拼前缀）")
    p.add_argument("--gpu-csv", default=None,
                   help="服务端旁路采样的 gpu_stats.csv（sys_monitor.py "
                        "--standalone --clocks-ecc 输出），探针窗口聚合用")
    p.add_argument("--result-dir", default=None,
                   help="结果根目录（默认 RESULTS_DIR）")
    p.add_argument("--run-id", default=None, help="运行标识（缺省按时间）")
    p.add_argument("--probe-timeout", type=int, default=1500,
                   help="单探针超时秒数")
    p.add_argument("--lite", action="store_true", help="4090 小体量矩阵")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印收敛引擎建议，不跑探针")
    return p.parse_args()


def parse_config(spec):
    cfg = {}
    for part in spec.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        key = {"mem": "gpu_memory_utilization", "seqs": "max_num_seqs",
               "kv": "kv_cache_dtype", "chunked": "enable_chunked_prefill",
               "batched": "max_num_batched_tokens"}.get(k, k)
        if key in ("gpu_memory_utilization",):
            cfg[key] = float(v)
        elif key in ("max_num_seqs", "max_num_batched_tokens"):
            cfg[key] = int(v)
        elif key == "enable_chunked_prefill":
            cfg[key] = v.lower() in ("on", "true", "1")
        else:
            cfg[key] = v
    return cfg


def detect_round(cfg):
    if "enable_chunked_prefill" in cfg or "max_num_batched_tokens" in cfg:
        return "3"
    if cfg.get("kv_cache_dtype") not in (None, "auto"):
        return "2"
    return "1"


def main():
    args = parse_args()
    cfg = parse_config(args.config)
    if not cfg:
        sys.exit(f"错误：无法解析 --config '{args.config}'，格式如 "
                 f"mem=0.95,seqs=128,kv=auto")
    round_tag = args.round or detect_round(cfg)
    cid = f"R{round_tag}-" + config_label(cfg)
    run_root = args.result_dir or RESULTS_DIR
    run_id = args.run_id or dt.datetime.now().strftime("run-opt-%Y%m%d")
    run_dir = os.path.join(run_root, run_id)
    summary_path = os.path.join(run_dir, "summary.csv")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"config": cfg, "config_id": cid, "round": round_tag},
                  fh, ensure_ascii=False, indent=2)

    rows = load_summary(summary_path)
    if any(r.get("config_id") == cid for r in rows):
        print(f"[{cid}] 已有评分记录，跳过探针（--resume 逻辑内置）")
    elif not args.dry_run:
        print(f"\n{'='*72}\n[{cid}] 双探针执行中...")
        print(f"  请确认服务已按此参数启动: "
              f"{serve_command(cfg, args.serve_base)}")
        t0 = dt.datetime.now()
        results = run_probes(args, os.path.join(run_dir, cid))
        t1 = dt.datetime.now()
        gpu_util, mem_pct, leak = gpu_window_stats(
            args.gpu_csv, t0.isoformat(), t1.isoformat())
        for name, m in results.items():
            m["gpu_util_mean"] = gpu_util
            m["gpu_mem_pct_max"] = mem_pct
            m["leak_warning"] = leak
        verdict = evaluate(results)
        row = {"round": round_tag, "config_id": cid,
               "verdict": verdict["verdict"],
               "score": f"{verdict['score']:.1f}"
                        if verdict["score"] is not None else "",
               "gpu_memory_utilization": cfg.get("gpu_memory_utilization"),
               "max_num_seqs": cfg.get("max_num_seqs"),
               "kv_cache_dtype": cfg.get("kv_cache_dtype"),
               "enable_chunked_prefill": cfg.get("enable_chunked_prefill"),
               "max_num_batched_tokens": cfg.get("max_num_batched_tokens"),
               "short_thr": results["short"].get("output_throughput"),
               "long_thr": results["long"].get("output_throughput"),
               "ttft_p99_ms": verdict["summary"].get("ttft_p99_ms"),
               "mem_pct_max": verdict["summary"].get("mem_pct_max"),
               "gpu_util_mean": verdict["summary"].get("gpu_util_mean"),
               "short_status": results["short"].get("status"),
               "long_status": results["long"].get("status"),
               "leak_warning": verdict["summary"].get("leak_warning"),
               "problems": " | ".join(verdict["problems"]),
               "suggestions": " | ".join(verdict["suggestions"]),
               "timestamp": dt.datetime.now().isoformat(timespec="seconds")}
        append_summary(summary_path, row)
        rows.append(row)
        print(f"\n[{cid}] verdict={row['verdict']} "
              f"score={row['score'] or '-'}")
        for s in verdict["suggestions"]:
            print(f"    建议: {s}")
        for p in verdict["problems"]:
            print(f"    问题: {p}")
    else:
        print("[dry-run] 跳过探针执行")

    print(f"\n{'='*72}\n[收敛引擎] 读取评分表（{len(rows)} 条记录）:")
    lines, exit_code = next_action(rows, lite=args.lite,
                                   serve_base=args.serve_base)
    for line in lines:
        print(line)
    print(f"评分表: {summary_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()