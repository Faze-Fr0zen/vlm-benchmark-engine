"""模型基准测试报告自动生成脚本（零第三方依赖）。

读取 P2 压测结果（summary.csv + run_config.json + metadata）与可选的
P3 稳定性结果（stability_report.json + monitor_events.log），结合
hardware_env.json 环境信息，生成一份 Markdown 报告草稿：
  环境表 / 模型与部署配置 / 矩阵结果总览 / 核心指标表（每组合取推荐
  速率点）/ 膝点与饱和点判定 / 稳定性红线结果 / 效率指标 / 附录索引。
人工结论章节留 TODO 占位，由测试人员补填。

判定口径与 plot_results.py 一致：
  - 排队临界 rate：P99 TTFT 首次 >= 最低速率基线 2 倍
  - 吞吐饱和 rate：output throughput 首次达峰值 95%
  - 推荐速率 = 二者较小者；无则取该组合最大吞吐的速率点

用法示例::

    python generate_report.py --env hardware_env.json \\
        --run /vllm-workspace/results/vllm-bench-result/run-20260814-023631 \\
        --stability /vllm-workspace/results/vllm-bench-result/run-20260817-052818 \\
        --figures-dir figures --output report.md

依赖：Python 3.9+ 标准库。
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

TODO = "TODO"


def fmt_num(value, digits=1):
    if value is None:
        return TODO
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return TODO
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def fmt_opt(value):
    if value is None or value == "":
        return TODO
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def fmt_unit(value, unit):
    v = fmt_opt(value)
    return TODO if v == TODO else f"{v} {unit}"


def load_env(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_summary(run_dir):
    path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(path):
        print(f"[warn] 未找到 {path}，P2 结果为空")
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("input_len")]


def load_json_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] 解析失败 {path}: {exc}")
        return None


def to_rate(value):
    if isinstance(value, str) and value.strip().lower() in ("inf", "infinity"):
        return float("inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rate_label(rate):
    if rate is None:
        return TODO
    return "inf" if math.isinf(rate) else str(int(rate) if rate == int(rate) else rate)


def safe_float(row, key):
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def analyze_rows(rows):
    """按 (phase, in, out) 分组，每组按速率排序，计算推荐点/膝点/峰值。"""
    combos = {}
    for r in rows:
        key = (r.get("phase") or "short", int(r["input_len"]), int(r["output_len"]))
        combos.setdefault(key, []).append(r)
    for key in combos:
        combos[key].sort(key=lambda r: to_rate(r.get("request_rate")) or 0.0)

    results = []
    for (phase, in_len, out_len), series in sorted(combos.items()):
        pairs_ttft = [(to_rate(r.get("request_rate")), safe_float(r, "p99_ttft_ms"))
                      for r in series]
        pairs_thr = [(to_rate(r.get("request_rate")), safe_float(r, "output_throughput"))
                     for r in series]
        onset = queue_onset(pairs_ttft, pairs_thr)
        best = max(series, key=lambda r: safe_float(r, "output_throughput") or 0.0)
        if onset is not None:
            pick = next((r for r in series
                         if (to_rate(r.get("request_rate")) or 0.0) == onset), best)
        else:
            pick = best
        results.append({
            "phase": phase, "input_len": in_len, "output_len": out_len,
            "onset": onset, "pick": pick, "series": series,
            "peak_thr": safe_float(best, "output_throughput"),
        })
    return results


def queue_onset(ttft_pairs, thr_pairs):
    """排队临界 rate：TTFT 突变点或吞吐饱和点，取较小者（与 plot_results.py 一致）。"""
    finite = [p for p in ttft_pairs if p[0] is not None and not math.isinf(p[0])
              and p[1] is not None]
    if finite and finite[0][1] > 0:
        base = finite[0][1]
        for rate, val in finite[1:]:
            if val >= base * 2:
                ttft_onset = rate
                break
        else:
            ttft_onset = None
    else:
        ttft_onset = None
    fthr = [p for p in thr_pairs if p[0] is not None and not math.isinf(p[0])
            and p[1] is not None and p[1] > 0]
    if len(fthr) >= 2:
        peak = max(v for _, v in fthr)
        thr_onset = next((rate for rate, val in fthr if val >= peak * 0.95), None)
    else:
        thr_onset = None
    cands = [x for x in (ttft_onset, thr_onset) if x is not None]
    return min(cands) if cands else None


def status_counts(rows):
    from collections import Counter
    return Counter((r.get("status") or "?").split("(")[0] for r in rows)


def phase_peak(analyses, phase):
    peaks = [a["peak_thr"] for a in analyses if a["phase"] == phase
             and a["peak_thr"] is not None]
    return max(peaks) if peaks else None


def gpu_peak_util(run_dir):
    """扫描 run 目录下所有组 gpu_stats.csv，返回全局 GPU 利用率峰值。"""
    peak = None
    for phase_dir in ("short", "long"):
        base = os.path.join(run_dir, phase_dir)
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            path = os.path.join(base, name, "gpu_stats.csv")
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        v = _first_number(row.get("util_pct") or "")
                        if v is not None:
                            peak = max(peak, v) if peak is not None else v
            except OSError:
                continue
    return peak


def _first_number(field):
    m = re.search(r"-?\d+(?:\.\d+)?", field)
    return float(m.group()) if m else None


def ttft_ratio_worst(analyses):
    """所有组合中「推荐速率点 P99 TTFT / 最低速率基线」的最大倍数。"""
    worst = None
    for a in analyses:
        series = a["series"]
        finite = [safe_float(r, "p99_ttft_ms") for r in series]
        finite = [v for v in finite if v is not None and v > 0]
        if len(finite) < 2:
            continue
        base = finite[0]
        pick_ttft = safe_float(a["pick"], "p99_ttft_ms")
        if pick_ttft is None:
            continue
        ratio = pick_ttft / base
        worst = max(worst, ratio) if worst is not None else ratio
    return worst


def acceptance_eval(analyses, rows, stability, n_err, short_peak, long_peak,
                    gpu_peak, lines):
    """按 acceptance_lines 逐项判定，返回 (表行, 是否全部达标)。"""
    if not lines:
        return None, None

    def line(label, actual, threshold, unit="", reverse=False, digits=2):
        def fmt(v):
            if v is None:
                return TODO
            return fmt_num(v, digits) if isinstance(v, (int, float)) else fmt_opt(v)

        if threshold is None:
            return [label, fmt(actual) + (f" {unit}" if unit and actual is not None else ""),
                    TODO, TODO]
        if actual is None:
            verdict = TODO
        else:
            ok = actual <= threshold if not reverse else actual >= threshold
            verdict = "达标" if ok else "**不达标**"
        return [label,
                (fmt(actual) + f" {unit}").strip() if actual is not None else TODO,
                (fmt(threshold) + f" {unit}").strip(), verdict]

    rows_out = [
        line("短文本极限吞吐 (tok/s)", short_peak, lines.get("short_peak_tok_s"), "tok/s", reverse=True),
        line("长文本极限吞吐 (tok/s)", long_peak, lines.get("long_peak_tok_s"), "tok/s", reverse=True),
        line("推荐点 P99 TTFT / 基线最大倍数", ttft_ratio_worst(analyses),
             lines.get("p99_ttft_ratio_max"), "×"),
    ]
    total_req = sum(safe_float(r, "num_requests") or 0 for r in rows)
    total_fail = sum(safe_float(r, "num_failed") or 0 for r in rows)
    fail_rate = (total_fail / total_req * 100) if total_req else None
    rows_out.append(line("请求失败率 (%)", fail_rate,
                         lines.get("max_failure_rate_pct"), "%"))
    bad = sum(1 for r in rows if r.get("status") not in ("OK", ""))
    rows_out.append(line("异常组数 (FAILED/OOM/TIMEOUT)", bad,
                         lines.get("max_bad_groups"), "组", digits=0))

    redlines = "未测"
    if stability is not None:
        trends = stability.get("trends") or {}
        passed = all(v is None or v is False for v in trends.values()) and n_err == 0
        redlines = "全部通过" if passed else "存在触发"
    redline_line = ["P3 稳定性红线", redlines, TODO, TODO]
    want = lines.get("stability_redlines_ok")
    if want is not None:
        redline_line[2] = "全部通过" if want else "不要求"
        if stability is not None:
            redline_line[3] = "达标" if (redlines == "全部通过") == want else "**不达标**"
    rows_out.append(redline_line)
    rows_out.append(line("GPU 利用率峰值 (%)", gpu_peak,
                         lines.get("min_gpu_util_pct"), "%", reverse=True))

    defined = [r for r in rows_out if r[2] != TODO]
    all_pass = defined and all(r[3] == "达标" for r in defined)
    return rows_out, all_pass


def count_events(log_path):
    errors = warns = 0
    if not log_path or not os.path.exists(log_path):
        return errors, warns
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in csv.reader(fh):
            if len(line) >= 2:
                level = line[1].upper()
                if level == "ERROR":
                    errors += 1
                elif level == "WARN":
                    warns += 1
    return errors, warns


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [str(c) if c is not None else "" for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section(title, level=1):
    return f"\n{'#' * level} {title}\n"


def gen_report(args, env):
    rows = load_summary(args.run)
    analyses = analyze_rows(rows)
    run_config = load_json_if_exists(os.path.join(args.run, "run_config.json"))
    metas = sorted(glob.glob(os.path.join(args.run, "metadata_*.json")))
    meta = load_json_if_exists(metas[-1]) if metas else None
    stability = load_json_if_exists(
        os.path.join(args.stability, "stability_report.json")) if args.stability else None

    out = []
    out.append("# 模型基准测试报告（自动生成草稿）")
    out.append("")
    out.append("> 本文件由 `generate_report.py` 自动生成，机器可读部分已填充，")
    out.append("> 人工结论章节（TODO）需测试人员补填后发布。")
    out.append("")
    out.append("| 字段 | 内容 |")
    out.append("|---|---|")
    vendor = (env.get("vendor") or TODO)
    out.append(f"| 报告编号 | TODO |")
    acc0 = (env.get("hardware") or {}).get("accelerator") or {}
    model0 = (env.get("model_deployment") or {}).get("model") or {}
    out.append(f"| 被测厂商 / 芯片 | {vendor} / {fmt_opt(acc0.get('name'))} |")
    out.append(f"| 被测模型 | {fmt_opt(model0.get('name'))} |")
    out.append(f"| 测试日期 | {fmt_opt((env.get('report_meta') or {}).get('date'))} |")
    out.append(f"| 测试人 | {fmt_opt((env.get('report_meta') or {}).get('tester'))} |")
    out.append("")

    # 0 摘要
    out.append(section("0. 摘要（一页结论）"))
    out.append(section("0.1 被测对象", 2))
    acc = (env.get("hardware") or {}).get("accelerator") or {}
    stack = (env.get("hardware") or {}).get("software_stack") or {}
    model = (env.get("model_deployment") or {}).get("model") or {}
    ea = (env.get("model_deployment") or {}).get("engine_args") or {}
    ba = (env.get("model_deployment") or {}).get("bench_args") or {}
    engine0 = stack.get("inference_engine") or {}
    out.append(md_table(["项", "值"], [
        ["加速卡", f"{fmt_opt(acc.get('name'))} × {fmt_opt(acc.get('cards_used'))} 卡"],
        ["软件栈", f"{fmt_opt(engine0.get('name'))} {fmt_opt(engine0.get('version'))}"],
        ["模型", f"{fmt_opt(model.get('name'))}（{fmt_unit(model.get('params_b'), 'B')} / {fmt_opt(model.get('quantization'))}）"],
        ["部署", f"TP={fmt_opt(ea.get('tensor_parallel_size'))}"],
    ]))
    out.append("")
    short_peak = phase_peak(analyses, "short")
    long_peak = phase_peak(analyses, "long")
    out.append(section("0.2 核心结论", 2))
    out.append(md_table(["项目", "结论"], [
        ["极限吞吐（tok/s，短文本）", fmt_num(short_peak)],
        ["极限吞吐（tok/s，长文本）", fmt_num(long_peak)],
        ["推荐请求速率 / 并发", f"见 3.4 各组合膝点；{TODO}"],
        ["P99 TTFT / TPOT（推荐点）", f"{TODO} / {TODO}"],
        ["稳定性结论", f"{TODO}（24h 红线见第 4 章）"],
        ["综合评级", TODO],
    ]))
    out.append("")

    # 1 环境
    out.append(section("1. 测试环境"))
    server = (env.get("hardware") or {}).get("server") or {}
    out.append(section("1.1 服务器硬件", 2))
    out.append(md_table(["项", "值"], [
        ["服务器型号", fmt_opt(server.get("model"))],
        ["CPU", fmt_opt(server.get("cpu"))],
        ["内存", fmt_unit(server.get("memory_gb"), "GB")],
        ["存储", fmt_opt(server.get("nvme"))],
        ["操作系统 / 内核", f"{fmt_opt(server.get('os'))} / {fmt_opt(server.get('kernel'))}"],
        ["容器运行时", fmt_opt(server.get("docker"))],
    ]))
    out.append("")
    out.append(section("1.2 加速卡", 2))
    out.append(md_table(["项", "值"], [
        ["芯片型号", fmt_opt(acc.get("name"))],
        ["架构 / 制程", f"{fmt_opt(acc.get('arch'))} / {fmt_unit(acc.get('process_nm'), 'nm')}"],
        ["算力 FP16/BF16/FP8/INT8", f"{fmt_num(acc.get('fp16_tflops'))} / {fmt_num(acc.get('bf16_tflops'))} / {fmt_num(acc.get('fp8_tflops'))} / {fmt_num(acc.get('int8_tops'))} TFLOPS/TOPS"],
        ["显存容量 / 带宽", f"{fmt_unit(acc.get('hbm_capacity_gb'), 'GB')} / {fmt_unit(acc.get('hbm_bandwidth_gbps'), 'GB/s')}"],
        ["单卡 TDP", fmt_unit(acc.get("tdp_w"), "W")],
        ["装机 / 用卡数", f"{fmt_opt(acc.get('cards_installed'))} / {fmt_opt(acc.get('cards_used'))}"],
        ["卡间互联", f"{fmt_opt(acc.get('interconnect_type'))} / {fmt_opt(acc.get('interconnect_bandwidth'))}"],
        ["拓扑说明", fmt_opt(acc.get("topology_note"))],
    ]))
    out.append("")
    out.append(section("1.3 软件栈", 2))
    runtime = stack.get("runtime") or {}
    engine = stack.get("inference_engine") or {}
    out.append(md_table(["层", "名称", "版本", "备注"], [
        ["驱动", fmt_opt(stack.get("driver_version")), "", ""],
        ["运行时", fmt_opt(runtime.get("name")), fmt_opt(runtime.get("version")), ""],
        ["推理引擎", fmt_opt(engine.get("name")), f"{fmt_opt(engine.get('version'))}（{fmt_opt(engine.get('fork'))}）", fmt_opt(engine.get("notes"))],
        ["容器镜像", fmt_opt(stack.get("container_image")), "", ""],
        ["计算库", fmt_opt(stack.get("compute_lib")), "", ""],
    ]))
    out.append("")
    probe = (env.get("hardware") or {}).get("gpu_probe") or {}
    out.append(section("1.5 工具链", 2))
    out.append(md_table(["项", "值"], [
        ["压测脚本版本", f"{TODO}（git commit）"],
        ["压测工具", f"{fmt_opt(ba.get('bench_tool'))} {fmt_opt(ba.get('bench_tool_version'))}"],
        ["监控方式", f"sys_monitor.py，GPU 探测={fmt_opt(probe.get('short_name'))}，间隔{TODO} s"],
        ["结果目录", args.run],
    ]))
    out.append("")

    # 2 模型与部署
    out.append(section("2. 被测模型与部署配置"))
    out.append(section("2.1 模型信息", 2))
    out.append(md_table(["项", "值"], [
        ["模型名称", fmt_opt(model.get("name"))],
        ["参数量", fmt_unit(model.get("params_b"), "B")],
        ["量化", fmt_opt(model.get("quantization"))],
        ["上下文长度", fmt_opt(model.get("context_len"))],
        ["Tokenizer", fmt_opt(model.get("tokenizer"))],
        ["权重路径", fmt_opt(model.get("model_path"))],
    ]))
    out.append("")
    out.append(section("2.2 推理引擎参数", 2))
    ea = (env.get("model_deployment") or {}).get("engine_args") or {}
    out.append(md_table(["参数", "值"], [
        ["tensor-parallel-size", fmt_opt(ea.get("tensor_parallel_size"))],
        ["pipeline-parallel-size", fmt_opt(ea.get("pipeline_parallel_size"))],
        ["gpu-memory-utilization", fmt_opt(ea.get("gpu_memory_utilization"))],
        ["max-num-seqs", fmt_opt(ea.get("max_num_seqs"))],
        ["max-num-batched-tokens", fmt_opt(ea.get("max_num_batched_tokens"))],
        ["enable-chunked-prefill", fmt_opt(ea.get("enable_chunked_prefill"))],
        ["kv-cache-dtype", fmt_opt(ea.get("kv_cache_dtype"))],
        ["enable-prefix-caching", fmt_opt(ea.get("enable_prefix_caching"))],
        ["其他 flag", fmt_opt(ea.get("other_flags"))],
    ]))
    out.append("")
    ba = (env.get("model_deployment") or {}).get("bench_args") or {}
    out.append(section("2.3 压测参数", 2))
    out.append(md_table(["参数", "值"], [
        ["请求数（短/长）", f"{fmt_opt(ba.get('num_prompts_short'))} / {fmt_opt(ba.get('num_prompts_long'))}"],
        ["并发上限（短/长）", f"{fmt_opt(ba.get('max_concurrency_short'))} / {fmt_opt(ba.get('max_concurrency_long'))}"],
        ["速率档位", f"short={TODO}，long={TODO}"],
        ["强制 flag", "--ignore-eos 等"],
    ]))
    out.append("")

    # 3 P2 结果
    out.append(section("3. 基准矩阵结果（Phase 2）"))
    counts = status_counts(rows)
    total = sum(counts.values())
    out.append(section("3.2 结果总览", 2))
    out.append(md_table(["状态", "组数"], [
        [k, v] for k, v in sorted(counts.items())
    ] + [["合计", total]]))
    out.append("")
    failed = [r for r in rows if r.get("status") not in ("OK", "")]
    if failed:
        out.append("失败/异常组明细：")
        out.append("")
        out.append(md_table(["组", "状态", "原因"], [
            [f"in{r['input_len']}_out{r['output_len']}_rate{r['request_rate']}",
             r.get("status") or "", (r.get("error") or "")[:80]] for r in failed]))
        out.append("")
    out.append(section("3.3 核心指标表（每组合推荐速率点）", 2))
    out.append("> 推荐速率 = 膝点/饱和点（3.4 判定口径）；全量逐组数据见附录 B。")
    out.append("")
    rows3 = []
    for a in analyses:
        p = a["pick"]
        rows3.append([
            a["input_len"], a["output_len"], rate_label(a["onset"]),
            fmt_num(safe_float(p, "output_throughput")),
            fmt_num(safe_float(p, "request_throughput"), 3),
            fmt_num(safe_float(p, "p99_ttft_ms")),
            fmt_num(safe_float(p, "p99_tpot_ms")),
            fmt_num(safe_float(p, "num_failed"), 0),
        ])
    out.append(md_table(["输入", "输出", "推荐速率", "吞吐(tok/s)", "吞吐(req/s)",
                         "P99 TTFT(ms)", "P99 TPOT(ms)", "失败数"], rows3))
    out.append("")
    out.append(section("3.4 膝点 / 饱和点判定", 2))
    out.append("> 判定口径：① 排队临界 rate = P99 TTFT 首次 ≥ 最低速率基线 2 倍；"
               "② 吞吐饱和 rate = 吞吐首次达峰值 95%；取二者较小者。")
    out.append("")
    rows4 = []
    for a in analyses:
        ttft_onset = None
        thr_onset = None
        series = a["series"]
        ttft_pairs = [(to_rate(r.get("request_rate")), safe_float(r, "p99_ttft_ms"))
                      for r in series]
        thr_pairs = [(to_rate(r.get("request_rate")), safe_float(r, "output_throughput"))
                     for r in series]
        finite = [p for p in ttft_pairs if p[0] is not None and not math.isinf(p[0])
                  and p[1] is not None]
        if finite and finite[0][1] > 0:
            for rate, val in finite[1:]:
                if val >= finite[0][1] * 2:
                    ttft_onset = rate
                    break
        fthr = [p for p in thr_pairs if p[0] is not None and not math.isinf(p[0])
                and p[1] is not None and p[1] > 0]
        if len(fthr) >= 2:
            peak = max(v for _, v in fthr)
            for rate, val in fthr:
                if val >= peak * 0.95:
                    thr_onset = rate
                    break
        rows4.append([
            a["input_len"], a["output_len"],
            rate_label(ttft_onset), rate_label(thr_onset),
            rate_label(a["onset"]) if a["onset"] is not None else "无（取峰值速率）",
        ])
    out.append(md_table(["输入", "输出", "排队临界 rate", "吞吐饱和 rate",
                         "结论（推荐 rate）"], rows4))
    out.append("")

    # 4 P3 稳定性
    out.append(section("4. 稳定性验证（Phase 3）"))
    events = os.path.join(args.stability, "monitor_events.log") if args.stability else ""
    n_err, n_warn = count_events(events)
    if stability:
        trends = stability.get("trends") or {}
        latency = stability.get("latency_ms") or {}
        out.append(section("4.1 施压配置", 2))
        out.append(md_table(["项", "值"], [
            ["模式", fmt_opt(stability.get("mode"))],
            ["时长", f"{fmt_num(stability.get('duration_s'), 0)} s"],
            ["请求速率", fmt_opt((load_json_if_exists(
                os.path.join(args.stability, "run_config.json")) or {}).get("rate"))],
            ["最大并发", fmt_opt((load_json_if_exists(
                os.path.join(args.stability, "run_config.json")) or {}).get("max_concurrency"))],
            ["长度分布", TODO],
        ]))
        out.append("")
        out.append(section("4.2 红线监控结果", 2))
        out.append(md_table(["红线", "判定标准", "结果", "说明"], [
            ["温度触顶降频", "temp ≥ 85℃ 且 SM 时钟 < 最大 90%", TODO, "见 monitor_events.log"],
            ["显存泄漏", "后 1/2 时段增速 ≥ 100 MB/min", trend_str(trends.get("mem_leak")), ""],
            ["ECC 错误", "uncorrected 计数上升", TODO, ""],
            ["Silent Crash", "心跳连续 3 次失败", TODO, ""],
            ["吞吐衰减", "长时段下行趋势", trend_str(trends.get("throughput_decay")), ""],
            ["队列堆积", "waiting 持续增长", trend_str(trends.get("queue_growth")), ""],
        ]))
        out.append("")
        out.append(section("4.3 稳定性结论", 2))
        out.append(f"- 事件统计（自动）：ERROR {n_err} 条 / WARN {n_warn} 条")
        out.append(f"- 请求（自动）：总 {fmt_num(stability.get('requests_ok'), 0)}，"
                   f"失败率 {fmt_num(stability.get('failure_rate'), 4)}%，"
                   f"平均 {fmt_num(stability.get('req_per_s'), 3)} req/s，"
                   f"P99 E2E {fmt_num(latency.get('e2e_p99'))} ms")
        out.append("- 总体判定：TODO（通过 / 有条件通过 / 不通过）")
        out.append("- 说明：TODO")
        out.append("")
    else:
        out.append("> 未提供 `--stability`，本节需人工填写。")
        out.append("")

    # 5 调优结论
    out.append(section("5. 调优结论与配置推荐"))
    out.append(section("5.1 达标线判定（自动）", 2))
    lines = (env.get("acceptance_lines") or {})
    if lines:
        out.append("> 达标线配置于 `hardware_env.json` 的 `acceptance_lines`；"
                   "`TODO` 表示未配置阈值或数据缺失，需人工判定。")
        out.append("")
        gpu_peak = gpu_peak_util(args.run)
        acc_rows, all_pass = acceptance_eval(analyses, rows, stability, n_err,
                                             short_peak, long_peak, gpu_peak, lines)
        if acc_rows:
            out.append(md_table(["指标", "实测值", "达标线", "判定"], acc_rows))
            out.append("")
            out.append(f"- 汇总判定（自动）："
                       f"{'**全部达标**' if all_pass else '存在未达标项或待人工判定'}")
            out.append("")
    else:
        out.append("> 未配置 `acceptance_lines`，本节需人工填写。")
        out.append("")
    out.append(section("5.2 推荐引擎参数（人工）", 2))
    out.append("> 基于 3.4 膝点与 5.1 判定结果填写。")
    out.append("")
    out.append(section("5.3 硬件建议（人工）", 2))
    out.append(section("5.4 效率与性价比（自动估算）", 2))
    cards = acc.get("cards_used")
    tdp = acc.get("tdp_w")
    out.append(md_table(["指标", "值"], [
        ["每卡吞吐（tok/s/卡，短文本）",
         fmt_num(short_peak / cards) if short_peak and cards else TODO],
        ["每瓦吞吐（tok/s/W，满载估算）",
         fmt_num(short_peak / (cards * tdp)) if short_peak and cards and tdp else TODO],
        ["长文本每卡吞吐（tok/s/卡）",
         fmt_num(long_peak / cards) if long_peak and cards else TODO],
        ["与参考平台比值", TODO],
    ]))
    out.append("")

    # 附录
    out.append(section("附录 A 部署踩坑与兼容性问题记录"))
    out.append(md_table(["编号", "日期", "现象", "根因", "规避/解决", "状态"], [
        ["A-001", TODO, TODO, TODO, TODO, "打开/关闭"],
    ]))
    out.append("")
    out.append(section("附录 B 原始数据索引"))
    figs = ""
    if args.figures_dir and os.path.isdir(args.figures_dir):
        figs = ", ".join(sorted(os.listdir(args.figures_dir)))
    out.append(md_table(["数据", "路径", "说明"], [
        ["P2 汇总表", os.path.join(args.run, "summary.csv"), "全量逐组指标"],
        ["P2 各组详情", args.run, "result.json / test.log / 监控文件"],
        ["P3 报告", os.path.join(args.stability, "stability_report.json")
         if args.stability else TODO, ""],
        ["P3 时序数据", os.path.join(args.stability, "stability_series.csv")
         if args.stability else TODO, ""],
        ["图表", figs or TODO, ""],
        ["环境信息", args.env or TODO, ""],
    ]))
    out.append("")
    out.append(section("附录 C 复现步骤"))
    out.append("```bash")
    out.append("# 1. 启动服务（示例）")
    out.append("# TODO: vllm serve ... --served-model-name <模型名> --tensor-parallel-size N")
    out.append("")
    out.append("# 2. 全量矩阵")
    out.append(f"python3 run_matrix.py --phase all --warmup --model <模型名> \\")
    out.append(f"  --gpu-monitor-cmd <注册表短名>")
    out.append("")
    out.append("# 3. 稳定性验证")
    out.append("python3 run_stability.py --duration 86400 --rate <P2 极限吞吐×70~80%>")
    out.append("")
    out.append("# 4. 画图")
    out.append("python3 plot_results.py --results-dir <结果目录>")
    out.append("python3 plot_stability.py --results-dir <P3 run 目录>")
    out.append("```")
    out.append("")

    return "\n".join(out)


def trend_str(value):
    if value is None:
        return TODO
    if isinstance(value, bool):
        return "触发" if value else "通过"
    return fmt_opt(value)


def parse_args():
    p = argparse.ArgumentParser(description="模型基准测试报告自动生成（读 summary.csv / "
                                            "stability_report.json / hardware_env.json）")
    p.add_argument("--env", required=True, help="hardware_env.json 环境信息文件")
    p.add_argument("--run", required=True, help="P2 结果 run 目录（含 summary.csv）")
    p.add_argument("--stability", default=None, help="P3 稳定性 run 目录（含 stability_report.json）")
    p.add_argument("--figures-dir", default=None, help="图表目录（附录索引用）")
    p.add_argument("--output", default=None, help="输出 Markdown 路径（缺省 report_<run_id>.md）")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.env):
        sys.exit(f"错误：环境信息文件不存在: {args.env}")
    if not os.path.exists(os.path.join(args.run, "summary.csv")):
        sys.exit(f"错误：{args.run} 下未找到 summary.csv")
    env = load_env(args.env)
    output = args.output or f"report_{os.path.basename(args.run.rstrip(os.sep))}.md"
    content = gen_report(args, env)
    with open(output, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"报告草稿已生成: {output}")


if __name__ == "__main__":
    main()