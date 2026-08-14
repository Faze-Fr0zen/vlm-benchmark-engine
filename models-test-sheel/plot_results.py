"""vLLM 压测结果解析与画图脚本。

扫描 run_matrix.py 输出的结果目录，解析各测试目录下的 result.json
（缺省回退到 summary.csv），按《模型测试方案书》第四节规范生成图表：
  1. 吞吐-延迟权衡图（Throughput vs Latency，标出膝点）
  2. 并发饱和度图（Request Rate vs Throughput + P99 TTFT，标出排队临界点）
  3. 长文本性能衰减图（Input Length vs Prefill(TTFT)，含失败/OOM 标记）
  4. 长文本 GPU 显存压力图（Input Length vs 峰值显存占用）
 5. gpu_series/ 下可选输出每组测试的 GPU 指标时序图

用法示例::

    python plot_results.py --results-dir results/run-20260810-120000
    python plot_results.py --results-dir results/run-20260810-120000 --out-dir figures
    python plot_results.py --results-dir results/run-20260810-120000 --gpu-series

依赖：matplotlib（pip install matplotlib）。
"""

import argparse
import csv
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

TEST_DIR_RE = re.compile(r"in(\d+)_out(\d+)_rate(.+)")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
RATE_THRESHOLD_LONG = 16384

CJK_FONTS = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
    "Source Han Sans SC", "PingFang SC",
]
PCT_ALIASES = {
    0.5: ["50%", "p50", "0.5", "50"],
    0.9: ["90%", "p90", "0.9", "90"],
    0.99: ["99%", "p99", "0.99", "99", "99.0%"],
}


def setup_fonts(custom=None):
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    if custom:
        chosen = [custom] if custom in available else None
        if chosen is None:
            print(f"[warn] 字体 '{custom}' 不存在，使用候选字体")
    if chosen is None:
        chosen = [name for name in CJK_FONTS if name in available]
    if chosen:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = chosen
    else:
        print("[warn] 未找到中文字体，中文标签可能显示为方块（可 --font 指定）")
    plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    p = argparse.ArgumentParser(description="vLLM 压测结果解析与画图")
    p.add_argument("--results-dir", required=True,
                   help="结果目录（run_matrix.py 输出，含 summary.csv）")
    p.add_argument("--out-dir", default="figures", help="图表输出目录")
    p.add_argument("--run", default=None,
                   help="多个 run 时选择其一（缺省取最新/第一个）")
    p.add_argument("--gpu-series", action="store_true",
                   help="额外输出每组测试的 GPU 指标时序图")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--font", default=None, help="指定中文字体名")
    return p.parse_args()


def to_rate(value):
    if isinstance(value, str) and value.strip().lower() in ("inf", "infinity"):
        return float("inf")
    return float(value)


def rate_label(rate):
    return "inf" if math.isinf(rate) else str(int(rate) if rate == int(rate) else rate)


def find_runs(results_dir):
    if os.path.exists(os.path.join(results_dir, "summary.csv")):
        return [results_dir]
    runs = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "summary.csv")):
            runs.append(path)
    return runs


def csv_index(run_dir):
    path = os.path.join(run_dir, "summary.csv")
    index = {}
    if not os.path.exists(path):
        return index
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("phase"), str(row.get("input_len") or ""),
                   str(row.get("output_len") or ""), str(row.get("request_rate") or ""))
            index[key] = row
    return index


def extract_metrics(result):
    m = {}

    def num(*keys):
        for k in keys:
            v = result.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def pct(base_keys, p):
        for bk in base_keys:
            obj = result.get(bk)
            if isinstance(obj, dict):
                for cand in PCT_ALIASES[p]:
                    v = obj.get(cand)
                    if isinstance(v, (int, float)):
                        return float(v)
        return None

    m["output_throughput"] = num("output_throughput")
    m["request_throughput"] = num("total_request_throughput", "request_throughput")
    m["mean_ttft_ms"] = num("mean_ttft_ms")
    m["p99_ttft_ms"] = pct(["percentiles_ttft_ms", "ttft_percentiles",
                            "latency_percentiles"], 0.99) or num("p99_ttft_ms")
    m["mean_tpot_ms"] = num("mean_tpot_ms")
    m["p99_tpot_ms"] = pct(["percentiles_tpot_ms", "tpot_percentiles",
                            "latency_percentiles"], 0.99) or num("p99_tpot_ms")
    m["mean_e2e_ms"] = num("mean_e2e_latency_ms", "mean_e2e_ms")
    m["num_requests"] = num("num_requests")
    m["num_failed"] = num("num_failed_requests", "num_failed") or 0.0
    m["total_tokens"] = num("total_tokens")
    return m


def load_gpu_stats(test_dir):
    path = os.path.join(test_dir, "gpu_stats.csv")
    stat = None
    if not os.path.exists(path):
        return None
    util_vals, mem_util_vals, mem_ratios = [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            util = _first_number(row.get("util_pct") or "")
            mem_util = _first_number(row.get("mem_util_pct") or "")
            used = _first_number(row.get("mem_used_mib") or "")
            total = _first_number(row.get("mem_total_mib") or "")
            if util is not None:
                util_vals.append(util)
            if mem_util is not None:
                mem_util_vals.append(mem_util)
            if used is not None and total:
                mem_ratios.append(used / total)
    if util_vals or mem_util_vals or mem_ratios:
        stat = {
            "peak_util": max(util_vals) if util_vals else None,
            "peak_mem_util": max(mem_util_vals) if mem_util_vals else None,
            "peak_mem_pct": max(mem_ratios) * 100 if mem_ratios else None,
        }
    return stat


def _first_number(field):
    m = _NUM_RE.search(field)
    return float(m.group()) if m else None


def parse_test_dir(path, run_id, index):
    name = os.path.basename(path)
    m = TEST_DIR_RE.fullmatch(name)
    if not m:
        return None
    input_len, output_len, rate_str = int(m.group(1)), int(m.group(2)), m.group(3)
    parent = os.path.basename(os.path.dirname(path))
    if "long" in parent:
        phase = "long"
    elif "short" in parent:
        phase = "short"
    else:
        phase = "long" if input_len >= RATE_THRESHOLD_LONG else "short"

    record = {
        "run_id": run_id, "phase": phase, "input_len": input_len,
        "output_len": output_len, "rate": to_rate(rate_str),
        "test_dir": path, "status": "", "error": "", "leak_warning": "",
        "metrics": {}, "gpu": load_gpu_stats(path),
    }

    res_path = os.path.join(path, "result.json")
    if os.path.exists(res_path):
        try:
            with open(res_path, "r", encoding="utf-8") as fh:
                record["metrics"] = extract_metrics(json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] 解析失败 {res_path}: {exc}")

    row = index.get((phase, str(input_len), str(output_len), str(rate_str)))
    if row:
        record["status"] = row.get("status") or ""
        record["error"] = row.get("error") or ""
        record["leak_warning"] = row.get("leak_warning") or ""
        if not record["metrics"]:
            record["metrics"] = {k: _safe_float(row.get(k)) for k in (
                "output_throughput", "request_throughput", "mean_ttft_ms",
                "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
                "mean_e2e_ms", "num_requests", "num_failed", "total_tokens")}
    if not record["metrics"] and not record["status"]:
        return None
    return record


def _safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_records(run_dir):
    index = csv_index(run_dir)
    records = []
    run_id = os.path.basename(run_dir)
    for phase_name in sorted(os.listdir(run_dir)):
        phase_dir = os.path.join(run_dir, phase_name)
        if not os.path.isdir(phase_dir) or phase_name.startswith("."):
            continue
        for name in sorted(os.listdir(phase_dir)):
            path = os.path.join(phase_dir, name)
            if os.path.isdir(path):
                rec = parse_test_dir(path, run_id, index)
                if rec:
                    records.append(rec)
    return records


def split_phases(records):
    return ([r for r in records if r["phase"] == "short"],
            [r for r in records if r["phase"] == "long"])


def series_by_combo(records):
    combos = {}
    for r in records:
        key = (r["input_len"], r["output_len"])
        combos.setdefault(key, []).append(r)
    for key in combos:
        combos[key].sort(key=lambda r: r["rate"])
    return combos


def rate_x_positions(rates):
    finite = sorted(float(r) for r in rates if not math.isinf(r))
    max_fin = max(finite) if finite else 1.0
    pos = {}
    for r in finite:
        pos[r] = math.log10(r) if r > 0 else 0.0
    for r in rates:
        if math.isinf(r):
            pos[r] = math.log10(max_fin * 2.5)
    return pos, math.log10(max_fin * 2.5) if any(math.isinf(r) for r in rates) else None


def x_positions_for_series(series):
    rates = [r["rate"] for r in series]
    return rate_x_positions(rates)[0]


def plot_knee(ax, pts_x, pts_y):
    if len(pts_x) < 3 or any(v is None for v in pts_x + pts_y):
        return
    xs, ys = list(pts_x), list(pts_y)
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    if x1 <= x0:
        return
    denom = math.hypot(x1 - x0, y1 - y0)
    if denom == 0:
        return
    dists = []
    for x, y in zip(xs[1:-1], ys[1:-1]):
        dists.append(abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom)
    if not dists:
        return
    idx = max(range(len(dists)), key=lambda i: dists[i])
    if dists[idx] <= 1e-6:
        return
    ax.scatter([xs[idx + 1]], [ys[idx + 1]], marker="*", s=220, zorder=5,
               color="green", edgecolors="black")
    ax.annotate("膝点(推荐配置)", (xs[idx + 1], ys[idx + 1]),
                textcoords="offset points", xytext=(8, 8), fontsize=8,
                color="green")


def queue_onset_point(pairs):
    finite = [p for p in pairs if not math.isinf(p[0])]
    if not finite or not finite[0][1]:
        return None
    base = finite[0][1]
    if base <= 0:
        return None
    return_onset = None
    for rate, val in finite[1:]:
        if val is not None and val >= base * 2:
            return_onset = rate
            break
    if return_onset is None or math.isinf(return_onset):
        for rate, val in pairs:
            if math.isinf(rate) and val is not None and val >= base * 2:
                return_onset = rate
    if return_onset is None:
        return None
    for rate, val in pairs:
        if rate == return_onset:
            return rate
    return return_onset


def clean_points(series, key):
    pts = [(r["rate"], r["metrics"].get(key)) for r in series]
    pts = [(r, v) for r, v in pts if v not in (None, 0)]
    if len(pts) < 1:
        return [], [], []
    rates = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    return rates, vals, rate_x_positions(rates)[0]


def as_2d(axes, n_rows, n_cols):
    if hasattr(axes, "reshape"):
        return axes.reshape(n_rows, n_cols)
    import numpy as _np
    return _np.reshape(axes, (n_rows, n_cols))


def save(fig, path, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {path}")


def chart_throughput_latency(records, args, tag):
    combos = series_by_combo(records)
    inputs = sorted({c[0] for c in combos})
    outputs = sorted({c[1] for c in combos})
    if not inputs:
        return
    n_rows = len(inputs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.4 * n_rows + 1))
    axes = as_2d(axes, n_rows, 2)
    colors = plt.cm.tab10.colors
    for row, i_len in enumerate(inputs):
        for col, metric in enumerate(("p99_ttft_ms", "p99_tpot_ms")):
            ax = axes[row][col]
            plotted = False
            for o_idx, o_len in enumerate(outputs):
                key = (i_len, o_len)
                if key not in combos:
                    continue
                rates, vals, _ = clean_points(combos[key], metric)
                xs = list(rates)
                thrs = [r["metrics"].get("output_throughput") for r in combos[key]]
                pts = [(r, t, v) for r, t, v in zip(rates, thrs, vals)
                       if t is not None and t > 0]
                if not pts:
                    continue
                xs, thrs, vals = zip(*pts)
                plotted = True
                color = colors[o_idx % len(colors)]
                ax.plot(thrs, vals, marker="o", markersize=5, color=color,
                        label=f"out={o_len // 1024}K")
                for r, t, v in zip(xs, thrs, vals):
                    ax.annotate(rate_label(r), (t, v), fontsize=6.5,
                                xytext=(4, 4), textcoords="offset points", alpha=0.75)
                if metric == "p99_ttft_ms":
                    plot_knee(ax, list(thrs), list(vals))
            if plotted:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"Input={i_len // 1024}K  {metric}")
            ax.set_ylabel("ms (log)")
            if row == n_rows - 1:
                ax.set_xlabel("Output Throughput (tokens/s, log)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{tag} 吞吐-延迟权衡图（P99，各点标注请求速率）", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, os.path.join(args.out_dir, f"throughput_latency_{tag}.png"), args)


def chart_saturation(records, args, tag):
    combos = series_by_combo(records)
    inputs = sorted({c[0] for c in combos})
    outputs = sorted({c[1] for c in combos})
    if not inputs or not outputs:
        return
    n_rows, n_cols = len(inputs), len(outputs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols + 1, 3.6 * n_rows))
    axes = as_2d(axes, n_rows, n_cols)
    for row, i_len in enumerate(inputs):
        for col, o_len in enumerate(outputs):
            ax = axes[row][col]
            ser = combos.get((i_len, o_len)) or []
            if not ser:
                ax.set_visible(False)
                continue
            rates = [r["rate"] for r in ser]
            pos, inf_pos = rate_x_positions(rates)
            thr_pairs = [(r["rate"], r["metrics"].get("output_throughput")) for r in ser]
            ttft_pairs = [(r["rate"], r["metrics"].get("p99_ttft_ms")) for r in ser]

            thr_x = [pos[r] for r, v in thr_pairs if v is not None and v > 0]
            thr_y = [v for r, v in thr_pairs if v is not None and v > 0]
            ttft_x = [pos[r] for r, v in ttft_pairs if v is not None and v > 0]
            ttft_y = [v for r, v in ttft_pairs if v is not None and v > 0]
            if not thr_x and not ttft_x:
                ax.set_visible(False)
                continue

            if thr_x:
                ax.plot(thr_x, thr_y, marker="o", color="#1f77b4",
                        label="Output Throughput")
            for r, v in thr_pairs:
                if v is not None and v > 0:
                    ax.annotate(rate_label(r), (pos[r], v), fontsize=6.5,
                                xytext=(4, 4), textcoords="offset points", alpha=0.75)

            ax2 = ax.twinx()
            if ttft_x:
                ax2.plot(ttft_x, ttft_y, marker="s", ls="--", color="#d62728",
                         label="P99 TTFT")

            onset = queue_onset_point(ttft_pairs)
            if onset is not None and onset in pos:
                ax.axvline(pos[onset], color="gray", ls=":", lw=1.5)
                ax.text(pos[onset], 0.06, "排队临界", rotation=90, fontsize=8,
                        color="gray", transform=ax.get_xaxis_transform())

            xticks, xtlabels = [], []
            for r in rates:
                xticks.append(pos[r])
                xtlabels.append("inf" if math.isinf(r) else rate_label(r))
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtlabels, fontsize=8)
            if thr_x:
                ax.set_yscale("log")
            if ttft_x:
                ax2.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"In={i_len // 1024}K / Out={o_len // 1024}K")
            if col == 0:
                ax.set_ylabel("Output Throughput (log)")
            if col == n_cols - 1:
                ax2.set_ylabel("P99 TTFT (ms, log)")
            if row == n_rows - 1:
                ax.set_xlabel("Request Rate")
    handles = [plt.Line2D([0], [0], marker="o", ls="-", color="#1f77b4"),
               plt.Line2D([0], [0], marker="s", ls="--", color="#d62728")]
    labels = ["Output Throughput", "P99 TTFT"]
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{tag} 并发饱和度图（曲线拐点附近即排队临界区）", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, os.path.join(args.out_dir, f"concurrency_saturation_{tag}.png"), args)


def chart_long_ttft(records, args):
    records = [r for r in records if not math.isinf(r["rate"])]
    if not records:
        return
    combos = {}
    for r in records:
        key = (r["output_len"], r["rate"])
        combos.setdefault(key, []).append(r)
    outputs = sorted({r["output_len"] for r in records})
    rates = sorted({r["rate"] for r in records})
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(len(outputs), 2, figsize=(14, 3.8 * len(outputs)))
    axes = as_2d(axes, len(outputs), 2)
    for row, o_len in enumerate(outputs):
        ax = axes[row][0]
        ax.set_xscale("log")
        ax.set_yscale("log")
        for r_idx, rate in enumerate(rates):
            ser = sorted(combos.get((o_len, rate), []), key=lambda x: x["input_len"])
            pts = [(r["input_len"], r["metrics"].get("p99_ttft_ms"))
                   for r in ser if r["metrics"].get("p99_ttft_ms")]
            if not pts:
                continue
            xs, ys = zip(*pts)
            color = colors[r_idx % len(colors)]
            ax.plot(xs, ys, marker="o", markersize=5, color=color, label=f"rate={rate_label(rate)}")
            for x, y in zip(xs, ys):
                failed = any(r["input_len"] == x and rate == r["rate"]
                             and (r["status"].startswith("FAILED") or r["status"] in ("OOM", "TIMEOUT")
                                  or (r["metrics"].get("num_failed") or 0) > 0)
                             for r in ser)
                if failed:
                    ax.annotate("失败", (x, y), xytext=(0, 10), fontsize=8,
                                color="red", textcoords="offset points",
                                ha="center")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Out={o_len // 1024}K  P99 TTFT (≈Prefill 耗时)")
        ax.set_ylabel("P99 TTFT (ms, log)")

        axf = axes[row][1]
        fail_records = [(r["input_len"], r["rate"],
                         (r["metrics"].get("num_failed") or 0))
                        for r in records if r["output_len"] == o_len]
        fail_records = [(i_len, rate, n) for i_len, rate, n in fail_records if n > 0]
        if fail_records:
            inputs = sorted({r[0] for r in fail_records})
            width = 0.8 / max(len(rates), 1)
            for r_idx, rate in enumerate(rates):
                counts = [(i_len, n) for i_len, r, n in fail_records if r == rate]
                if counts:
                    xs = [inputs.index(i) + r_idx * width for i, n in counts]
                    ys = [n for i, n in counts]
                    axf.bar(xs, ys, width=width, color=colors[r_idx % len(colors)],
                            label=f"rate={rate_label(rate)}")
                    for x, y in zip(xs, ys):
                        axf.annotate(str(int(y)), (x, y), xytext=(0, 3), fontsize=7,
                                     ha="center", textcoords="offset points")
            axf.set_xticks(range(len(inputs)))
            axf.set_xticklabels([f"{i // 1024}K" for i in inputs])
        axf.set_ylabel("失败请求数")
        axf.set_xlabel("Input Length")
        axf.grid(True, axis="y", alpha=0.3)
        axf.set_title(f"Out={o_len // 1024}K  失败/超时请求数")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(rates), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("长文本性能衰减图（Input Length vs P99 TTFT / 失败数）", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, os.path.join(args.out_dir, "long_context_prefill_long.png"), args)


def chart_long_gpu(records, args):
    records = [r for r in records if r.get("gpu") and r["gpu"].get("peak_mem_pct") is not None]
    if not records:
        print("[skip] 无 long 阶段 GPU 采样数据，跳过显存压力图")
        return
    outputs = sorted({r["output_len"] for r in records})
    rates = sorted({r["rate"] for r in records})
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(len(outputs), 1, figsize=(12, 3.6 * len(outputs)),
                             squeeze=False)
    for row, o_len in enumerate(outputs):
        ax = axes[row][0]
        ax.set_xscale("log")
        for r_idx, rate in enumerate(rates):
            ser = sorted([r for r in records if r["output_len"] == o_len and r["rate"] == rate],
                         key=lambda x: x["input_len"])
            pts = [(r["input_len"], r["gpu"]["peak_mem_pct"]) for r in ser]
            if not pts:
                continue
            xs, ys = zip(*pts)
            color = colors[r_idx % len(colors)]
            ax.plot(xs, ys, marker="o", markersize=5, color=color,
                    label=f"rate={rate_label(rate)}")
        ax.axhline(90, color="red", ls=":", lw=1.2, alpha=0.7)
        ax.text(0.02, 91.5, "90% 显存红线", fontsize=8, color="red",
                transform=ax.get_yaxis_transform())
        ax.set_title(f"Out={o_len // 1024}K  峰值 GPU 显存占用")
        ax.set_ylabel("峰值显存占用 (%)")
        ax.set_xlabel("Input Length (log)")
        ax.set_ylim(0, 105)
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(len(rates), 1),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("长文本 KV Cache 显存压力图（nvidia-smi 采样）", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, os.path.join(args.out_dir, "long_context_gpu_mem_long.png"), args)


def chart_gpu_series(records, args):
    for rec in records:
        gpu_csv = os.path.join(rec["test_dir"], "gpu_stats.csv")
        if not os.path.exists(gpu_csv):
            continue
        t, util, mem_util, used, total = [], [], [], [], []
        with open(gpu_csv, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                u = _first_number(row.get("util_pct") or "")
                mu = _first_number(row.get("mem_util_pct") or "")
                mem = _first_number(row.get("mem_used_mib") or "")
                tot = _first_number(row.get("mem_total_mib") or "")
                if u is None and mem is None:
                    continue
                t.append(float(row["timestamp"]))
                util.append(u if u is not None else float("nan"))
                mem_util.append(mu if mu is not None else float("nan"))
                used.append(mem if mem is not None else float("nan"))
                total.append(tot if tot is not None else float("nan"))
        if not t:
            continue
        t0 = t[0]
        t = [x - t0 for x in t]
        sys_csv = os.path.join(rec["test_dir"], "sys_stats.csv")
        st, cpu, smem, smem_total = [], [], [], []
        if os.path.exists(sys_csv):
            with open(sys_csv, "r", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    c = _first_number(row.get("cpu_util_pct") or "")
                    used_m = _first_number(row.get("sys_mem_used_mib") or "")
                    tot_m = _first_number(row.get("sys_mem_total_mib") or "")
                    if c is None and used_m is None:
                        continue
                    st.append(float(row["timestamp"]))
                    cpu.append(c if c is not None else float("nan"))
                    smem.append(used_m if used_m is not None else float("nan"))
                    smem_total.append(tot_m if tot_m is not None else float("nan"))
            if st:
                s0 = min(st)
                st = [x - s0 for x in st]
        has_sys = bool(st)
        fig, axes = plt.subplots(3 if has_sys else 2, 1,
                                 figsize=(10, 7.5 if has_sys else 5.5),
                                 sharex=True)
        axes[0].plot(t, util, color="#1f77b4", lw=1)
        axes[0].set_ylabel("GPU 利用率 (%)")
        axes[0].set_ylim(0, 105)
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(t, used, color="#d62728", lw=1)
        if any(not math.isnan(v) and v > 0 for v in total):
            axes[1].plot(t, total, color="gray", ls="--", lw=1, alpha=0.7,
                         label="总显存")
            axes[1].legend(loc="upper left", fontsize=8)
        axes[1].set_ylabel("显存使用 (MiB)")
        axes[1].grid(True, alpha=0.3)
        axm = axes[1].twinx()
        if any(not math.isnan(v) for v in mem_util):
            axm.plot(t, mem_util, color="#2ca02c", lw=0.8, alpha=0.8,
                     label="显存带宽利用率")
            axm.set_ylabel("显存带宽利用率 (%)")
            axm.set_ylim(0, 105)
            axes[1].legend(loc="upper left", fontsize=8)
            axm.legend(loc="upper right", fontsize=8)
        if has_sys:
            axes[2].plot(st, cpu, color="#ff7f0e", lw=1)
            axes[2].set_ylabel("CPU 利用率 (%)")
            axes[2].set_ylim(0, 105)
            axes[2].grid(True, alpha=0.3)
            axs = axes[2].twinx()
            if any(not math.isnan(v) and v > 0 for v in smem_total):
                axs.plot(st, [v / m * 100 if m else float("nan")
                              for v, m in zip(smem, smem_total)],
                         color="#9467bd", ls="--", lw=1, alpha=0.8,
                         label="系统内存占用")
                axs.set_ylabel("系统内存占用 (%)")
                axs.set_ylim(0, 105)
                axs.legend(loc="upper right", fontsize=8)
            axes[2].legend(loc="upper left", fontsize=8)
        axes[-1].set_xlabel("测试耗时 (s)")
        if rec.get("leak_warning"):
            fig.text(0.5, 0.01, f"泄漏告警: {rec['leak_warning']}",
                     ha="center", fontsize=9, color="red")
        fig.suptitle(f"{rec['run_id']} / {os.path.basename(rec['test_dir'])} "
                     f"({rec['status'] or 'unknown'})", fontsize=10)
        fig.tight_layout()
        save(fig, os.path.join(args.out_dir, "gpu_series",
                               os.path.basename(rec["test_dir"]) + ".png"), args)


def write_metrics_csv(records, args):
    path = os.path.join(args.out_dir, "summary_metrics.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    fields = ["run_id", "phase", "input_len", "output_len", "rate",
              "output_throughput", "p99_ttft_ms", "mean_ttft_ms",
              "p99_tpot_ms", "mean_tpot_ms", "num_failed", "status",
              "leak_warning",
              "peak_gpu_util_pct", "peak_gpu_mem_util_pct", "peak_gpu_mem_pct"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "run_id": r["run_id"], "phase": r["phase"],
                "input_len": r["input_len"], "output_len": r["output_len"],
                "rate": rate_label(r["rate"]),
                "output_throughput": r["metrics"].get("output_throughput"),
                "p99_ttft_ms": r["metrics"].get("p99_ttft_ms"),
                "mean_ttft_ms": r["metrics"].get("mean_ttft_ms"),
                "p99_tpot_ms": r["metrics"].get("p99_tpot_ms"),
                "mean_tpot_ms": r["metrics"].get("mean_tpot_ms"),
                "num_failed": r["metrics"].get("num_failed"),
                "status": r["status"],
                "leak_warning": r.get("leak_warning") or "",
                "peak_gpu_util_pct": r["gpu"].get("peak_util") if r.get("gpu") else None,
                "peak_gpu_mem_util_pct": r["gpu"].get("peak_mem_util") if r.get("gpu") else None,
                "peak_gpu_mem_pct": r["gpu"].get("peak_mem_pct") if r.get("gpu") else None,
            })
    print(f"[csv] {path}")


def main():
    args = parse_args()
    setup_fonts(args.font)

    runs = find_runs(args.results_dir)
    if not runs:
        sys.exit(f"错误：在 {args.results_dir} 下未找到含 summary.csv 的 run 目录")
    for i, r in enumerate(runs, 1):
        print(f"  可用 run [{i}]: {r}")
    if args.run:
        chosen = next((r for r in runs if os.path.basename(r) == args.run), None)
        if chosen is None:
            sys.exit(f"错误：未找到 run '{args.run}'")
        run_dir = chosen
    else:
        run_dir = runs[0]
        if len(runs) > 1:
            print(f"[info] 检测到多个 run，默认使用: {run_dir}（--run 可切换）")

    records = load_records(run_dir)
    if not records:
        sys.exit("错误：run 目录下未解析到任何测试结果")
    short_records, long_records = split_phases(records)
    print(f"解析到 {len(records)} 条记录（short={len(short_records)}，long={len(long_records)}）")
    failed = [r for r in records if r["status"].startswith("FAILED") or r["status"] == "OOM"]
    if failed:
        print("警告：以下测试失败/OOM：")
        for r in failed:
            print(f"  - {os.path.basename(r['test_dir'])}: {r['status']}")

    write_metrics_csv(records, args)
    if short_records:
        chart_throughput_latency(short_records, args, "short")
        chart_saturation(short_records, args, "short")
    if long_records:
        chart_throughput_latency(long_records, args, "long")
        chart_saturation(long_records, args, "long")
        chart_long_ttft(long_records, args)
        chart_long_gpu(long_records, args)
    if args.gpu_series:
        chart_gpu_series(records, args)
    print(f"完成。图表已输出至 {args.out_dir}")


if __name__ == "__main__":
    main()