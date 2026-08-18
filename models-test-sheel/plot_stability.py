"""Phase 3 长跑稳定性结果画图脚本（run_stability.py 配套）。

扫描 run_stability.py 输出的 run 目录（含 stability_series.csv），为每个
run 生成一张时序图，覆盖四条红线 + 队列/吞吐：

  面板 1: GPU 利用率 / 温度 / 功耗
  面板 2: 显存使用（泄漏观察）
  面板 3: vLLM waiting / running 队列（堆积观察）
  面板 4: SM 时钟 vs 最大时钟（降频观察）+ 心跳

monitor_events.log 中的 ERROR 事件画红色竖线（≤5 条时附文字），WARN 画
橙色虚线；stability_report.json 的汇总信息写进图顶标题。

用法示例::

    python plot_stability.py --results-dir run-20260817-030000
    python plot_stability.py --results-dir /vllm-workspace/results --out-dir figures
    python plot_stability.py --results-dir run-20260817-030000 --dpi 150 --font "Microsoft YaHei"

依赖：matplotlib（与 plot_results.py 相同）。
"""

import argparse
import csv
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_results import setup_fonts, save, _first_number

_EVENT_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}) \[(\w+)\] (.*)")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 稳定性结果画图")
    p.add_argument("--results-dir", required=True,
                   help="结果目录（P3 run 目录，或包含多个 P3 run 的父目录）")
    p.add_argument("--out-dir", default="figures", help="图表输出目录")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--font", default=None, help="指定中文字体名")
    return p.parse_args()


def find_runs(results_dir):
    """返回含 stability_series.csv 的 run 目录列表。"""
    runs = []
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "stability_series.csv")):
            runs.append(path)
    if not runs and os.path.exists(os.path.join(results_dir, "stability_series.csv")):
        runs = [results_dir]
    return runs


def iso_to_epoch(stamp):
    try:
        return int(__import__("datetime").datetime.fromisoformat(stamp).timestamp())
    except ValueError:
        return None


def load_series(run_dir):
    rows = []
    with open(os.path.join(run_dir, "stability_series.csv"), "r",
              encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ts = _first_number(row.get("ts") or "")
            if ts is None:
                continue
            rows.append((ts, row))
    if not rows:
        return None
    t0_epoch = rows[0][0]
    out = {k: [] for k in rows[0][1]}
    for ts, row in rows:
        x = (ts - t0_epoch) / 60.0
        for k, v in row.items():
            n = _first_number(v or "")
            out[k].append((x, n))
    return t0_epoch, out


def load_events(run_dir, series_t0):
    """解析 monitor_events.log -> [(相对分钟, level, msg)]。"""
    path = os.path.join(run_dir, "monitor_events.log")
    events = []
    if not os.path.exists(path):
        return events
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = _EVENT_RE.match(line.strip())
            if not m:
                continue
            epoch = iso_to_epoch(m.group(1))
            if epoch is None:
                continue
            events.append(((epoch - series_t0) / 60.0, m.group(2), m.group(3)))
    events.sort()
    return events


def load_report(run_dir):
    path = os.path.join(run_dir, "stability_report.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _vals(series, key):
    return [(x, v) for x, v in series.get(key, []) if v is not None]


def _line(ax, series, key, color, label, **kw):
    pts = _vals(series, key)
    if not pts:
        return False
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, label=label, **kw)
    return True


def chart_run(run_dir, args):
    loaded = load_series(run_dir)
    if loaded is None:
        print(f"[skip] {run_dir}: stability_series.csv 无可解析数据")
        return None
    t0_epoch, series = loaded
    report = load_report(run_dir)
    events = load_events(run_dir, t0_epoch)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    ax0, ax1, ax2, ax3 = axes

    # 面板 1: 利用率 / 温度 / 功耗
    _line(ax0, series, "gpu_util_pct", "#1f77b4", "GPU 利用率 %")
    _line(ax0, series, "gpu_temp_c", "#d62728", "温度 ℃")
    ax0.set_ylim(0, 105)
    axp = ax0.twinx()
    if _line(axp, series, "gpu_power_w", "#ff7f0e", "功耗 W"):
        axp.set_ylabel("功耗 (W)")
    ax0.set_ylabel("利用率 % / 温度 ℃")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8)

    # 面板 2: 显存
    ax1 = axes[1]
    _line(ax1, series, "gpu_mem_used_mib", "#d62728", "显存使用 MiB")
    _line(ax1, series, "gpu_mem_total_mib", "gray", "总显存 MiB", ls="--", alpha=0.7)
    ax1.set_ylabel("显存 (MiB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # 面板 3: 队列
    ax2 = axes[2]
    has_q = _line(ax2, series, "num_requests_waiting", "#1f77b4", "waiting 队列")
    has_q = _line(ax2, series, "num_requests_running", "#2ca02c", "running 队列") or has_q
    ax2.set_ylabel("vLLM 请求队列")
    ax2.grid(True, alpha=0.3)
    if has_q:
        ax2.legend(loc="upper right", fontsize=8)

    # 面板 4: 时钟 + 心跳
    ax3 = axes[3]
    has_c = _line(ax3, series, "clocks_sm_mhz", "#9467bd", "SM 时钟 MHz")
    has_c = _line(ax3, series, "clocks_max_mhz", "gray", "最大时钟 MHz",
                  ls="--", alpha=0.7) or has_c
    axh = ax3.twinx()
    hb = _vals(series, "heartbeat_ok")
    if hb:
        xs, ys = zip(*hb)
        axh.plot(xs, ys, ls="", marker=".", ms=3, color="red", label="心跳")
        axh.set_ylim(-0.2, 1.2)
        axh.set_yticks([0, 1])
        axh.set_yticklabels(["fail", "ok"], fontsize=7)
    ax3.set_ylabel("SM 时钟 (MHz)")
    ax3.set_xlabel("运行时间 (min)")
    ax3.grid(True, alpha=0.3)
    if has_c:
        ax3.legend(loc="upper right", fontsize=8)

    # 红线事件
    errors = [(x, m) for x, lv, m in events if lv == "ERROR"]
    warns = [(x, m) for x, lv, m in events if lv == "WARN"]
    for x, msg in warns:
        for ax in axes:
            ax.axvline(x, color="orange", ls=":", lw=1, alpha=0.8)
    for x, msg in errors:
        for ax in axes:
            ax.axvline(x, color="red", ls="--", lw=1.2)
        if len(errors) <= 5:
            ax2.text(x, ax2.get_ylim()[1] * 0.9, f"ERROR\n{x:.0f}min", fontsize=7,
                     color="red", ha="center", va="top")

    # 汇总标题
    lines = [os.path.basename(run_dir)]
    if report:
        r = report
        trends = r.get("trends") or {}
        parts = [f"mode={r.get('mode', '?')}",
                 f"{r.get('duration_s', 0) / 60.0:.0f}min",
                 f"req/s={r.get('req_per_s', '?')}",
                 f"ok={r.get('requests_ok')} fail={r.get('requests_failed')}"
                 f" ({r.get('failure_rate', 0) * 100:.0f}%)"]
        for label, v in (("衰减", trends.get("throughput_decay")),
                         ("队列", trends.get("queue_growth")),
                         ("泄漏", trends.get("mem_leak"))):
            if v:
                parts.append(f"[{label}]{v}")
        lines.append(" | ".join(parts))
        lat = r.get("latency_ms") or {}
        if lat:
            lines.append(f"TTFT mean={lat.get('ttft_mean', '?')}ms "
                         f"p99={lat.get('ttft_p99', '?')}ms | "
                         f"TPOT mean={lat.get('tpot_mean', '?')}ms "
                         f"p99={lat.get('tpot_p99', '?')}ms | "
                         f"E2E mean={lat.get('e2e_mean', '?')}ms "
                         f"p99={lat.get('e2e_p99', '?')}ms")
    if errors:
        lines.append(f"红线事件: {len(errors)} ERROR / {len(warns)} WARN")
    fig.suptitle("\n".join(lines), fontsize=11, y=0.995, ha="left",
                 x=0.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(args.out_dir, f"{os.path.basename(run_dir)}_stability.png")
    save(fig, out, args)
    return {"run_id": os.path.basename(run_dir), "report": report,
            "n_errors": len(errors), "n_warns": len(warns),
            "series": series}


def write_summary_csv(rows, args):
    path = os.path.join(args.out_dir, "stability_summary.csv")
    fields = ["run_id", "mode", "duration_s", "requests_ok", "requests_failed",
              "failure_rate", "req_per_s", "n_errors", "n_warns",
              "ttft_mean_ms", "ttft_p99_ms", "tpot_mean_ms", "tpot_p99_ms",
              "throughput_decay", "queue_growth", "mem_leak",
              "gpu_util_max_pct", "temp_max_c", "mem_used_max_mib",
              "waiting_max"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            r = row["report"] or {}
            trends = r.get("trends") or {}
            s = row["series"]
            util_max = max((v for _, v in _vals(s, "gpu_util_pct")), default="")
            temp_max = max((v for _, v in _vals(s, "gpu_temp_c")), default="")
            mem_max = max((v for _, v in _vals(s, "gpu_mem_used_mib")), default="")
            wait_max = max((v for _, v in _vals(s, "num_requests_waiting")), default="")
            lat = r.get("latency_ms") or {}
            w.writerow({
                "run_id": row["run_id"], "mode": r.get("mode", ""),
                "duration_s": r.get("duration_s", ""),
                "requests_ok": r.get("requests_ok", ""),
                "requests_failed": r.get("requests_failed", ""),
                "failure_rate": r.get("failure_rate", ""),
                "req_per_s": r.get("req_per_s", ""),
                "n_errors": row["n_errors"], "n_warns": row["n_warns"],
                "ttft_mean_ms": lat.get("ttft_mean", ""),
                "ttft_p99_ms": lat.get("ttft_p99", ""),
                "tpot_mean_ms": lat.get("tpot_mean", ""),
                "tpot_p99_ms": lat.get("tpot_p99", ""),
                "throughput_decay": trends.get("throughput_decay") or "",
                "queue_growth": trends.get("queue_growth") or "",
                "mem_leak": trends.get("mem_leak") or "",
                "gpu_util_max_pct": util_max, "temp_max_c": temp_max,
                "mem_used_max_mib": mem_max, "waiting_max": wait_max,
            })
    print(f"[csv] {path}")


def main():
    args = parse_args()
    setup_fonts(args.font)
    runs = find_runs(args.results_dir)
    if not runs:
        sys.exit(f"错误：在 {args.results_dir} 下未找到含 stability_series.csv 的 run 目录")
    print(f"发现 {len(runs)} 个 P3 run")
    rows = []
    for r in runs:
        print(f"  处理: {r}")
        row = chart_run(r, args)
        if row:
            rows.append(row)
    if rows:
        write_summary_csv(rows, args)
        print(f"完成。图表已输出至 {args.out_dir}")


if __name__ == "__main__":
    main()
