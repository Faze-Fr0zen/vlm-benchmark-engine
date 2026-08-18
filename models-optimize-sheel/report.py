"""调优指南报告生成器：收敛旅程回放 + 评分表 + 验收对比。

用法::

    python report.py --results-dir <probe_only 结果根目录> \\
                     [--p2-before <默认配置 run-p2 目录>] \\
                     [--p2-after  <锁定配置 run-p2 目录>]

输入:
  - --results-dir: probe_only.py 输出的 run-opt-* 目录（含 summary.csv）
  - --p2-before/--p2-after: 可选，验收对比数据源（run_matrix.py 输出，
    before=默认配置基线，after=锁定配置重跑）

输出（--out-dir）:
  - opt_summary.csv     所有配置合并评分表（按 score 降序）
  - best_serve_cmd.txt  锁定配置的完整 vllm serve 命令
  - report.md           调优指南（锁定配置 + 收敛旅程回放 + 评分表 +
                        验收对比收益）
"""

import argparse
import csv
import os

import probe_only as po

SERVE_BASE_DEFAULT = ("vllm serve /models/Qwen3-0.6B "
                      "--served-model-name qwen3-0.6b --dtype half "
                      "--max-model-len 32768")


def collect_runs(results_dir):
    runs = []
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "summary.csv")) and _is_score_table(path):
            runs.append(path)
    if not runs and os.path.exists(
            os.path.join(results_dir, "summary.csv")) \
            and _is_score_table(results_dir):
        runs = [results_dir]
    return runs


def _is_score_table(dir_path):
    """只收 probe_only 评分表（含 config_id 列），排除 run_matrix 输出。"""
    try:
        with open(os.path.join(dir_path, "summary.csv"),
                  "r", encoding="utf-8") as fh:
            return "config_id" in (fh.readline() or "")
    except OSError:
        return False


def load_summary(run_dir):
    rows = []
    with open(os.path.join(run_dir, "summary.csv"),
              "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["_run"] = os.path.basename(run_dir)
            rows.append(row)
    return rows


def replay_convergence(rows, lite, serve_base):
    """按实际执行顺序回放收敛引擎，返回每步 (config_id, 引擎文本行列表)。"""
    steps = []
    sim = []
    seen = set()
    for row in rows:
        cid = row.get("config_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        sim.append(row)
        lines, _ = po.next_action(sim, lite=lite, serve_base=serve_base)
        steps.append((cid, lines))
    return steps


def extract_locked_cmd(steps):
    """从最后一步的引擎输出提取最终 serve 命令。"""
    for _, lines in reversed(steps):
        for i, line in enumerate(lines):
            if "[锁定]" in line and i + 1 < len(lines):
                return lines[i + 1].strip()
    return None


def md_escape(text):
    return str(text).replace("|", "\\|")


def render_convergence(steps):
    lines = ["## 收敛旅程回放", "",
             "> 以下为每次探针完成后收敛引擎的实际输出，完整复现调优过程。",
             ""]
    for cid, engine_lines in steps:
        lines.append(f"### {cid}")
        lines.append("")
        lines.append("```")
        for el in engine_lines:
            lines.append(el)
        lines.append("```")
        lines.append("")
    return lines


def render_score_table(rows):
    lines = ["## 配置评分表（按 score 降序）", "",
             "| run | 轮次 | 配置 | verdict | score | 短吞吐 | 长吞吐 | "
             "TTFT p99(ms) | 显存峰值% | GPU均% | 问题 | 建议 |", "|---|"]
    for r in rows:
        s = po.to_float(r, "score")
        s_str = f"{s:.1f}" if s is not None else "-"
        lines.append(
            f"| {r['_run']} | {r.get('round') or '-'} | {r['config_id']} | "
            f"{r['verdict']} | {s_str} | "
            f"{r.get('short_thr') or '-'} | {r.get('long_thr') or '-'} | "
            f"{r.get('ttft_p99_ms') or '-'} | {r.get('mem_pct_max') or '-'} | "
            f"{r.get('gpu_util_mean') or '-'} | "
            f"{md_escape(r.get('problems') or '-')} | "
            f"{md_escape(r.get('suggestions') or '-')} |")
    lines.append("")
    return lines


def compare_p2(before_dir, after_dir):
    """对比两个 run_matrix 输出目录（默认配置 vs 锁定配置）。"""
    def load(dir_path):
        path = os.path.join(dir_path, "summary.csv")
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        ok = [r for r in rows if r.get("status") == "OK"]
        short = [r for r in ok if r.get("phase") == "short"]
        long = [r for r in ok if r.get("phase") == "long"]
        metrics = {
            "n_short": len(short), "n_long": len(long),
            "n_oom": sum(1 for r in rows
                         if r.get("status") in ("OOM", "FAILED")),
            "thr_short": _median(short, "output_throughput"),
            "thr_long": _median(long, "output_throughput"),
            "ttft_p99": _median(short, "p99_ttft_ms"),
            "tpot_p99": _median(short, "p99_tpot_ms"),
            "mem_pct_max": _max_of(rows, "gpu_mem_pct_max"),
            "gpu_util": _mean_of(rows, "gpu_util_mean"),
            "leak_warn": sum(1 for r in rows if r.get("gpu_leak_warning")),
        }
        return metrics

    def _median(rows, key):
        vals = sorted(po.to_float(r, key) for r in rows
                      if po.to_float(r, key) is not None)
        if not vals:
            return None
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    def _max_of(rows, key):
        vals = [po.to_float(r, key) for r in rows
                if po.to_float(r, key) is not None]
        return max(vals) if vals else None

    def _mean_of(rows, key):
        vals = [po.to_float(r, key) for r in rows
                if po.to_float(r, key) is not None]
        return sum(vals) / len(vals) if vals else None

    return load(before_dir), load(after_dir)


def render_acceptance(before, after, before_dir, after_dir):
    if not before or not after:
        return ["## 验收对比", "",
                "> 未提供 --p2-before/--p2-after，跳过验收对比。", ""]
    lines = ["## 验收对比（默认配置 vs 锁定配置）", "",
             f"- before: `{os.path.basename(before_dir)}`"
             f"（默认配置基线）",
             f"- after : `{os.path.basename(after_dir)}`"
             f"（锁定配置重跑）",
             "",
             "| 指标 | before | after | 变化 |", "|---|"]

    def pct(new, old):
        if old is None or new is None:
            return "-"
        if old == 0:
            return "-"
        return f"{(new - old) / old * 100:+.1f}%"

    def fmt(v):
        return f"{v:.1f}" if v is not None else "-"

    for label, key, desc in (
            ("short 组中位吞吐( tok/s )", "thr_short", "up"),
            ("long 组中位吞吐( tok/s )", "thr_long", "up"),
            ("TTFT p99(ms)", "ttft_p99", "down"),
            ("TPOT p99(ms)", "tpot_p99", "down"),
            ("显存峰值%(max)", "mem_pct_max", "down"),
            ("GPU 均利用%(mean)", "gpu_util", "up")):
        b, a = before.get(key), after.get(key)
        lines.append(f"| {desc} {label} | {fmt(b)} | {fmt(a)} | {pct(a, b)} |")
    lines += [
        f"| OOM/失败组数 | {before['n_oom']} | {after['n_oom']} | "
        f"{'固定' if before['n_oom'] == after['n_oom'] else '变化'} |",
        f"| 显存泄漏警告组数 | {before['leak_warn']} | {after['leak_warn']} | "
        f"{'无' if after['leak_warn'] == 0 else '存在'} |",
        "",
    ]
    return lines


def main():
    p = argparse.ArgumentParser(description="调优指南报告生成器")
    p.add_argument("--results-dir", required=True,
                   help="probe_only 结果根目录（含 run-opt-* 子目录）")
    p.add_argument("--out-dir", default="opt_report",
                   help="报告输出目录（默认 ./opt_report）")
    p.add_argument("--p2-before", default=None,
                   help="验收对比 before：默认配置的 run_matrix 输出目录")
    p.add_argument("--p2-after", default=None,
                   help="验收对比 after：锁定配置的 run_matrix 输出目录")
    p.add_argument("--serve-base", default=SERVE_BASE_DEFAULT,
                   help="基础 serve 命令前缀（拼调优参数）")
    p.add_argument("--lite", action="store_true",
                   help="4090 小体量矩阵（收敛回放使用）")
    args = p.parse_args()

    runs = collect_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"未在 {args.results_dir} 找到任何 summary.csv")
    all_rows = []
    for run_dir in runs:
        all_rows += load_summary(run_dir)
    all_rows.sort(key=lambda r: po.to_float(r, "score") or -1.0,
                  reverse=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "opt_summary.csv")
    fields = ["_run"] + [f for f in po.SUMMARY_FIELDS
                         if f not in ("problems", "suggestions")] \
        + ["problems", "suggestions"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[report] 评分表已写入: {out_csv}")

    accepted = po.accept_rows(all_rows)
    steps = replay_convergence(all_rows, args.lite, args.serve_base)
    locked_cmd = extract_locked_cmd(steps)
    lines = ["# Serve 参数调优报告", "",
             "## 锁定配置", ""]
    if locked_cmd:
        lines += ["- 完整 serve 命令:", "```", locked_cmd, "```", ""]
        with open(os.path.join(args.out_dir, "best_serve_cmd.txt"),
                  "w", encoding="utf-8") as fh:
            fh.write(locked_cmd + "\n")
    else:
        lines += ["> 收敛尚未完成（无锁定命令），请继续按 probe_only.py 输出",
                  "> 跑完剩余组合后再生成报告。", ""]
    if accepted:
        best = accepted[0]
        lines += [f"- 共 {len(accepted)} 组 ACCEPT，最高分: "
                  f"`{best['config_id']}`  score: {best['score']}"]
    else:
        lines += ["> 无 ACCEPT 记录，锁定命令取自收敛引擎最终输出。", ""]

    lines += render_convergence(steps)
    lines += render_score_table(all_rows)

    before = after = None
    if args.p2_before and args.p2_after:
        before, after = compare_p2(args.p2_before, args.p2_after)
        lines += render_acceptance(before, after,
                                   args.p2_before, args.p2_after)
    else:
        lines += ["## 验收对比", "",
                  "> 未提供 --p2-before/--p2-after，跳过验收对比。", ""]

    out_md = os.path.join(args.out_dir, "report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[report] 报告已写入: {out_md}")
    if locked_cmd:
        print(f"[report] 最佳命令已写入: "
              f"{os.path.join(args.out_dir, 'best_serve_cmd.txt')}")


if __name__ == "__main__":
    main()