"""vLLM 阶梯压测矩阵自动化执行脚本（大模型全量版，适配国产 GPU）。

按《模型测试方案书》Phase 2 定义的全量矩阵，逐组调用 `vllm bench serve`：
  - 2.1 短/中文本: 输入 1K/4K/8K x 输出 1K/4K x 速率 1/8/16/32/64/128/inf
  - 2.2 长文本:    输入 16K/32K/64K/128K x 输出 1K/4K x 速率 1/4/8/16
每组测试强制追加 --ignore-eos 与 --no-enable-prefix-caching。

监控（sys_monitor.py）默认采集 CPU/系统内存；需要 GPU 指标时用
--gpu-monitor-cmd 显式指定探测命令（nvidia-smi/npu-smi/任意工具均可），
不探测、不依赖任何 GPU 厂商工具。实时写 sys_stats.csv 与可选的
gpu_stats.csv，monitor_events.log 记录 OOM 告警与日志错误扫描，
测试结束做内存泄漏回归检测并回填 summary 的 leak_warning。

模型/tokenizer 路径：脚本顶部 MODEL / TOKENIZER / MODEL_PATH 变量统一
管理（切换模型只改变量）；压测前校验 tokenizer 与模型权重目录存在性。

结果存储与推送：结果默认写入 /vllm-workspace/results（RESULTS_DIR，
独立 git 仓库）；每组测试成功（退出码 0）且生成 result.json 后自动
commit + push（分支动态获取、push 前 pull --rebase、无变更跳过、
失败仅打印警告不阻塞）。

运行小模型快速验证时请使用 run_matrix_lite.py（矩阵/参数不同，监控相同）。

用法示例::

    python run_matrix.py                          # 执行全量矩阵（74 组）
    python run_matrix.py --phase short --warmup   # 仅 2.1 短/中文本矩阵
    python run_matrix.py --dry-run                # 打印命令不执行
    python run_matrix.py --resume                 # 跳过已有 result.json 的测试

依赖仅为 Python 3.9+ 标准库（CPU 采样优先 psutil，缺省回退 /proc）。
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time

from sys_monitor import SystemMonitor, make_probe

MODEL = "qwen3-0.6b"               # 压测模型名（--model 默认，必须与 vLLM serve 的 --served-model-name 一致，切换模型只改这里）
TOKENIZER = "/models/Qwen3-0.6B"     # tokenizer 本地绝对路径（离线模式必须）
MODEL_PATH = "/models/Qwen3-0.6B"    # 模型权重目录（仅做存在性校验）
RESULTS_DIR = "/vllm-workspace/results/vllm-bench-result"  # 压测结果统一存放目录（独立 git 仓库）
DEFAULT_BASE_URL = "http://localhost:8000"

SHORT_INPUTS = [1024, 4096, 8192]
SHORT_OUTPUTS = [1024, 4096]
SHORT_RATES = ["1", "8", "16", "32", "64", "128", "inf"]
LONG_INPUTS = [16384, 32768, 65536, 131072]
LONG_OUTPUTS = [1024, 4096]
LONG_RATES = ["1", "4", "8", "16"]

DEFAULT_NUM_PROMPTS_SHORT = 200
DEFAULT_NUM_PROMPTS_LONG = 20
MAX_CONCURRENT_SHORT = 32
MAX_CONCURRENT_LONG = 16

SUMMARY_FIELDS = [
    "run_id", "phase", "input_len", "output_len", "request_rate", "num_prompts",
    "status", "error", "output_throughput", "request_throughput",
    "total_tokens", "input_tokens", "output_tokens",
    "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
    "mean_e2e_ms", "p99_e2e_ms", "num_requests", "num_failed",
    "leak_warning", "start_time", "end_time", "duration_s",
]

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

DEFAULT_FLAGS = {"--random-input", "--num-prompts", "--ignore-eos",
                 "--no-enable-prefix-caching", "--result-dir",
                 "--max-concurrency", "--tokenizer"}


def split_ints(value):
    parts = [x for x in re.split(r"[, ]+", value) if x]
    return [int(x) for x in parts]


def parse_args():
    p = argparse.ArgumentParser(description="vLLM 阶梯压测矩阵自动化执行（大模型全量版）")
    p.add_argument("--model", default=MODEL, help="模型名称（服务端已加载的模型）")
    p.add_argument("--tokenizer", default=TOKENIZER,
                   help="tokenizer 本地绝对路径（离线模式必须，默认取脚本顶部 TOKENIZER 变量）")
    p.add_argument("--model-path", default=MODEL_PATH,
                   help="模型权重本地路径（仅做存在性校验，默认取脚本顶部 MODEL_PATH 变量）")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM 服务地址")
    p.add_argument("--vllm-cmd", default="vllm", help="vllm 可执行文件或命令")
    p.add_argument("--phase", choices=["all", "short", "long"], default="all",
                   help="执行阶段：short=2.1 短/中文本矩阵，long=2.2 长文本矩阵")
    p.add_argument("--input-lens", type=split_ints, default=None,
                   help="自定义输入长度列表，如 '1024,4096'（默认按阶段取矩阵定义）")
    p.add_argument("--output-lens", type=split_ints, default=None,
                   help="自定义输出长度列表")
    p.add_argument("--request-rates", nargs="*", default=None,
                   help="自定义速率列表，支持 inf")
    p.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS_SHORT,
                   help="短/中文本测试请求数")
    p.add_argument("--num-prompts-long", type=int, default=DEFAULT_NUM_PROMPTS_LONG,
                   help="长文本测试请求数")
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="限制客户端最大并发，防止 Prefill 队列堆积导致 TTFT 爆炸；"
                        "不传则自动按阶段取默认值（short=32，long=16），传 0 表示不限制")
    p.add_argument("--result-dir", default=None,
                   help="结果根目录（默认取脚本顶部 RESULTS_DIR，服务器上为 "
                        "/vllm-workspace/results/vllm-bench-result，本地调试可用 --result-dir results 覆盖）")
    p.add_argument("--run-id", default=None, help="本次运行标识（缺省按时间生成）")
    p.add_argument("--timeout", type=int, default=7200, help="单组测试超时秒数")
    p.add_argument("--sleep-between", type=float, default=5.0, help="相邻测试间隔秒数")
    p.add_argument("--warmup", action="store_true", help="矩阵前先跑一组小规模预热")
    p.add_argument("--resume", action="store_true", help="跳过已有 result.json 的测试")
    p.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    p.add_argument("--verbose", action="store_true", help="打印完整命令")
    p.add_argument("--no-gpu-monitor", action="store_true",
                   help="禁用 GPU 采样（仅当传了 --gpu-monitor-cmd 时才有意义，"
                        "CPU/内存监控始终开启）")
    p.add_argument("--gpu-interval", type=float, default=10.0, help="监控采样间隔秒数")
    p.add_argument("--gpu-ids", default=None, help="限制监控的 GPU 编号，如 '0,1'（缺省采集所有）")
    p.add_argument("--gpu-monitor-cmd", default=None,
                   help="自定义 GPU 探测命令，输出每行逗号分隔数字列 "
                        "util,mem_used,mem_total[,power,temp]（可与 --gpu-regex 搭配）")
    p.add_argument("--gpu-regex", default=None,
                   help="自定义 GPU 探测正则，命名组 util/mem_used/mem_total/power/temp")
    p.add_argument("--sys-mem-warn-mb", type=int, default=2048,
                   help="系统可用内存告警阈值（MiB），低于则报 OOM 风险")
    p.add_argument("--gpu-mem-warn-pct", type=float, default=95.0,
                   help="GPU 显存利用率告警阈值（%）")
    return p.parse_args()


def build_cases(args):
    if args.phase in ("all", "short"):
        inputs = args.input_lens or SHORT_INPUTS
        outputs = args.output_lens or SHORT_OUTPUTS
        rates = args.request_rates or SHORT_RATES
        for i_len in inputs:
            for o_len in outputs:
                for rate in rates:
                    yield {"phase": "short", "input_len": i_len,
                           "output_len": o_len, "request_rate": rate}
    if args.phase in ("all", "long"):
        inputs = args.input_lens or LONG_INPUTS
        outputs = args.output_lens or LONG_OUTPUTS
        rates = args.request_rates or LONG_RATES
        for i_len in inputs:
            for o_len in outputs:
                for rate in rates:
                    yield {"phase": "long", "input_len": i_len,
                           "output_len": o_len, "request_rate": rate}


def case_dir_name(case):
    return f"in{case['input_len']}_out{case['output_len']}_rate{case['request_rate']}"


def probe_vllm_flags(vllm_cmd_list):
    """探测 vLLM bench serve 支持的参数（新版本 --help 只出分组概览，
    需用 --help=all 才能拿到全量 flag）。"""
    help_text = ""
    ok = False
    for probe_args in (["bench", "serve", "--help=all"],
                       ["bench", "serve", "--help"]):
        try:
            proc = subprocess.run(vllm_cmd_list + probe_args,
                                  capture_output=True, text=True, timeout=90)
            text = proc.stdout + "\n" + proc.stderr
            if proc.returncode in (0, 2) and len(text) > len(help_text):
                help_text = text
        except Exception as exc:
            print(f"[warn] 无法探测 vLLM bench serve 参数: {exc}")
    flags = set(re.findall(r"--[\w-]+", help_text))
    flags.discard("--help")
    if flags:
        return flags, True
    print("[warn] 探测失败，暂按常见参数默认值处理（实际运行时将重新探测）")
    return set(DEFAULT_FLAGS), False


def build_command(case, num_prompts, max_concurrency, args, flags, test_dir):
    cmd = args.vllm_cmd.split() + ["bench", "serve"]
    if "--model" in flags:
        cmd += ["--model", args.model]
    else:
        cmd += [args.model]
    cmd += [
        "--base-url", args.base_url,
        "--dataset-name", "random",
        "--request-rate", str(case["request_rate"]),
    ]
    if "--tokenizer" in flags and args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    else:
        print("[warn] 当前 vLLM 不支持 --tokenizer，离线场景可能无法加载 tokenizer")
    if "--num-prompts" in flags:
        cmd += ["--num-prompts", str(num_prompts)]
    else:
        print("[warn] 当前 vLLM 不支持 --num-prompts，将使用其默认请求数")
    if "--max-concurrency" in flags:
        cmd += ["--max-concurrency", str(max_concurrency)]
    else:
        print("[warn] 当前 vLLM 不支持 --max-concurrency，"
              "将无法限制客户端并发，TTFT 可能被 Prefill 队列拉爆")
    if "--random-input" in flags:
        cmd += ["--random-input", str(case["input_len"]),
                "--random-output", str(case["output_len"])]
    elif "--random-input-len" in flags:
        cmd += ["--random-input-len", str(case["input_len"]),
                "--random-output-len", str(case["output_len"])]
    elif "--random-input-mean" in flags:
        cmd += ["--random-input-mean", str(case["input_len"]), "--random-input-stdev", "0",
                "--random-output-mean", str(case["output_len"]), "--random-output-stdev", "0"]
    else:
        print("[warn] 当前 vLLM 不支持随机输入长度参数，将使用其默认输入长度")
    if "--ignore-eos" in flags:
        cmd.append("--ignore-eos")
    else:
        print("[warn] 当前 vLLM 不支持 --ignore-eos，已跳过")
    if "--no-enable-prefix-caching" in flags:
        cmd.append("--no-enable-prefix-caching")
    else:
        print("[warn] 当前 vLLM 不支持 --no-enable-prefix-caching，prefix caching 使用默认行为")
    if "--save-result" in flags:
        cmd.append("--save-result")
    else:
        print("[warn] 当前 vLLM 不支持 --save-result，将不会写出 result.json，指标将缺失")
    if "--result-dir" in flags:
        cmd += ["--result-dir", test_dir, "--result-file", "result.json"]
    elif "--result-filename" in flags:
        cmd += ["--result-filename", os.path.join(test_dir, "result.json")]
    else:
        cmd += ["--result-file", os.path.join(test_dir, "result.json")]
    return cmd


def extract_metrics(result):
    m = {}

    def num(*keys):
        for k in keys:
            v = result.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def pct(base_keys, aliases):
        for bk in base_keys:
            obj = result.get(bk)
            if isinstance(obj, dict):
                for cand in aliases:
                    v = obj.get(cand)
                    if isinstance(v, (int, float)):
                        return float(v)
        return None

    p99_ttft = pct(["percentiles_ttft_ms", "ttft_percentiles", "latency_percentiles"],
                   ["99%", "p99", "0.99", "99", "99.0%"])
    p99_tpot = pct(["percentiles_tpot_ms", "tpot_percentiles", "latency_percentiles"],
                   ["99%", "p99", "0.99", "99", "99.0%"])
    p99_e2e = pct(["percentiles_e2e_latency_ms", "e2e_percentiles", "latency_percentiles"],
                  ["99%", "p99", "0.99", "99", "99.0%"])
    m.update({
        "output_throughput": num("output_throughput"),
        "request_throughput": num("total_request_throughput", "request_throughput"),
        "total_tokens": num("total_tokens"),
        "input_tokens": num("input_tokens"),
        "output_tokens": num("output_tokens"),
        "mean_ttft_ms": num("mean_ttft_ms"),
        "p99_ttft_ms": p99_ttft or num("p99_ttft_ms"),
        "mean_tpot_ms": num("mean_tpot_ms"),
        "p99_tpot_ms": p99_tpot or num("p99_tpot_ms"),
        "mean_e2e_ms": num("mean_e2e_latency_ms", "mean_e2e_ms"),
        "p99_e2e_ms": p99_e2e or num("p99_e2e_latency_ms", "p99_e2e_ms"),
        "num_requests": num("num_requests"),
        "num_failed": num("num_failed_requests", "num_failed") or 0.0,
    })
    return m


def _first_number(field):
    m = _NUM_RE.search(field)
    return float(m.group()) if m else None


def validate_paths(args):
    """压测前校验 tokenizer / 模型权重路径存在性，缺失立即报错退出。"""
    for label, path in (("tokenizer", args.tokenizer),
                        ("模型权重", args.model_path)):
        if path and not os.path.isdir(path):
            sys.exit(f"错误：{label} 路径不存在: {path}\n"
                     f"请检查脚本顶部 TOKENIZER / MODEL_PATH 变量或 --tokenizer / "
                     f"--model-path 参数")


def git_push_results(args, run_id, case, test_dir):
    """压测成功后将单组 result.json 提交并推送（独立 results git 仓库）。

    - 仓库不存在或非 git 目录时静默跳过
    - 分支名用 git branch --show-current 动态获取，不硬编码
    - push 前先 pull --rebase 防远端新提交导致拒绝
    - 无变更（git add 后无 staged diff）跳过 commit/push，避免空提交
    - commit message 含模型/输入/输出/速率及 run_id，保证幂等可追溯
    - 任何 git 失败只打印警告，不阻塞压测主流程
    """
    repo = args.result_dir
    if not os.path.isdir(os.path.join(repo, ".git")):
        return
    rel = os.path.join(run_id, case["phase"], case_dir_name(case),
                       "result.json").replace(os.sep, "/")
    rel_abs = os.path.join(repo, *rel.split("/"))
    if not os.path.exists(rel_abs):
        print(f"[warn] 未生成结果文件 {rel}，跳过 git 推送")
        return

    model = str(args.model).replace("/", "_")
    msg = (f"bench: auto push {model}_{case['input_len']}_{case['output_len']}"
           f"_rate{case['request_rate']} [{run_id}]")

    def git(*argv):
        return subprocess.run(["git", "-C", repo] + list(argv),
                              capture_output=True, text=True, timeout=120)

    try:
        branch = (git("branch", "--show-current").stdout or "").strip()
        if not branch:
            return
        git("pull", "--rebase", "origin", branch)
        add = git("add", rel)
        if add.returncode != 0:
            print(f"[warn] git add {rel} 失败，跳过推送: {add.stderr.strip()}")
            return
        if git("diff", "--cached", "--quiet").returncode == 0:
            return
        commit = git("commit", "-m", msg)
        if commit.returncode != 0:
            print(f"[warn] git commit 失败，跳过推送: {commit.stderr.strip()}")
            return
        push = git("push", "origin", branch)
        if push.returncode != 0:
            print(f"[warn] git push 失败（不影响压测）: {push.stderr.strip()}")
        else:
            print(f"[git] 已推送 {rel} -> {branch} ({msg})")
    except Exception as exc:
        print(f"[warn] git 推送异常（不影响压测）: {exc}")


def run_one(case, args, flags, run_id, run_dir, index, total):
    test_dir = os.path.join(run_dir, case["phase"], case_dir_name(case))
    if case["phase"] == "long":
        num_prompts = args.num_prompts_long
        max_concurrency = args.max_concurrency if args.max_concurrency \
            else MAX_CONCURRENT_LONG
    else:
        num_prompts = args.num_prompts
        max_concurrency = args.max_concurrency if args.max_concurrency \
            else MAX_CONCURRENT_SHORT
    if args.resume and os.path.exists(os.path.join(test_dir, "result.json")):
        print(f"[{index}/{total}] {case_dir_name(case)} 已存在 result.json，跳过")
        return

    cmd = build_command(case, num_prompts, max_concurrency, args, flags, test_dir)
    if args.dry_run or args.verbose:
        print("  cmd: " + " ".join(cmd))
    if args.dry_run:
        return
    os.makedirs(test_dir, exist_ok=True)

    probe = args.gpu_probe if not args.no_gpu_monitor else None
    monitor = SystemMonitor(test_dir, interval=args.gpu_interval,
                            gpu_ids=args.gpu_ids, gpu_probe=probe,
                            probe_name=args.gpu_probe_name,
                            sys_mem_warn_mb=args.sys_mem_warn_mb,
                            gpu_mem_warn_pct=args.gpu_mem_warn_pct)
    monitor.start()

    start_time = dt.datetime.now()
    started = time.time()
    log_path = os.path.join(test_dir, "test.log")
    meta = {"run_id": run_id, "command": cmd, "start_time": start_time.isoformat()}
    try:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                status = "TIMEOUT"
            else:
                status = "OK" if proc.returncode == 0 else f"FAILED(rc={proc.returncode})"
        duration = time.time() - started

        if status != "OK":
            tail = ""
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                tail = "\n".join(fh.readlines()[-200:])
            if re.search(r"(?i)out of memory|\bcuda out of memory\b|\boom\b", tail):
                status = "OOM"
            print(f"    {status}，日志尾部：")
            print("    " + "\n    ".join(tail[-1200:].splitlines()[-8:]))
            error = tail[-1200:].strip()
        else:
            error = ""
    finally:
        monitor.stop()
        monitor.join(timeout=5)

    end_time = dt.datetime.now()
    leak_warning = monitor.finish()

    row = {
        "run_id": run_id, "phase": case["phase"],
        "input_len": case["input_len"], "output_len": case["output_len"],
        "request_rate": str(case["request_rate"]), "num_prompts": num_prompts,
        "status": status, "error": error,
        "leak_warning": leak_warning,
        "start_time": start_time.isoformat(), "end_time": end_time.isoformat(),
        "duration_s": f"{duration:.1f}",
    }
    result_path = os.path.join(test_dir, "result.json")
    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as fh:
                row.update(extract_metrics(json.load(fh)))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] 解析 {result_path} 失败: {exc}")
    meta["end_time"] = end_time.isoformat()
    meta["duration_s"] = round(duration, 1)
    meta["status"] = status
    if leak_warning:
        meta["leak_warning"] = leak_warning
    with open(os.path.join(test_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    thr = row.get("output_throughput")
    ttft = row.get("p99_ttft_ms")
    line = (f"[{index}/{total}] {case['phase']:>5} in={case['input_len']:>6} "
            f"out={case['output_len']:>5} rate={case['request_rate']:>3} -> {status}")
    if thr is not None:
        line += f" | out_thr={float(thr):.1f} tok/s"
    if ttft is not None:
        line += f" | p99_ttft={float(ttft):.0f} ms"
    if leak_warning:
        line += f" | [泄漏] {leak_warning}"
    print(line)
    with open(os.path.join(run_dir, "summary.csv"), "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writerow(row)
        fh.flush()
    if status == "OK" and os.path.exists(result_path):
        git_push_results(args, run_id, case, test_dir)
    return row


def export_run_metadata(run_dir, run_id, args, total, rows, run_started):
    """整个 run 结束后按当前时间导出带时间戳的 metadata_*.json（run 级汇总）。"""
    from collections import Counter
    finished = dt.datetime.now()
    counter = Counter((r.get("status") or "") for r in rows)
    meta = {
        "run_id": run_id,
        "script": os.path.basename(sys.argv[0]),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "model_path": args.model_path,
        "base_url": args.base_url,
        "result_dir": run_dir,
        "matrix_total": total,
        "completed": len(rows),
        "status_count": dict(counter),
        "started_at": run_started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": round((finished - run_started).total_seconds(), 1),
    }
    path = os.path.join(run_dir, f"metadata_{finished.strftime('%Y%m%d-%H%M%S')}.json")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        return path
    except OSError as exc:
        print(f"[warn] 导出 run 元数据失败: {exc}")
        return None


def run_warmup(args, flags, run_id):
    case = {"phase": "warmup", "input_len": 1024, "output_len": 1024,
            "request_rate": "1"}
    print("[warmup] 预热：1024 in / 1024 out，5 个请求（与短矩阵首组对齐）")
    warmup_cmd = build_command(case, 5, 1, args, flags,
                               os.path.join(args.result_dir, run_id, ".warmup"))
    try:
        proc = subprocess.run(warmup_cmd, capture_output=True, text=True,
                              timeout=300)
        print(f"[warmup] 预热完成 rc={proc.returncode}")
    except subprocess.TimeoutExpired:
        print("[warmup] 预热超时（忽略，继续矩阵）")
    except Exception as exc:
        print(f"[warmup] 预热失败（忽略，继续矩阵）: {exc}")


def main():
    args = parse_args()
    args.result_dir = args.result_dir or RESULTS_DIR
    if not args.dry_run and not shutil.which(args.vllm_cmd.split()[0]):
        sys.exit(f"错误：未找到 vllm 命令 '{args.vllm_cmd}'，"
                 f"请确认在 venv 中执行（--dry-run 可跳过检查）")
    if not args.dry_run:
        validate_paths(args)
        print(f"[info] 压测结果目录: {args.result_dir}（tokenizer: {args.tokenizer}，"
              f"模型权重: {args.model_path}）")

    vllm_cmd_list = args.vllm_cmd.split()
    print("vLLM bench serve 参数探测中 ...")
    flags, probe_ok = probe_vllm_flags(vllm_cmd_list)
    print("vLLM 支持参数: ignore-eos=" + str("--ignore-eos" in flags) +
          ", no-enable-prefix-caching=" + str("--no-enable-prefix-caching" in flags))

    cases = list(build_cases(args))
    if not cases:
        sys.exit("错误：矩阵为空，请检查 --phase/--input-lens/--request-rates 参数")
    total = len(cases)
    print(f"矩阵规模：{total} 组测试（short={sum(1 for c in cases if c['phase']=='short')}，"
          f"long={sum(1 for c in cases if c['phase']=='long')}）")

    args.gpu_probe = None
    args.gpu_probe_name = None
    if args.gpu_monitor_cmd:
        args.gpu_probe = make_probe(args.gpu_monitor_cmd, args.gpu_regex)
        if args.gpu_probe is not None:
            args.gpu_probe_name = f"custom({args.gpu_monitor_cmd})"
            print(f"GPU 采样使用: {args.gpu_monitor_cmd}（间隔 {args.gpu_interval}s）")
    if args.gpu_probe is None:
        print("[info] 未配置 --gpu-monitor-cmd，仅监控 CPU/系统内存，"
              "不采样 GPU（需要 GPU 指标时用 --gpu-monitor-cmd 指定，"
              "输出格式: 每行逗号分隔数字列 util,mem_used,mem_total[,power,temp]）")
    print(f"监控输出: 每组测试目录下的 sys_stats.csv / monitor_events.log"
          f"{' / gpu_stats.csv' if args.gpu_probe is not None else ''}；"
          f"泄漏检测: 系统内存≥50MB/min、显存≥100MB/min")
    if os.path.isdir(os.path.join(args.result_dir, ".git")):
        print(f"[info] {args.result_dir} 为 git 仓库，每组测试成功后自动 "
              f"commit + push（失败不影响压测）")

    run_id = args.run_id or dt.datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_started = dt.datetime.now()
    run_dir = os.path.join(args.result_dir, run_id)
    if not args.dry_run:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as fh:
            config = {k: v for k, v in vars(args).items()
                      if not k.startswith("gpu_probe")}
            json.dump(config, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS).writeheader()

    if args.warmup and not args.dry_run:
        run_warmup(args, flags, run_id)
        time.sleep(2)

    completed = []
    for idx, case in enumerate(cases, 1):
        row = run_one(case, args, flags, run_id, run_dir, idx, total)
        if row:
            completed.append(row)
        if idx < total and args.sleep_between > 0 and not args.dry_run:
            time.sleep(args.sleep_between)

    print(f"\n全部完成：{total} 组。结果目录: {run_dir}")
    print(f"汇总: {os.path.join(run_dir, 'summary.csv')}")
    if not args.dry_run:
        meta_path = export_run_metadata(run_dir, run_id, args, total,
                                        completed, run_started)
        if meta_path:
            print(f"元数据: {meta_path}")
    print("下一步: python plot_results.py --results-dir " + run_dir)


if __name__ == "__main__":
    main()