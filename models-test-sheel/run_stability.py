"""Phase 3 极限稳定性与长跑验证（自定义轻量施压器 + 红线监控）。

按《模型测试方案书》Phase 3 要求，两种运行模式：

  P3-1 高并发饱和施压（1~2 小时）：
    --rate inf --max-concurrency <P2 极限并发> --duration 7200
    观察 Request Throughput 是否衰减、num_requests_waiting 是否无限堆积。

  P3-2 24h 生产模拟长跑：
    --rate <极限吞吐 70%~80% 对应 req/s> --duration 86400
    随机均匀长短文本混合（短 512~1K / 长 4K~8K，各 50%）。

监控红线：
  - GPU 温度触顶降频（temp 超阈 且 clocks.sm << clocks.max.sm）
  - 显存占用持续上升不回落（内存泄漏，回归斜率判定）
  - ECC 错误计数增加（uncorrected）
  - Silent Crash（HTTP 心跳连续失败）

数据源：随机生成（长度按分布采样 + 随机 token，tokenizer 校准），
流式窗口只保留在飞请求，内存 O(并发数 x 单请求)，24h 无内存压力。

依赖：Python 3.9+ 标准库 + transformers（tokenizer 生成）；GPU 探测
复用 sys_monitor 的 CustomProbe / GPU_PROBES。
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime as dt

from sys_monitor import CustomProbe, GPU_PROBES, _first_number

MODEL = "qwen3-0.6b"                  # 模型名，必须与 vllm serve --served-model-name 一致
TOKENIZER_PATH = "/models/Qwen3-0.6B"  # tokenizer 本地路径
BASE_URL = "http://localhost:8000"    # vllm 服务地址
RESULTS_DIR = "/vllm-workspace/results/vllm-bench-result"  # 结果根目录

# 默认两档长度分布（A 方案：短/长各 50%）
SHORT_IN_RANGE = (512, 1024)
LONG_IN_RANGE = (4096, 8192)
SHORT_RATIO = 0.5
OUT_RANGE = (512, 2048)               # 输出长度随机范围

# 红线阈值
TEMP_WARN_C = 85.0                    # 温度告警阈值
TEMP_DEGRADE_RATIO = 0.9             # clocks.sm < max*ratio 且温度超阈 -> 判定降频
MEM_LEAK_SLOPE_MB_MIN = 100.0        # 显存泄漏增速阈值
WAIT_QUEUE_SLOPE_THRESH = 0.5        # waiting 队列每分钟增长阈值（堆积判定）
HEARTBEAT_FAILS = 3                  # 连续心跳失败次数 -> Silent Crash
MIN_POINTS_FOR_TREND = 8             # 趋势判定最少采样点

METRIC_NAMES = ["num_requests_waiting", "num_requests_running",
                "num_requests_swapped", "num_requests_running_prefill",
                "num_requests_running_decode"]


# ---------------------------------------------------------------------------
# vLLM /metrics 解析
# ---------------------------------------------------------------------------

def fetch_vllm_metrics(base_url):
    """拉取 vLLM Prometheus 指标，返回 {指标名: 数值}。失败返回 {}。"""
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        # vLLM 0.26+ 指标带 label: vllm:num_requests_waiting{engine="0",model_name="..."} 0.0
        m = re.match(r"vllm:(\w+)(?:\{[^}]*\})?\s+([\d.e+-]+)", line)
        if m and m.group(1) in METRIC_NAMES:
            out[m.group(1)] = float(m.group(2))
    return out


def fetch_heartbeat(base_url):
    """心跳探测：GET /v1/models，返回 True/False。"""
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=15) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 请求生成（随机文本 + tokenizer 校准）
# ---------------------------------------------------------------------------

def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    except Exception as exc:
        print(f"[warn] 无法加载 tokenizer（{exc}），退化用纯 ASCII 随机文本")
        return None


class PromptGenerator:
    def __init__(self, tokenizer, rng):
        self.tok = tokenizer
        self.rng = rng
        self.vocab_size = len(tokenizer) if tokenizer is not None else 0
        self._special = set(tokenizer.all_special_ids) if tokenizer is not None else set()

    def gen(self, target_len):
        """生成约 target_len token 的 prompt 文本（随机 token id -> decode）。"""
        if self.tok is None:
            chars = "abcdefghijklmnopqrstuvwxyz0123456789,.;:!? 你我他它是一个有的和不在"
            text = "".join(self.rng.choice(chars) for _ in range(max(16, target_len * 3)))
            return text
        ids = []
        while len(ids) < target_len:
            cand = self.rng.randrange(self.vocab_size)
            if cand in self._special:
                continue
            ids.append(cand)
        return self.tok.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 红线监控线程
# ---------------------------------------------------------------------------

class StabilityMonitor(threading.Thread):
    """采样 GPU（含 clocks/ECC）+ vLLM metrics + 心跳，写时序 CSV 与事件日志。"""

    def __init__(self, test_dir, base_url, interval=15.0, gpu_probe=None,
                 gpu_probe_name=None, temp_warn_c=TEMP_WARN_C):
        super().__init__(daemon=True, name="stability-monitor")
        self.test_dir = test_dir
        self.base_url = base_url
        self.interval = interval
        self.gpu_probe = gpu_probe
        self.gpu_probe_name = gpu_probe_name
        self.temp_warn_c = temp_warn_c
        self._stop_evt = threading.Event()
        self._rows = []          # 采样行（含时间戳）
        self._events = []        # (时间, 级别, 消息)
        self._ecc_last = None    # ECC uncorrected 上次值
        self._hb_fails = 0
        self._clock_probe = None
        self._ecc_probe = None
        if gpu_probe:
            self._clock_probe = CustomProbe(
                "nvidia-smi --query-gpu=clocks.sm,clocks.max.sm "
                "--format=csv,noheader,nounits", raw=True)
            self._ecc_probe = CustomProbe(
                "nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total,"
                "ecc.errors.corrected.aggregate.total "
                "--format=csv,noheader,nounits", raw=True)

    def stop(self):
        self._stop_evt.set()

    def _event(self, level, msg):
        stamp = dt.now().isoformat(timespec="seconds")
        self._events.append((stamp, level, msg))
        with open(os.path.join(self.test_dir, "monitor_events.log"), "a",
                  encoding="utf-8") as fh:
            fh.write(f"{stamp} [{level}] {msg}\n")
        print(f"[{level}] {msg}")

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self._sample()
            except Exception as exc:
                self._event("WARN", f"监控采样异常: {exc}")
            self._stop_evt.wait(self.interval)

    def _sample(self):
        t = time.time()
        row = {"ts": f"{t:.1f}"}

        # GPU 常规指标
        if self.gpu_probe:
            g = self.gpu_probe.sample()
            if g:
                u, _, mu, mt, pw, tc = g[0]
                row["gpu_util_pct"] = u if u is not None else ""
                row["gpu_mem_used_mib"] = mu if mu is not None else ""
                row["gpu_mem_total_mib"] = mt if mt is not None else ""
                row["gpu_power_w"] = pw if pw is not None else ""
                row["gpu_temp_c"] = tc if tc is not None else ""
            # clocks（降频检测）
            cl = self._clock_probe.sample() if self._clock_probe else []
            if cl:
                row["clocks_sm_mhz"] = cl[0][0]
                row["clocks_max_mhz"] = cl[0][1]
            # ECC
            ec = self._ecc_probe.sample() if self._ecc_probe else []
            if ec and ec[0][0] is not None:
                row["ecc_uncorrected"] = ec[0][0]
                cur = float(ec[0][0])
                if self._ecc_last is not None and cur > self._ecc_last:
                    self._event("ERROR",
                                f"ECC uncorrected 计数增加: {self._ecc_last:.0f} -> {cur:.0f}")
                self._ecc_last = cur
            if ec and len(ec[0]) > 1 and ec[0][1] is not None:
                row["ecc_corrected"] = ec[0][1]

        # vLLM 队列指标
        vm = fetch_vllm_metrics(self.base_url)
        for k in METRIC_NAMES:
            if k in vm:
                row[k] = vm[k]

        # 心跳（Silent Crash）
        ok = fetch_heartbeat(self.base_url)
        row["heartbeat_ok"] = 1 if ok else 0
        if ok:
            self._hb_fails = 0
        else:
            self._hb_fails += 1
            if self._hb_fails >= HEARTBEAT_FAILS:
                self._event("ERROR", "Silent Crash：HTTP 心跳连续失败 "
                                     f"{self._hb_fails} 次，服务可能已不可用")

        # 温度/降频判定
        tc = row.get("gpu_temp_c")
        cs, cm = row.get("clocks_sm_mhz"), row.get("clocks_max_mhz")
        if tc not in ("", None) and float(tc) >= self.temp_warn_c:
            if cm not in ("", None) and cs not in ("", None):
                try:
                    if float(cs) < float(cm) * TEMP_DEGRADE_RATIO:
                        self._event("ERROR",
                                    f"GPU 温度 {tc}°C 超阈且已降频 "
                                    f"(SM {cs} / max {cm} MHz)")
                    else:
                        self._event("WARN", f"GPU 温度 {tc}°C 超阈（未降频）")
                except ValueError:
                    pass
            else:
                self._event("WARN", f"GPU 温度 {tc}°C 超阈（无 clocks 数据）")

        # 写入内存并定期落盘
        self._rows.append(row)
        self._flush()

    def _flush(self):
        path = os.path.join(self.test_dir, "stability_series.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self._rows[-1].keys()))
            if new:
                w.writeheader()
            # 只写自上次 flush 以来的新增行（避免重复累积）
            for r in self._rows[self._flushed:]:
                w.writerow(r)
            self._flushed = len(self._rows)

    _sample_flush = 200
    _flushed = 0

    _sample_flush = 200

    # ------------------------------------------------------------------
    # 趋势判定（结束时调用）
    # ------------------------------------------------------------------

    def _series(self, key):
        pts = []
        path = os.path.join(self.test_dir, "stability_series.csv")
        if not os.path.exists(path):
            return pts
        with open(path, "r", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            for r in rd:
                v = r.get(key)
                if v not in ("", None) and v is not None:
                    try:
                        pts.append((float(r["ts"]), float(v)))
                    except (KeyError, ValueError):
                        pass
        return pts

    @staticmethod
    def _slope(pts):
        if len(pts) < MIN_POINTS_FOR_TREND:
            return None
        t0 = pts[0][0]
        xs = [(t - t0) / 60.0 for t, _ in pts]
        ys = [v for _, v in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx == 0:
            return None
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return sxy / sxx

    def finalize(self):
        """返回红线判定摘要 dict（供 report 使用）。"""
        findings = {}
        # 显存泄漏：后 1/2 时段斜率
        mem = self._series("gpu_mem_used_mib")
        if len(mem) >= MIN_POINTS_FOR_TREND:
            half = max(MIN_POINTS_FOR_TREND, len(mem) // 2)
            sl = self._slope(mem[-half:])
            rise = mem[-1][1] - mem[-half][1]
            if sl is not None and sl >= MEM_LEAK_SLOPE_MB_MIN:
                findings["mem_leak"] = (f"显存疑似泄漏: 后段增速 {sl:.0f} MB/min, "
                                        f"累计上升 {rise:.0f} MB")
            else:
                findings["mem_leak"] = None
        # waiting 堆积：全时段斜率（不足 10 分钟不做判定，短测过渡段斜率无意义）
        wq = self._series("num_requests_waiting")
        if len(wq) >= MIN_POINTS_FOR_TREND:
            span = wq[-1][0] - wq[0][0]
            if span >= 600.0:
                sl = self._slope(wq)
                if sl is not None and sl >= WAIT_QUEUE_SLOPE_THRESH:
                    findings["queue_growth"] = (f"waiting 队列疑似无限堆积: "
                                                f"增速 {sl:.2f} req/min")
                else:
                    findings["queue_growth"] = None
        # 吞吐衰减：由施压器侧统计（见 LoadGenerator.finalize）
        return findings


# ---------------------------------------------------------------------------
# 轻量施压器（流式请求 + 速率控制）
# ---------------------------------------------------------------------------

class PromptPool:
    """预生成 prompt 池：后台线程持续生成，施压线程取用，生成与发送解耦。

    避免高并发下 tokenizer.decode 长 prompt 阻塞窗口补充、造成 GPU 断流。
    """

    def __init__(self, gen, min_size=16, max_size=48, fillers=2):
        self._gen = gen
        self._min = min_size
        self._max = max_size
        self._pool = []
        self._cv = threading.Condition()
        self._stop = threading.Event()
        for _ in range(fillers):
            threading.Thread(target=self._fill, daemon=True).start()

    def _fill(self):
        rng = random.Random()
        while not self._stop.is_set():
            with self._cv:
                if len(self._pool) >= self._max:
                    self._cv.wait(0.2)
                    continue
            in_len = rng.randint(*SHORT_IN_RANGE) if rng.random() < SHORT_RATIO \
                else rng.randint(*LONG_IN_RANGE)
            try:
                prompt = self._gen.gen(in_len)
            except Exception:
                time.sleep(0.1)
                continue
            with self._cv:
                self._pool.append((in_len, prompt))
                self._cv.notify()

    def get(self):
        with self._cv:
            while not self._pool:
                self._cv.wait(0.1)
            return self._pool.pop()


class LoadGenerator(threading.Thread):
    """并发窗口持续发请求，SSE 流式解析 TTFT/TPOT/E2E，统计吞吐与失败。"""

    def __init__(self, base_url, model, gen, out_range, rate, max_concurrency,
                 duration, dry_run=False):
        super().__init__(daemon=True, name="load-gen")
        self.base_url = base_url
        self.model = model
        self.gen = gen
        self.out_range = out_range
        self.rate = rate                 # req/s；inf 表示不限速（并发窗口决定）
        self.max_concurrency = max_concurrency
        self.duration = duration
        self.dry_run = dry_run
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._in_flight = 0
        self._n_ok = 0
        self._n_fail = 0
        self._n_timed_out = 0
        self._latencies = []             # (t, e2e_ms, ttft_ms, tpot_ms, out_tok)
        self._started_at = time.time()
        self._rate_window = []           # 滑动窗口: (t, ok)
        self._pool = PromptPool(gen)

    # ---- 指标（线程安全）----
    def _snapshot(self):
        with self._lock:
            return (self._n_ok, self._n_fail, self._n_timed_out,
                    list(self._latencies))

    def throughput_now(self):
        """最近 60s 完成请求数 -> req/s。"""
        now = time.time()
        with self._lock:
            self._rate_window = [(t, k) for t, k in self._rate_window
                                 if now - t < 60.0]
            return sum(1 for _, k in self._rate_window if k)

    def _report_one(self, t0, ttft, tpot, e2e_ms, out_tok, ok):
        now = time.time()
        with self._lock:
            if ok:
                self._n_ok += 1
                self._rate_window.append((now, 1))
                self._latencies.append((t0, e2e_ms, ttft, tpot, out_tok))
            else:
                self._n_fail += 1
                self._rate_window.append((now, 0))

    # ---- 单个请求 ----
    def _send_one(self, rng):
        in_len, prompt = self._pool.get()
        out_len = rng.randint(*self.out_range)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": out_len,
            "ignore_eos": True,
            "stream": True,
            "temperature": 0.0,
        }
        if self.dry_run:
            print(f"[dry-run] in={in_len} out={out_len} prompt_chars={len(prompt)}")
            time.sleep(0.01)
            self._report_one(time.time(), 0, 0, 0, out_len, True)
            return
        t0 = time.time()
        ttft = tpot = None
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            first_tok_t = None
            out_tokens = 0
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk_txt = line[5:].strip()
                    if chunk_txt == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_txt)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if first_tok_t is None and choices and \
                            (choices[0].get("text") or choices[0].get("delta")):
                        first_tok_t = time.time()
                        ttft = (first_tok_t - t0) * 1000.0
                    if choices and choices[0].get("text"):
                        out_tokens += 1
                    if "usage" in chunk:
                        usage = chunk["usage"]
                        if usage.get("completion_tokens") is not None:
                            out_tokens = max(out_tokens, usage["completion_tokens"])
                        if ttft is None:
                            ttft = 0.0
            e2e = (time.time() - t0) * 1000.0
            if out_tokens > 1 and ttft is not None:
                tpot = (e2e - ttft) / (out_tokens - 1) if out_tokens > 1 else 0.0
            self._report_one(t0, ttft, tpot, e2e, out_tokens, True)
        except Exception as exc:
            self._report_one(t0, None, None,
                             (time.time() - t0) * 1000.0, 0, False)
            if self._n_fail % 50 == 1:
                print(f"[warn] 请求失败 #{self._n_fail}: {exc}")

    def run(self):
        rng = random.Random(20260814)
        interval = 1.0 / self.rate if self.rate and math.isfinite(self.rate) else 0.0
        next_submit = time.time()
        while not self._stop_evt.is_set():
            if interval > 0:
                next_submit = max(next_submit, time.time())
                if time.time() < next_submit:
                    self._stop_evt.wait(0.05)
                    continue
                next_submit += interval
            with self._lock:
                if self._in_flight >= self.max_concurrency:
                    self._stop_evt.wait(0.05)
                    continue
                self._in_flight += 1

            def done(self=self):
                with self._lock:
                    self._in_flight -= 1
            # 直接同步发送（并发由 in_flight 计数 + 多个 worker 承担）
            threading.Thread(target=self._run_worker, args=(done,),
                             daemon=True).start()
            if (time.time() - self._started_at) > self.duration:
                break
        # 等待在飞请求结束
        while not self._stop_evt.is_set():
            with self._lock:
                if self._in_flight == 0:
                    break
            time.sleep(0.5)

    def _run_worker(self, done):
        try:
            self._send_one(random.Random())
        finally:
            done()

    def stop(self):
        self._stop_evt.set()

    def finalize(self):
        """吞吐衰减判定：前半 vs 后半平均吞吐。"""
        ok, fail, to, lats = self._snapshot()
        if ok < 40 or not lats:
            return {"throughput_decay": None,
                    "latency": None, "summary": (ok, fail, to, len(lats))}
        # 运行不足 10 分钟不做趋势判定（短测样本内长短请求分布不均，纯属噪声）
        if (lats[-1][0] - lats[0][0]) < 600.0:
            return {"throughput_decay": None,
                    "latency": None, "summary": (ok, fail, to, len(lats))}
        half = len(lats) // 2
        first_half = [e for _, e, *_ in lats[:half]]
        second_half = [e for _, e, *_ in lats[half:]]
        t0 = lats[0][0]
        t1 = lats[half - 1][0] if half > 1 else lats[-1][0]
        span = max((t1 - t0) / 60.0, 1e-6)
        thr_first = half / span
        span2 = max((lats[-1][0] - lats[half][0]) / 60.0, 1e-6) if half < len(lats) else span
        thr_second = (len(lats) - half) / span2
        decay = None
        if thr_first > 0 and thr_second < thr_first * 0.8:
            decay = (f"吞吐衰减: 前半 {thr_first:.1f} req/min -> "
                     f"后半 {thr_second:.1f} req/min (降 {100 * (1 - thr_second / thr_first):.0f}%)")
        return {"throughput_decay": decay,
                "latency": lats, "summary": (ok, fail, to, len(lats))}


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 极限稳定性与长跑验证")
    p.add_argument("--base-url", default=BASE_URL, help="vLLM 服务地址")
    p.add_argument("--model", default=MODEL, help="模型名（须与 served-model-name 一致）")
    p.add_argument("--tokenizer", default=TOKENIZER_PATH, help="tokenizer 路径")
    p.add_argument("--duration", type=int, default=7200,
                   help="运行时长（秒），如 7200（1h）/ 86400（24h）")
    p.add_argument("--rate", type=float, default=float("inf"),
                   help="请求速率 req/s（P3-1 用 inf 压满；P3-2 用极限吞吐的 70%~80%）")
    p.add_argument("--max-concurrency", type=int, default=64,
                   help="并发窗口上限（P3-1 饱和施压的关键参数）")
    p.add_argument("--short-input", nargs=2, type=int, default=list(SHORT_IN_RANGE),
                   help="短文本输入长度范围 [min max]")
    p.add_argument("--long-input", nargs=2, type=int, default=list(LONG_IN_RANGE),
                   help="长文本输入长度范围 [min max]")
    p.add_argument("--short-ratio", type=float, default=SHORT_RATIO,
                   help="短文本占比（默认 0.5，其余为长文本）")
    p.add_argument("--out-range", nargs=2, type=int, default=list(OUT_RANGE),
                   help="输出长度范围 [min max]")
    p.add_argument("--metrics-interval", type=float, default=15.0,
                   help="监控采样间隔秒数")
    p.add_argument("--temp-warn", type=float, default=TEMP_WARN_C,
                   help="温度告警阈值（℃）")
    p.add_argument("--gpu-monitor-cmd", default="nvidia",
                   help="GPU 探测（注册表短名或完整命令；不传则不采样 GPU）")
    p.add_argument("--result-dir", default=None, help="结果根目录（默认 RESULTS_DIR）")
    p.add_argument("--run-id", default=None, help="运行标识（默认按时间生成）")
    p.add_argument("--dry-run", action="store_true", help="只演示请求生成，不真正压测")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    global SHORT_IN_RANGE, LONG_IN_RANGE, SHORT_RATIO, OUT_RANGE, TOKENIZER_PATH
    SHORT_IN_RANGE = tuple(args.short_input)
    LONG_IN_RANGE = tuple(args.long_input)
    SHORT_RATIO = args.short_ratio
    OUT_RANGE = tuple(args.out_range)
    TOKENIZER_PATH = args.tokenizer

    result_root = args.result_dir or RESULTS_DIR
    stage = "p3-1" if math.isinf(args.rate) else "p3-2"
    run_id = args.run_id or dt.now().strftime(f"run-{stage}-%Y%m%d-%H%M%S")
    test_dir = os.path.join(result_root, run_id)
    if not args.dry_run:
        os.makedirs(test_dir, exist_ok=True)
        with open(os.path.join(test_dir, "run_config.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({k: v for k, v in vars(args).items()},
                      fh, ensure_ascii=False, indent=2)

    mode = "P3-1 饱和施压" if math.isinf(args.rate) else "P3-2 混合长跑"
    print(f"[{run_id}] {mode}: rate={args.rate} 并发={args.max_concurrency} "
          f"时长={args.duration}s")
    print(f"长度分布: 短{SHORT_IN_RANGE}x{SHORT_RATIO:.0%} "
          f"/ 长{LONG_IN_RANGE}x{1 - SHORT_RATIO:.0%}, "
          f"输出{OUT_RANGE}")

    # GPU 探测
    gpu_probe, gpu_name = None, None
    if args.gpu_monitor_cmd and not args.dry_run:
        cmd = GPU_PROBES.get(args.gpu_monitor_cmd, args.gpu_monitor_cmd)
        gpu_probe = CustomProbe(cmd)
        if gpu_probe.sample():
            gpu_name = f"{args.gpu_monitor_cmd}({cmd.split()[0]})"
            print(f"GPU 采样使用: {gpu_name}（间隔 {args.metrics_interval}s）")
        else:
            print(f"[warn] GPU 探测失败: {args.gpu_monitor_cmd}，仅监控系统指标")

    monitor = None
    if not args.dry_run:
        monitor = StabilityMonitor(test_dir, args.base_url,
                                   interval=args.metrics_interval,
                                   gpu_probe=gpu_probe,
                                   gpu_probe_name=gpu_name,
                                   temp_warn_c=args.temp_warn)
        monitor.start()

    tok = None if args.dry_run else _load_tokenizer()
    gen = PromptGenerator(tok, random.Random(1))
    loader = LoadGenerator(args.base_url, args.model, gen, OUT_RANGE,
                           args.rate, args.max_concurrency, args.duration,
                           dry_run=args.dry_run)
    loader.start()

    start = time.time()
    last_report = start
    try:
        while loader.is_alive():
            time.sleep(5)
            now = time.time()
            if now - last_report >= 300:
                last_report = now
                thr = loader.throughput_now()
                ok, fail, to, nl = loader._snapshot()
                print(f"[{now - start:7.0f}s] 完成 {ok} 失败 {fail} "
                      f"| 近60s {thr} req/s | 在飞 "
                      f"{loader._in_flight if hasattr(loader, '_in_flight') else '?'}")
    except KeyboardInterrupt:
        print("\n[interrupt] 收到 Ctrl+C，正在收尾…")
        loader.stop()
        monitor.stop() if monitor else None
    loader.join(timeout=60)

    if monitor:
        monitor.stop()
        monitor.join(timeout=10)
        findings = monitor.finalize()
    else:
        findings = {}

    res = loader.finalize()
    ok, fail, to, nl = res["summary"]
    elapsed = max(time.time() - start, 1e-6)
    report = {
        "run_id": run_id,
        "mode": mode,
        "duration_s": round(elapsed, 1),
        "requests_ok": ok,
        "requests_failed": fail,
        "failure_rate": round(fail / max(ok + fail, 1), 4),
        "req_per_s": round(ok / elapsed, 3),
        "trends": {"throughput_decay": res["throughput_decay"],
                   "queue_growth": findings.get("queue_growth"),
                   "mem_leak": findings.get("mem_leak")},
    }
    # 延迟分位数
    lats = res["latency"] or []
    if lats:
        e2es = sorted(e for _, e, *_ in lats)
        ttfts = sorted(t for _, _, t, _, _ in lats if t is not None)
        tdots = sorted(p for _, _, _, p, _ in lats if p is not None)

        def pct(vals, q):
            if not vals:
                return None
            idx = min(len(vals) - 1, int(q * len(vals)))
            return round(vals[idx], 2)
        report["latency_ms"] = {
            "e2e_mean": round(sum(e2es) / len(e2es), 2),
            "e2e_p99": pct(e2es, 0.99),
            "ttft_mean": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "ttft_p99": pct(ttfts, 0.99),
            "tpot_mean": round(sum(tdots) / len(tdots), 2) if tdots else None,
            "tpot_p99": pct(tdots, 0.99),
        }

    if not args.dry_run:
        with open(os.path.join(test_dir, "stability_report.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print("\n======= 稳定性报告 =======")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告: {test_dir}/stability_report.json")
    else:
        print(f"\n[dry-run] 已生成 {ok} 个请求示例，未真实压测")


if __name__ == "__main__":
    main()
