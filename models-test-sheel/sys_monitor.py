"""通用系统监控模块（零第三方依赖），供 run_matrix.py / run_matrix_lite.py 共用。

在每组压测期间后台采样并写入测试目录：
  sys_stats.csv        CPU 利用率 / 系统内存 / Swap
  gpu_stats.csv        仅当配置了 --gpu-monitor-cmd 时生成
  monitor_events.log   告警事件（OOM、日志异常、泄漏判定）

默认只监控 CPU 与系统内存，不探测任何 GPU 工具；需要 GPU 指标时，
由调用方显式传入探测命令（如 nvidia-smi / npu-smi / 任意国产卡工具），
格式约定为每行逗号分隔数字列：
  util,mem_used,mem_total[,power,temp]
或通过命名组正则指定：util/mem_used/mem_total/power/temp。

泄漏判定：对采样序列做线性回归，斜率超过阈值且拟合优度 R2>=0.5、
累计增长超过阈值时判定“疑似泄漏”并在测试结束时回填 summary 的 leak_warning 列。
"""

import csv
import os
import re
import subprocess
import threading
import time

ERROR_PATTERNS = [
    r"(?i)\bout of memory\b",
    r"(?i)\bcuda out of memory\b",
    r"(?i)\boom\b",
    r"(?i)failed to allocate",
    r"(?i)\btraceback\b",
    r"(?i)\bexception\b",
    r"(?i)connection refused",
    r"(?i)connection reset",
    r"(?i)empty response",
    r"(?i)worker .*(crash|died|failed)",
    r"(?i)request failed with status",
    r"HTTP request .* (500|502|503|504)\b",
]

GPU_STATS_HEADER = ["timestamp", "gpu", "util_pct", "mem_util_pct",
                    "mem_used_mib", "mem_total_mib", "power_w", "temp_c"]
SYS_STATS_HEADER = ["timestamp", "cpu_util_pct", "sys_mem_used_mib",
                    "sys_mem_total_mib", "sys_mem_avail_mib",
                    "swap_used_mib", "swap_total_mib"]
EVENT_HEADER = ["time", "level", "message"]

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _first_number(field):
    m = _NUM_RE.search(str(field))
    return float(m.group()) if m else None


class CpuMemSampler:
    def __init__(self):
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None
        self._proc_last = None

    def sample(self):
        if self._psutil is not None:
            try:
                cpu = self._psutil.cpu_percent(interval=None)
                mem = self._psutil.virtual_memory()
                swap = self._psutil.swap_memory()
                return (cpu, mem.used / 1048576, mem.total / 1048576,
                        mem.available / 1048576, swap.used / 1048576,
                        swap.total / 1048576)
            except Exception:
                pass
        if os.path.exists("/proc/stat") and os.path.exists("/proc/meminfo"):
            return self._sample_proc()
        return None

    def _sample_proc(self):
        try:
            with open("/proc/stat", "r") as fh:
                fields = fh.readline().split()[1:]
            vals = [int(x) for x in fields]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            cpu = None
            if self._proc_last:
                d_idle = idle - self._proc_last[0]
                d_total = total - self._proc_last[1]
                if d_total > 0:
                    cpu = 100.0 * (1 - d_idle / d_total)
            self._proc_last = (idle, total)
            mem = {}
            with open("/proc/meminfo", "r") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem[parts[0].rstrip(":")] = int(parts[1]) / 1024
            total_mem = mem.get("MemTotal")
            if total_mem is None:
                return None
            avail = mem.get("MemAvailable")
            used = total_mem - (avail if avail is not None else 0)
            swap_total = mem.get("SwapTotal", 0)
            swap_free = mem.get("SwapFree", swap_total)
            return (cpu, used, total_mem, avail, swap_total - swap_free,
                    swap_total)
        except Exception:
            return None


class CustomProbe:
    def __init__(self, cmd, regex=None, raw=False):
        self.cmd = cmd.split()
        self.regex = re.compile(regex) if regex else None
        self.raw = raw  # True: 返回原始数字列表（列数不定），由调用方自行解包

    def sample(self):
        try:
            proc = subprocess.run(self.cmd, capture_output=True, text=True,
                                  timeout=20)
            if proc.returncode != 0:
                return []
            text = proc.stdout + "\n" + proc.stderr
            return self.parse(text)
        except Exception:
            return []

    def parse(self, text):
        rows = []
        if self.regex:
            for m in self.regex.finditer(text):
                g = m.groupdict()
                rows.append((_first_number(g.get("util")),
                             None,
                             _first_number(g.get("mem_used")),
                             _first_number(g.get("mem_total")),
                             _first_number(g.get("power")),
                             _first_number(g.get("temp"))))
        else:
            for line in text.strip().splitlines():
                nums = [_first_number(x) for x in line.split(",")]
                if not nums or nums[0] is None:
                    continue
                if self.raw:
                    rows.append(nums)
                elif len(nums) >= 3:
                    rows.append((nums[0], None, nums[1], nums[2],
                                 nums[3] if len(nums) > 3 else None,
                                 nums[4] if len(nums) > 4 else None))
        return rows


GPU_PROBES = {
    # 新增厂商: 在此注册探测命令，--gpu-monitor-cmd 直接传名字即可
    "nvidia": ("nvidia-smi --query-gpu=utilization.gpu,memory.used,"
               "memory.total,power.draw,temperature.gpu "
               "--format=csv,noheader,nounits"),
    "npu-smi": ("npu-smi info -t usage -i 0"),  # 占位示例，按实际工具调整
}


def make_probe(gpu_cmd=None, gpu_regex=None):
    """按名称（GPU_PROBES 注册表）或直接命令字符串创建 GPU 探测。
    新增厂商: 在 GPU_PROBES 加一行 + --gpu-monitor-cmd 换名字即可。"""
    if gpu_cmd:
        cmd = GPU_PROBES.get(gpu_cmd, gpu_cmd)
        probe = CustomProbe(cmd, gpu_regex)
        if probe.sample():
            return probe
        print(f"[warn] 自定义 GPU 探测命令执行失败或无法解析: {gpu_cmd}")
    return None


def _linreg(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2


def detect_leak(series, label, min_slope_mb_min):
    pts = [(s[0], s[1]) for s in series if s[1] is not None]
    if len(pts) < 8:
        return None
    t0 = pts[0][0]
    xs = [(t - t0) / 60.0 for t, _ in pts]
    ys = [v for _, v in pts]
    fit = _linreg(xs, ys)
    if fit is None:
        return None
    slope, r2 = fit
    rise = ys[-1] - ys[0]
    if slope >= min_slope_mb_min and r2 >= 0.5 and \
            rise >= max(ys[0] * 0.02, 64):
        return (f"{label}疑似内存泄漏: 增速 {slope:.0f} MB/min, "
                f"累计上升 {rise:.0f} MB")
    return None


class SystemMonitor(threading.Thread):
    def __init__(self, test_dir, interval=10.0, gpu_ids=None,
                 gpu_probe=None, probe_name=None,
                 sys_mem_warn_mb=2048, gpu_mem_warn_pct=95.0):
        super().__init__(daemon=True, name="sys-monitor")
        self.test_dir = test_dir
        self.interval = interval
        self.gpu_ids = gpu_ids
        self.probe = gpu_probe
        self.probe_name = probe_name
        self.sys_mem_warn_mb = sys_mem_warn_mb
        self.gpu_mem_warn_pct = gpu_mem_warn_pct
        self._stop_evt = threading.Event()
        self._cpu = CpuMemSampler()
        self._gpu_idx = 0
        self._mem_warned = False
        self._gpu_warned = set()
        self._sys_mem_series = []
        self._gpu_mem_series = {}
        self._gpu_allowed = {int(x) for x in gpu_ids.split(",")} \
            if gpu_ids else None
        self._log_fh = None
        self._log_offset = 0
        self._sampled = 0
        self.events_path = os.path.join(test_dir, "monitor_events.log")

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self._sample()
            except Exception as exc:
                self._event("WARN", f"监控采样异常: {exc}")
            self._stop_evt.wait(self.interval)

    def _sample(self):
        self._sampled += 1
        ts = time.time()
        sys_row = self._cpu.sample()
        if sys_row is not None:
            with open(os.path.join(self.test_dir, "sys_stats.csv"), "a",
                      newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if self._sampled == 1:
                    writer.writerow(SYS_STATS_HEADER)
                writer.writerow([f"{ts:.1f}"] + list(sys_row))
            avail = sys_row[3]
            if avail is not None and avail < self.sys_mem_warn_mb \
                    and not self._mem_warned:
                self._mem_warned = True
                self._event("WARN", f"系统可用内存低于 {self.sys_mem_warn_mb} MiB"
                                    f"（当前 {avail:.0f} MiB），接近 OOM")
            if sys_row[1] is not None:
                self._sys_mem_series.append((ts, sys_row[1]))

        if self.probe is not None:
            rows = self.probe.sample()
            seen = set()
            self._gpu_idx = 0
            with open(os.path.join(self.test_dir, "gpu_stats.csv"), "a",
                      newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if self._sampled == 1:
                    writer.writerow(GPU_STATS_HEADER)
                for row in rows:
                    if self._gpu_allowed is not None:
                        gpu_id = self._gpu_idx
                        self._gpu_idx += 1
                        if gpu_id not in self._gpu_allowed:
                            continue
                    else:
                        gpu_id = len(seen)
                    seen.add(gpu_id)
                    writer.writerow([f"{ts:.1f}", gpu_id] + ["" if v is None
                                                             else f"{v:.1f}"
                                                             for v in row])
                    if row[2] is not None and row[3]:
                        self._gpu_mem_series.setdefault(gpu_id, []).append(
                            (ts, row[2]))
                    if row[1] is not None and \
                            row[1] >= self.gpu_mem_warn_pct and \
                            gpu_id not in self._gpu_warned:
                        self._gpu_warned.add(gpu_id)
                        self._event("WARN",
                                    f"GPU{gpu_id} 显存利用率达 {row[1]:.0f}% "
                                    f"（阈值 {self.gpu_mem_warn_pct}%），"
                                    f"接近显存 OOM")
        self._scan_log()

    def _scan_log(self):
        path = os.path.join(self.test_dir, "test.log")
        try:
            if not os.path.exists(path):
                return
            if self._log_fh is None:
                self._log_fh = open(path, "r", encoding="utf-8",
                                    errors="replace")
                self._log_offset = 0
            self._log_fh.seek(self._log_offset)
            chunk = self._log_fh.read(65536)
            self._log_offset = self._log_fh.tell()
            for line in chunk.splitlines():
                if not line.strip():
                    continue
                for pat in ERROR_PATTERNS:
                    if re.search(pat, line):
                        self._event("ERROR",
                                    f"压测日志异常: {line.strip()[:180]}")
                        break
        except Exception:
            pass

    def _event(self, level, message):
        first = not os.path.exists(self.events_path)
        with open(self.events_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if first:
                writer.writerow(EVENT_HEADER)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                             level, message])

    def finish(self):
        warnings = []
        for label, series, threshold in (
                ("系统内存", self._sys_mem_series, 50.0),
                ("GPU0显存", self._gpu_mem_series.get(0, []), 100.0)):
            w = detect_leak(series, label, threshold)
            if w:
                warnings.append(w)
                self._event("WARN", w)
        for gpu_id, series in self._gpu_mem_series.items():
            if gpu_id == 0:
                continue
            w = detect_leak(series, f"GPU{gpu_id}显存", 100.0)
            if w:
                warnings.append(w)
                self._event("WARN", w)
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None
        return "; ".join(warnings) or ""