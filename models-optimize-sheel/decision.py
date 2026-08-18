"""调优决策闭环：按《模型性能优化方案书》4.2 监控→调优映射表实现。

对一次探针结果（short+long 双探针 + GPU 采样）给出：
  1. verdict   : ACCEPT / REJECT / WARN
  2. suggestions: 针对观测现象的具体调优动作（4.2 表格映射）
  3. score     : 0~100 的配置评分（决策用，供收敛排序）

评分原则（与 4.2 闭环一致，先红线后性能）：
  - 硬性淘汰（REJECT）：任一探针失败/OOM/请求失败、显存峰值 > 95%
    （KV Cache 溢出风险）、TTFT P99 > 5s（Prefill 排队失控）
  - 软告警（WARN）：
      * GPU 利用率 < 70%            -> 建议 ↑ max-num-batched-tokens / 检查 enforce-eager
      * GPU 利用率 > 90% 且吞吐不涨  -> 算力天花板，接受该配置
      * TTFT P99 周期性尖峰          -> 建议开启 chunked-prefill
  - 评分：短探针 output_throughput 相对本阶段最优归一化（×60）
          + 长探针 output_throughput 归一化（×20）
          + 延迟项：TTFT/TTPOT 超阈值按比例扣分（×20）
"""


TTFT_WARN_MS = 2000.0      # TTFT P99 软告警阈值
TTFT_REJECT_MS = 5000.0    # TTFT P99 硬淘汰阈值
MEM_REJECT_PCT = 95.0      # 显存峰值硬淘汰阈值
UTIL_LOW_PCT = 70.0        # 利用率过低告警
UTIL_HIGH_PCT = 90.0       # 利用率过高但吞吐不涨告警
TPOT_WARN_MS = 500.0       # TPOT P99 软告警阈值

SUGGEST_MAP = {
    "mem_high": "↓ max-num-seqs 或检查 kv-cache-dtype 是否降级（KV Cache 溢出）",
    "util_low": "↑ max-num-batched-tokens；检查是否误开 enforce-eager（batch 不够大/CPU 调度瓶颈）",
    "util_high_flat": "算力已达物理天花板，接受当前极限或增加 TP 卡数",
    "ttft_spike": "开启 enable-chunked-prefill；或重启 Serve 清理显存碎片",
    "tpot_high": "↓ tensor-parallel-size 或检查卡间互联带宽（多卡场景）",
}


def evaluate(results, best_throughput=None):
    """results: {"short": metrics, "long": metrics}（双探针输出）。
    返回 dict: verdict / score / suggestions / metrics 摘要。"""
    short = results.get("short") or {}
    long = results.get("long") or {}
    suggestions = []
    problems = []

    def check_ok(name, m):
        if not m:
            return True
        status = m.get("status")
        if status not in (None, "OK"):
            problems.append(f"{name}探针 {status}")
            return False
        failed = m.get("num_failed") or 0
        if failed > 0:
            problems.append(f"{name}探针失败请求 {failed}")
            return False
        return True

    check_ok("short", short)
    check_ok("long", long)

    mem_max = max([short.get("gpu_mem_pct_max") or 0,
                   long.get("gpu_mem_pct_max") or 0])
    if mem_max > MEM_REJECT_PCT:
        problems.append(f"显存峰值 {mem_max:.0f}% > {MEM_REJECT_PCT:.0f}%")
        suggestions.append(SUGGEST_MAP["mem_high"])

    ttft = max([short.get("p99_ttft_ms") or 0,
                long.get("p99_ttft_ms") or 0])
    if ttft > TTFT_REJECT_MS:
        problems.append(f"TTFT P99 {ttft:.0f}ms > {TTFT_REJECT_MS:.0f}ms")
    elif ttft > TTFT_WARN_MS:
        suggestions.append(SUGGEST_MAP["ttft_spike"])

    for m in (short, long):
        if not m:
            continue
        util = m.get("gpu_util_mean")
        if util is not None and util < UTIL_LOW_PCT:
            suggestions.append(SUGGEST_MAP["util_low"])
            break

    verdict = "ACCEPT"
    if problems:
        verdict = "REJECT"
    elif suggestions:
        verdict = "WARN"

    score = None
    if verdict != "REJECT":
        base = best_throughput or 1.0
        thr = (short.get("output_throughput") or 0.0) / base
        long_thr = (long.get("output_throughput") or 0.0)
        score = 60.0 * min(thr, 1.2) + 20.0 * min(long_thr / 500.0, 1.0)
        if ttft > TTFT_WARN_MS:
            score -= 10.0 * min((ttft - TTFT_WARN_MS) / TTFT_REJECT_MS, 1.0)
        tpot = max([short.get("p99_tpot_ms") or 0,
                    long.get("p99_tpot_ms") or 0])
        if tpot > TPOT_WARN_MS:
            score -= 5.0
            suggestions.append(SUGGEST_MAP["tpot_high"])
        score = max(0.0, min(score, 100.0))

    summary = {
        "short_thr": short.get("output_throughput"),
        "long_thr": long.get("output_throughput"),
        "ttft_p99_ms": ttft or None,
        "mem_pct_max": mem_max or None,
        "gpu_util_mean": short.get("gpu_util_mean"),
        "short_status": short.get("status"),
        "long_status": long.get("status"),
        "leak_warning": short.get("leak_warning") or long.get("leak_warning"),
    }
    return {"verdict": verdict, "score": score,
            "suggestions": suggestions, "problems": problems,
            "summary": summary}