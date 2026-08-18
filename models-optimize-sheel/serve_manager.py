"""Serve 参数映射与显存释放检测（半自动模式，服务由用户手动启停）。

本文件不再管理服务进程生命周期（start/restart/stop 已移除）——
测试与服务分离后，vllm serve 由操作者按收敛引擎给出的命令手动启动，
调优脚本只做两件事：
  - config_to_flags(): serve 参数 dict -> CLI 参数列表（供 probe_only
    输出完整启动命令时拼装）
  - wait_gpu_freed(): 服务停止后检测显存是否回落，避免上组参数残留
    导致下一组误判 OOM

跨平台提示：显存释放检测默认走 nvidia-smi；无 nvidia-smi 的国产卡
环境可给 wait_gpu_freed() 传 gpu_free_cmd 指定任意探测命令（输出每行
第一个数字为已用显存 MiB），或传空字符串跳过释放等待。
"""

import re
import subprocess
import time


DEFAULT_FREE_CMD = ("nvidia-smi --query-gpu=memory.used "
                    "--format=csv,noheader,nounits")

# 调优参数 -> vLLM serve CLI 参数映射（None 值/False 不追加）
SERVE_FLAG_MAP = {
    "gpu_memory_utilization": ("--gpu-memory-utilization", None),
    "max_num_seqs": ("--max-num-seqs", None),
    "kv_cache_dtype": ("--kv-cache-dtype", None),
    "max_model_len": ("--max-model-len", None),
    "max_num_batched_tokens": ("--max-num-batched-tokens", None),
    "tensor_parallel_size": ("--tensor-parallel-size", None),
    "enable_chunked_prefill": ("--enable-chunked-prefill", "FLAG_ONLY"),
    "enforce_eager": ("--enforce-eager", "FLAG_ONLY"),
    "disable_prefix_caching": ("--no-enable-prefix-caching", "FLAG_ONLY"),
    "disable_speculative": ("--speculative-config", "SKIP"),  # 4090/国产卡跳过
}


def config_to_flags(config):
    """把 config 字典（serve 参数名->值）转成 CLI 参数列表。"""
    flags = []
    for key, value in config.items():
        if key not in SERVE_FLAG_MAP:
            print(f"[warn] 忽略未知 serve 参数: {key}")
            continue
        cli, mode = SERVE_FLAG_MAP[key]
        if mode == "SKIP":
            continue
        if mode == "FLAG_ONLY":
            if value:
                flags.append(cli)
        elif value is not None:
            flags += [cli, str(value)]
    return flags


def wait_gpu_freed(gpu_free_cmd=DEFAULT_FREE_CMD, timeout=120,
                   free_threshold_mib=200, poll_interval=10):
    """等待显存回落（旧服务实例退出后）。无探测命令或探测失败则跳过。

    返回 True 表示已释放（或无法探测），False 表示超时未释放。
    """
    if not gpu_free_cmd:
        print("[serve] 未配置显存探测命令，跳过显存释放等待")
        return True
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            out = subprocess.run(
                gpu_free_cmd.split(), capture_output=True,
                text=True, timeout=15).stdout
            vals = [float(x) for x in
                    re.findall(r"-?\d+(?:\.\d+)?", out)][:8]
            last = vals[0] if vals else None
            if last is not None and last < free_threshold_mib:
                return True
        except Exception as exc:
            print(f"[serve] 显存探测失败（跳过等待）: {exc}")
            return True
        time.sleep(poll_interval)
    print(f"[serve] 警告: 等待 {timeout}s 显存仍未释放"
          f"{f'（当前 {last} MiB）' if last is not None else ''}，"
          f"继续尝试（新实例将自行申请显存）")
    return False