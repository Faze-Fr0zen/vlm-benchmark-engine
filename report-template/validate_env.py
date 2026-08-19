"""hardware_env.json 校验脚本（零第三方依赖）。

检查环境信息 JSON 的字段完整性与类型，输出 ERROR / WARN / INFO 三级结果：
  - ERROR：缺失或类型错误，会导致报告无意义或脚本异常
  - WARN ：未填，对应章节将输出 TODO 或不做自动判定
  - INFO ：字段拼写检查（未知键防笔误）

用法::

    python validate_env.py --env hardware_env_ascend.json

退出码：0 = 无 ERROR；1 = 存在 ERROR。
"""

import argparse
import json
import sys

NUM = (int, float)
BOOL = bool

LEVEL_ORDER = {"INFO": 0, "WARN": 1, "ERROR": 2}

# (字段路径, 期望类型或 None=任意, 缺失级别, 说明)
RULES = [
    ("vendor", str, "ERROR", "厂商名（报告标题与输出文件名用）"),
    ("report_meta.report_id", str, "WARN", "报告编号"),
    ("report_meta.tester", str, "WARN", "测试人"),
    ("report_meta.date", str, "WARN", "测试日期"),
    ("report_meta.location", str, "INFO", "测试地点"),
    ("report_meta.test_plan_version", str, "INFO", "测试方案版本"),
    ("hardware.server.model", str, "INFO", "服务器型号"),
    ("hardware.server.cpu", str, "INFO", "CPU 型号"),
    ("hardware.server.cpu_cores", NUM, "WARN", "CPU 核数（调优指南 CPU 建议用）"),
    ("hardware.server.memory_gb", NUM, "WARN", "内存容量（调优指南内存建议用）"),
    ("hardware.server.nvme", str, "INFO", "存储"),
    ("hardware.server.os", str, "INFO", "操作系统"),
    ("hardware.server.kernel", str, "INFO", "内核版本"),
    ("hardware.server.docker", str, "INFO", "容器环境"),
    ("hardware.accelerator.name", str, "ERROR", "加速卡型号（报告关键信息）"),
    ("hardware.accelerator.arch", str, "INFO", "架构"),
    ("hardware.accelerator.process_nm", NUM, "INFO", "制程"),
    ("hardware.accelerator.fp16_tflops", NUM, "INFO", "FP16 算力"),
    ("hardware.accelerator.bf16_tflops", NUM, "INFO", "BF16 算力"),
    ("hardware.accelerator.fp8_tflops", NUM, "INFO", "FP8 算力"),
    ("hardware.accelerator.int8_tops", NUM, "INFO", "INT8 算力"),
    ("hardware.accelerator.hbm_capacity_gb", NUM, "WARN",
     "单卡显存（显存估算 / 达标判定用）"),
    ("hardware.accelerator.hbm_bandwidth_gbps", NUM, "INFO", "显存带宽"),
    ("hardware.accelerator.tdp_w", NUM, "WARN", "单卡功耗（调优指南功耗预算用）"),
    ("hardware.accelerator.cards_installed", NUM, "INFO", "装机卡数"),
    ("hardware.accelerator.cards_used", NUM, "WARN", "测试用卡数（效率指标用）"),
    ("hardware.accelerator.interconnect_type", str, "INFO", "互联类型"),
    ("hardware.accelerator.interconnect_bandwidth", str, "INFO", "互联带宽"),
    ("hardware.accelerator.topology_note", str, "INFO", "拓扑说明"),
    ("hardware.software_stack.driver_version", str, "WARN", "驱动版本"),
    ("hardware.software_stack.runtime.name", str, "INFO", "运行时名称"),
    ("hardware.software_stack.runtime.version", str, "INFO", "运行时版本"),
    ("hardware.software_stack.inference_engine.name", str, "INFO", "推理引擎"),
    ("hardware.software_stack.inference_engine.version", str, "WARN", "推理引擎版本"),
    ("hardware.software_stack.inference_engine.fork", str, "INFO", "引擎分支"),
    ("hardware.software_stack.inference_engine.notes", str, "INFO", "引擎备注"),
    ("hardware.software_stack.container_image", str, "INFO", "容器镜像"),
    ("hardware.software_stack.compute_lib", str, "INFO", "计算库"),
    ("hardware.gpu_probe.short_name", str, "INFO", "GPU 探测短名"),
    ("hardware.gpu_probe.command", str, "WARN", "GPU 探测命令（压测 GPU 采样用）"),
    ("hardware.gpu_probe.regex", str, "INFO", "探测正则"),
    ("hardware.gpu_probe.notes", str, "INFO", "探测备注"),
    ("model_deployment.model.name", str, "ERROR", "模型名"),
    ("model_deployment.model.params_b", NUM, "WARN", "参数量（调优指南权重估算用）"),
    ("model_deployment.model.quantization", str, "INFO", "量化方式"),
    ("model_deployment.model.context_len", NUM, "INFO", "上下文长度"),
    ("model_deployment.model.tokenizer", str, "WARN", "tokenizer 路径"),
    ("model_deployment.model.model_path", str, "INFO", "权重路径"),
    ("model_deployment.engine_args.tensor_parallel_size", NUM, "INFO", "张量并行"),
    ("model_deployment.engine_args.pipeline_parallel_size", NUM, "INFO", "流水线并行"),
    ("model_deployment.engine_args.gpu_memory_utilization", NUM, "INFO", "显存利用率"),
    ("model_deployment.engine_args.max_num_seqs", NUM, "INFO", "并发上限"),
    ("model_deployment.engine_args.max_num_batched_tokens", NUM, "INFO", "单批 token"),
    ("model_deployment.engine_args.enable_chunked_prefill", BOOL, "INFO", "分块预填充"),
    ("model_deployment.engine_args.kv_cache_dtype", str, "INFO", "KV Cache 精度"),
    ("model_deployment.engine_args.enable_prefix_caching", BOOL, "INFO", "前缀缓存"),
    ("model_deployment.engine_args.other_flags", str, "INFO", "其他参数"),
    ("model_deployment.bench_args.num_prompts_short", NUM, "INFO", "短文本请求数"),
    ("model_deployment.bench_args.num_prompts_long", NUM, "INFO", "长文本请求数"),
    ("model_deployment.bench_args.max_concurrency_short", NUM, "INFO", "短文本并发"),
    ("model_deployment.bench_args.max_concurrency_long", NUM, "INFO", "长文本并发"),
    ("model_deployment.bench_args.bench_tool", str, "INFO", "压测工具"),
    ("model_deployment.bench_args.bench_tool_version", str, "INFO", "压测工具版本"),
    ("acceptance_lines.short_peak_tok_s", NUM, "WARN", "短文本极限吞吐达标线"),
    ("acceptance_lines.long_peak_tok_s", NUM, "WARN", "长文本极限吞吐达标线"),
    ("acceptance_lines.p99_ttft_ratio_max", NUM, "INFO", "TTFT 基线倍数上限"),
    ("acceptance_lines.max_failure_rate_pct", NUM, "WARN", "失败率上限"),
    ("acceptance_lines.max_bad_groups", NUM, "WARN", "异常组数上限"),
    ("acceptance_lines.stability_redlines_ok", BOOL, "INFO", "稳定性红线要求"),
    ("acceptance_lines.min_gpu_util_pct", NUM, "WARN", "GPU 利用率下限"),
]

# 每个对象允许的键集合（_comment 恒允许），用于拼写检查
KNOWN_KEYS = {
    "": {"vendor", "report_meta", "hardware", "model_deployment",
         "acceptance_lines", "_comment"},
    "report_meta": {"report_id", "tester", "date", "location",
                    "test_plan_version"},
    "hardware": {"server", "accelerator", "software_stack", "gpu_probe"},
    "hardware.server": {"model", "cpu", "cpu_cores", "memory_gb", "nvme",
                        "os", "kernel", "docker"},
    "hardware.accelerator": {"name", "arch", "process_nm", "fp16_tflops",
                             "bf16_tflops", "fp8_tflops", "int8_tops",
                             "hbm_capacity_gb", "hbm_bandwidth_gbps",
                             "tdp_w", "cards_installed", "cards_used",
                             "interconnect_type", "interconnect_bandwidth",
                             "topology_note"},
    "hardware.software_stack": {"driver_version", "runtime",
                                "inference_engine", "container_image",
                                "compute_lib"},
    "hardware.software_stack.runtime": {"name", "version"},
    "hardware.software_stack.inference_engine": {"name", "version", "fork",
                                                 "notes"},
    "hardware.gpu_probe": {"short_name", "command", "regex", "notes"},
    "model_deployment": {"model", "engine_args", "bench_args"},
    "model_deployment.model": {"name", "params_b", "quantization",
                               "context_len", "tokenizer", "model_path"},
    "model_deployment.engine_args": {"tensor_parallel_size",
                                     "pipeline_parallel_size",
                                     "gpu_memory_utilization",
                                     "max_num_seqs",
                                     "max_num_batched_tokens",
                                     "enable_chunked_prefill",
                                     "kv_cache_dtype",
                                     "enable_prefix_caching",
                                     "other_flags"},
    "model_deployment.bench_args": {"num_prompts_short", "num_prompts_long",
                                    "max_concurrency_short",
                                    "max_concurrency_long", "bench_tool",
                                    "bench_tool_version"},
    "acceptance_lines": {"short_peak_tok_s", "long_peak_tok_s",
                         "p99_ttft_ratio_max", "max_failure_rate_pct",
                         "max_bad_groups", "stability_redlines_ok",
                         "min_gpu_util_pct", "_comment"},
}


def get_path(obj, path):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None, False
        if part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def check_spelling(obj, path=""):
    """递归检查未知键（防笔误），返回 [(path, key)]。"""
    found = []
    if not isinstance(obj, dict):
        return found
    allowed = KNOWN_KEYS.get(path, set())
    for key in obj:
        if allowed and key not in allowed:
            found.append((path or "<root>", key))
        found += check_spelling(obj[key], f"{path}.{key}" if path else key)
    return found


def is_missing(value):
    return value is None or value == "" or value == []


def main():
    p = argparse.ArgumentParser(description="hardware_env.json 校验脚本")
    p.add_argument("--env", required=True, help="hardware_env.json 路径")
    args = p.parse_args()

    try:
        with open(args.env, "r", encoding="utf-8") as fh:
            env = json.load(fh)
    except OSError as exc:
        print(f"[ERROR] 无法读取文件: {exc}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] JSON 语法错误（行 {exc.lineno} 列 {exc.colno}）: "
              f"{exc.msg}")
        sys.exit(1)

    issues = []
    for path, expected, level, desc in RULES:
        value, exists = get_path(env, path)
        if not exists or is_missing(value):
            issues.append((level, path, f"未填写（{desc}）"))
        elif expected is not None and not isinstance(value, expected):
            issues.append(("ERROR", path,
                           f"类型应为 {expected.__name__}，实际 "
                           f"{type(value).__name__}（{desc}）"))

    for path, key in check_spelling(env):
        issues.append(("WARN", f"{path}.{key}", "未知字段（拼写检查，疑似笔误）"))

    lines = env.get("acceptance_lines") or {}
    if all(is_missing(v) for k, v in lines.items()
           if k != "_comment"):
        issues.append(("WARN", "acceptance_lines",
                       "未配置任何达标线，报告 5.1 节将全部输出 TODO 人工判定"))

    issues.sort(key=lambda item: -LEVEL_ORDER[item[0]])
    for level, path, msg in issues:
        print(f"[{level}] {path}: {msg}")

    n_error = sum(1 for level, _, _ in issues if level == "ERROR")
    n_warn = sum(1 for level, _, _ in issues if level == "WARN")
    print(f"\n共 {len(issues)} 项：ERROR {n_error} / WARN {n_warn}")
    if n_error:
        print("校验不通过：存在 ERROR，建议修复后使用")
        sys.exit(1)
    if n_warn:
        print("校验通过（有 WARN，对应章节将输出 TODO，不影响脚本运行）")
    else:
        print("校验通过")


if __name__ == "__main__":
    main()
