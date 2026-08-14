"""vLLM 轻量压测矩阵脚本（小模型 / 单机小卡流程验证用）。

逻辑与监控完全复用 run_matrix.py（含 sys_monitor 的系统监控、OOM 告警、
日志扫描与泄漏检测），仅覆盖矩阵规模与默认参数，适合 4090 等 1~2 卡
小模型（如 Qwen3-0.6B）先跑通流程：

  短/中文本矩阵: 输入 512/1K/2K x 输出 256/512/1K x 速率 1/4/8/16/32
  长文本矩阵:    输入 4K/8K/16K  x 输出 512/1K    x 速率 1/4/8

用法示例::

    python run_matrix_lite.py --model Qwen/Qwen3-0.6B --phase short --warmup
    python run_matrix_lite.py --model Qwen/Qwen3-0.6B --phase long --num-prompts-long 10
    python run_matrix_lite.py --dry-run

全部参数同 run_matrix.py（--base-url/--resume/--gpu-monitor-cmd 等），
生成的结果目录结构与 plot_results.py 完全兼容。
"""

import run_matrix as rm

rm.SHORT_INPUTS = [1024, 2048, 4096]
rm.SHORT_OUTPUTS = [512, 1024, 2048]
rm.SHORT_RATES = ["4", "8", "16", "32", "inf"]
rm.LONG_INPUTS = [8192, 16384, 32768]
rm.LONG_OUTPUTS = [1024, 2048]
rm.LONG_RATES = ["4", "8", "16"]

rm.DEFAULT_NUM_PROMPTS_SHORT = 120
rm.DEFAULT_NUM_PROMPTS_LONG = 16
rm.MAX_CONCURRENT_SHORT = 64
rm.MAX_CONCURRENT_LONG = 32

if __name__ == "__main__":
    rm.main()