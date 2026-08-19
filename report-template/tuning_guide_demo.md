# Qwen3-0.6B 参数调优指南（演示厂商（本地测试））


# 0. 摘要（一页结论）

| 项 | 值 |
|---|---|
| 模型 | Qwen3-0.6B |
| 厂商/芯片 | 演示厂商（本地测试） / RTX 4090 |
| 报告编号 | TODO |
| 测试人 | 测试员 |
| 日期 | 2026-08-19 |


## 0.1 结论速览（自动）

| 项目 | 结论 |
|---|---|
| 锁定配置 | R3-mem=0.95,seqs=128,kv=fp8,chunked=off,batched=8192 |
| 最高评分 | 89.3 |
| 短探针吞吐 (tok/s) | 1840.6 |
| 长探针吞吐 (tok/s) | 197.3 |
| TTFT p99 (ms) | 438.0 |
| 显存峰值 (%) | 70.0 |
| GPU 平均利用率 (%) | 96.0 |
| 验收收益 | TODO |


## 0.2 关键结论（人工）

- 推荐配置是否可用：TODO（可直接投产 / 有条件 / 不推荐）
- 制约因素：TODO（显存 / 并发 / 长文本 / 软件栈兼容）
- 与默认配置相比的收益与代价：TODO


# 1. 调优目标与方法（人工简述）

- 目标：TODO（在固定模型与硬件上寻找最佳 `vllm serve` 参数组合）
- 方法：分层收敛（Serve 参数分组独立扫描，不做全排列）+ 探针模式（每组双探针 short/long，<15 分钟快速评分）+ 决策闭环（方案书 4.2 映射表）
- 收敛矩阵：Round 1 显存×并发网格（9 组）→ Round 2 kv-dtype 补扫（2 组）→ Round 2.5 复活点验证（可选）→ Round 3 调度 A/B（6 组）→ 锁定；详见 `models-optimize-sheel/STEP_BY_STEP.md`


# 2. 调优结果


## 2.1 锁定配置（自动）

> 完整启动命令（来自 `best_serve_cmd.txt`）：

```bash
vllm serve /models/Qwen3-0.6B --served-model-name qwen3-0.6b --dtype half --max-model-len 32768 --gpu-memory-utilization 0.95 --max-num-seqs 128 --kv-cache-dtype fp8 --max-num-batched-tokens 8192 --disable-log-requests
```


## 2.2 关键参数取值说明（自动 + 人工）

| 参数 | 取值 | 作用 | 取值依据（人工） |
|---|---|---|---|
| --served-model-name | qwen3-0.6b | 服务对外模型名 | TODO |
| --dtype | half | 权重计算精度（影响显存与吞吐） | TODO |
| --max-model-len | 32768 | 最大上下文长度 | TODO |
| --gpu-memory-utilization | 0.95 | 显存利用率上限（KV Cache 预算来源） | TODO |
| --max-num-seqs | 128 | 并发序列上限（调度窗口） | TODO |
| --kv-cache-dtype | fp8 | KV Cache 精度（fp8 省显存） | TODO |
| --max-num-batched-tokens | 8192 | 单批次最大 token 数（Prefill 粒度） | TODO |
> 未识别的参数（无内置说明）：--disable-log-requests，作用与取值依据请在正文补充。


## 2.3 配置评分表（自动，按 score 降序）

> 数据源：`data\opt-demo\opt_summary.csv`（共 11 组）

| run | 轮次 | 配置 | verdict | score | 短吞吐 | 长吞吐 | TTFT p99(ms) | 显存峰值% | GPU均% | 问题 | 建议 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run-opt-20260818-103001 | R1 | R1-mem=0.9,seqs=64 | ACCEPT | 70.2 | 1532.4 | 214.7 | 512.0 | 52.0 | 91.0 | - | 升 batched 试高吞吐 |
| run-opt-20260818-103001 | R1 | R1-mem=0.95,seqs=64 | ACCEPT | 78.4 | 1688.9 | 208.3 | 468.0 | 58.0 | 93.0 | - | - |
| run-opt-20260818-103001 | R1 | R1-mem=0.98,seqs=64 | REJECT | - | - | 183.2 | 9800.0 | 99.0 | 87.0 | vllm out of memory | 显存天花板，不再尝试更高档 |
| run-opt-20260818-103001 | R1 | R1-mem=0.95,seqs=128 | ACCEPT | 84.6 | 1793.5 | 201.1 | 487.0 | 74.0 | 95.0 | - | 并发高吞吐好 |
| run-opt-20260818-103001 | R1 | R1-mem=0.98,seqs=128 | REJECT | - | - | 180.9 | 13200.0 | 99.0 | 85.0 | OOM + 请求失败 | 显存天花板 |
| run-opt-20260818-103001 | R2 | R2-mem=0.95,seqs=128,kv=fp8 | ACCEPT | 88.1 | 1821.7 | 198.6 | 442.0 | 71.0 | 96.0 | - | fp8 省显存提并发 |
| run-opt-20260818-103001 | R2.5 | R2.5-mem=0.98,seqs=64,kv=fp8 | REJECT | - | - | 190.5 | 12400.0 | 99.0 | 88.0 | 复活点仍 OOM | 维持 auto 判定的显存天花板 |
| run-opt-20260818-103001 | R3 | R3-mem=0.95,seqs=128,kv=fp8,chunked=off,batched=4096 | ACCEPT | 87.2 | 1805.2 | 199.4 | 450.0 | 70.0 | 95.0 | - | - |
| run-opt-20260818-103001 | R3 | R3-mem=0.95,seqs=128,kv=fp8,chunked=off,batched=8192 | ACCEPT | 89.3 | 1840.6 | 197.3 | 438.0 | 70.0 | 96.0 | - | 最优 batched |
| run-opt-20260818-103001 | R3 | R3-mem=0.95,seqs=128,kv=fp8,chunked=on,batched=8192 | ACCEPT | 86.0 | 1781.2 | 201.7 | 475.0 | 71.0 | 94.0 | - | - |
| run-opt-20260818-103001 | R3 | R3-mem=0.95,seqs=128,kv=fp8,chunked=off,batched=16384 | ACCEPT | 88.5 | 1833.9 | 198.1 | 441.0 | 70.0 | 95.0 | - | batched 触 16K 上限，可手工扩展 |


# 3. 验收对比（默认配置 vs 锁定配置）

| 指标 | before | after | 变化 |
|---|---|---|---|
| | |   | 指 | 标 |   | | |   | b | e | f | o | r | e |   | | |   | a | f | t | e | r |   | | |   | 变 | 化 |   | | |
| | | - | - | - | | | - | - | - | | | - | - | - | | | - | - | - | | |
| | |   | 验 | 收 | 对 | 比 | 数 | 据 |   | | |   | T | O | D | O |   | | |   | T | O | D | O |   | | |   | T | O | D | O |   | | |

- 收益结论（人工复核）：TODO（吞吐/时延/显存的实际改善与业务意义）


# 4. 生产部署建议


## 4.1 服务器硬件资源配置建议（自动估算 + 人工复核）

> 基于锁定配置实测与经验公式估算，**需按实际业务负载复核**。

| 资源 | 建议 | 依据 |
|---|---|---|
| GPU 型号 | RTX 4090 | 硬件环境 |
| 最小卡数（张量并行） | 1 | 权重体积 ÷ 显存利用率 ÷ 单卡容量 |
| 推荐卡数（含冗余） | 2 | 最小卡数 + 1（故障/多实例备用） |
| 单卡显存建议 | 24 GB | 权重估算 1 GB（0.6B × 2.0 B/参）；锁定配置实测显存峰值 70% × 24 GB ≈ 16.8 GB/卡；按 ×1.25 余量取常见容量档；估算值需人工复核 |
| CPU 核数建议 | ≥ 8 核 | 经验：每卡 ≥4 核 + 调度开销 |
| 内存建议 | ≥ 32 GB | 经验：≥ 4×权重，且 ≥ 32 GB |
| 存储建议 | NVMe ≥ 2 GB | 模型权重 ×2 + 日志/KV 落盘余量 |
| 互联要求 | 单卡无特殊要求 | 张量并行跨卡通信 |
| 功耗/散热预算 | ≥ 420 W | 350.0 W/卡 × 卡数 × 1.2 冗余 |


## 4.2 生产环境部署建议（人工清单）

- [ ] 部署形态：TODO（单机多卡张量并行 / 多实例负载均衡 / 容器化）
- [ ] 高可用：TODO（主备切换 / 多副本 + 流量分配）
- [ ] 健康检查与告警：TODO（`/v1/models` 探活、GPU 温度/显存/利用率阈值）
- [ ] 扩缩容策略：TODO（按吞吐目标与排队时延指标）
- [ ] 故障恢复：TODO（OOM 重启、卡故障替换、日志保留策略）
- [ ] 安全与合规：TODO（模型鉴权、接口白名单、数据不落盘）


# 5. 调优过程问题与决策记录（自动）

> 从评分表提取 REJECT/WARN 配置及其原因，作为调优过程证据。

| 配置 | verdict | 问题 | 建议 |
|---|---|---|---|
| R1-mem=0.98,seqs=64 | REJECT | vllm out of memory | 显存天花板，不再尝试更高档 |
| R1-mem=0.98,seqs=128 | REJECT | OOM + 请求失败 | 显存天花板 |
| R2.5-mem=0.98,seqs=64,kv=fp8 | REJECT | 复活点仍 OOM | 维持 auto 判定的显存天花板 |


# 附录 A 复现步骤

> 完整操作手册见 `models-optimize-sheel/STEP_BY_STEP.md`；收敛引擎规则与评分判据见 `probe_only.py` / `decision.py`。


# 附录 B 原始数据索引

| 数据 | 路径 | 说明 |
|---|---|---|
| 配置评分表 | data\opt-demo\opt_summary.csv | 所有配置逐组指标与判定 |
| 锁定启动命令 | data\opt-demo\best_serve_cmd.txt | 最佳 vLLM serve 命令（交付物） |
| 收敛旅程回放 | data\opt-demo\report.md | report.py 输出的完整回放（含引擎每步输出） |
