# 模型基准测试报告（自动生成草稿）

> 本文件由 `generate_report.py` 自动生成，机器可读部分已填充，
> 人工结论章节（TODO）需测试人员补填后发布。

| 字段 | 内容 |
|---|---|
| 报告编号 | TODO |
| 被测厂商 / 芯片 | TODO / TODO |
| 被测模型 | TODO |
| 测试日期 | TODO |
| 测试人 | TODO |


# 0. 摘要（一页结论）


## 0.1 被测对象

| 项 | 值 |
|---|---|
| 加速卡 | TODO × TODO 卡 |
| 软件栈 | vllm TODO |
| 模型 | TODO（TODO / TODO） |
| 部署 | TP=TODO |


## 0.2 核心结论

| 项目 | 结论 |
|---|---|
| 极限吞吐（tok/s，短文本） | 4865.9 |
| 极限吞吐（tok/s，长文本） | TODO |
| 推荐请求速率 / 并发 | 见 3.4 各组合膝点；TODO |
| P99 TTFT / TPOT（推荐点） | TODO / TODO |
| 稳定性结论 | TODO（24h 红线见第 4 章） |
| 综合评级 | TODO |


# 1. 测试环境


## 1.1 服务器硬件

| 项 | 值 |
|---|---|
| 服务器型号 | TODO |
| CPU | TODO |
| 内存 | TODO |
| 存储 | TODO |
| 操作系统 / 内核 | TODO / TODO |
| 容器运行时 | TODO |


## 1.2 加速卡

| 项 | 值 |
|---|---|
| 芯片型号 | TODO |
| 架构 / 制程 | TODO / TODO |
| 算力 FP16/BF16/FP8/INT8 | TODO / TODO / TODO / TODO TFLOPS/TOPS |
| 显存容量 / 带宽 | TODO / TODO |
| 单卡 TDP | TODO |
| 装机 / 用卡数 | TODO / TODO |
| 卡间互联 | TODO / TODO |
| 拓扑说明 | TODO |


## 1.3 软件栈

| 层 | 名称 | 版本 | 备注 |
|---|---|---|---|
| 驱动 | TODO |  |  |
| 运行时 | TODO | TODO |  |
| 推理引擎 | vllm | TODO（TODO） | TODO |
| 容器镜像 | TODO |  |  |
| 计算库 | TODO |  |  |


## 1.5 工具链

| 项 | 值 |
|---|---|
| 压测脚本版本 | TODO（git commit） |
| 压测工具 | vllm bench serve TODO |
| 监控方式 | sys_monitor.py，GPU 探测=TODO，间隔TODO s |
| 结果目录 | data\run-20260814-023631 |


# 2. 被测模型与部署配置


## 2.1 模型信息

| 项 | 值 |
|---|---|
| 模型名称 | TODO |
| 参数量 | TODO |
| 量化 | TODO |
| 上下文长度 | TODO |
| Tokenizer | TODO |
| 权重路径 | TODO |


## 2.2 推理引擎参数

| 参数 | 值 |
|---|---|
| tensor-parallel-size | TODO |
| pipeline-parallel-size | TODO |
| gpu-memory-utilization | TODO |
| max-num-seqs | TODO |
| max-num-batched-tokens | TODO |
| enable-chunked-prefill | 否 |
| kv-cache-dtype | TODO |
| enable-prefix-caching | 否 |
| 其他 flag | TODO |


## 2.3 压测参数

| 参数 | 值 |
|---|---|
| 请求数（短/长） | TODO / TODO |
| 并发上限（短/长） | TODO / TODO |
| 速率档位 | short=TODO，long=TODO |
| 强制 flag | --ignore-eos 等 |


# 3. 基准矩阵结果（Phase 2）


## 3.2 结果总览

| 状态 | 组数 |
|---|---|
| OK | 45 |
| 合计 | 45 |


## 3.3 核心指标表（每组合推荐速率点）

> 推荐速率 = 膝点/饱和点（3.4 判定口径）；全量逐组数据见附录 B。

| 输入 | 输出 | 推荐速率 | 吞吐(tok/s) | 吞吐(req/s) | P99 TTFT(ms) | P99 TPOT(ms) | 失败数 |
|---|---|---|---|---|---|---|---|
| 1024 | 512 | 32 | 4524.2 | 8.836 | 48.8 | 12.9 | 0 |
| 1024 | 1024 | 16 | 3845.7 | 3.756 | 73.4 | 15.6 | 0 |
| 1024 | 2048 | 16 | 3204.4 | 1.565 | 100.8 | 19.3 | 0 |
| 2048 | 512 | 16 | 2607.0 | 5.092 | 209.4 | 23.0 | 0 |
| 2048 | 1024 | 16 | 2579.6 | 2.519 | 200.8 | 23.9 | 0 |
| 2048 | 2048 | 4 | 2152.3 | 1.051 | 16492.6 | 30.9 | 0 |
| 4096 | 512 | 4 | 1315.0 | 2.568 | 8825.7 | 29.9 | 0 |
| 4096 | 1024 | 4 | 1323.7 | 1.293 | 25895.9 | 32.1 | 0 |
| 4096 | 2048 | 4 | 1227.0 | 0.599 | 58212.3 | 47.2 | 0 |


## 3.4 膝点 / 饱和点判定

> 判定口径：① 排队临界 rate = P99 TTFT 首次 ≥ 最低速率基线 2 倍；② 吞吐饱和 rate = 吞吐首次达峰值 95%；取二者较小者。

| 输入 | 输出 | 排队临界 rate | 吞吐饱和 rate | 结论（推荐 rate） |
|---|---|---|---|---|
| 1024 | 512 | TODO | 32 | 32 |
| 1024 | 1024 | TODO | 16 | 16 |
| 1024 | 2048 | TODO | 16 | 16 |
| 2048 | 512 | 16 | 16 | 16 |
| 2048 | 1024 | 32 | 16 | 16 |
| 2048 | 2048 | TODO | 4 | 4 |
| 4096 | 512 | TODO | 4 | 4 |
| 4096 | 1024 | TODO | 4 | 4 |
| 4096 | 2048 | TODO | 4 | 4 |


# 4. 稳定性验证（Phase 3）


## 4.1 施压配置

| 项 | 值 |
|---|---|
| 模式 | P3-1 饱和施压 |
| 时长 | 3689 s |
| 请求速率 | inf |
| 最大并发 | 64 |
| 长度分布 | TODO |


## 4.2 红线监控结果

| 红线 | 判定标准 | 结果 | 说明 |
|---|---|---|---|
| 温度触顶降频 | temp ≥ 85℃ 且 SM 时钟 < 最大 90% | TODO | 见 monitor_events.log |
| 显存泄漏 | 后 1/2 时段增速 ≥ 100 MB/min | TODO |  |
| ECC 错误 | uncorrected 计数上升 | TODO |  |
| Silent Crash | 心跳连续 3 次失败 | TODO |  |
| 吞吐衰减 | 长时段下行趋势 | TODO |  |
| 队列堆积 | waiting 持续增长 | TODO |  |


## 4.3 稳定性结论

- 事件统计（自动）：ERROR 0 条 / WARN 0 条
- 请求（自动）：总 3832，失败率 0.0000%，平均 1.039 req/s，P99 E2E 80181.7 ms
- 总体判定：TODO（通过 / 有条件通过 / 不通过）
- 说明：TODO


# 5. 调优结论与配置推荐


## 5.1 达标线判定（自动）

> 达标线配置于 `hardware_env.json` 的 `acceptance_lines`；`TODO` 表示未配置阈值或数据缺失，需人工判定。

| 指标 | 实测值 | 达标线 | 判定 |
|---|---|---|---|
| 短文本极限吞吐 (tok/s) | 4865.87 tok/s | TODO | TODO |
| 长文本极限吞吐 (tok/s) | TODO | TODO | TODO |
| 推荐点 P99 TTFT / 基线最大倍数 | 2.46 × | 2.00 × | **不达标** |
| 请求失败率 (%) | TODO | 0.00 % | TODO |
| 异常组数 (FAILED/OOM/TIMEOUT) | 0 组 | 0 组 | 达标 |
| P3 稳定性红线 | 全部通过 | 全部通过 | 达标 |
| GPU 利用率峰值 (%) | 100.00 % | 80.00 % | 达标 |

- 汇总判定（自动）：存在未达标项或待人工判定


## 5.2 推荐引擎参数（人工）

> 基于 3.4 膝点与 5.1 判定结果填写。


## 5.3 硬件建议（人工）


## 5.4 效率与性价比（自动估算）

| 指标 | 值 |
|---|---|
| 每卡吞吐（tok/s/卡，短文本） | TODO |
| 每瓦吞吐（tok/s/W，满载估算） | TODO |
| 长文本每卡吞吐（tok/s/卡） | TODO |
| 与参考平台比值 | TODO |


# 附录 A 部署踩坑与兼容性问题记录

| 编号 | 日期 | 现象 | 根因 | 规避/解决 | 状态 |
|---|---|---|---|---|---|
| A-001 | TODO | TODO | TODO | TODO | 打开/关闭 |


# 附录 B 原始数据索引

| 数据 | 路径 | 说明 |
|---|---|---|
| P2 汇总表 | data\run-20260814-023631\summary.csv | 全量逐组指标 |
| P2 各组详情 | data\run-20260814-023631 | result.json / test.log / 监控文件 |
| P3 报告 | data\run-20260817-052818\stability_report.json |  |
| P3 时序数据 | data\run-20260817-052818\stability_series.csv |  |
| 图表 | concurrency_saturation_short.png, gpu_series, run-20260817-052818_stability.png, stability_summary.csv, summary_metrics.csv, throughput_latency_short.png |  |
| 环境信息 | report-template\hardware_env.json |  |


# 附录 C 复现步骤

```bash
# 1. 启动服务（示例）
# TODO: vllm serve ... --served-model-name <模型名> --tensor-parallel-size N

# 2. 全量矩阵
python3 run_matrix.py --phase all --warmup --model <模型名> \
  --gpu-monitor-cmd <注册表短名>

# 3. 稳定性验证
python3 run_stability.py --duration 86400 --rate <P2 极限吞吐×70~80%>

# 4. 画图
python3 plot_results.py --results-dir <结果目录>
python3 plot_stability.py --results-dir <P3 run 目录>
```
