# 模型基准测试报告

| 字段 | 内容 |
|---|---|
| 报告编号 | TODO（如 `MT-2026-001`） |
| 被测厂商 / 芯片 | TODO |
| 被测模型 | TODO |
| 测试日期 | TODO |
| 测试人 | TODO |
| 关联测试方案 | 《模型测试方案书》Phase 1~3 |

> 使用说明：本模板为**填写式**通用模板，适用于各国产 GPU 厂商（昇腾 / 海光 /
> 寒武纪 / 沐曦 / 天数智芯 / 燧原 / 摩尔线程 等）。正文所有 `TODO` 为人工
> 填写项；「自动」标记的表格可由 `generate_report.py` 直接从
> `summary.csv` / `stability_report.json` / `hardware_env.json` 生成草稿。
> 环境信息建议统一维护在 `hardware_env.json`（每厂商一份），报告从该文件
> 取数，避免重复填写。

---

## 0. 摘要（一页结论）

### 0.1 被测对象

- 加速卡：TODO（芯片型号 × N 卡，互联方式）
- 软件栈：TODO（推理引擎及版本、驱动/运行时版本）
- 模型：TODO（名称、参数量、量化、上下文长度）
- 部署：TODO（TP/PP 大小、显存分配、容器或裸机）

### 0.2 核心结论（自动 + 人工）

| 项目 | 结论 |
|---|---|
| 极限吞吐（tok/s，短文本） | TODO |
| 极限吞吐（tok/s，长文本） | TODO |
| 推荐请求速率（req/s） | TODO（膝点/饱和点，见 3.4） |
| 推荐并发上限 | TODO |
| P99 TTFT（推荐点，短/长文本） | TODO |
| P99 TPOT（推荐点） | TODO |
| 稳定性结论 | TODO（24h 红线是否全部通过） |
| 综合评级 / 推荐度 | TODO（建议/有条件建议/不推荐，附一句话理由） |

### 0.3 配置推荐速览（自动 + 人工）

| 项 | 推荐值 | 说明 |
|---|---|---|
| 引擎参数 | TODO | 见 5.1 完整清单 |
| 单机卡数 | TODO | |
| 显存要求 | TODO | 含 KV Cache 需求 |
| 网络/互联要求 | TODO | |
| 服务器配置 | TODO | CPU/内存/存储 |

---

## 1. 测试环境

> 环境信息与 `hardware_env.json` 保持一致；填表后回填 JSON，二者互为准。

### 1.1 服务器硬件（自动）

| 项 | 值 |
|---|---|
| 服务器型号 | TODO |
| CPU | TODO |
| 内存 | TODO |
| 存储（NVMe） | TODO |
| 操作系统 / 内核 | TODO |
| Docker / 容器运行时 | TODO |

### 1.2 加速卡（自动）

| 项 | 值 |
|---|---|
| 芯片型号 | TODO |
| 架构 / 制程 | TODO |
| 单卡算力（FP16 / BF16 / FP8 / INT8） | TODO |
| 显存容量 / 带宽 | TODO |
| 单卡 TDP | TODO |
| 装机卡数 / 本次用卡数 | TODO |
| 卡间互联（类型 / 带宽 / 拓扑） | TODO |
| 拓扑说明（如 8 卡全互联 / 2×4 分组） | TODO |

### 1.3 软件栈（自动）

| 层 | 名称 | 版本 | 备注 |
|---|---|---|---|
| 驱动 | TODO | | |
| 运行时（CANN / ROCm / 自研） | TODO | | |
| 推理引擎 | TODO | | 如 vllm-ascend、SGLang 等 |
| 容器镜像 | TODO | | |
| 依赖库（如 PyTorch / MindSpore） | TODO | | |

### 1.4 部署方式

| 项 | 值 |
|---|---|
| 部署形态 | TODO（Docker / 裸机 / K8s） |
| NUMA / CPU 绑定 | TODO |
| 卡间拓扑与 TP 对应关系 | TODO |
| 其他（电源模式、PCIe 直通等） | TODO |

### 1.5 工具链（自动）

| 项 | 值 |
|---|---|
| 压测脚本版本 | TODO（git commit） |
| 压测工具 | TODO（vllm bench serve 等） |
| 监控方式 | TODO（sys_monitor.py，采样间隔） |
| 结果目录 | TODO（run 目录路径） |

### 1.6 GPU 监控探测登记

> 对应 `sys_monitor.py` 的 `GPU_PROBES` 注册表；新厂商首次接入时在此登记。

| 项 | 值 |
|---|---|
| 注册表短名 | TODO（如 `npu-smi`） |
| 探测命令 | TODO |
| 解析正则（如有） | TODO |
| 输出格式说明 | TODO（字段顺序、单位） |
| 验证结果 | TODO（命令在 shell 中手动执行是否 OK） |

---

## 2. 被测模型与部署配置

### 2.1 模型信息（自动）

| 项 | 值 |
|---|---|
| 模型名称 | TODO |
| 参数量 | TODO |
| 量化 | TODO（FP16 / FP8 / INT4 等） |
| 上下文长度 | TODO |
| Tokenizer | TODO |
| 权重来源 | TODO（HuggingFace 仓库 / 厂商格式转换） |

### 2.2 推理引擎参数（自动，与 run_config.json 对齐）

| 参数 | 值 |
|---|---|
| tensor-parallel-size | TODO |
| pipeline-parallel-size | TODO |
| gpu-memory-utilization | TODO |
| max-num-seqs | TODO |
| max-num-batched-tokens | TODO |
| enable-chunked-prefill | TODO |
| kv-cache-dtype | TODO |
| enable-prefix-caching | TODO |
| 其他关键 flag | TODO |

### 2.3 压测参数（自动）

| 参数 | 值 |
|---|---|
| 请求数（短/长文本） | TODO |
| 客户端并发上限（短/长） | TODO |
| 速率档位 | TODO |
| 强制 flag（--ignore-eos 等） | TODO |

### 2.4 加载与正确性验证

| 项 | 结果 |
|---|---|
| 模型加载是否成功 | TODO |
| 首请求响应验证 | TODO |
| 输出内容 sanity check（中/英文、截断、乱码） | TODO |
| 多卡张量并行一致性 | TODO |

---

## 3. 基准矩阵结果（Phase 2）

### 3.1 矩阵说明（自动）

- 短/中文本：输入 `1K/4K/8K` × 输出 `1K/4K` × 速率 `1~inf`（74 组）
- 长文本：输入 `16K~128K` × 输出 `1K/4K` × 速率 `1~16`
- 完整矩阵定义见 `run_matrix.py` 顶部常量；本次实际执行规模：
  TODO（如 short 42 组 + long 32 组）

### 3.2 结果总览（自动）

| 状态 | 组数 |
|---|---|
| OK | TODO |
| FAILED | TODO |
| OOM | TODO |
| TIMEOUT | TODO |
| 合计 | TODO |

失败/异常组明细（自动）：

| 组 | 状态 | 原因 |
|---|---|---|
| TODO | | |

### 3.3 核心指标表（自动）

> 每个 输入×输出 组合取"推荐速率点"一行（膝点/饱和点，见 3.4）；
> 全量逐组数据见附录 B 的 `summary.csv`。

| 输入 | 输出 | 推荐速率 | 吞吐(tok/s) | 吞吐(req/s) | P99 TTFT(ms) | P99 TPOT(ms) | 失败数 |
|---|---|---|---|---|---|---|---|
| TODO | | | | | | | |

### 3.4 膝点 / 饱和点判定（自动 + 人工）

> 判定口径（与 `plot_results.py` 一致）：
> ① 排队临界 rate = P99 TTFT 首次 ≥ 最低速率基线 2 倍；
> ② 吞吐饱和 rate = output throughput 首次达峰值 95%；
> 取二者较小者。人工复核后填入结论。

| 输入 | 输出 | 排队临界 rate | 吞吐饱和 rate | 结论（推荐 rate） |
|---|---|---|---|---|
| TODO | | | | |

### 3.5 图表索引（自动）

| 图表 | 路径 | 说明 |
|---|---|---|
| 吞吐-延迟权衡图 | TODO | TTFT/TPOT 子图，标膝点 |
| 并发饱和度图 | TODO | 标排队临界 rate |
| 长文本性能衰减图 | TODO | 输入长度 vs P99 TTFT |
| 长文本显存压力图 | TODO | 输入长度 vs 峰值显存 |
| GPU 时序图 | TODO | 每组监控时序（可选） |

---

## 4. 稳定性验证（Phase 3）

### 4.1 施压配置（自动）

| 项 | 值 |
|---|---|
| 模式 | TODO（P3-1 饱和施压 / P3-2 24h 生产模拟） |
| 时长 | TODO |
| 请求速率 | TODO |
| 最大并发 | TODO |
| 长度分布 | TODO（短/长比例与区间） |

### 4.2 红线监控结果（自动 + 人工）

| 红线 | 判定标准 | 结果 | 说明 |
|---|---|---|---|
| 温度触顶降频 | temp ≥ 85℃ 且 SM 时钟 < 最大 90% | TODO 通过/触发 | |
| 显存泄漏 | 后 1/2 时段增速 ≥ 100 MB/min | TODO 通过/触发 | |
| ECC 错误 | uncorrected 计数上升 | TODO 通过/触发 | |
| Silent Crash | 心跳连续 3 次失败 | TODO 通过/触发 | |
| 吞吐衰减 | 长时段吞吐下行趋势 | TODO 通过/触发 | |
| 队列堆积 | waiting 队列持续增长 | TODO 通过/触发 | |

### 4.3 稳定性结论（人工）

- 总体判定：TODO（通过 / 有条件通过 / 不通过）
- 关键数据（自动）：总请求 TODO，失败率 TODO%，平均 req/s TODO，
  P99 E2E TODO ms
- 说明：TODO

---

## 5. 调优结论与配置推荐

> 本部分是报告的核心交付物：先用 5.1 达标线判定"能不能用"，再用
> 3.4 膝点 + 4 稳定性结果给出"怎么配"。

### 5.1 达标线判定（自动）

> 达标线在 `hardware_env.json` 的 `acceptance_lines` 配置，由
> `generate_report.py` 自动判定；未配置或数据缺失的行输出 TODO 需人工判定。
> 判定口径：短/长文本极限吞吐取各组最大 output_throughput；TTFT 取所有
> 组合中「推荐速率点 P99 TTFT / 最低速率基线」的最大倍数；失败率 =
> num_failed/num_requests；异常组 = FAILED/OOM/TIMEOUT 组数；稳定性红线 =
> P3 trends 全部通过且无 ERROR 事件；GPU 利用率 = 全 run 组级峰值最大值。

| 指标 | 实测值 | 达标线 | 判定 |
|---|---|---|---|
| 短文本极限吞吐 (tok/s) | TODO | | 达标/不达标 |
| 长文本极限吞吐 (tok/s) | TODO | | |
| 推荐点 P99 TTFT / 基线最大倍数 | TODO | ≤ 2.0 | |
| 请求失败率 (%) | TODO | | |
| 异常组数 (FAILED/OOM/TIMEOUT) | TODO | | |
| P3 稳定性红线 | TODO | 全部通过 | |
| GPU 利用率峰值 (%) | TODO | | |

汇总判定：TODO（全部达标 / 存在未达标项 / 待人工判定），
不达标项对业务的影响：TODO。

### 5.2 推荐引擎参数（人工）

| 参数 | 推荐值 | 依据 |
|---|---|---|
| tensor-parallel-size | TODO | |
| gpu-memory-utilization | TODO | |
| max-num-seqs | TODO | |
| max-num-batched-tokens | TODO | |
| chunked prefill | TODO | |
| kv-cache-dtype | TODO | |
| 其他 | TODO | |

### 5.2 硬件建议（人工）

| 项 | 建议 | 依据 |
|---|---|---|
| 单机卡数 / TP 配置 | TODO | |
| 显存需求（模型权重 + KV Cache） | TODO | |
| 互联要求 | TODO | |
| 服务器配置 | TODO | |

### 5.3 效率与性价比（自动 + 人工）

| 指标 | 值 |
|---|---|
| 每卡吞吐（tok/s/卡） | TODO |
| 每瓦吞吐（tok/s/W，满载时） | TODO |
| 每卡并发承载（req/s/卡） | TODO |
| 与参考平台（如 4090/NVIDIA）比值 | TODO |

### 5.4 已知问题与规避（人工）

| 问题 | 影响 | 规避/解决 |
|---|---|---|
| TODO | | |

---

## 附录 A 部署踩坑与兼容性问题记录

> 厂商兼容性差异、驱动/框架版本问题、算子缺失等均记录于此，作为选型
> 与下次测试的依据。每次测试新增条目，形成厂商问题库。

| 编号 | 日期 | 现象 | 根因 | 规避/解决 | 状态 |
|---|---|---|---|---|---|
| A-001 | TODO | | | | 打开/关闭 |

## 附录 B 原始数据索引

| 数据 | 路径 | 说明 |
|---|---|---|
| P2 汇总表 | TODO（summary.csv） | 全量逐组指标 |
| P2 各组详情 | TODO（run 目录） | result.json / test.log / 监控文件 |
| P3 报告 | TODO（stability_report.json） | |
| P3 时序数据 | TODO（stability_series.csv） | |
| 图表 | TODO（figures/） | |
| 环境信息 | TODO（hardware_env.json） | |

## 附录 C 复现步骤

> 记录本次测试的完整复现命令（含服务启动命令、压测命令、画图命令），
> 保证他人可一键复现。

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