# vLLM Serve 参数调优（models-optimize-sheel）

按《模型性能优化方案书》实现的服务端参数调优工具集，用于在固定模型、
固定硬件上寻找最佳 `vllm serve` 参数组合。与 `models-test-sheel`（基准
测试）配套：**先调优锁定配置 → 再用测试脚本跑全量矩阵验收**。

调优核心思想（方案书 1.2 / 3.2）：

- **分层收敛**：Serve 参数分组独立扫描，不做全排列（探针优先）
- **探针模式**：每个 Serve 配置只跑两组短时基准（<15 分钟），快速评分
- **决策闭环**：观测现象 → 瓶颈定位 → 调优动作（方案书 4.2 映射表）
- **收敛引擎**：每次探针跑完自动输出进度 / 下一组完整启动命令 / 晋级
  结论（确定性规则，非 AI 决策）
- **半自动执行**：服务由操作者手动启停（避免脚本跨端实时控制），探针、
  评分、收敛判定全部自动化
- **小体量适配**：4090 等消费级卡用 `--lite`（组合与探针减半）

## 整体架构

```
┌─ 服务端（vLLM 所在机器）────────────────────────┐
│  vllm serve ...（操作者手动启动，命令由收敛引擎给出）│
│  sys_monitor.py --standalone（旁路采样 gpu_stats.csv）
└──────────────────────────┬─────────────────────┘
                           ▼
┌─ 测试端（本目录）───────────────────────────────┐
│  probe_only.py --config "mem=0.95,seqs=128,..."  │
│    ├ 双探针压测（复用 run_matrix.py 命令构建/解析）│
│    ├ 评分（decision.py 方案书 4.2）→ summary.csv │
│    └ 收敛引擎 → 下一组启动命令 / 晋级判定 / 锁定   │
│  report.py → 调优指南报告 + 验收对比              │
└─────────────────────────────────────────────────┘
```

调优流程：**Round 1 显存×并发网格 → Round 2 kv-dtype 补扫 →（fp8 胜时
Round 2.5 复活点验证）→ Round 3 调度 A/B → 锁定配置 → run_matrix.py
全量验收**。详细操作步骤见 **[STEP_BY_STEP.md](STEP_BY_STEP.md)**。

## 文件说明

| 文件 | 作用 |
|---|---|
| `probe_only.py` | 双探针执行 + 评分 + **收敛引擎**（自动输出下一步，主入口） |
| `serve_manager.py` | 调优参数→CLI 映射表 + `wait_gpu_freed()` 显存释放检测 |
| `decision.py` | 决策闭环：方案书 4.2 规则映射 + 配置评分函数 |
| `report.py` | 调优指南报告：收敛旅程回放 + 评分表 + `--p2-before/--p2-after` 验收对比 |
| `STEP_BY_STEP.md` | 调优操作手册（按轮次复制命令执行） |

依赖：Python 3.9+ 标准库；`probe_only.py` 复用 `models-test-sheel/` 下的
`run_matrix.py`（同目录部署，脚本自动加入 `sys.path`）。

## 公共约定（关键）

- **调优参数定义在 `serve_manager.py` 顶部 `SERVE_FLAG_MAP`**：新增或修改
  可调参数只改映射表（`FLAG_ONLY` 型参数如 `enable_chunked_prefill` 只需
  布尔值）；投机解码相关参数默认跳过（4090/国产卡按方案书不调）
- **收敛矩阵与判据内置在 `probe_only.py`**（与 STEP_BY_STEP.md 第 2 节
  规则表同一套，改规则必须两处同步）
- **`--config` 短名**：`mem=`（gpu-memory-utilization）、`seqs=`
  （max-num-seqs）、`kv=`（kv-cache-dtype）、`chunked=`（enable-chunked-
  prefill）、`batched=`（max-num-batched-tokens）；轮次自动识别
  （kv 非 auto = Round 2，chunked/batched 存在 = Round 3，其余 Round 1）
- **幂等续跑**：按 config_id 跳过已有评分，中断后重跑同一条命令即恢复
- **结果目录**：默认 `/vllm-workspace/results/vllm-optimize-result`
  （脚本顶部 `RESULTS_DIR` 常量，可 `--result-dir` 覆盖），每个
  `run-opt-*` 下 `summary.csv` 为累积评分表
- **GPU 指标**：服务端旁路采样 `gpu_stats.csv` 拷回后用 `--gpu-csv` 传入，
  按探针起止时间窗自动聚合（平均利用率 / 显存峰值 / 泄漏检测）

---

## 1. 探针执行与收敛引擎（probe_only.py）

### 关键信息

每轮流程（详见 STEP_BY_STEP.md）：

1. 服务端启动旁路采样：`sys_monitor.py --standalone --clocks-ecc`
2. 按收敛引擎输出的命令手动启动 `vllm serve`，等 `/v1/models` 就绪
3. 测试端跑 `probe_only.py --config "<当前服务参数>" --gpu-csv <gpu_stats.csv>`
4. 引擎输出进度 + 下一组完整启动命令；停掉当前服务（`wait_gpu_freed`
   确认显存回落）→ 回到第 2 步，直到引擎输出 `[锁定]`

探针定义（`probe_only.py` 顶部）：

| 探针 | 输入 | 输出 | rate | 请求数 | 并发 | 测什么 |
|---|---|---|---|---|---|---|
| short | 1K | 512 | inf | 120（lite 60） | 64 | 调度上限 / 最大吞吐 |
| long | 32K | 1K | 1 | 16（lite 8） | 1 | 显存上限 / Prefill 压力 |

收敛矩阵（全量版 17 组 + 复活点）：

| 轮次 | 组合 | 判据 |
|---|---|---|
| Round 1 | 9 组：mem(0.90/0.95/0.98) × seqs(64/128/256)，kv=auto | 全测完取 ACCEPT 最高分（同分取更激进档）为 top1；REJECT 的 mem 档以上记"显存天花板"不再尝试 |
| Round 2 | 2 组：top1 + kv=auto/fp8 | fp8 胜 → Round 2.5；否则 → Round 3 |
| Round 2.5 | 1~2 组：Round 1 REJECT 点以 fp8 重测 | 救回则锁 fp8，救不回维持 auto |
| Round 3 | 6 组：top1 基础上 chunked(off/on) × batched(4K/8K/16K) | B 最优 > baseline 才替换；最优 batched 触 16K 上限提示可扩展 |
| 锁定 | — | 输出最终命令 → run_matrix.py 全量验收 |

4090 小体量（`--lite`）：Round 1 = 4 组（mem 0.90/0.95 × seqs 64/128，
跳过 0.98/256，24GB 极易 OOM）；Round 3 = 4 组（batched 4K/8K）。

### 决策闭环（decision.py，方案书 4.2）

| 观测现象 | 判定 | 后续动作 |
|---|---|---|
| 探针失败 / 请求失败 / OOM | REJECT | 淘汰；显存>95% 建议降 seqs |
| 显存峰值 > 95% | REJECT | 淘汰；记录显存天花板 |
| TTFT P99 > 5s | REJECT | 淘汰（Prefill 排队失控） |
| GPU 利用率 < 70% | WARN | 不参与排序；建议升 batched |
| 其余正常 | ACCEPT | 参与评分排序 |

评分（0~100，越高越好）：short 探针吞吐归一化 ×60 + long 探针吞吐 ×20
+ TTFT/TTPOT 超阈值扣分 ×20。

### 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | 必填 | 当前服务参数，逗号分隔短名，如 `mem=0.95,seqs=128,kv=auto` |
| `--round` | 自动 | 手动指定轮次（一般不需要） |
| `--serve-base` | 顶部 `DEFAULT_SERVE_BASE` | 基础 serve 命令前缀（引擎输出启动命令时拼接） |
| `--model` / `--tokenizer` | `qwen3-0.6b` / `/models/Qwen3-0.6B` | 探针用模型名 / tokenizer 路径 |
| `--base-url` | `http://localhost:8000` | vLLM 服务地址 |
| `--vllm-cmd` | `vllm` | vllm 可执行文件 |
| `--gpu-csv` | 无 | 服务端旁路采样 `gpu_stats.csv`（探针窗口聚合） |
| `--result-dir` | 顶部 `RESULTS_DIR` | 结果根目录 |
| `--run-id` | 按时间（`run-opt-YYYYMMDD-HHMMSS`） | 运行标识 |
| `--probe-timeout` | 1500 | 单探针超时（秒） |
| `--lite` | 关 | 4090 小体量矩阵 |
| `--dry-run` | 关 | 只跑收敛引擎，不执行探针 |

### 代码示例

```bash
# Round 1 第一组（命令由引擎首次运行时给出）
python probe_only.py --config "mem=0.9,seqs=64,kv=auto" \
    --gpu-csv gpu_stats.csv

# 中途断了重跑同一条命令（幂等跳过已有，直接输出下一步）
python probe_only.py --config "mem=0.95,seqs=128,kv=auto"

# 只查当前进度（不跑探针）
python probe_only.py --config "mem=0.95,seqs=128,kv=auto" --dry-run
```

### 显存释放检测

```bash
python -c "import serve_manager as s; s.wait_gpu_freed()"
```

---

## 2. 调优指南报告（report.py）

### 关键信息

- `opt_summary.csv`：所有配置评分表（按 score 降序）
- `best_serve_cmd.txt`：锁定配置的完整 `vllm serve` 命令（交付物
  《最佳生产启动命令》）
- `report.md`：调优指南（锁定配置 + **收敛旅程回放**——逐组复现收敛
  引擎当时的输出 + 评分表 + **验收对比**）

验收对比数据流：`--p2-before` 用锁定前（默认配置）跑的 run-p2，
`--p2-after` 用锁定配置重跑的 run-p2，量化吞吐/时延/显存/GPU 利用率收益。

### 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--results-dir` | 必填 | probe_only 结果根目录（自动扫描 run-opt-* 的 summary.csv，自动排除 run_matrix 输出） |
| `--p2-before` | 无 | 验收 before：默认配置的 run_matrix 输出目录 |
| `--p2-after` | 无 | 验收 after：锁定配置的 run_matrix 输出目录 |
| `--out-dir` | `opt_report` | 报告输出目录 |
| `--serve-base` | 顶部常量 | 基础 serve 命令前缀 |
| `--lite` | 关 | 收敛回放用小体量矩阵 |

### 代码示例

```bash
python report.py --results-dir /vllm-workspace/results/vllm-optimize-result \
    --p2-before /vllm-workspace/results/vllm-bench-result/run-p2-baseline \
    --p2-after  /vllm-workspace/results/vllm-bench-result/run-p2-locked
```

---

## 3. 与 models-test-sheel 的配合

锁定配置后，验收与出报告走测试脚本（详见 STEP_BY_STEP.md 第 5 节）：

```bash
# 1. 按引擎 [锁定] 命令启动服务（服务端旁路采样继续开着）
# 2. 全量矩阵验收（P2）
python run_matrix.py --warmup --gpu-csv <拷回的 gpu_stats.csv>
# 3. 稳定性验证（P3-2，rate 取 P2 极限吞吐的 70%~80%）
python run_stability.py --duration 86400 --rate 0.9 --max-concurrency 64 \
    --gpu-csv <gpu_stats.csv>
# 4. 画图出报告
python plot_results.py --results-dir ... --gpu-series --gpu-csv <gpu_stats.csv>
```

跨平台注意（方案书 1.2 / 2.1）：国产卡上若某参数导致启动失败，优先检查
该卡对 vLLM 特定 Backend（FlashAttention / Triton）的兼容情况；`fp8`
KV Cache、Graph 捕获失败时按方案书降级为 `auto` / `enforce-eager`。