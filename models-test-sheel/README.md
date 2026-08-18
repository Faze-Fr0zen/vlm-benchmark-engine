# vLLM 压测与监控（models-test-sheel）

国产 GPU 服务器（昇腾 / 海光 / 寒武纪 / 沐曦 / 天数智芯 / 燧原等）的
模型基准测试工具集。全部脚本零第三方依赖（压测本体），GPU 厂商无统一
工具链，探测命令走登记制注册表。测试结果供 `report-template/generate_report.py`
自动生成测试报告。

## 整体架构

```
┌─ 压测执行器（零依赖）────────────────┐
│  run_matrix.py / run_matrix_lite.py │  矩阵压测（Phase 2）
│  run_stability.py                   │  长跑稳定性（Phase 3）
└──────────────────┬──────────────────┘
                   │ 每步自动启动监控
                   ▼
┌─ 系统监控 ──────────────────────────┐
│  sys_monitor.py（被压测脚本 import） │  系统/GPU 采样、告警、泄漏检测
│  GPU_PROBES 注册表（厂商探测登记）   │
└──────────────────┬──────────────────┘
                   ▼
┌─ 结果产物 ──────────────────────────┐
│  summary.csv / stability_report.json│  数据（证据）
│  组目录：result.json + 监控 csv/log │
└──────────────────┬──────────────────┘
                   ▼
┌─ 画图 ─────────────────────────────┐
│  plot_results.py / plot_stability.py│  图（证据可视化）
└────────────────────────────────────┘
```

测试流程：**矩阵压测 → 长跑稳定性 → 画图**，系统监控内嵌于前两步自动执行。

## 文件说明

| 文件 | 作用 |
|---|---|
| `run_matrix.py` | P2 阶梯压测矩阵（大模型版，74 组） |
| `run_matrix_lite.py` | P2 快速矩阵（小模型版，63 组） |
| `run_stability.py` | P3 极限稳定性与长跑验证（含红线监控） |
| `sys_monitor.py` | 共享监控模块：GPU 探测注册表、告警、泄漏检测、事件日志 |
| `plot_results.py` | P2 结果画图（含膝点自动判定） |
| `plot_stability.py` | P3 稳定性趋势画图 |

## 公共约定（关键）

- **模型配置在脚本顶部常量**：`MODEL`（须与 vLLM `--served-model-name` 一致）、
  `TOKENIZER`（本地绝对路径）、`MODEL_PATH`、`BASE_URL`、`RESULTS_DIR`
  （服务器上为 `/vllm-workspace/results/vllm-bench-result`），切换模型只改顶部
- **命令构造自动适配 vLLM 版本**：0.26+ 需 `vllm bench serve --model`，
  老版本用 `--save-result`，脚本自动探测并修正，无需手改
- **强制 `--ignore-eos`**：保证输出 token 数达标，不依赖模型主动收尾
- **结果目录按 run 组织**：`<RESULTS_DIR>/run-<时间戳>/`，组目录含
  `result.json`（bench 原始结果）、`metadata.json`、`sys_stats.csv`、
  `gpu_stats.csv`、`monitor_events.log`、`test.log`；根目录 `summary.csv`
  汇总全部 24 列指标
- **监控告警阈值**：OOM = 显存 ≥ 95%；内存告警 = 剩余 < 2048 MiB；
  泄漏判定 = 系统内存增速 ≥ 50 MB/min、显存增速 ≥ 100 MB/min

---

## 1. 矩阵压测（Phase 2）

### 关键信息

- `run_matrix.py`（大模型版）74 组：短/中文本 3×2×7=42 组
  （输入 1024/4096/8192 × 输出 1024/4096 × 速率 1/8/16/32/64/128/inf）
  + 长文本 4×2×4=32 组（输入 16K~128K × 输出 1024/4096 × 速率 1/4/8/16）
- `run_matrix_lite.py`（小模型版）63 组：短文本为主，长文本仅 2 组大速率
- 每组合自动：启动 vLLM（自动适配版本参数）→ 压测 → 采样 GPU/内存 →
  结果写入 `summary.csv` 与组目录
- 支持断点续跑（`--resume` 跳过已有结果）与自定义矩阵（`--input-lens` 等）

### 使用方法

1. 改脚本顶部 `MODEL` / `TOKENIZER` / `RESULTS_DIR` 常量
2. 先 `--dry-run` 检查要执行的命令
3. 正式执行（建议 `--warmup` 先预热）；中途断掉后带 `--resume` 续跑
4. 跑完检查根目录 `summary.csv` 行数与状态

### 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 顶部 `MODEL` | 模型名，须与 served-model-name 一致 |
| `--tokenizer` | 顶部 `TOKENIZER` | tokenizer 本地绝对路径（离线必须） |
| `--model-path` | 顶部 `MODEL_PATH` | 模型权重路径（仅存在性校验） |
| `--base-url` | `http://localhost:8000` | vLLM 服务地址 |
| `--vllm-cmd` | `vllm` | vllm 可执行文件或命令 |
| `--phase` | `all` | `short`=短/中文本矩阵，`long`=长文本矩阵 |
| `--input-lens` | 按阶段矩阵 | 自定义输入长度，如 `1024,4096` |
| `--output-lens` | 按阶段矩阵 | 自定义输出长度 |
| `--request-rates` | 按阶段矩阵 | 自定义速率列表，支持 `inf` |
| `--num-prompts` | 200 | 短/中文本每组请求数 |
| `--num-prompts-long` | 20 | 长文本每组请求数 |
| `--max-concurrency` | short=32 / long=16 | 客户端并发上限（防 Prefill 堆积 TTFT 爆炸），0=不限制 |
| `--result-dir` | 顶部 `RESULTS_DIR` | 结果根目录（本地调试用 `results` 覆盖） |
| `--run-id` | 按时间 | 本次运行标识 |
| `--timeout` | 7200 | 单组测试超时（秒） |
| `--sleep-between` | 5.0 | 相邻测试间隔（秒） |
| `--warmup` | 关 | 矩阵前先跑一组小规模预热 |
| `--resume` | 关 | 跳过已有 `result.json` 的组 |
| `--dry-run` | 关 | 只打印命令不执行 |
| `--verbose` | 关 | 打印完整命令 |
| `--gpu-monitor-cmd` | 无 | GPU 探测命令（见第 3 节） |
| `--gpu-ids` | 全部 | 限制监控的 GPU，如 `0,1` |
| `--gpu-interval` | 10.0 | 监控采样间隔（秒） |
| `--no-gpu-monitor` | 关 | 禁用 GPU 采样（CPU/内存监控始终开启） |
| `--sys-mem-warn-mb` | 2048 | 系统剩余内存告警阈值 |
| `--gpu-mem-warn-pct` | 95.0 | 显存占用告警百分比（OOM 判定） |

### 代码示例

```bash
# 只打印命令，核对无误后再执行
python run_matrix.py --dry-run --verbose

# 正式跑全矩阵（带预热 + GPU 监控）
python run_matrix.py --warmup --gpu-monitor-cmd nvidia

# 只跑长文本阶段、自定义矩阵、限制并发
python run_matrix.py --phase long --input-lens 16384,32768 --request-rates 1 4 8 \
    --max-concurrency 16

# 中断后续跑（跳过已完成组）
python run_matrix.py --resume

# 小模型快速矩阵（lite 版，用法相同）
python run_matrix_lite.py --run-id run-lite-01
```

---

## 2. 长跑性能监控（Phase 3，run_stability.py）

### 关键信息

按《模型测试方案书》Phase 3 两种模式：

- **P3-1 高并发饱和施压（1~2 小时）**：`--rate inf` + P2 极限并发，观察
  吞吐是否衰减、waiting 队列是否无限堆积
- **P3-2 24h 生产模拟长跑**：`--rate` 取极限吞吐的 70%~80%，随机长短文本
  混合（短 512~1K / 长 4K~8K 各 50%，可配）

监控红线（写入 `monitor_events.log`）：温度触顶降频（≥85℃ 且时钟下降）、
显存泄漏（后 1/2 时段增速 ≥ 100 MB/min）、ECC 错误上升、Silent Crash
（心跳连续 3 次失败）。趋势判定（吞吐衰减 / 队列堆积 / 显存泄漏）汇总进
`stability_report.json`，供报告脚本直接使用。

产物：`stability_series.csv`（时间序列）、`monitor_events.log`、
`stability_report.json`、`run_config.json`。

### 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--base-url` | `http://localhost:8000` | vLLM 服务地址 |
| `--model` | 顶部 `MODEL` | 模型名（须与 served-model-name 一致） |
| `--tokenizer` | 顶部 `TOKENIZER_PATH` | tokenizer 路径 |
| `--duration` | 7200 | 运行时长（秒），如 7200（1h）/ 86400（24h） |
| `--rate` | inf | 请求速率 req/s；P3-1 用 inf 压满，P3-2 用极限吞吐的 70%~80% |
| `--max-concurrency` | 64 | 并发窗口上限（P3-1 饱和施压关键参数） |
| `--short-input` | `512 1024` | 短文本输入长度范围 [min max] |
| `--long-input` | `4096 8192` | 长文本输入长度范围 [min max] |
| `--short-ratio` | 0.5 | 短文本占比，其余为长文本 |
| `--out-range` | `512 2048` | 输出长度范围 [min max] |
| `--metrics-interval` | 15.0 | 监控采样间隔（秒） |
| `--temp-warn` | 85.0 | 温度告警阈值（℃） |
| `--gpu-monitor-cmd` | nvidia | GPU 探测（注册表短名或完整命令；不传则不采样） |
| `--result-dir` | 顶部 `RESULTS_DIR` | 结果根目录 |
| `--run-id` | 按时间 | 运行标识 |
| `--dry-run` | 关 | 只演示请求生成，不真正压测 |

### 代码示例

```bash
# P3-1 高并发饱和施压 2 小时（并发取 P2 极限值）
python run_stability.py --duration 7200 --rate inf --max-concurrency 64

# P3-2 24h 生产模拟长跑（rate 取极限吞吐的 70%~80% 对应 req/s）
python run_stability.py --duration 86400 --rate 12.0 --max-concurrency 32

# 自定义长度分布与 GPU 探测
python run_stability.py --duration 3600 --short-input 256 512 --long-input 2048 4096 \
    --short-ratio 0.7 --gpu-monitor-cmd "mthreads-gmi -q"
```

---

## 3. 系统监控（sys_monitor.py）

### 关键信息

被压测脚本 import 的共享模块（不单独运行）：

- **零第三方依赖**：只解析子进程 stdout 的浮点列，不依赖厂商工具库
- **GPU_PROBES 注册表**：内置各厂商探测短名（如 `nvidia`、`mthreads`、
  `smi` 等），或传完整命令；探测输出须为逗号分隔数字列，约定第 2~3 列
  为 `util,mem_used`（可带 `mem_total`、`power`、`temp`），否则不采样
- **采样落盘**：`sys_stats.csv`（CPU/内存/负载）+ `gpu_stats.csv`
  （util,mem_util,mem_used,mem_total,power,temp）
- **事件日志** `monitor_events.log`：按 `时间 级别 来源 内容` 追加
  告警与红线事件（ERROR/WARN），供第 4 节画图与报告脚本统计
- **告警与泄漏检测**：OOM（显存 ≥ 95%）、内存告警（剩余 < 2048 MiB）、
  显存泄漏（增速 ≥ 100 MB/min）、系统内存泄漏（增速 ≥ 50 MB/min）

GPU 探测参数（`--gpu-monitor-cmd` / `--gpu-ids` / `--gpu-interval` /
`--no-gpu-monitor` / `--sys-mem-warn-mb` / `--gpu-mem-warn-pct`）见第 1 节
参数表，`run_matrix.py` 与 `run_stability.py` 通用。

### 代码示例

```bash
# 登记制：短名（注册表内置）
python run_matrix.py --gpu-monitor-cmd nvidia --gpu-ids 0,1 --gpu-interval 5

# 自定义探测命令（输出逗号分隔数字列，第 2~3 列为 util,mem_used）
python run_matrix.py --gpu-monitor-cmd "/usr/bin/mthreads-gmi -q"

# 厂商探测登记示例（sys_monitor.py 内 GPU_PROBES）
# "custom_vendor": "smi -x | python parse.py"  →  --gpu-monitor-cmd custom_vendor
```

---

## 4. 画图（plot_results.py / plot_stability.py）

### 关键信息

- `plot_results.py`：P2 结果画图，输出 4 张主图 +
  `gpu_series/` 逐组 GPU 时序；主图含膝点自动判定标注：
  排队临界 rate = P99 TTFT 首次 ≥ 最低速率基线 2 倍；
  吞吐饱和 rate = 吞吐首次达峰值 95%（判定口径与报告脚本一致）
- `plot_stability.py`：P3 趋势画图，输出 `<run_id>_stability.png`
  （吞吐/时延/队列/显存/温度五联图）+ `stability_summary.csv`

### 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--results-dir` | 必填 | 结果根目录（含 `summary.csv`） |
| `--run` | 全部 | 只画指定 run（如 `run-20260814-023631`） |
| `--gpu-series` | 关 | 同时输出逐组 GPU 时序图 |
| `--out-dir` | `figures` | 图表输出目录 |
| `--dpi` | 150 | 图片分辨率 |
| `--font` | 自动 | 指定中文字体名（绘图机缺字体时用） |

`plot_stability.py` 参数同 `plot_results.py`，无 `--run` / `--gpu-series`，
`--results-dir` 指向 P3 结果目录。

### 代码示例

```bash
# P2 全量画图（含膝点标注与 GPU 时序）
python plot_results.py --results-dir /vllm-workspace/results/vllm-bench-result \
    --gpu-series --out-dir figures

# 只画某个 run
python plot_results.py --results-dir results --run run-20260814-023631 --out-dir figures

# P3 稳定性趋势
python plot_stability.py --results-dir results --out-dir figures

# 指定中文字体（画图机缺字体时）
python plot_results.py --results-dir results --font "SimHei"
```