# vLLM 压测矩阵自动化脚本

一套通用的 vLLM 服务压力测试与系统监控方案：按预定义矩阵逐组调用
`vllm bench serve` 压测本地/远程 vLLM 服务，每组测试期间后台采样系统
指标（CPU、内存、可选的 GPU），实时告警 OOM 与日志异常，测试结束做
内存泄漏回归检测，最后统一解析结果出报告图表。

面向国产 GPU（昇腾、海光等）：**不依赖、不探测任何 GPU 厂商工具**。
默认仅监控 CPU 与系统内存；需要 GPU 指标时由用户显式提供探测命令
（`--gpu-monitor-cmd`），内置厂商注册表（`nvidia` / `npu-smi` 等），
新增厂商只需在 `GPU_PROBES` 注册表加一行、参数换名字即可。

## 文件说明

| 文件 | 作用 |
|---|---|
| `run_matrix.py` | 全量压测矩阵执行器（大模型版，74 组测试） |
| `run_matrix_lite.py` | 轻量矩阵执行器（小模型/流程验证版，63 组），逻辑与监控完全复用 `run_matrix.py` |
| `sys_monitor.py` | 共享系统监控模块（无第三方依赖），含 `GPU_PROBES` 厂商注册表，被上面两个脚本 import |
| `plot_results.py` | 结果解析与画图（依赖 matplotlib） |
| `plot_stability.py` | Phase 3 稳定性时序图（P3 专用，依赖 matplotlib） |
| `run_stability.py` | Phase 3 长跑稳定性施压器（自定义轻量客户端 + 红线监控，零第三方依赖） |
| `requirements.txt` | 画图依赖（运行压测本体不需要） |

## 目录布局（容器内）

脚本与数据分离存放：

```
/vllm-workspace/
├── scripts/
│   └── models-test-sheel/    # 代码仓库根（本文所有 python 命令都在此目录执行）
│       ├── run_matrix.py
│       ├── run_matrix_lite.py
│       ├── sys_monitor.py
│       ├── plot_results.py
│       └── README.md
├── results/
│   └── vllm-bench-result/    # 压测结果统一存放（自行归档，git 无关）
└── .vscode/
    └── launch.json           # VSCode 调试配置（workspace 根打开 /vllm-workspace）
```

## 前置条件

1. **vLLM 服务已启动**，例如（当前环境：单卡 4090 跑小模型）：

   ```bash
   vllm serve /models/Qwen3-0.6B \
     --served-model-name qwen3-0.6b \
     --tensor-parallel-size 1 \
     --host 0.0.0.0 --port 8000 \
     --gpu-memory-utilization 0.9
   ```
   脚本只负责压测，通过 `--base-url` 连接服务，不关心服务如何启动。
   大模型多卡场景示例：`vllm serve zai-org/GLM-5.2-FP8
   --tensor-parallel-size 8 --enable-chunked-prefill --port 8000`。

   > **重要（vLLM 0.26+）**：`vllm bench serve` 不再接受位置参数模型名，
   > 必须用 `--model` flag，且**值必须与服务端 `--served-model-name` 完全
   > 一致**，否则请求返回 `404 Not Found`、GPU 空转。脚本会自动探测并
   > 适配；切换模型时同步修改 `run_matrix.py` 顶部 `MODEL` 变量。

2. **Python 3.9+**，建议在 venv 中运行（压测脚本本身零第三方依赖，
   但需要能执行 `vllm bench serve` 的 python 环境）；也可直接用容器内
   系统 Python 3.12（VSCode 调试时选择该解释器即可）：

   ```bash
   cd /vllm-workspace/scripts/models-test-sheel
   python3 -m venv .venv && source .venv/bin/activate   # Linux（可选）
   # 或 .venv\Scripts\activate                          # Windows
   ```

3. （可选）需要画图时安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

4. **模型与结果目录就绪**：

   - 模型权重与 tokenizer 在 `/models/Qwen3-0.6B`（脚本顶部 `MODEL_PATH` /
     `TOKENIZER` 变量，路径不存在会立即报错）
   - `/vllm-workspace/results/vllm-bench-result` 为压测结果统一存放目录
     （自动创建；数据自行下载归档）

## 快速开始

```bash
# 上传（本地执行）：放到 scripts/models-test-sheel 目录
scp run_matrix.py run_matrix_lite.py sys_monitor.py plot_results.py \
    requirements.txt README.md user@<服务器IP>:/vllm-workspace/scripts/models-test-sheel/

# 容器内：进入脚本目录后执行（下同）
cd /vllm-workspace/scripts/models-test-sheel

# 1. 先 dry-run 检查生成的命令（不校验路径、不创建目录）
python3 run_matrix_lite.py --phase short --dry-run

# 2. 小模型流程验证（单卡 4090，先跑短矩阵；模型/tokenizer 取脚本顶部变量）
python3 run_matrix_lite.py --phase short --warmup

# 3. 需要 GPU 指标时加上探测命令（短名走注册表，或整串命令）
python3 run_matrix_lite.py --phase short --warmup --gpu-monitor-cmd nvidia
python3 run_matrix_lite.py --phase short --warmup \
  --gpu-monitor-cmd "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits"

# 4. 大模型全量矩阵（国产 GPU；切换模型时修改 run_matrix.py 顶部 MODEL 变量）
python3 run_matrix.py --phase short --warmup --model zai-org/GLM-5.2-FP8

# 5. 跑完画图（结果目录指向 vllm-bench-result）
python3 plot_results.py --results-dir /vllm-workspace/results/vllm-bench-result
```

结果默认落在 `/vllm-workspace/results/vllm-bench-result/<run-id>/`
（脚本顶部 `RESULTS_DIR` 变量；本地调试可用 `--result-dir results` 覆盖）。

## 模型与路径配置（脚本顶部变量）

所有可调配置集中在 `run_matrix.py` 顶部，**切换模型只需改这些变量**，
无需改动命令本体：

```python
MODEL = "qwen3-0.6b"                 # 压测模型名（--model 默认，必须与
                                     #   vllm serve 的 --served-model-name 一致）
TOKENIZER = "/models/Qwen3-0.6B"     # tokenizer 本地绝对路径（离线模式必须）
MODEL_PATH = "/models/Qwen3-0.6B"    # 模型权重目录（仅做存在性校验）
RESULTS_DIR = "/vllm-workspace/results/vllm-bench-result"  # 压测结果统一存放目录
```

压测命令执行前会**校验 tokenizer 与模型权重目录存在**，路径不存在立即报错
退出（`--dry-run` 跳过校验）。也可用 `--tokenizer` / `--model-path` 临时覆盖。

## vLLM 版本适配（自动探测）

脚本启动时会执行 `vllm bench serve --help=all` 探测该版本支持的参数，
**自动适配 vLLM 语法差异**（本项目已实测 vLLM 0.26.0）：

| 旧版写法 | 0.26+ 写法 | 脚本行为 |
|---|---|---|
| `bench serve <模型名>`（位置参数） | `bench serve --model <模型名>` | 探测到 `--model` 即切换 |
| `--random-input/--random-output` | `--random-input-len/--random-output-len` | 三个分支逐一尝试 |
| `--no-enable-prefix-caching` | 已移除 | 探测不到则跳过并警告 |
| `--result-dir/--result-file` 即可落盘 | **必须显式 `--save-result`** 才写 result.json | 自动追加 `--save-result` |
| `--help` 列出全部参数 | `--help` 仅分组概览 | 改用 `--help=all` 探测 |

> **易踩坑**：vLLM 0.26+ 不加 `--save-result` 时，`vllm bench serve`
> 退出码为 0 但**不生成 result.json**，导致 summary 指标全空、无法画图。
> 脚本已自动处理；手动跑 bench 时注意补上。

## 服务器部署注意事项

1. **单卡 4090 可满足 lite 矩阵**（Qwen3-0.6B）：权重仅 ~1.3GB（FP16），
   显存充裕；服务启动时预分配显存（`--gpu-memory-utilization 0.9`），
   因此 gpu_stats.csv 的显存曲线接近平线属**正常现象**（看"分配"而非
   "活动"），GPU 利用率曲线才是忙闲信号
2. **vLLM 服务必须与卡数一致启动**：当前环境 `CUDA_VISIBLE_DEVICES=0`
   配 `--tensor-parallel-size 1`；若配置 TP=2 而第二张卡未启用，服务会
   启动失败
3. **速率 32 × 并发 64 时 P99 TTFT 升高是预期排队现象**：这正是压测要
   观测的排队临界；想先验证流程可只跑 `--request-rates 4 8 16`
4. **results 目录**：`/vllm-workspace/results/vllm-bench-result` 为结果
   统一存放目录，脚本自动创建
5. **dry-run 是最安全的预检**：只打印命令，不校验路径、不创建任何
   目录/文件

## 测试矩阵

每组测试强制追加 `--ignore-eos`；`--no-enable-prefix-caching` 按版本
自动适配（0.26+ 已移除该参数，走默认行为）。

### run_matrix.py（大模型全量，74 组）

| 阶段 | 输入长度 | 输出长度 | 请求速率 (req/s) | 组数 |
|---|---|---|---|---|
| short | 1K / 4K / 8K | 1K / 4K | 1, 8, 16, 32, 64, 128, inf | 42 |
| long | 16K / 32K / 64K / 128K | 1K / 4K | 1, 4, 8, 16 | 32 |

默认参数：请求数 short=200（`--num-prompts`）、long=20（`--num-prompts-long`）；
客户端并发上限 short=32、long=16（`--max-concurrency 0` 表示不限制）。

### run_matrix_lite.py（小模型轻量，63 组）

4090 单卡跑 0.6B 小模型，尺寸已调大以保证 GPU 有压力：

| 阶段 | 输入长度 | 输出长度 | 请求速率 (req/s) | 组数 |
|---|---|---|---|---|
| short | 1K / 2K / 4K | 512 / 1K / 2K | 4, 8, 16, 32, inf | 45 |
| long | 8K / 16K / 32K | 1K / 2K | 4, 8, 16 | 18 |

默认参数：请求数 short=120、long=16；并发上限 short=64、long=32。

其余所有参数与 `run_matrix.py` 完全一致。

## Phase 3 长跑稳定性验证（run_stability.py）

按《模型测试方案书》Phase 3 要求：在 P2 找出的膝点/饱和点基础上，
对服务做**长时间稳定性施压**，监控四条红线（温度降频 / 显存泄漏 /
ECC 错误 / Silent Crash）。

**与 P2 的关键区别**：P2 用 `vllm bench serve`（一次性预生成全部请求，
24h 会爆内存）；P3 用**自定义轻量施压器**（`run_stability.py`）——
流式生成一个发一个，内存只保留在飞窗口，长短文本按分布随机混合。

### 两种运行模式

| 模式 | 命令 | 观察点 |
|---|---|---|
| P3-1 高并发饱和施压 | `--rate inf --max-concurrency <并发> --duration 7200` | 吞吐是否衰减、waiting 队列是否无限堆积 |
| P3-2 24h 生产模拟长跑 | `--rate <极限吞吐 70~80%> --duration 86400` | 24h 红线监控 |

### 用法示例

```bash
cd /vllm-workspace/scripts/models-test-sheel

# 0. 真实短测（强烈建议先跑 1~5 分钟验证链路，再上长跑）
python3 run_stability.py --duration 60 --rate 4 --max-concurrency 16

# P3-1 饱和施压 1 小时（P2 极限并发点，如 64）
python3 run_stability.py --duration 3600 --rate inf --max-concurrency 64

# P3-2 24h 混合长跑（速率取 P2 极限吞吐的 70%~80%，如 3 req/s）
python3 run_stability.py --duration 86400 --rate 3 --max-concurrency 64
```

### 长度分布（A 方案：短/长各 50%）

| 档位 | 输入长度 | 占比 | 可调参数 |
|---|---|---|---|
| 短 | 512 ~ 1K | 50% | `--short-input 512 1024 --short-ratio 0.5` |
| 长 | 4K ~ 8K | 50% | `--long-input 4096 8192` |
| 输出 | 512 ~ 2K（随机） | — | `--out-range 512 2048` |

### 监控红线（stability_series.csv + monitor_events.log）

| 红线 | 采样来源 | 判定 |
|---|---|---|
| 温度触顶降频 | nvidia-smi（temp + clocks.sm/max） | temp≥85℃ 且 SM 时钟 < 最大 90% → ERROR |
| 显存泄漏 | gpu_stats 序列回归 | 后 1/2 时段增速 ≥100 MB/min |
| ECC 计数增加 | nvidia-smi ECC 查询 | uncorrected 计数上升即 ERROR |
| Silent Crash | GET /v1/models 心跳 | 连续 3 次失败 → ERROR |

### 输出产物

- `stability_series.csv`：全时段时序采样（GPU / 队列 / 心跳）
- `monitor_events.log`：红线事件流
- `stability_report.json`：汇总报告（请求数、失败率、req/s、延迟分位数、
  吞吐衰减判定、队列堆积判定、显存泄漏判定）
- `run_config.json`：本次运行参数

> 注意：`--rate` 单位是 **req/s**（不是 tok/s）。P3-2 的速率从 P2 的
> `request_throughput` 列取值（如极限 3.8 req/s，取 70%~80% ≈ 3 req/s）。

## 参数说明（两个脚本通用）

### 基本参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | 顶部 `MODEL` 变量（`qwen3-0.6b`） | 模型名（**必须与服务端 `--served-model-name` 一致**，否则 404） |
| `--tokenizer` | 顶部 `TOKENIZER` 变量（`/models/Qwen3-0.6B`） | tokenizer 本地绝对路径，离线模式必须；压测前做存在性校验 |
| `--model-path` | 顶部 `MODEL_PATH` 变量 | 模型权重目录，仅做存在性校验 |
| `--base-url` | `http://localhost:8000` | vLLM 服务地址（服务解耦部署时改为宿主机可达地址） |
| `--vllm-cmd` | `vllm` | vllm 可执行命令（如 `--vllm-cmd "python3 -m vllm"`） |
| `--phase` | `all` | `short` / `long` / `all` |
| `--result-dir` | 顶部 `RESULTS_DIR` 变量（`/vllm-workspace/results/vllm-bench-result`） | 结果根目录（本地调试可用相对路径覆盖） |
| `--run-id` | 按时间生成 | 本次运行标识（`run-YYYYMMDD-HHMMSS`） |
| `--timeout` | 7200 | 单组测试超时秒数（超时记为 TIMEOUT 并继续） |
| `--sleep-between` | 5.0 | 相邻测试间隔秒数 |
| `--warmup` | 关 | 矩阵前先跑一组小规模预热（1024 in/out，5 请求，与短矩阵首组对齐） |
| `--resume` | 关 | 跳过已有 `result.json` 的测试（断点续跑） |
| `--dry-run` | 关 | 只打印命令不执行（跳过路径校验，也可配合 `--verbose`） |
| `--verbose` | 关 | 打印每条完整命令 |

### 矩阵自定义

| 参数 | 说明 |
|---|---|
| `--input-lens` | 自定义输入长度，如 `--input-lens "1024,4096"` |
| `--output-lens` | 自定义输出长度 |
| `--request-rates` | 自定义速率列表，支持 `inf`，如 `--request-rates 1 16 64` |
| `--num-prompts` / `--num-prompts-long` | 短/长文本测试请求数 |
| `--max-concurrency` | 客户端最大并发；不传按阶段默认（32/16），`0` 表示不限制 |

### 系统监控参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gpu-monitor-cmd` | 无 | **GPU 探测：传注册表短名（如 `nvidia`）或完整命令字符串**（见下节）。不传则只监控 CPU/系统内存 |
| `--gpu-regex` | 无 | 配合上面的命令，用命名组正则解析输出 |
| `--gpu-ids` | 所有 | 限制监控的 GPU 编号，如 `--gpu-ids 0,1`（仅对行式输出有效） |
| `--gpu-interval` | 10 | 采样间隔秒数（低速率组 10s 一次采样大概率采到 GPU 空闲，属正常） |
| `--no-gpu-monitor` | 关 | 传了 `--gpu-monitor-cmd` 后仍想禁采 GPU 时使用（CPU/内存始终采样） |
| `--sys-mem-warn-mb` | 2048 | 系统可用内存告警阈值（MiB），低于则记 WARN 事件 |
| `--gpu-mem-warn-pct` | 95.0 | 显存利用率告警阈值（%） |

## GPU 监控说明

### 厂商注册表（换厂商只改一行）

`sys_monitor.py` 内置 `GPU_PROBES` 注册表，`--gpu-monitor-cmd` 传**短名**
即可，无需再写整串命令：

```python
GPU_PROBES = {
    "nvidia": ("nvidia-smi --query-gpu=utilization.gpu,memory.used,"
               "memory.total,power.draw,temperature.gpu "
               "--format=csv,noheader,nounits"),
    "npu-smi": ("npu-smi info -t usage -i 0"),  # 按实际工具调整
}
```

- 用短名：`--gpu-monitor-cmd nvidia`
- 新增厂商：注册表加一行 + 参数换名字即可，**脚本代码零改动**
- 也仍支持直接传完整命令字符串（未命中注册表时原样执行）

### 输出格式约定

探测命令输出为**每行逗号分隔的数字列**：

```
util,mem_used,mem_total[,power,temp]
```

单位约定：util/power/temp 为原样数值（% / W / ℃），显存单位 MiB
（与 nvidia-smi `--format=csv,noheader,nounits` 输出一致）。

命令输出格式特殊时用 `--gpu-regex` 兜底，要求命名组
`util / mem_used / mem_total / power / temp`：

```bash
--gpu-monitor-cmd "some-custom-tool" \
--gpu-regex "GPU.(?P<util>\d+)%.*?(?P<mem_used>\d+)/(?P<mem_total>\d+)"
```

注意：

- 命令会以 `split()` 拆分执行，含空格参数需注意；带管道的命令请写成
  `bash -c "..."` 形式。
- 命令输出**每行代表一张卡**，多卡时一行一卡；`--gpu-ids` 仅对这类
  行式输出生效。
- 命令失败或无法解析时会打印警告并降级为纯 CPU/内存监控，不影响压测。
- **不配置此参数就等于没有 GPU 采样**，这是设计使然——脚本不探测任何
  厂商工具，保持通用性。
- 注册表查询失败/未配时,探测失败会打印 `[warn]` 并回退默认参数集。

## 监控输出（每组测试目录内）

| 文件 | 内容 |
|---|---|
| `sys_stats.csv` | 时间戳、CPU 利用率、系统内存 used/total/avail（MiB）、swap used/total |
| `gpu_stats.csv` | 时间戳、GPU 编号、利用率、显存带宽利用率、used/total（MiB）、功耗、温度（仅配置了 `--gpu-monitor-cmd` 时生成） |
| `monitor_events.log` | 告警事件：时间、级别（WARN/ERROR）、消息 |
| `test.log` | 该组 `vllm bench serve` 的完整输出 |
| `result.json` | vLLM 官方 bench 结果（0.26+ 需 `--save-result`，脚本已自动处理；未完成时为 ERROR/OOM 状态则缺失或为空） |
| `metadata.json` | 命令、起止时间、时长、状态、泄漏告警 |

### 实时告警（写入 monitor_events.log）

- **显存 OOM 风险**：显存利用率 ≥ `--gpu-mem-warn-pct`（默认 95%）
- **系统 OOM 风险**：可用内存 < `--sys-mem-warn-mb`（默认 2048 MiB）
- **压测日志异常扫描**：按增量读取 `test.log`，命中以下模式即记 ERROR
  事件：`out of memory` / `cuda out of memory` / `failed to allocate` /
  `traceback` / `exception` / `connection refused|reset` / `empty response` /
  worker crash / `request failed with status` / HTTP 500/502/503/504
- 每组失败时自动检查日志尾部，含 OOM 关键字的组状态记为 `OOM`，
  其余为 `FAILED(rc=...)` / `TIMEOUT`

### 内存泄漏检测（测试结束后）

对整组采样序列做线性回归，同时满足以下条件判定"疑似泄漏"，
结果写入 summary 的 `leak_warning` 列及 monitor_events.log：

- 增速 ≥ 阈值：系统内存 50 MB/min、每卡显存 100 MB/min
- 拟合优度 R² ≥ 0.5（增长平稳而非偶然波动）
- 累计上升 ≥ max(起始值 × 2%, 64 MB)

> 采样点 <8 时跳过检测，因此只有跑得久的组（如 rate=1 低速率）才会
> 触发评估；低速率组长时段的"可用内存下降"多为 page cache / KV cache
> 预分配所致，非 Python 泄漏，结合趋势图综合判断。

## 结果目录结构

```
/vllm-workspace/results/vllm-bench-result/run-YYYYMMDD-HHMMSS/
├── run_config.json          # 本次运行的全部参数
├── summary.csv              # 所有组的汇总表（进度实时追加）
├── metadata_<时间戳>.json    # run 级汇总元数据（整个矩阵跑完后生成，含模型/
│                            #   tokenizer、矩阵规模、各状态组数、起止时间）
├── .warmup/                 # --warmup 预热组（不参与统计）
├── short/
│   ├── in1024_out1024_rate1/
│   │   ├── result.json  test.log  metadata.json
│   │   ├── sys_stats.csv  monitor_events.log
│   │   └── gpu_stats.csv            # 仅配置了 GPU 命令时
│   ├── in1024_out1024_rate8/ ...
│   └── ...
└── long/
    ├── in16384_out1024_rate1/ ...
    └── ...
```

各组产物（`result.json` / `summary.csv` / 监控文件 / 日志 / 元数据）
全部落在结果目录下，自行下载归档（不经过 git）。

`summary.csv` 列（部分）：`run_id, phase, input_len, output_len,
request_rate, num_prompts, status, error, output_throughput,
request_throughput, total_tokens, mean/p99_ttft_ms, mean/p99_tpot_ms,
mean/p99_e2e_ms, num_requests, num_failed, leak_warning, start_time,
end_time, duration_s`。

## 结果画图

以下命令均在 `/vllm-workspace/scripts/models-test-sheel` 目录执行（图表
输出到 `figures/`；`--results-dir` 指向结果仓库）：

```bash
python3 plot_results.py --results-dir /vllm-workspace/results/vllm-bench-result   # 最新 run 出图
python3 plot_results.py --results-dir /vllm-workspace/results/vllm-bench-result --run run-20260811-103801
python3 plot_results.py --results-dir /vllm-workspace/results/vllm-bench-result/run-20260811-103801   # 直接指到 run 目录
python3 plot_results.py --results-dir /vllm-workspace/results/vllm-bench-result --gpu-series          # 额外输出每组时序图
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--results-dir` | （必填） | 结果目录（run 根目录或包含多个 run 的上级目录） |
| `--out-dir` | `figures` | 图表输出目录 |
| `--run` | 最新 | 多个 run 时指定其一（`run-YYYYMMDD-HHMMSS`） |
| `--gpu-series` | 关 | 额外生成每组测试的监控时序图 |
| `--dpi` | 150 | 图分辨率 |
| `--font` | 自动 | 指定中文字体名（如 `--font "Noto Sans CJK SC"`） |

输出（`figures/` 下）：

| 文件 | 内容 |
|---|---|
| `throughput_latency_short.png` / `_long.png` | 吞吐-延迟权衡图，TTFT/TPOT 子图均标出膝点（推荐配置） |
| `concurrency_saturation_short.png` / `_long.png` | 速率-吞吐 + P99 TTFT，标出排队临界 rate 竖线 |
| `long_context_prefill_long.png` | 长文本性能衰减图（输入长度 vs P99 TTFT + Max Concurrent Requests） |
| `long_context_gpu_mem_long.png` | 长文本显存压力图（输入长度 vs 峰值显存占用） |
| `gpu_series/*.png` | 每组测试的监控时序图：GPU 利用率 / 显存 / CPU+系统内存 |
| `summary_metrics.csv` | 每组合并后的指标表格（含峰值 GPU 指标与泄漏告警） |

说明：

- **排队临界 rate 的判定**（饱和图竖线）：优先取 P99 TTFT 首次 ≥ 最低速率
  基线 2 倍的速率（排队拉高延迟）；若 TTFT 始终不翻倍，则回退取 output
  throughput 首次达到峰值 95% 的速率（吞吐饱和点）。两者都不满足则不画线。
  例：0.6B 实测 4K 输入组临界 rate=4、1K 输入组 16~32。
- **Max Concurrent Requests** 取自 result.json 的 `max_concurrent_requests`
  （vLLM 调度器观测到的并发运行峰值，直接反映 KV Cache 容量压力）；随输入
  长度增长而下降即 KV Cache 压力上升的表现。
- **Prefill 耗时**以 P99 TTFT 近似（vLLM 0.26 的 result.json 无独立
  prefill 字段）。

- 无 GPU 数据的 run（未配置 `--gpu-monitor-cmd`）时序图自动退化为纯
  CPU/系统内存面板，其余图不受影响，`summary_metrics.csv` 的 GPU 列为空。
- `rate=inf` 的点会画在 X 轴末尾位置并单独标注。
- 中文标签自动匹配系统字体（Microsoft YaHei / SimHei / Noto Sans CJK /
  Source Han Sans 等）；赛博服务器无图形环境正常画图（Agg 后端），
  缺中文字体时可用 `--font` 指定。
- 图例与标题已做布局分层（`bbox_to_anchor` + `tight_layout`），多子图
  时顶部不再重叠。

### Phase 3 结果画图（plot_stability.py）

P3 数据格式与 P2 不同（无 summary.csv），用独立脚本，一个 P3 run 一张
四面板时序图（GPU 利用率/温度/功耗、显存、waiting/running 队列、SM
时钟 + 心跳），ERROR 红线事件画红色竖线，报告摘要写入图顶：

```bash
python3 plot_stability.py --results-dir run-20260817-030000            # 单个 P3 run
python3 plot_stability.py --results-dir /vllm-workspace/results --out-dir figures   # 多个 run 一起
```

输出：`<run_id>_stability.png` + `stability_summary.csv`（每个 run 一行，
含趋势判定结果、ERROR/WARN 数、GPU 峰值/温度/队列峰值）。

## 实用技巧
- **先用 dry-run 再跑**：`--dry-run --verbose` 打印全部命令，确认
  `vllm bench serve` 参数与本地 vLLM 版本吻合后再执行（dry-run 跳过
  路径校验，且不创建任何目录）。
- **切换模型**：修改 `run_matrix.py` 顶部 `MODEL` / `TOKENIZER` /
  `MODEL_PATH` 三个变量（对应服务端 served-model-name、本地 tokenizer
  目录、模型权重目录），命令本体无需改动。
- **中途修改/补跑某组**：`--resume` 跳过已有 `result.json` 的组；
  想强制重跑某组可先删该组目录再执行。
- **长时运行挂后台**（服务器上）：

  ```bash
  cd /vllm-workspace/scripts/models-test-sheel
  tmux new -s bench
  python3 run_matrix.py --phase all --warmup 2>&1 | tee bench.log
  # 断开 tmux 继续跑；断网/中断后用 --resume 续跑
  ```

- **断点续跑**：中断后重跑同样参数并加 `--resume`，已有 `result.json`
  的组自动跳过；`summary.csv` 汇总会自动续写打开新文件头，可用
  run_config.json 核对。
- **中途查看进度**：`tail -f /vllm-workspace/results/vllm-bench-result/<run-id>/summary.csv`。
- **飞控并发**：`--max-concurrency 0` 取消客户端并发上限（吞吐压满），
  但有 prefill 堆积拉爆 TTFT 的风险，仅在需要极限吞吐时使用。
- **只跑特定组合**：`--input-lens "1024" --output-lens "1024" --request-rates 1 16`。

## VSCode 调试（Run and Debug）

`.vscode/launch.json` 放在 `/vllm-workspace/.vscode/`，当前全部指向
**lite 版本**（`scripts/models-test-sheel/run_matrix_lite.py`），适配单卡
4090 小体量流程测试。**VSCode 打开的文件夹应为 `/vllm-workspace`**
（`program` 用 `${workspaceFolder}/scripts/models-test-sheel/...` 定位脚本）：

| 配置 | 用途 |
|---|---|
| `lite: dry-run (short)` | 断点查看 short 命令生成，不实跑 |
| `lite: dry-run (all 63)` | 查看全部 63 组命令 |
| `lite: real short + warmup (GPU 监控)` | 正式跑 short 45 组 + 预热，GPU 采样走注册表短名 `nvidia` |
| `lite: real all 63 (GPU 监控)` | 全矩阵 |
| `lite: single combo` | 单组 × 3 请求，快速单步调试监控逻辑 |
| `plot results` | 调试结果画图（指向 vllm-bench-result） |
| `attach` | 附加到已运行的压测进程 |

使用要点：

- 先选择解释器：Ctrl+Shift+P → **Python: Select Interpreter** → 选容器内
  Python 3.12；`launch.json` 的 `python` 为
  `${command:python.interpreterPath}`，跟随所选解释器，无需改路径
- debugpy 由 Python 扩展自动使用，容器内无需手动管理
- **调试前先启动 vLLM 服务**（见前置条件），否则真实运行的组会 FAILED；
  用 `dry-run` 配置下断点查 `build_command` 是最安全的调试入口

**attach 附加调试**（在已运行的压测进程中下断点，无需重启压测）：

```bash
cd /vllm-workspace/scripts/models-test-sheel && source .venv/bin/activate
python3 -m debugpy --listen 5678 --wait-for-client \
    run_matrix_lite.py --phase short --warmup
```

再从 VSCode 选择 `attach` 配置（F5）连接 5678。当内置配置与本地环境
不一致时，可直接编辑 `launch.json`（改 `program`、`--base-url`、
`--gpu-monitor-cmd` 等）。

## 常见问题排查

| 现象 | 处理 |
|---|---|
| `错误：未找到 vllm 命令` | 未在 venv 中执行，或 `--vllm-cmd` 路径不对 |
| `错误：tokenizer 路径不存在` | 检查脚本顶部 `TOKENIZER` / `MODEL_PATH` 变量与 `/models/` 下实际目录是否一致 |
| `POST /v1/completions 404` 且 GPU 空转 | **模型名不匹配**：脚本 `--model` 必须等于服务端 `--served-model-name`（如 `qwen3-0.6b`），对照 `vllm serve` 的启动参数修正 `MODEL` 变量 |
| 测试全 OK 但 result.json 缺失、summary 指标全空 | vLLM 0.26+ 需 `--save-result`（脚本已自动追加）；手动跑 bench 时补 `--save-result --result-file xxx.json` |
| `FileNotFoundError .../scripts/run_matrix_lite.py` | launch.json 的 `program` 与目录布局不一致（脚本现在在 `scripts/models-test-sheel/`），换容器后记得同步 `.vscode/launch.json` |
| 显存曲线平线、GPU 利用率有波动 | 正常：vLLM 启动时预分配显存（gpu-memory-utilization），显存看"分配"、利用率看"忙闲" |
| 低速率组（rate=1/4/8）GPU 采样几乎全 0 | 正常：0.6B 单请求几十 ms，10s 采样间隔大概率采到空闲；看 `inf`/高 rate 组验证 GPU 满载 |
| `无法探测 vLLM bench serve 参数` | 按常见参数默认值继续，实际运行前会重新探测；确认 vllm 在 PATH 中 |
| GPU 探测命令失败警告 | 命令在 shell 中手动执行确认输出；注意 `split()` 拆分与管道问题；短名需先在 `GPU_PROBES` 注册 |
| 某组 OOM | 看看该组 gpu_stats.csv 与 monitor_events.log 的显存告警；长文本显存压力图和 summary 的 `num_failed` 定位 |
| 中文变方块 | 服务器装中文字体或用 `--font` 指定已有字体 |
| 想改矩阵规模 | 用 `--input-lens/--output-lens/--request-rates` 覆盖，或直接改脚本顶部 SHORT_*/LONG_* 常量 |

## 指标口径（result.json 解析）

- `output_throughput`：输出 token 吞吐（tok/s）
- `ttft / tpot / e2e` 的 mean / p99：解析 `percentiles_ttft_ms` 等
  百分位字典或平铺字段，兼容新旧 vLLM bench 输出格式
- `num_failed`：失败请求数（`num_failed_requests`）
- 网络/服务不可用导致进程非零退出时，状态为 `FAILED(rc=...)`，
  日志含 OOM 关键字时为 `OOM`，超过 `--timeout` 为 `TIMEOUT`

## 依赖

- 压测 + 监控：Python 3.9+ 标准库（CPU 采样优先 psutil，缺失时回退
  读 /proc，零安装成本）
- 画图：`matplotlib>=3.7`（见 requirements.txt）