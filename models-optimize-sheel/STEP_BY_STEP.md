# Serve 参数调优操作手册（半自动模式）

> 本手册与 `probe_only.py` 收敛引擎配套。**你不需要思考下一步该测什么**——
> 每次探针跑完，引擎会直接输出：本轮进度 + 下一组完整启动命令 + 晋级结论。
> 你只需要：复制命令 → 启动服务 → 确认探针 → 停服务等显存释放。

---

## 0. 部署与准备（一次性）

### 服务端（跑 vLLM 的机器）

| 文件 | 用途 |
|---|---|
| `sys_monitor.py`（models-test-sheel） | 旁路 GPU 采样，输出 `gpu_stats.csv` |

### 测试端（跑探针的机器）

| 文件 | 用途 |
|---|---|
| `probe_only.py` | 双探针执行 + 评分 + 收敛引擎 |
| `serve_manager.py` | 参数映射 + 显存释放检测 |
| `decision.py` | 评分规则 |
| `models-test-sheel/run_matrix.py` | 探针命令构建 / 指标解析 / GPU 窗口聚合 |

> `probe_only.py` 通过相对路径 `../models-test-sheel` 引用 `run_matrix.py`，
> 两个目录必须同级部署。Python 3.9+ 标准库，无第三方依赖。

### 准备步骤

1. 部署两个目录到测试端，`sys_monitor.py` 拷到服务端（或直接部署同目录）
2. 确认服务端与测试端时钟同步（秒级即可，两端各跑一次 `date` 对一下）
3. 确认模型已就位（默认 `/models/Qwen3-0.6B`，tokenizer 本地路径）

---

## 1. 每轮标准流程（循环执行）

每轮（每组配置）只有 4 步，全部由收敛引擎驱动：

```
┌─────────────────────────────────────────────────────────┐
│ 0. 启动旁路采样（服务端，只做一次，整轮调优持续开着）      │
│    python sys_monitor.py --out-dir /tmp/monitor \        │
│        --interval 5 --gpu-monitor-cmd nvidia --clocks-ecc│
│    （Ctrl+C 结束；记录启动时刻，见第 4 节对时）            │
└────────────────────────────┬────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│ 1. 复制收敛引擎输出的"启动命令"，在服务端手动启动 vllm     │
│    serve（等 /v1/models 就绪，日志可重定向到文件）          │
└────────────────────────────┬────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│ 2. 测试端跑探针（命令引擎同时给出）                        │
│    python probe_only.py --config "mem=0.95,seqs=128,..." │
│    （自动跑 short + long 双探针，评分写入评分表）           │
└────────────────────────────┬────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│ 3. 看收敛引擎输出 → 拿到下一组启动命令 → 停掉当前服务        │
│    （用 serve_manager 的显存释放检测确认回落，避免 OOM 残留）│
└────────────────────────────┬────────────────────────────┘
                             ▼
                        回到第 1 步，直到引擎输出 [锁定]
```

### 显存释放检查（换组前必做）

服务手动停掉后，跑：

```bash
python -c "import serve_manager as s; s.wait_gpu_freed()"
```

输出 `已释放`（或探测失败跳过）即可换下一组。若一直不回落（等 120s
超时警告），说明有残留进程：`nvidia-smi` 找 PID 手动清掉。

### 断点续跑

`summary.csv` 是累积的。中断后重新执行同一条 `probe_only.py --config ...`
命令会自动跳过已完成配置（提示"已有评分记录"），并重新输出当前进度和
下一组命令。**不需要 --resume 参数，天然幂等**。

---

## 2. 收敛矩阵与引擎规则（probe_only.py 内置）

| 轮次 | 目的 | 组合 | 判据（引擎自动执行） |
|---|---|---|---|
| **Round 1** | 显存×并发网格定位 | 9 组：mem(0.90/0.95/0.98) × seqs(64/128/256)，kv=auto | 9 组全测完 → 取 ACCEPT 中 score 最高者（同分取更激进档：mem↑、seqs↑）为 top1；出现 REJECT 的 mem 档位记录"显存天花板"，以上档位不再尝试 |
| **Round 2** | kv-dtype 补扫 | 2 组：top1 + kv=auto / kv=fp8 | fp8 score > auto → **Round 2.5** 复活点验证；否则 → Round 3 |
| **Round 2.5** | 复活点验证（fp8 赢时触发） | 1~2 组：Round 1 中 REJECT 的大显存/高并发点，以 fp8 重测 | fp8 成功救回 REJECT 点 → 记录"复活点"，后续锁定配置带 fp8；救不回 → 维持 auto |
| **Round 3** | 调度 A/B 控制变量 | 6 组：top1 基础上 chunked(off/on) × batched(4K/8K/16K) | 全测完 → B 最优 score > baseline(top1) 才替换，否则锁定 Round1 top1；最优 batched 触到 16K 上限时提示可手工扩展 |
| **锁定** | 收敛完成 | — | 输出最终 serve 命令 → 用锁定配置重跑 run_matrix.py 全量矩阵验收 |

**4090 小体量版**（`--lite`）：Round 1 = mem(0.90/0.95) × seqs(64/128) = 4 组
（跳过 0.98/256，24GB 极易 OOM）；Round 3 = chunked × batched(4K/8K) = 4 组；
探针减半（short 60 / long 8 请求）。

### 评分判据（decision.py，方案书 4.2）

| 观测现象 | 判定 | 后续动作（引擎） |
|---|---|---|
| 探针失败 / 请求失败 / OOM | REJECT | 淘汰；显存>95% 建议降 seqs |
| 显存峰值 > 95% | REJECT | 淘汰；记录显存天花板 |
| TTFT P99 > 5s | REJECT | 淘汰（Prefill 排队失控） |
| GPU 利用率 < 70% | WARN | 不参与 top1 排序，建议升 batched |
| 其余正常 | ACCEPT | 参与评分排序 |

---

## 3. 每轮示例（全量版）

### Round 1 第一组

```bash
# 服务端（收敛引擎输出的启动命令）:
vllm serve /models/Qwen3-0.6B --served-model-name qwen3-0.6b --dtype half \
    --max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs 64 \
    --kv-cache-dtype auto

# 测试端:
python probe_only.py --config "mem=0.9,seqs=64,kv=auto" \
    --gpu-csv /path/to/gpu_stats.csv
```

输出节选：

```
[R1-mem=0.9,seqs=64] verdict=ACCEPT score=82.5
[收敛引擎] 读取评分表（1 条记录）:
[Round 1] 已完成 1/9，下一组（8 组剩余）:
  启动命令: vllm serve ... --gpu-memory-utilization 0.9 --max-num-seqs 128 ...
  [判定] 启动后运行: python probe_only.py --config "mem=0.9,seqs=128,kv=auto"
```

→ 复制命令换下一组。全程不需要任何决策。

### Round 2 / 2.5 / 3 的入口

引擎会依次自动给出（Round 1 完成 → Round 2 第一组 → 判定 fp8 胜负 →
Round 2.5 或 Round 3 → … → 锁定）。中途任何中断，重跑最后一条
`probe_only.py --config ...` 即可恢复现场。

---

## 4. GPU 采样与事后对齐（gpu_stats.csv）

### 采样（服务端）

```bash
python sys_monitor.py --out-dir /tmp/monitor --interval 5 \
    --gpu-monitor-cmd nvidia --clocks-ecc
```

- `--clocks-ecc`：追加采样时钟与 ECC 计数（throttle/纠错判定用）
- 输出 `gpu_stats.csv`，头部含时间戳；启动时会打印本机当前时间用于对时
- 调优全程保持运行，结束时 Ctrl+C；**换组不重启采样**

### 对齐（测试端）

每轮探针跑完后把 `gpu_stats.csv` 拷回测试端，用 `--gpu-csv` 传入：
`probe_only.py` 会按探针实际起止时间窗口自动聚合 GPU 指标
（平均利用率、显存峰值、泄漏检测）写入评分表。

### 服务端无 nvidia-smi（国产卡）

- 采样：`--gpu-monitor-cmd <任意探测命令>`（输出每行第一个数字为已用
  显存 MiB），或注册表短名
- 释放检测：`serve_manager.py` 的 `wait_gpu_freed(gpu_free_cmd=...)`
  同样支持自定义命令

---

## 5. 验收（Round 4）

锁定配置后：

```bash
# 服务端: 按引擎输出的 [锁定] 命令启动服务（普通启动，不用旁路参数）
# 服务端: 旁路采样照常开着

# 测试端: 全量矩阵验收（新目录 run-p2-*）
python run_matrix.py --warmup

# 稳定性（P3-2, rate 取 P2 极限吞吐的 70%~80%）
python run_stability.py --duration 86400 --rate 0.9 --max-concurrency 64 \
    --gpu-csv <拷回的 gpu_stats.csv>

# 画图出报告
python plot_results.py --results-dir ... --gpu-csv <全局 gpu_stats.csv>
```

### 生成调优指南报告

```bash
python report.py --results-dir <probe_only 结果根目录> \
    --p2-before <默认配置 run-p2 目录> \
    --p2-after  <锁定配置 run-p2 目录>
```

> `--p2-before` 用**锁定前**（默认配置）跑的 run-p2（如 `run-20260814-023631`），
> `--p2-after` 用锁定配置重跑的 run-p2。输出：
> `opt_summary.csv`（评分表）+ `best_serve_cmd.txt`（锁定命令）+
> `report.md`（锁定配置 + 收敛旅程回放 + 验收收益对比）。

---

## 6. 故障排查

| 现象 | 处理 |
|---|---|
| 服务起不来 / 探针全部 FAILED | 看服务端日志（OOM / 参数不支持）；`/v1/models` 探活；全部 REJECT 时引擎会提示手工降 seqs=32 重测一组验证 |
| 探针 OOM（服务端日志含 out of memory） | 引擎记 REJECT + 显存天花板，正常流程继续 |
| 显存不释放 | `nvidia-smi` 查残留进程，手动 kill；再跑 `wait_gpu_freed` 确认 |
| 评分表里漏了某组 | 重跑该组 `probe_only.py --config ...` 自动补测（幂等跳过已有） |
| GPU 列是空的 | `--gpu-csv` 路径不对或窗口对不上（时间同步问题）；检查采样是否开着 |
| 收敛引擎卡在"全部 REJECT" | 按引擎建议人工干预一次，然后重跑探针恢复流程 |

---

## 7. 与旧全自动版的区别

| 项 | 旧版（全自动，已删除） | 新版（probe_only.py 半自动） |
|---|---|---|
| 服务启停 | 脚本自动重启 + 交互确认 | **操作者手动启停**（按引擎给的命令） |
| 监控 | 测试脚本内联 GPU 采样 | **服务端旁路采样**（sys_monitor --standalone），事后 --gpu-csv 对齐 |
| 下一步 | 脚本内循环 | **引擎输出下一组完整命令**，人复制执行 |
| 显存释放 | 自动等待 | `wait_gpu_freed()` 一条命令 |
| 断点 | `--resume` | 天然幂等（按 config_id 跳过） |
| 验收 | — | report.py `--p2-before/--p2-after` 量化收益 |
