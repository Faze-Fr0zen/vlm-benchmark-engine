# 模型基准测试报告生成（report-template）

面向国产 GPU 厂商（昇腾 / 海光 / 寒武纪 / 沐曦 / 天数智芯 / 燧原等）的
通用基准测试报告体系：一份填写式模板 + 一份环境信息 JSON + 两个自动生成
脚本（基准测试报告 / 参数调优指南），数据口径与 `models-test-sheel` 压测
产物、`models-optimize-sheel` 调优产物完全对齐。

## 整体架构

```
hardware_env.json（每厂商一份）────────────────┐
summary.csv（P2 压测结果）──────────┐          ├→ generate_report.py → 报告
stability_report.json（P3）─────────┤          │
opt_summary.csv + best_serve_cmd.txt ├→ generate_tuning_guide.py → 调优指南
（report.py 调优产物）              │
────────────────────────────────────┤
           机器部分自动填充，人工补填 TODO 结论后发布
```

三层分离：**环境信息**（JSON，机器可读）→ **测试/调优数据**（CSV/TXT，证据）→
**报告结论**（Markdown，人写的判断），三者互不污染。

## 文件说明

| 文件 | 作用 |
|---|---|
| `report_template.md` | 填写式主模板（基准测试报告）：摘要 → 环境 → 模型部署 → P2 矩阵 → P3 稳定性 → 达标线判定 → 配置推荐 → 附录 |
| `hardware_env.json` | 环境信息模板，**每厂商复制一份**（如 `hardware_env_ascend.json`），含硬件/软件栈/GPU 探测登记/达标线/模型参数 |
| `generate_report.py` | 基准测试报告生成：机器可读部分自动填充，结论留 TODO |
| `generate_tuning_guide.py` | 参数调优指南生成：锁定配置/评分表/验收对比/硬件资源估算自动填充 |
| `validate_env.py` | `hardware_env.json` 字段完整性与类型校验（含拼写检查） |
| `report_demo.md` | 基准测试报告示例草稿（本地示例数据生成） |
| `tuning_guide_demo.md` | 参数调优指南示例草稿（`data/opt-demo/` 示例数据生成） |

## 脚本关键信息

- **基准测试报告**：数据源 `summary.csv`（必填）+ `stability_report.json` / `monitor_events.log`（可选）+ `hardware_env.json`（必填）；判定口径与 `plot_results.py` 一致（排队临界 rate = P99 TTFT 首次 ≥ 最低速率基线 2 倍；吞吐饱和 = 吞吐首次达峰值 95%）
- **参数调优指南**：数据源 `models-optimize-sheel/report.py` 产物（`opt_summary.csv` + `best_serve_cmd.txt`）+ `hardware_env.json`；锁定配置取 ACCEPT 最高分；硬件资源建议按锁定配置实测显存峰值与权重体积估算
- **零第三方依赖**（Python 3.9+ 标准库），两脚本共享 `generate_report.py` 的格式化工具

## 报告生成流程

1. 复制 `hardware_env.json` 为厂商版本，填写环境信息与 `acceptance_lines` 达标线
2. **先用 `validate_env.py` 校验**（见下节），通过后再继续
3. 在厂商服务器跑测试（见 `models-test-sheel/README.md`：矩阵压测 + 长跑稳定性）
4. 下载结果目录，生成报告草稿
5. 补填草稿中所有 TODO（结论、推荐配置、踩坑记录），对照 `report_template.md` 核对章节完整性后发布

## 环境文件校验（validate_env.py）

```bash
python validate_env.py --env hardware_env_ascend.json
```

校验字段完整性与类型，输出三级结果（退出码 0=无 ERROR）：

| 级别 | 含义 | 处理 |
|---|---|---|
| ERROR | 缺失或类型错误（vendor / 加速卡型号 / 模型名 / 数值写成字符串等） | 必须修复 |
| WARN | 未填，对应章节输出 TODO 或不自动判定（显存/功耗/参数量/达标线等） | 建议补填 |
| INFO | 可选字段未填 | 可忽略 |

含未知字段拼写检查（如把 `hbm_capacity_gb` 写成 `hbm_capacity`），防止笔误导致报告静默缺数据。

## 参数说明

| 参数 | 必填 | 说明 |
|---|---|---|
| `--env` | 是 | `hardware_env.json` 路径（每厂商一份） |
| `--run` | 是 | P2 结果 run 目录（含 `summary.csv`） |
| `--stability` | 否 | P3 稳定性 run 目录（含 `stability_report.json`），不传则第 4 章留人工 |
| `--figures-dir` | 否 | 图表目录，写入附录 B 索引 |
| `--output` | 否 | 输出路径，缺省 `report_<run_id>.md` |

## 达标线判定（acceptance_lines）

| 字段 | 含义 | 判定口径 |
|---|---|---|
| `short_peak_tok_s` | 短文本极限吞吐达标线 | 实测 = 各组最大 output_throughput |
| `long_peak_tok_s` | 长文本极限吞吐达标线 | 同上 |
| `p99_ttft_ratio_max` | TTFT 基线倍数上限（默认 2.0） | 实测 = 所有组合中「推荐速率点 P99 TTFT / 最低速率基线」最大倍数 |
| `max_failure_rate_pct` | 请求失败率上限（%） | num_failed / num_requests |
| `max_bad_groups` | 异常组数上限 | FAILED/OOM/TIMEOUT 组数 |
| `stability_redlines_ok` | P3 红线是否必须全部通过 | trends 全通过且无 ERROR 事件 |
| `min_gpu_util_pct` | 满载点 GPU 利用率下限（%） | 全 run 组级峰值最大值 |

字段留 `null` 则该行输出 TODO 由人工判定。

## 代码示例

```bash
# 每厂商环境文件
cp hardware_env.json hardware_env_ascend.json

# 生成基准测试报告草稿（带 P3 稳定性与图表索引）
python generate_report.py \
    --env hardware_env_ascend.json \
    --run /vllm-workspace/results/vllm-bench-result/run-20260814-023631 \
    --stability /vllm-workspace/results/vllm-bench-result/run-20260817-052818 \
    --figures-dir figures \
    --output report_ascend.md

# 仅 P2 数据（P3 章节留人工）
python generate_report.py --env hardware_env_ascend.json \
    --run run-20260814-023631 --output report_ascend_p2.md
```

## 参数调优指南（generate_tuning_guide.py）

### 数据流与章节

```
models-optimize-sheel/report.py ──→ opt_summary.csv + best_serve_cmd.txt ──┐
hardware_env.json（params_b / 显存 / TDP）─────────────────────────────────┼─→ tuning_guide_<厂商>.md
run_matrix 输出（--p2-before / --p2-after，验收对比）─────────────────────┘
```

章节：0 摘要（结论速览 + 人工结论）→ 1 目标与方法 → 2 调优结果（锁定配置 /
参数说明 / 评分表）→ 3 验收对比（默认 vs 锁定）→ 4 生产部署建议（硬件资源
配置估算 + 部署清单）→ 5 调优过程问题记录 → 附录。机器部分自动填充，
结论留 TODO。

### 参数说明

| 参数 | 必填 | 说明 |
|---|---|---|
| `--opt-dir` | 是 | `report.py` 输出目录（含 `opt_summary.csv` + `best_serve_cmd.txt`） |
| `--env` | 是 | `hardware_env.json` 路径（模型 `params_b` 用于权重估算） |
| `--p2-before` | 否 | 验收 before：默认配置的 run_matrix 输出目录 |
| `--p2-after` | 否 | 验收 after：锁定配置的 run_matrix 输出目录 |
| `--output` | 否 | 输出路径，缺省 `tuning_guide_<厂商>.md` |

### 硬件资源估算口径（4.1 节，自动）

- 权重体积 = 参数量 × 每参数字节（fp16/bf16=2、fp8/int8=1、int4=0.5）
- 单卡显存需求 = 锁定配置实测显存峰值 × 单卡容量，与权重估算取大 ×1.25 余量，向上取整到常见容量档
- 卡数 = 权重体积 ÷ 显存利用率 ÷ 单卡容量；CPU/内存/功耗按经验公式
- 以上为估算，报告内标注需人工复核

### 代码示例

```bash
python generate_tuning_guide.py \
    --opt-dir opt_report \
    --env hardware_env_ascend.json \
    --p2-before ../results/run-p2-baseline \
    --p2-after  ../results/run-p2-locked \
    --output tuning_guide_ascend.md
```

### 本地验证（demo 数据）

```bash
python generate_tuning_guide.py --opt-dir data\opt-demo \
    --env data\opt-demo\hardware_env_demo.json \
    --p2-before data\smoke\run-smoke-p2 --p2-after data\smoke\run-smoke-p2 \
    --output tuning_guide_demo.md
```

## 附：多厂商对比怎么做

本报告不内置横向对比章节（各厂商测试条件、软件栈版本难以归一）。
需要对比时，从各厂商报告的 0.2 / 5.4 节与调优指南 3 节手工取数汇总，
且必须写明对比条件（同模型/量化/上下文/请求分布）与归一化口径
（按卡数/算力/功耗）。踩坑与兼容性问题记录在基准测试报告附录 A，
形成厂商问题库。