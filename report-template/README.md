# 模型基准测试报告生成（report-template）

面向国产 GPU 厂商（昇腾 / 海光 / 寒武纪 / 沐曦 / 天数智芯 / 燧原等）的
通用基准测试报告体系：一份填写式模板 + 一份环境信息 JSON + 一个自动生成
脚本，数据口径与 `models-test-sheel` 压测产物完全对齐。

## 整体架构

```
hardware_env.json（每厂商一份）──┐
summary.csv（P2 压测结果）──────┼──→ generate_report.py ──→ report_<厂商>.md（草稿）
stability_report.json（P3）─────┘                                  │
                                         人工补填 TODO 结论后发布 → report_template.md（标准结构）
```

三层分离：**环境信息**（JSON，机器可读）→ **测试数据**（CSV/JSON，证据）→
**报告结论**（Markdown，人写的判断），三者互不污染。

## 文件说明

| 文件 | 作用 |
|---|---|
| `report_template.md` | 填写式主模板：摘要 → 环境 → 模型部署 → P2 矩阵 → P3 稳定性 → 达标线判定 → 配置推荐 → 附录 |
| `hardware_env.json` | 环境信息模板，**每厂商复制一份**（如 `hardware_env_ascend.json`），含硬件/软件栈/GPU 探测登记/达标线 |
| `generate_report.py` | 自动生成报告草稿：机器可读部分自动填充，结论留 TODO |
| `report_demo.md` | 本地示例数据生成的示例草稿 |

## 脚本关键信息

- **数据源**：`summary.csv`（必填）+ `stability_report.json` / `monitor_events.log`（可选）+ `hardware_env.json`（必填）
- **判定口径**（与 `plot_results.py` 一致）：排队临界 rate = P99 TTFT 首次 ≥ 最低速率基线 2 倍；吞吐饱和 = 吞吐首次达峰值 95%；推荐速率取二者较小者
- **达标线判定**：`acceptance_lines` 配置量化阈值，自动判定"达标/不达标"，让报告有明确结论
- **零第三方依赖**（Python 3.9+ 标准库）

## 报告生成流程

1. 复制 `hardware_env.json` 为厂商版本，填写环境信息与 `acceptance_lines` 达标线
2. 在厂商服务器跑测试（见 `models-test-sheel/README.md`：矩阵压测 + 长跑稳定性）
3. 下载结果目录，生成报告草稿
4. 补填草稿中所有 TODO（结论、推荐配置、踩坑记录），对照 `report_template.md` 核对章节完整性后发布

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

# 生成报告草稿（带 P3 稳定性与图表索引）
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

## 附：多厂商对比怎么做

本报告不内置横向对比章节（各厂商测试条件、软件栈版本难以归一）。
需要对比时，从各厂商报告的 0.2 / 5.4 节手工取数汇总，且必须写明对比
条件（同模型/量化/上下文/请求分布）与归一化口径（按卡数/算力/功耗）。
踩坑与兼容性问题记录在附录 A，形成厂商问题库。