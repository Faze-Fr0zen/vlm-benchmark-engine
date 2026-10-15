# vLLM 国产 GPU 基准测试与参数调优工具集

面向国产 GPU 厂商（昇腾 / 海光 / 寒武纪 / 沐曦 / 天数智芯 / 燧原等）的
模型基准测试、Serve 参数调优与报告生成一体化工具集。本仓库由三个独立组件
组成，各组件自带 README，串成一条完整流水线：**调优锁定配置 → 全量矩阵验收
→ 报告生成**。

## 整体架构

```
┌─ models-optimize-sheel（调优）─────────────┐
│  探针收敛引擎 → 锁定最佳 vllm serve 配置    │
│  产出：opt_summary.csv + best_serve_cmd.txt │
└────────────────────┬───────────────────────┘
                     ▼
┌─ models-test-sheel（测试）─────────────────┐
│  run_matrix.py P2 矩阵压测                 │
│  run_stability.py P3 稳定性长跑            │
│  plot_results.py 画图                      │
│  产出：summary.csv + stability_report.json │
└────────────────────┬───────────────────────┘
                     ▼
┌─ report-template（报告）───────────────────┐
│  generate_report.py → 基准测试报告         │
│  generate_tuning_guide.py → 参数调优指南   │
│  hardware_env.json 环境信息 + 达标线判定    │
└────────────────────────────────────────────┘
```

## 组件总览

| 组件 | 职责 | 关键产出 | README |
|---|---|---|---|
| [models-optimize-sheel](models-optimize-sheel/) | vLLM Serve 参数调优：探针 + 收敛引擎，分层收敛出锁定配置 | `opt_summary.csv`、`best_serve_cmd.txt`、`report.md` | [查看](models-optimize-sheel/README.md) |
| [models-test-sheel](models-test-sheel/) | 基准测试：P2 矩阵压测、P3 长跑稳定性、系统监控、画图 | `summary.csv`、`stability_report.json`、图表 | [查看](models-test-sheel/README.md) |
| [report-template](report-template/) | 报告生成：基准测试报告 + 参数调优指南（客户版） | `report_<厂商>.md`、`tuning_guide_<厂商>.md` | [查看](report-template/README.md) |

## 工作流程

1. **调优**（models-optimize-sheel）：在固定模型、固定硬件上，用探针收敛引擎
   分层扫描 Serve 参数（显存×并发 → kv-dtype → 调度），锁定最佳启动命令
   `best_serve_cmd.txt`
2. **测试**（models-test-sheel）：按锁定配置跑全量 P2 矩阵压测 + P3 长跑
   稳定性验证，产出 `summary.csv` 与 `stability_report.json`
3. **报告**（report-template）：用 `hardware_env.json` + 测试/调优产物自动
   生成基准测试报告与参数调优指南，机器可读部分自动填充，结论由人工补填

三者数据口径完全对齐：`report-template` 的判定逻辑与 `models-test-sheel`
的 `plot_results.py` 一致，调优指南直接消费调优产物的结果。

## 环境要求

- Python 3.9+，全部脚本零第三方依赖（标准库）
- 测试脚本须在国产 GPU 服务器上运行；GPU 探测走登记制注册表（各厂商短名）
- 模型配置（MODEL / TOKENIZER / 路径）在各自脚本顶部常量修改

## 快速开始

各组件用法、参数表与代码示例见对应 README：

- 调优操作手册：[models-optimize-sheel/STEP_BY_STEP.md](models-optimize-sheel/STEP_BY_STEP.md)
- 压测与监控：[models-test-sheel/README.md](models-test-sheel/README.md)
- 报告生成：[report-template/README.md](report-template/README.md)
