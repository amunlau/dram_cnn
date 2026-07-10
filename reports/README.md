# reports/

本目录存放《基于CNN的内存故障预测算法设计报告》及配套材料。

| 文件/目录 | 说明 |
|-----------|------|
| `基于CNN的内存故障预测算法设计报告.md` | 主报告 V3.0（九章清晰结构；算法设计与验收） |
| `算法开发与实验结果.md` | 实验设计、评价方法与 **2026-07-10 实测结果**（含配图） |
| `algorithm_scripts_index.md` | Shell / Pipeline 入口与复现命令 |
| `figures/` | Mermaid 图源（`.mmd`）与实验配图（`fig7_*.png`） |

**阅读顺序**：主报告 → `算法开发与实验结果.md` → `algorithm_scripts_index.md` → `figures/` 对应图源。

**复现**：`export PYTHONPATH=src`；命令见 `algorithm_scripts_index.md` 与 `configs/experiments/exp_full_f1_ddp.yaml`。正式指标见 `outputs/tables/full_all_strategies_deployment_kpi.csv`（主实验部署阶段 F1 已达标）；配图见 `reports/figures/fig7_*.png`；细节见《算法开发与实验结果》。
