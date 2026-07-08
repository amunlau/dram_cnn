# reports/

本目录存放《基于CNN的内存故障预测算法设计报告》（V2.2）及配套材料。

| 文件/目录 | 说明 |
|-----------|------|
| `基于CNN的内存故障预测算法设计报告.md` | 主报告（九章正文，无附录；正文 ≥30000 字） |
| `算法开发与实验结果.md` | 实验验证与结果分析（Notebook 01–05 输出、验收结论） |
| `algorithm_scripts_index.md` | Shell / Pipeline 入口与复现命令 |
| `figures/` | 报告引用的 Mermaid 图源（图 1–7） |

**阅读顺序**：主报告 → `算法开发与实验结果.md` → `algorithm_scripts_index.md` → `figures/` 对应图源。

**复现**：`export PYTHONPATH=src`；命令与 `scripts/*.sh`、`configs/experiments/exp_full_f1_ddp.yaml` 一致。主 KPI 验收目标为部署阶段评价 **F1≥0.6**（`exp_full_f1_ddp`）；主实验 deployment F1=**0.761**，验收已通过（详见《算法开发与实验结果》）。
