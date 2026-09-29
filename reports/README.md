# reports/

本目录存放《基于CNN的内存故障预测算法设计报告》及配套材料。

| 文件/目录 | 说明 |
|-----------|------|
| `GridCNN-Deep_FP32后处理剪枝与量化专题报告.md` | **FP32 后处理专题**（融合无偏置、BN-γ/非结构化剪枝、对称 INT8 原理与对比，万字级） |
| `GridCNN-Deep_20260827_BNγ通道剪枝交付摘要.md` | **现行交付**摘要（BN-γ 0.40 + 非结构化 30%；扫参对比） |
| `GridCNN-Deep_20260827交付与INT8结果摘要.md` | 上一版仅非结构化交付摘要（τ=0.52；INT8 F1≈0.757） |
| `基于CNN的内存故障预测算法设计报告.md` | 主报告 V3.0（九章清晰结构；算法设计与验收） |
| `算法开发与实验结果.md` | 实验设计、评价方法与 **2026-07-10 实测结果**（含配图） |
| `GridCNN-Deep_W8A8量化报告.md` | W8A8 PTQ 专题（含 packed INT8；固定部署 τ 口径） |
| `GridCNN-Deep_Base-new_官方对称INT8_PTQ_QAT对比_20260918/` | **Base-new** 官方对称 INT8 FP32/PTQ/QAT（**v2：BN 折叠后 FakeQuant**；QDQ 无 BN；val 重搜 τ） |
| `GridCNN剪枝对比实验报告.md` | 三种粒度剪枝对比实测表（主指标=val 重搜 τ） |
| `GridCNN剪枝对比说明.md` | 三种方法思路、评估口径、注意事项与复现 |
| `GridCNN剪枝对比实验_summary.json` | 上述实验的机器可读汇总 |
| `GridCNN先剪枝后量化对比报告.md` | 先剪枝后 W8A8 完整对比大表（valτ 主指标 + 固定τ对照） |
| `GridCNN先剪枝后量化说明.md` | 串联流水线、数据口径、P=1.0 说明与复现 |
| `GridCNN先剪枝后量化对比_summary.json` | 机器可读汇总（含 `data_audit`） |
| `GridCNN先剪枝后量化对比表.csv` | 表格 CSV（便于 Excel） |
| `algorithm_scripts_index.md` | Shell / Pipeline 入口与复现命令 |
| `figures/` | Mermaid 图源（`.mmd`）与实验配图（`fig7_*.png`） |

**阅读顺序**：主报告 → `算法开发与实验结果.md` → `algorithm_scripts_index.md` → 量化 / 剪枝 / 先剪枝后量化专题 → `figures/`。

**交付总目录**：[`delivery/`](../delivery/README.md)。**现行子包**：`delivery/bnch040-u30-symint8-20260827/`（2026-08-27；BN-γ=0.40 + 非结构化 30% + 整网 INT8；P≈0.924 / R≈0.639 / τ=0.50）。上一版仅非结构化 `delivery/nobias-symint8-apstop-20260827/` 与历史 `delivery/nobias-symint8-20260818/` 请勿与现行数字混表。


**剪枝与量化评估口径（摘要）**：

- 数据：实验 YAML 指定的 **time_stratified 全量** `splits`（主实验 `data/splits`，约 24958/5348/5348）。
- 剪枝与「先剪枝后量化」**主表**：各模型在 val 上重搜 F1 最优阈值 → 再评 test。
- 单独 W8A8 专题报告历史数字多为**固定基线部署 τ**（与训练 `best_threshold.json` 一致）；与剪枝串联表的 valτ 主指标勿混读。
- 固定高 τ 下量化模型可能出现 Precision=1.0（FP=0），见先剪枝后量化说明。

**复现**：`export PYTHONPATH=src`；命令见 `algorithm_scripts_index.md` 与 `configs/experiments/exp_full_f1_ddp.yaml`。正式主实验指标见 `outputs/tables/full_all_strategies_deployment_kpi.csv`；细节见《算法开发与实验结果》。
