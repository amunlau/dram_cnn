# outputs/

本目录用于存放 **笔记本分析脚本生成的图表与导出表**，以及可选的手工报告。训练、推理、评估的**主要输出**仍在 `runs/<run_name>/`（权重、`predictions.csv`、`metrics.json` 等）。

## 子目录说明

| 路径 | 用途 |
|------|------|
| `figures/` | 笔记本保存的图片（分布图、对比图等） |
| `tables/` | 笔记本导出的 CSV / 汇总表 |
| `reports/` | 可选的 Markdown / HTML 简报 |
| `notebooks/` | 按笔记本名称分子目录的中间结果（若脚本中使用；**非** 源码目录 `notebooks/`） |

运行笔记本**01–03** 后写入上述目录。**04** 写入 `tables/full_f1_time_vs_sid_deployment_kpi.csv`；**05** 写入 `tables/full_recall_vs_precision_deployment_kpi.csv`；二者合并为 `tables/full_all_strategies_deployment_kpi.csv`（`python3 -m dram2cnn.utils.deployment_kpi` 或笔记本04/05 汇总单元）。**2026-07-10 全量重新执行后**，`full_all_strategies_deployment_kpi.csv` 已含六组 部署阶段指标（）；报告配图另见 `figures/fig7_*.png` 与 `reports/figures/`。端到端主要输出仍在 `runs/`。

## 与版本库同步

默认 `.gitignore` 会忽略 `outputs/` 下生成的大文件，仅保留 `.gitkeep` 与本说明。需要长期保留某次分析结果时，可复制到 `docs/` 或独立归档目录。

批量清理：`bash scripts/clean.sh`（`--notebooks` 清除笔记本单元格输出；`--purge-indexes` 删除可再生全量索引）。
