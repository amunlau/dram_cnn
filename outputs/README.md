# outputs/

本目录用于存放 **Notebook 分析脚本生成的图表与导出表**，以及可选的手工报告。训练、推理、评估的**主产物**仍在 `runs/<run_name>/`（权重、`predictions.csv`、`metrics.json` 等）。

## 子目录约定

| 路径 | 用途 |
|------|------|
| `figures/` | Notebook 保存的图片（分布图、对比图等） |
| `tables/` | Notebook 导出的 CSV / 汇总表 |
| `reports/` | 可选的 Markdown / HTML 简报 |
| `notebooks/` | 按 Notebook 名称分子目录的中间产物（若脚本中使用；**非** 源码目录 `notebooks/`） |

运行 Notebook **01–03** 后写入上述目录。**04** 写入 `tables/full_f1_time_vs_sid_deployment_kpi.csv`；**05** 写入 `tables/full_recall_vs_precision_deployment_kpi.csv`；二者合并为 `tables/full_all_strategies_deployment_kpi.csv`（`python3 -m dram2cnn.utils.deployment_kpi` 或 Notebook 04/05 汇总单元）。端到端主产物仍在 `runs/`。

## 与版本库同步

默认 `.gitignore` 会忽略 `outputs/` 下生成的大文件，仅保留 `.gitkeep` 与本说明。需要长期保留某次分析结果时，可复制到 `docs/` 或独立制品存储。

一键清理：`bash scripts/clean.sh`（`--notebooks` 清除 Notebook 单元格输出；`--purge-indexes` 删除可再生全量索引）。
