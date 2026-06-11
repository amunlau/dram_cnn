# outputs/

本目录用于存放 **Notebook 分析脚本生成的图表与导出表**，以及可选的手工报告。训练、推理、评估的**主产物**仍在 `runs/<run_name>/`（权重、`predictions.csv`、`metrics.json` 等）。

## 子目录约定

| 路径 | 用途 |
|------|------|
| `figures/` | Notebook 保存的图片（分布图、对比图等） |
| `tables/` | Notebook 导出的 CSV / 汇总表 |
| `reports/` | 可选的 Markdown / HTML 简报 |
| `notebooks/` | 按 Notebook 名称分子目录的中间产物（若脚本中使用） |

运行 `notebooks/01_eda.ipynb`、`02_feature_analysis.ipynb`、`03_error_analysis.ipynb` 后，会向上述目录写入文件。端到端流水线请用 `notebooks/04_run_pipelines.ipynb`，其主产物在 `runs/` 而非本目录。

## 与版本库同步

默认 `.gitignore` 会忽略 `outputs/` 下生成的大文件，仅保留 `.gitkeep` 与本说明。需要长期保留某次分析结果时，可复制到 `docs/` 或独立制品存储。

一键清理：`bash scripts/clean.sh`（`--notebooks` 清除 Notebook 单元格输出；`--purge-indexes` 删除可再生全量索引）。
