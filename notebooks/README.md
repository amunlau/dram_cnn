# Notebooks

| 文件 | 用途 |
|------|------|
| `01_eda.ipynb` | 原始三表（`mcelog`、`tickets`、索引表）体量与列概览 |
| `02_feature_analysis.ipynb` | 读取 `data/processed` 或 `data/splits` 的 `X`（`[N,C,H,W]`），展平后做按标签的粗统计与示例直方图 |
| `03_error_analysis.ipynb` | 基于 `predictions.csv` 的混淆统计与 FP/FN 清单导出 |
| `04_run_pipelines.ipynb` | 以 marker 断点续跑 `scripts/*.sh` 的端到端流程 |

各 Notebook 的 `PROJECT_ROOT` 同时支持在仓库根目录或本目录下打开。流水线与配置说明以根目录 **`README.md`** 与 **`docs/tianchi_pipeline.md`** 为准（默认 **`grid_representation: spatial_heatmap`**；`exp_baseline` 等为 **`legacy_rc_et`** 冒烟配置）。

**演示数据**：默认 `index_features_path` 指向 `data/index_stubs/features_5m_index_demo.csv`。若需按当前工单/mcelog 重建（保证含正负样本），在项目根执行：`python scripts/build_demo_index.py`。导入 `dram2cnn` 的 Notebook（02/03）若遇陈旧校验报错，先运行首格的 `sys.path` 与 `sys.modules` 清理，或 **Restart kernel** 后自上而下执行。

**预处理耗时**：全量 `preprocess` 仍以**顺序读盘**为主；Linux 上可对「按 sid 排序」与「填网格」两步并行（`configs/data.yaml` 的 `preprocess_workers` / `preprocess_parallel_frac`）。见 `data/processed/preprocess_manifest.json` 中 `preprocess_events_finalize_parallel`、`preprocess_grid_fill_parallel`。
