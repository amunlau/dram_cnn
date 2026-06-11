# data/

- **`raw/`**：`mcelog.csv`、`trouble_tickets.csv`、`inventory.csv`；全量索引表由 `bash scripts/build_index.sh full` 写入此目录。
- **`index_stubs/`**：冒烟用小样本索引。
  - **默认配置使用**：`features_5m_index_demo.csv`（5 分钟，`bash scripts/build_index.sh smoke`）
  - **Horizon 对比 stub**：`features_index_h1800s_step600s_smoke.csv`、`features_index_h3600s_step1800s_smoke.csv`（由 `build_horizon_indexes.sh smoke` 生成）
  - **5 分钟别名**：`features_index_h300s_step300s_smoke.csv` 与 `features_5m_index_demo.csv` **内容相同**（horizon 脚本会复制 demo 表，避免重复构建）
  重建：
  ```bash
  export PYTHONPATH=src
  bash scripts/build_index.sh smoke
  bash scripts/build_horizon_indexes.sh smoke   # 批量 5min/30min/1h 冒烟索引
  ```
- **`processed/`**：`X.npy`、`y.npy`、`meta.csv`、`preprocess_manifest.json`；**`tabular.npy` 按配置可有可无**。
- **`splits/`**：train/val/test 切分及 `normalization_stats.json`，由 `scripts/preprocess.sh` 写入。

## 索引表与设计空间

| 命令 | 说明 |
|------|------|
| `bash scripts/build_index.sh smoke` | 推荐入口：写入 `features_5m_index_demo.csv`，~500 行 |
| `PYTHONPATH=src python3 scripts/build_index.py --mode smoke` | 同上；自定义 horizon 时用 `--out` 指定路径 |
| `bash scripts/build_index.sh full` 或 `build_index.py --mode full` | 全量扫描，输出至 `data/raw/features_index_h{H}s_step{S}s_full.csv` |
| `--label-horizon-seconds 1800` | 预测「未来 30 分钟是否故障」 |
| `--sample-interval-seconds 600` | 每 10 分钟一个预测锚点 |
| `--window-seconds 1800` | 特征窗长度（须与实验 `window_seconds` 一致；CLI 未指定时默认等于 horizon） |

生成后在 `configs/data.yaml` 或实验 yaml 的 `overrides.data.index_features_path` 中指向该 CSV 并 **重跑 preprocess**（`exp_full_best_ddp.yaml` 已默认指向 `data/raw/features_index_h300s_step300s_full.csv`）。

字段与形状见 `docs/data_dictionary.md`、`docs/data_processing_guide.md`。
