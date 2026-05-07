# data/

- **`raw/`**：`mcelog.csv`、`trouble_tickets.csv`、`inventory.csv`（可选，见 `inventory_path`）；全量竞赛索引表通常也放此处并由 `index_features_path` 指向。
- **`index_stubs/`**：**小样本索引**（如 `features_5m_index_demo.csv`），`configs/data.yaml` 默认指向它。可按当前工单/mcelog 重建：`PYTHONPATH=src python3 scripts/build_demo_index.py`（保证演示流程含正负标签）。
- **`processed/`**：`X.npy`、`y.npy`、`meta.csv`、`preprocess_manifest.json`；**`tabular.npy` 按配置可有可无**（spatial 且 inventory 拼入网格通道时常不写）。
- **`splits/`**：按时间的 train/val/test 切分及 `normalization_stats.json`，由 `scripts/preprocess.sh` 写入。

字段与形状说明见 `docs/data_dictionary.md` 与 `docs/tianchi_pipeline.md`。
