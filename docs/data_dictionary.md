# 数据字典

数据处理思路与流程的展开说明见 **`docs/data_processing_guide.md`**。

## 原始数据（`data/raw/`）

- `mcelog.csv`：至少含 `sid`, `row`, `col`, `error_type`, `error_time`。**空间热图模式**建议含 **`bankid`**（0–15；缺列则预处理视为 0）。可选列因数据源而异（如 `memoryid`, `rankid` 等）。
- `trouble_tickets.csv`：`sid`, `failure_type`, `failed_time`
- `inventory.csv`：`sid`, `server_manufacturer`, `DRAM_model`, `DIMM_number` → 固定类别 **15 维 one-hot**（见 `dram2cnn.data.inventory_encode`）
- 索引表（命名可变）：至少 **`sid`, `predict_time`**；时间列建议 **ISO 8601**（见 `docs/troubleshooting.md`「低年份」）
- 仓库默认 **`data/index_stubs/features_5m_index_demo.csv`**（`scripts/build_demo_index.py` 可重建）；全量训练请换真实索引路径

## 预处理产物（`data/processed/`）

- **`X.npy`**：`float32`，`[N, C, H, W]`  
  - **`spatial_heatmap`（默认）**：典型 **`C=31`**, **`H=128`**, **`W=32`**（16 bank 通道 + 15 维 inventory 平铺通道，当 `inventory_as_grid_channels: true`）；或 **`C=16`** + 独立 tabular（`inventory_as_grid_channels: false`）  
  - **`legacy_rc_et`**：典型 **`C=3`**（read/scrub/type0），**H/W** 由 `grid_max_*` 与 `model.grid_*` 限制
- **`tabular.npy`**：`float32`，`[N, F]`，**仅当**配置写出时存在（spatial 且 **`inventory_as_grid_channels: true`** 时**不写**，特征已在 **`X`**；legacy 或 spatial+独立 tabular 时常为 **`F=15`**；无 inventory 路径时为全 0）
- **`y.npy`**：`int64`，`[N]`
- **`meta.csv`**：`sid`, `predict_time`, `label`, `predict_ts`
- **`preprocess_manifest.json`**：`grid_shape`、`schema`、`grid_representation`、`tabular_dim`、`window_seconds`、`row_bin_size` / `col_bin_size`（spatial）、`inventory_as_grid_channels`、并行开关等

## 切分（`data/splits/`）

- **`X_{train,val,test}.npy`**、**`y_*.npy`**：与 **`meta_*.csv`** 行序一致
- **`tabular_{train,val,test}.npy`**：若 **`processed/`** 曾写出 **`tabular.npy`** 则存在；否则无此文件（与 **`model.tabular_dim=0`** 配套）
- **`normalization_stats.json`**：按 **`X_train`** 的 per-(C,H,W) 标准化；若存在 tabular split，另含按 **`tabular_train`** 逐特征标准化说明
