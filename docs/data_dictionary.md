# 数据字典

数据处理思路与流程的展开说明见 **`docs/data_processing_guide.md`**。

## 原始数据（`data/raw/`）

- `mcelog.csv`：至少含 `sid`, `row`, `col`, `error_type`, `error_time`。**空间热图模式**建议含 **`bankid`**（0–15；缺列则预处理视为 0）。可选列因数据源而异（如 `memoryid`, `rankid` 等）。
- `trouble_tickets.csv`：`sid`, `failure_type`, `failed_time`
- `inventory.csv`：`sid`, `server_manufacturer`, `DRAM_model`, `DIMM_number` → 固定类别 **15 维 one-hot**（见 `dram2cnn.data.inventory_encode`）
- 索引表（命名可变）：至少 **`sid`, `predict_time`**；时间列建议 **ISO 8601**（见 `docs/troubleshooting.md`「低年份」）
- 仓库默认 **`data/index_stubs/features_5m_index_demo.csv`**（`bash scripts/build_index.sh smoke` 可重建）；全量见 `scripts/build_index.sh full`

## 预处理产物（`data/processed/`）

- **`X.npy`**：`float32`，`[N, C, H, W]`  
  - **`spatial_heatmap`（默认）**：典型 **`C=31`**, **`H=128`**, **`W=32`**（16 bank 通道 + 15 维 inventory 平铺通道，当 `inventory_as_grid_channels: true`）；或 **`C=16`** + 独立 tabular（`inventory_as_grid_channels: false`）  
  - **`legacy_rc_et`**：典型 **`C=3`**（read/scrub/type0），**H/W** 由 `grid_max_*` 与 `model.grid_*` 限制
- **`tabular.npy`**：`float32`，`[N, F]`，**仅当**配置写出时存在。默认 **`spatial_heatmap` + `inventory_as_grid_channels: true`** 时 **不产生** `tabular.npy`（inventory 已拼入 **`X`** 通道；缺 inventory 时对应通道为 0）。legacy 或 spatial+独立 tabular 时常为 **`F=15`**。
- **`y.npy`**：`int64`，`[N]`
- **`meta.csv`**：`sid`, `predict_time`（1970 起算展示）, `label`, `predict_ts`（相对秒，≥ 0）
- **`preprocess_manifest.json`**：`grid_shape`、`schema`、`grid_representation`、`tabular_dim`、`index_features_path`、`window_seconds`、`label_horizon_seconds`（与 `label_backtrace_seconds` 同值）、`time_axis`（`relative_seconds_since_global_min`）、`time_epoch_offset_absolute`（全局 `t₀`）、`mcelog_max_rows`（冒烟截断行数，`null` 表示全文件）、`row_bin_size` / `col_bin_size`（spatial）、`inventory_as_grid_channels`、并行开关、`n_empty_grids`（窗口内无 mcelog 事件的样本数，>0 时应检查索引与 raw 对齐）等
- **索引 sidecar**（`*.csv.stats.json`）：与 manifest 对齐的 `time_epoch_offset_absolute`、`mcelog_scan_rows`；preprocess 会校验二者与当前配置一致

## 切分（`data/splits/`）

- **`X_{train,val,test}.npy`**、**`y_*.npy`**：与 **`meta_*.csv`** 行序一致
- **`tabular_{train,val,test}.npy`**：当 **`processed/`** 包含 **`tabular.npy`** 时存在；否则无此文件（与 **`model.tabular_dim=0`** 配套）
- **`normalization_stats.json`**：按 **`X_train`** 的 per-(C,H,W) 标准化；若存在 tabular split，另含按 **`tabular_train`** 逐特征标准化说明

## 模型输入契约（`configs/model.yaml` → GridCNN-Deep）

- **`X_*`** 形状须与 **`in_channels` × `grid_height` × `grid_width`** 一致（spatial 默认 **31×128×32**）
- **`tabular_dim=0`** 时训练/推理仅使用 **`X`**；**`tabular_dim=15`** 时需同时提供 **`tabular_*`**
- 修改 backbone 结构（`backbone_channels`、`blocks_per_stage` 等）**不必重跑 preprocess**，但须 **重新训练**；修改 C/H/W 则须重跑 preprocess
