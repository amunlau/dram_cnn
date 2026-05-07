# 天池 DRAM 网格 CNN 流水线

> **新手向完整叙述**（从原始表到 `X`/`y`、切分与标准化）：见 **[`data_processing_guide.md`](./data_processing_guide.md)**。本文保持精简规则与配置速查。

## 样本与标签

- **时间戳**：`predict_time`、`failed_time`、`error_time` 经 pandas 解析后统一转为 **Unix 秒**（与 pandas `datetime64[us]`/`[ns]` 无关，避免误用 `// 10**9` 导致时间错位）。
- **索引表**（`data.index_features_path`）：每行一条样本，至少含 `sid`, `predict_time`（与天池 `features_5m_*.csv` 一致）。可用全量表或子集（`max_index_rows`）。  
  - 仓库默认可指向 **`data/index_stubs/features_5m_index_demo.csv`**（小样本）；按当前 `data/raw` 重建、并保证含正负样本以便冒烟训练有意义：在仓库根执行 **`PYTHONPATH=src python3 scripts/build_demo_index.py`**。  
  - **`predict_time` 写入 CSV**：低年份（如竞赛数据中的 `0001-…`）请用 **ISO 8601 字符串**（四位年），避免 `strftime` 写成 `1-mm-dd` 导致解析错位、**`y` 全 0**。
- **特征窗口**：`[predict_time - window_seconds, predict_time)`。默认配置对齐天池 132973 报告：**6 小时**（`window_seconds=21600`）；亦可改为 300（5 分钟）用于与旧实验对齐。
- **网格表示**（`data.grid_representation`）：
  - **`spatial_heatmap`（默认）**：按 `row_bin_size`×`col_bin_size`（默认 **1024×32**）将逻辑 row/col 映射到 **H×W=128×32**（对应物理地址范围 131072×1024）；**bankid（0–15）** 作为 **16 个通道**在 bin 上累积计数，再对各通道 `log1p`。默认仅统计 **Read/Scrub CE**（`mcelog_error_type_filter: [1,2]`）。若 `inventory_as_grid_channels: true`，将 inventory **one-hot**（15 维）按通道复制为 **15×H×W**，与 16 个 bank 通道拼接，得到 **`in_channels=31`**，此时 **不写** `tabular.npy`，`model.tabular_dim=0`。若 `inventory_as_grid_channels: false`，则 **`in_channels=16`**，inventory 仍写入 `tabular.npy` 并在池化后与网格融合（`tabular_dim=15`）。
  - **`legacy_rc_et`**（调试）：对每个窗口内事件，将 **原始 row/col** 当作网格下标，`error_type==1/2/0` 分别计入 3 个通道（见旧版说明）；坐标裁剪由 `grid_max_row/col` 与 `model.grid_height/width` 限制。
- **inventory**：编码定义不变（`inventory_encode`）。在 spatial 且 `inventory_as_grid_channels` 时特征已进入 **X**；否则写入 `tabular.npy` 并在切分后按训练集标准化，再由 `GridCNN` 融合。
- **变换**：网格逐 cell `log1p` 后堆成 `[C,H,W]`，再经切分阶段按训练集 **逐 (C,H,W) 位置** 标准化（见 `splitter.normalize_split_features_by_train`）。
- **正样本规则**（对齐天池工单）：对配置的 `label_failure_types`，若存在同一 `sid` 的工单满足  
  `predict_ts ≤ failed_ts ≤ predict_ts + label_backtrace_seconds`，则 `y=1`，否则 `0`。

## 配置要点（`configs/data.yaml`）

| 字段 | 含义 |
|------|------|
| `grid_representation` | `spatial_heatmap`（默认）或 `legacy_rc_et` |
| `row_bin_size` / `col_bin_size` | spatial 分箱（默认 1024 / 32） |
| `num_bank_channels` | spatial 通道数（默认 16，对应 bankid） |
| `inventory_as_grid_channels` | spatial 下是否把 inventory 拼入 **X** 的通道维 |
| `mcelog_error_type_filter` | spatial 下计入 CE 的 `error_type` 列表（默认 `[1,2]`） |
| `index_features_path` | 索引 CSV |
| `window_seconds` | 例如 6h=21600 或 5min=300 |
| `label_backtrace_seconds` | 回溯窗口（秒）；报告示例 7 天=604800 |
| `label_failure_types` | 工单类型列表，如 `[1,2,3]` |
| `grid_max_row/col` | **legacy** 模式下 row/col 裁剪上限；spatial 下可忽略 |
| `inventory_path` | `inventory.csv`；`null` 时 one-hot 全 0 |
| `max_index_rows` / `mcelog_max_rows` | 调试用子样本 |
| `preprocess_workers` | **mcelog 按 sid 排序**与**填网格 X** 共用：`null` = `max(1, floor(os.cpu_count() × preprocess_parallel_frac))`；`1` = 全关闭并行 |
| `preprocess_parallel_frac` | 与上一项联用；**默认 0.6**（约使用 **60%** 逻辑 CPU 推算进程数，为系统留余量）；基线/全量可在实验 `overrides.data` 覆盖 |

**预处理耗时**：全量时主要时间在 **顺序读取 entire `mcelog.csv`** 写入按 sid 的 bucket，以及 **对每个 sid 的事件按时间排序合并**、再 **逐索引行填网格**。在 **Linux** 上，后两步均可 **多进程**（同一 `preprocess_workers` / `preprocess_parallel_frac`，`fork` + 写时复制）：① bucket 较多时对「按 sid 排序」并行（见 manifest `preprocess_events_finalize_parallel`）；② 索引行数较多时对「填 X」并行（`preprocess_grid_fill_parallel`）。非 Linux，或 sid 数 / 样本数低于阈值时自动退回单进程。详见 `preprocess_manifest.json`。

**注意**：本工程 **不使用** `window_days`；时间窗仅以 **`window_seconds`**（秒）配置，与 `validate_config` 一致。

## 模型（`configs/model.yaml`）

- `arch: grid_cnn`
- `in_channels`、`grid_height` / `grid_width` 必须与 preprocess 写出的 `X.npy` 及 `preprocess_manifest.json` 中 `grid_shape` 一致（spatial 默认 **31×128×32**）。
- `tabular_dim`：spatial 且 inventory 已拼入网格时为 **0**；legacy 或与 spatial+独立 tabular 并存时为 **15**。
