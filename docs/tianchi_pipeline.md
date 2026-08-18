# 天池 DRAM 网格 CNN 流水线

> **新手向完整叙述**（从原始表到 `X`/`y`、切分与标准化）：见 **[`data_processing_guide.md`](./data_processing_guide.md)**。本文保持精简规则与配置速查。

## 样本与标签

- **时间戳**：raw CSV 经 `time_axis` 平移为 **相对秒**（全局最早 raw 时刻 = 0）；`predict_ts` 与 mcelog 事件轴一致。`preprocess_manifest.json` 含 `time_epoch_offset_absolute`。
- **索引表**（`data.index_features_path`）：每行一条样本，至少含 `sid`, `predict_time`（**1970-01-01 起算 ISO**，读回数值即相对秒）。重建：**`bash scripts/build_index.sh smoke`**。
- **特征窗口**（半开）：`[predict_ts − window_seconds, predict_ts)`。默认 **W=300**；Horizon 实验 **W=H**（1800/3600），负样本步长见 附属统计文件 `sample_interval_seconds`（可与 W 不同）。
- **网格表示**（`data.grid_representation`）：
  - **`spatial_heatmap`（默认）**：按 `row_bin_size`×`col_bin_size`（默认 **1024×32**）将逻辑 row/col 映射到 **H×W=128×32**（对应物理地址范围 131072×1024）；**bankid（0–15）** 作为 **16 个通道**在 bin 上累积计数，再对各通道 `log1p`。默认仅统计 **Read/Scrub CE**（`mcelog_error_type_filter: [1,2]`）。若 `inventory_as_grid_channels: true`，将 inventory **one-hot**（15 维）按通道复制为 **15×H×W**，与 16 个 bank 通道拼接，得到 **`in_channels=31`**，此时 **不写** `tabular.npy`，`model.tabular_dim=0`。若 `inventory_as_grid_channels: false`，则 **`in_channels=16`**，inventory 仍写入 `tabular.npy` 并在池化后与网格融合（`tabular_dim=15`）。
  - **`legacy_rc_et`**（调试）：对每个窗口内事件，将 **原始 row/col** 当作网格下标，`error_type==1/2/0` 分别计入 3 个通道；坐标裁剪由 `grid_max_row/col` 与 `model.grid_height/width` 限制。
- **inventory**：编码定义不变（`inventory_encode`）。在 spatial 且 `inventory_as_grid_channels` 时特征已进入 **X**；否则写入 `tabular.npy` 并在切分后按训练集标准化，再由 **GridCNN-Deep** 融合。
- **变换**：网格逐 cell `log1p` 后堆成 `[C,H,W]`；切分阶段按训练集 **robust（median/IQR，IQR≈0 时改用 MAD/std）** 或 **zscore** 标准化，并可选 **clip 到 ±normalization_clip**（默认 20）。见 `splitter.normalize_split_features_by_train`。
- **正样本规则**（对应天池工单规则）：对配置的 `label_failure_types`，若存在同一 `sid` 的工单满足  
  `predict_ts ≤ failed_ts ≤ predict_ts + label_horizon_seconds`（字段别名 `label_backtrace_seconds`），则 `y=1`，否则 `0`。
- **索引自建时的负样本交叠**：全局负采样跳过会打成 \(y=1\) 的点，并 `exclude` 工单驱动候选的精确锚点；**不**挖空近故障整段时间窗；负样本不足时近故障 \(y=0\) 补入负类。详见 [`data_processing_guide.md`](./data_processing_guide.md) §3.5.2。

## 配置要点（`configs/data.yaml`）

| 字段 | 含义 |
|------|------|
| `grid_representation` | `spatial_heatmap`（默认）或 `legacy_rc_et` |
| `row_bin_size` / `col_bin_size` | spatial 分箱（默认 1024 / 32） |
| `num_bank_channels` | spatial 通道数（默认 16，对应 bankid） |
| `inventory_as_grid_channels` | spatial 下是否把 inventory 拼入 **X** 的通道维 |
| `mcelog_error_type_filter` | spatial 下计入 CE 的 `error_type` 列表（默认 `[1,2]`） |
| `index_features_path` | 索引 CSV |
| `window_seconds` | 特征窗 W（秒）；半开窗 `[predict_ts−W, predict_ts)`；默认 **300** |
| `label_horizon_seconds` | 预测 horizon H（秒）；闭区间标签窗 `[predict_ts, predict_ts+H]` |
| `label_backtrace_seconds` | 与 `label_horizon_seconds` 同义（别名；配置加载时双向同步） |
| `split_strategy` / `time_split` | 切分策略（默认 **`time_stratified`**）；比例默认 **0.7 / 0.15 / 0.15**（`exp_baseline` 覆盖为 0.6 / 0.2 / 0.2）；`min_pos_per_split` 默认 1 |
| `normalization_clip` | 归一化后 clip 到 ±值（默认 20；null 不裁剪） |
| `index_build` / `index_build_presets` | `dram2cnn.pipelines.run_build_index` 采样与平衡参数 |
| `label_failure_types` | 工单类型列表，如 `[1,2,3]` |
| `grid_max_row/col` | **legacy** 模式下 row/col 裁剪上限；spatial 下可忽略 |
| `inventory_path` | `inventory.csv`；`null` 时 one-hot 全 0 |
| `max_index_rows` / `mcelog_max_rows` | 调试用子样本 |
| `preprocess_workers` | **mcelog 按 sid 排序**与**填网格 X** 共用：显式整数固定 CPU 上限；`null` = `max(1, floor(os.cpu_count() × preprocess_parallel_frac))`；`1` = 全关闭并行 |
| `preprocess_parallel_frac` | 与上一项联用；**默认 0.6**（约 **60%** 逻辑 CPU）；可被 **`ram_budget_frac`** 进一步压低 |
| `ram_total_gib` / `ram_budget_frac` | 系统内存（非显存）；`data.yaml` 默认 **50%**；全量实验覆盖 **0.85** + `preprocess_workers: 1` |

**预处理耗时**：全量时主要时间在 **顺序读取 entire `mcelog.csv`** 写入按 sid 的 bucket，以及 **对每个 sid 的事件按时间排序合并**、再 **填网格**。在 **Linux** 上，排序阶段可多进程（预处理清单中的 `preprocess_events_finalize_parallel`）；**填网格**在小样本可并行，全量 compact（空窗丢弃且预分配 >8GiB）时多为**单进程**（`preprocess_grid_fill_parallel=false`）。非 Linux 或样本过少时退回单进程。详见 `preprocess_manifest.json`。

**注意**：本工程 **不使用** `window_days`；时间窗仅以 **`window_seconds`**（秒）配置，与 `validate_config` 一致。

## 模型（`configs/model.yaml` / `configs/model_smoke.yaml`）

- **`exp_baseline`**：`model_smoke.yaml`（轻量 backbone，冒烟默认）
- **`exp_full_f1_*` / `exp_full_recall_*` / horizon / Optuna**：`model.yaml`（GridCNN-Deep 默认）
- `arch: grid_cnn`
- `in_channels`、`grid_height` / `grid_width` 必须与预处理写出的 `X.npy` 及 `preprocess_manifest.json` 中 `grid_shape` 一致（spatial 默认 **31×128×32**）。
- **Backbone**：`stem_channels` + 4×(`blocks_per_stage` 卷积块 + MaxPool)；默认 **blocks_per_stage=2**，纯卷积堆叠，约 **11 个 Conv2d**。
- `backbone_channels`、`conv_kernel_size`、`head_hidden`、`dropout` 等可由 Optuna 搜索（`configs/hpo/hpo.yaml`）。
- `head_bias`：分类头 Linear 是否含 bias（**默认 true**）；无偏置硬件交付实验见 `exp_full_f1_nobias_*.yaml`（`false`）。
- `tabular_dim`：spatial 且 inventory 已拼入网格时为 **0**；legacy 或与 spatial+独立 tabular 并存时为 **15**。
