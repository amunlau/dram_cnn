# 常见问题

## 切换 horizon / 索引后 validate 报错

- **现象**：`preprocess_manifest.json 中 window_seconds=300 与当前配置 1800 不一致`（完整 validate / train 前）；或 preprocess 报 **`索引 sidecar label_horizon_seconds=300 与当前配置 1800 不一致`**。  
- **原因**：磁盘上 `data/processed` / 索引 sidecar 仍是 **baseline（5min）** 产物，与 `exp_horizon_30m.yaml` 等配置不匹配。  
- **处理**：`bash scripts/build_horizon_indexes.sh smoke`（或 `full`）→ 更新实验 YAML 的 `index_features_path` → `rm -rf data/processed/* data/splits/*` → **`bash scripts/preprocess.sh configs/experiments/exp_horizon_30m.yaml`**。  
- **配置级预检**（尚未对该 horizon 重跑 preprocess）：**`bash scripts/validate.sh --no-require-splits configs/experiments/exp_horizon_30m.yaml`**（跳过 manifest horizon 比对，仅检查 raw 路径与 processed 形状）。

## `use_weighted_sampler` 与 `positive_oversample_factor>1` 冲突

- **现象**：训练启动报 `use_weighted_sampler=true 与 positive_oversample_factor>1 互斥`。  
- **原因**：过采样扩展了 `IndexMappedDataset` 长度，但 `WeightedRandomSampler` 权重仍按原始 `y_train` 构造，会导致过采样 silently 失效。  
- **处理**：保持 `configs/train.yaml` 默认 **`use_weighted_sampler: false`**（与 `positive_oversample_factor` 联用；当前默认 **2**，baseline 为 **1**）。

## evaluate 报 predictions 与 splits_dir 不一致

- **现象**：`predictions 来自 splits_dir=... 与当前实验 ... 不一致`。  
- **处理**：对当前实验 YAML 重跑 **`bash scripts/infer.sh ...`** 后再 **`bash scripts/evaluate.sh ...`**，或切换为生成该 predictions 的实验配置。

## `X_train.npy` 与 model 的 C/H/W 不一致

重跑 `bash scripts/preprocess.sh <实验 yaml>`，并确认 `configs/model.yaml` 的 `grid_height/grid_width/in_channels` 与 `preprocess_manifest.json` 一致。

## 清空 processed/splits 后 `scripts/validate.sh` 报错

尚未重新预处理时，切分文件不存在属正常。可先 **`bash scripts/validate.sh configs/experiments/<exp>.yaml --no-require-splits`**（仅校验配置与原始文件路径），完成 preprocess 后再去掉 `--no-require-splits` 做完整校验。

## `ModuleNotFoundError: dram2cnn`

在仓库根目录执行，并设置 `PYTHONPATH=src`（脚本已默认设置）。

## 预处理内存 / 时间过长

- 调大 `mcelog_chunk_rows` 不一定减内存；可先设 `data.mcelog_max_rows`、`data.max_index_rows` 做子集试验。
- 全量训练前将二者设为 `null`。
- **耗时构成**：顺序扫描 **整表 mcelog** 写入 buckets → **按 sid 做时间排序** → **按索引行填网格 X**。后两段在 **Linux** 上均可按 `preprocess_workers` / `preprocess_parallel_frac` 多进程加速（`fork` + 写时复制）；`preprocess_workers: null` 时默认 **`preprocess_parallel_frac: 0.6`**（约 **60%** `os.cpu_count()` 逻辑核作为进程预算，见 `dram2cnn.utils.preprocess_parallelism`）；manifest 中 `preprocess_events_finalize_parallel` / `preprocess_grid_fill_parallel` 标明是否启用。**顺序读 CSV** 本身仍为单线程流式读取。Windows / macOS 对上述两段保持单进程。
- 若日志出现 fork 与多线程告警，可在运行预处理前设 **`export OMP_NUM_THREADS=1`**（减少 BLAS 线程与 fork 交互）。

## 推理 meta 列

`meta_test.csv` 至少含 `sid`, `label`；若含 `predict_time` 会一并写入 `predictions.csv`。

## `tabular_*.npy` 缺失或与 `model.tabular_dim` 不一致

- **`spatial_heatmap` + `inventory_as_grid_channels: true`**：预处理**不写** `tabular.npy`，此时 **`model.tabular_dim` 应为 0**，切分目录下**不应**存在 `tabular_*.npy`（若有则删 splits 或重跑 preprocess）。
- **`legacy_rc_et`** 或 **spatial 且 inventory 独立写入 tabular**：需要 **`tabular_*.npy`**，且 **`model.tabular_dim`** 必须等于 **`TABULAR_DIM`（15）**（与 `preprocess_manifest.json` 一致）；`load_experiment_config` 会对错误维度报错。
- **禁止混用**：`tabular_dim: 0` 的配置与含 **`tabular_*.npy`** 的切分目录不能搭配使用（或反之）。

## EDA Notebook 与 `data.yaml` 键名

`01_eda.ipynb` 中 mcelog 分块大小应使用 **`mcelog_chunk_rows`**（与 `configs/data.yaml` 一致，勿用 `mcelog_chunksize`）。Notebook 按该大小**分块流式读取直至文件末尾**，汇总 `error_type` 至 `outputs/tables/eda_mcelog_summary.csv`（全文件统计；`mcelog_chunk_rows` 只影响单次读入行数）。

## Notebook 单元格输出与源码不一致

Jupyter 会持久化单元格输出；若 Notebook 显示与当前源码不符的 traceback，请 **Kernel → Restart** 后自上而下重新执行，或使用菜单 **「清除所有输出」** 后再跑。配置校验使用 **`data.window_seconds`**。

## `03_error_analysis.ipynb` 仅分析已有 `predictions.csv`

若已把 `predictions.csv` 放在 `runs/<run_name>/.latest_infer_session/`（或平铺目录），Notebook **不会**强制 `load_experiment_config`（避免缺 `index_features_path` 时无法打开分析）。若需自动跑 preprocess/train/infer，仍会加载完整配置，此时必须齐备 `data/raw` 与索引表。

## 预处理 `pos_rate=0` 或训练日志里 AUC / PR-AUC 全是 nan

- **含义**：索引行与工单打标规则对不上，或 **train/val 某一档没有正样本**（仅在使用 `split_strategy: time` 且无分层时常见），验证集上 ROC/PR 无法定义。
- **冒烟模式**：默认 `time_stratified` + demo 索引（`min_positives≥5`）应保证 train/val/test 各有正例；若仍 nan，检查是否未重建索引或未重跑 preprocess。
- **全量 / 需可读指标时**：检查 `label_failure_types`、`label_horizon_seconds` 与索引时间；`mcelog_max_rows`/`mcelog_scan_rows` 须一致；重建索引：**`bash scripts/build_index.sh full`**，再重跑 preprocess。
- **全量索引正样本仍偏少**：查看 sidecar 中 `positive_candidates` 与 `ticket_sid_coverage_in_pos`。若后者极低，常见原因是工单 sid 与 mcelog 覆盖不一致，或故障时刻与 CE 事件时间轴错位。可适度增大 `positive_prefault_seconds`（近故障 hard neg）；**正样本**由 `positive_sample_interval_seconds` 在 **`[ft−H, ft]`** 决定（**ft 可晚于 mcelog 观测末尾**，此时特征窗可能为空，preprocess 仍保留 y=1）。
- **Notebook**：`04_run_pipelines.ipynb` 中 **preflight_split_labels** 会在 train 前打印各 split 正例数；全量训练前若 val 正例为 0，应修数据或调整切分策略后再训。

## 时间与 `predict_ts`

- **相对秒**：preprocess / build_index 会将 mcelog、tickets、索引中**全局最早** raw 时间作为 `t=0`；`meta.csv` 的 `predict_ts` 为 **≥ 0 的相对秒**（不再出现 `-6e10` 量级负数）。
- **`predict_time` 列**：展示为 `1970-01-01` 起的 ISO 时间（= 相对秒的可读形式），由 **`bash scripts/build_index.sh smoke`** 写出；sidecar **`*.csv.stats.json`** 含 `time_axis: relative_seconds_since_global_min`。
- **遗留索引**：若仍含「公元 0001」或公历 **2000+** 字符串，preprocess 会自动减 `t₀`（依据 sidecar / 字符串规则，非纯数值阈值）；建议重建索引后清空 `data/processed`、`data/splits` 重跑。
- **`mcelog_max_rows` / `mcelog_scan_rows`**：冒烟模式下二者须一致（默认均为 **800000**）；不一致时 preprocess 会 **报错**。仅扫描前 N 行时 `t₀` 与 sid 边界可能不完整，全量请设为 `null`。
- **工单 sid 无 mcelog**：索引构建会 **跳过** 无 CE 事件边界的 sid（无法定义 `t_min/t_max`）；有 mcelog 但故障晚于 `t_max` 时仍可在 `[ft−H, ft]` 生成正锚点（特征窗或为空）。
- **勿用** `strftime('%Y-%m-%d')` 写四位年缺失的 `1-mm-dd`，会导致解析错位、`y` 全 0。

## 索引里 `predict_time` 不是整 5 分钟（:05/:10）

- **含义**：本仓库索引在**正样本 label 窗口** `[ft−H, ft]` 内按 **`positive_sample_interval_seconds`（默认 60s）** 取锚点；**近故障 / 负样本区**按 **`sample_interval_seconds=300`** 在相对时间轴上取点，**不对齐墙钟整 5 分钟**。
- **是否正常**：是。标签/特征窗仍按相对秒定义：`[predict_ts−W, predict_ts)` 与 `[predict_ts, predict_ts+H]`。  
- **详情**：见 **`docs/data_processing_guide.md` §3.5.1**。若需模拟线上 cron 整点推理，需另行在索引构建阶段做墙钟 snap（当前未实现）。

## Jupyter 内核缓存了已导入的 `dram2cnn` 模块

- 配置校验 **只使用 `data.window_seconds`**。  
- **处理**：对 02/03 Notebook **先运行首格**（会 `del sys.modules` 中已加载的 `dram2cnn`），或 **Kernel → Restart** 后自上而下执行；并对 Notebook **清除输出** 避免过时的 traceback 误导。

## 训练结束找不到 `checkpoints/grid_cnn.pt`

- 验证 PR-AUC 为 nan 时，早停逻辑可能跳过保存；`engine/trainer.py` 使用 **`best_score=-inf`**、非有限指标归一化，并在仍无文件时 **保存最后一轮权重**。若仍失败，查看该次训练日志末尾是否有 rank0 报错。

## `exp_full_best_ddp` 与演示索引

- `exp_full_best_ddp.yaml` 取消 `max_index_rows` / `mcelog_max_rows` 截断，但 **`index_features_path` 仍来自 `data.yaml`**。
- 若仍指向 `data/index_stubs/` 演示表，**`run_preprocess` / `run_train` / `train_ddp.sh` 会 RuntimeError**（`assert_full_best_index_not_stub`），不会静默用 stub 跑全量。
- **全量训练**：`bash scripts/build_index.sh full` 后，在 `data.yaml` 或实验 `overrides.data` 中把 **`index_features_path`** 指到 `data/raw/features_index_*.csv`，再 **清空并重跑 preprocess**。  
- **训练启动时**：rank0 会打一行 **`数据摘要`**（`log_dataset_health_for_train`），含 `index_features_path`、`preprocess_manifest` 的 `n_samples`、train/val 行数与正例数；若路径含 `index_stubs` / `features_5m_index_demo` 会 **WARNING**（约 500 行），避免误以为已用天池全量表。

## 预处理前缺文件 / 长时间读 mcelog 后才失败

- **`run_preprocess`** 与 Notebook 均会先调用 **`check_preprocess_raw_inputs`**：在扫描 mcelog 前检查 `mcelog_path`、`tickets_path`、`index_features_path`（及可选 `inventory_path`）是否为可读文件。  
- **处理**：确认路径存在；演示索引可 **`bash scripts/build_index.sh smoke`**。

## `preprocess_manifest.json` 中 `n_empty_grids` > 0

- **含义**：该样本在 `[predict_ts − window_seconds, predict_ts)` 内无 mcelog 事件，对应 mcelog 通道为全零。
- **默认行为**（`drop_empty_mcelog_windows: true`）：仅丢弃 **y=0** 的空窗行；**正样本** 空窗仍保留（manifest：`n_empty_grids_kept_positive`）。
- **排查**：检查索引 sid 是否在 mcelog 中有覆盖、时间轴是否对齐；重建索引后清空 `data/processed`、`data/splits` 重跑 preprocess。

## 推理报错「无法解析阈值文件 best_threshold.json」

- **含义**：训练目录下 `best_threshold.json` 存在但 JSON 损坏或非数值；推理**不再**静默回退 0.5。  
- **处理**：删除损坏文件（将使用默认 0.5），或从 `best_val_metrics.json` / 验证日志恢复合理阈值后重写 JSON。

## `bash scripts/autotune.sh` 与多卡 DDP

- autotune 通过 **`scripts/train.sh` / `scripts/infer.sh`** 启动训练与推理，继承 `base.yaml` 的 gpu/ddp 默认。默认 **`exp_baseline`** 与 **`exp_horizon_*`** 在实验 overrides 中关闭了 train/infer 的 DDP 与 auto_batch，以保持冒烟速度。  
- 若将 `EXPS` 改为启用 DDP 的全量实验，autotune 会自动走 torchrun（≥2 卡时）。  
- **生产多卡**：也可直接使用 **`bash scripts/train_ddp.sh`** + **`bash scripts/infer.sh`**（全量实验 YAML）。

## 多卡 DDP 启动失败（NCCL / nproc 不匹配）

- **`train.sh` / `infer.sh` 报错 nproc > cuda_count**：将 **`base.ddp.nproc_per_node`**（或实验 overrides）设为不超过可见 GPU 数，或调整 **`CUDA_VISIBLE_DEVICES`**。
- **`device=cuda` 但无 GPU**：将 `base.device` 改为 `auto` 或 `cpu`，或检查驱动/CUDA 环境。

## 早停看的是哪一列？`best_f1` 与 `f1@0.5` 为何差很多？

- **早停**由 `configs/train.yaml` 的 **`train.early_stop_metric`** 决定（常见为 `avg_precision`），与日志里的 **`best_f1` 列名无关**：后者始终是「验证集上 **阈值搜索** 后的 F1」；**`f1@0.5`** 是固定 0.5 阈值下的 F1。二者不同属正常现象。  
- 每 epoch 日志中会额外打印 **`early_stop_metric` 的当前分数**（与 `epoch_metrics.csv` 中 `early_stop_score` 一致）；早停触发时会写明 **metric 名**、**patience** 与 **best_该指标**。  
- 验证集 **极小**（如 `N_val` 小于 32）时，ROC-AUC / PR-AP 易在首轮即达平台，随后连续 `patience` 轮无提升会早停，**不等于**训练/验证泄漏。

## 多卡冒烟想改成单卡

- 终端设置 **`CUDA_VISIBLE_DEVICES=0`** 后再执行 `bash scripts/train.sh ...`，当可见 GPU 数为 1 时脚本会走单进程而非 DDP。
