# 常见问题

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

- **`spatial_heatmap` + `inventory_as_grid_channels: true`**：预处理**不写** `tabular.npy`，此时 **`model.tabular_dim` 应为 0**，切分目录下**不应**残留旧的 `tabular_*.npy`（若有则删 splits 或重跑 preprocess）。
- **`legacy_rc_et`** 或 **spatial 且 inventory 独立写入 tabular**：需要 **`tabular_*.npy`**，且 **`model.tabular_dim`** 必须等于 **`TABULAR_DIM`（15）**（与 `preprocess_manifest.json` 一致）；`load_experiment_config` 会对错误维度报错。
- **禁止混用**：不要用 **`tabular_dim: 0`** 的训练配置去加载仍含 **`tabular_*.npy`** 的旧切分（或反之）。

## EDA Notebook 与 `data.yaml` 键名

`01_eda.ipynb` 中 mcelog 分块大小应使用 **`mcelog_chunk_rows`**（与 `configs/data.yaml` 一致，勿用 `mcelog_chunksize`）。Notebook 按该大小**分块流式读取直至文件末尾**，汇总 `error_type` 至 `outputs/tables/eda_mcelog_summary.csv`（全文件统计；`mcelog_chunk_rows` 只影响单次读入行数）。

## Notebook 已保存的旧报错 / 误导性输出

Jupyter 会持久化单元格输出；若曾用旧版代码或本地改过 `validate_config`，可能仍显示 **`window_days`** 等已不存在的报错。**建议**：对有关 Notebook 执行「清除输出」或运行仓库内已批量清理过的版本；当前校验只使用 **`data.window_seconds`**。

## `03_error_analysis.ipynb` 仅分析已有 `predictions.csv`

若已把 `predictions.csv` 放在 `runs/<run_name>/.latest_infer_session/`（或平铺目录），Notebook **不再强制** `load_experiment_config`（避免缺 `index_features_path` 时无法打开分析）。若需自动跑 preprocess/train/infer，仍会加载完整配置，此时必须齐备 `data/raw` 与索引表。

## 预处理 `pos_rate=0` 或训练日志里 AUC / PR-AUC 全是 nan

- **含义**：索引行与工单打标规则对不上，或 **train/val 某一档没有正样本**，验证集上 ROC/PR 无法定义。  
- **处理**：检查 `data.yaml` 中 `label_failure_types`、`label_backtrace_seconds` 与索引时间是否落在工单窗口内；冒烟数据可在项目根执行 **`PYTHONPATH=src python3 scripts/build_demo_index.py`** 重建 `data/index_stubs/features_5m_index_demo.csv`，再 **重跑 preprocess**（并删除旧 `data/processed`、`data/splits` 或清掉流水线 marker）。  
- **Notebook**：`04_run_pipelines.ipynb` 中 **preflight_split_labels** 会在 train 前打印各 split 正例数；若 val 为 0，应先修数据再打训练。

## 索引 CSV 里 `predict_time` 与「公元 0001 年」等低年份

- 用 **`strftime('%Y-%m-%d')`** 在部分 Python/pandas 版本下会得到 **`1-07-16`** 而非四位年，读回 CSV 后 **`predict_time` 解析错位**，打标全错。  
- **建议**：写入 CSV 时使用 **ISO 8601**（如 `0001-07-16T23:11:07` 或空格分隔的等价形式）；演示索引由 `scripts/build_demo_index.py` 按 ISO 写出。加载侧见 `tianchi_build.load_index_table`（`format="mixed"` 解析）。

## Jupyter 仍报 `window_days` 或陈旧 `validate_config`

- 当前仓库 **只使用 `data.window_seconds`**，源码中已无 `window_days`。  
- **处理**：对 02/03 Notebook **先运行首格**（会 `del sys.modules` 中已加载的 `dram2cnn`），或 **Kernel → Restart** 后自上而下执行；并对 Notebook **清除输出** 避免旧 traceback 误导。

## 训练结束找不到 `checkpoints/grid_cnn.pt`

- 历史问题：验证 PR-AUC 恒为 nan 时早停逻辑可能从未 `torch.save`。  
- **当前**：`engine/trainer.py` 已用 **`best_score=-inf`**、非有限指标归一化，并在仍无文件时 **保存最后一轮权重**；若仍失败，看该次训练日志末尾是否有 rank0 报错。

## `exp_full_best_ddp` 与演示索引

- `exp_full_best_ddp.yaml` 仅关闭 `max_index_rows` / `mcelog_max_rows`；**`index_features_path` 仍来自 `data.yaml`**。若默认指向演示表，全量流水线仍只会用演示行数。  
- **全量训练**：在 `data.yaml`（或实验 `overrides.data`）中把 **`index_features_path`** 指到真实全量索引 CSV，再 **重跑 preprocess**（否则 `data/processed` 与 `data/splits` 仍是旧索引规模）。  
- **训练启动时**：rank0 会打一行 **`数据摘要`**（`log_dataset_health_for_train`），含 `index_features_path`、`preprocess_manifest` 的 `n_samples`、train/val 行数与正例数；若路径含 `index_stubs` / `features_5m_index_demo` 会 **WARNING**，避免误以为已用天池全量表。

## 早停看的是哪一列？`best_f1` 与 `f1@0.5` 为何差很多？

- **早停**由 `configs/train.yaml` 的 **`train.early_stop_metric`** 决定（常见为 `avg_precision`），与日志里的 **`best_f1` 列名无关**：后者始终是「验证集上 **阈值搜索** 后的 F1」；**`f1@0.5`** 是固定 0.5 阈值下的 F1。二者不同属正常现象。  
- 每 epoch 日志中会额外打印 **`early_stop_metric` 的当前分数**（与 `epoch_metrics.csv` 中 `early_stop_score` 一致）；早停触发时会写明 **metric 名**、**patience** 与 **best_该指标**。  
- 验证集 **极小**（如 `N_val` 小于 32）时，ROC-AUC / PR-AP 易在首轮即达平台，随后连续 `patience` 轮无提升会早停，**不等于**训练/验证泄漏。

## 多卡冒烟想改成单卡

- 终端设置 **`CUDA_VISIBLE_DEVICES=0`** 后再执行 `bash scripts/train.sh ...`，当可见 GPU 数为 1 时脚本会走单进程而非 DDP。
