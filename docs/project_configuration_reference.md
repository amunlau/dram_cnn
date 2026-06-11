# DRAM 网格 CNN 项目：配置体系详解（参考手册）

> **文档目的**：将本仓库中与「可重复实验」相关的全部 YAML 配置，从**设计原因**、**适用情况**、**修改方法**到**与代码/脚本的对应关系**，梳理为一份可独立阅读的说明。阅读对象包括：需要切换天池全量数据的研究人员、需要调训练/不平衡策略的工程师、以及希望理解「为何 baseline 与 full_best 行为不同」的协作者。  
> **配套阅读**：数据处理概念与流程见 [`data_processing_guide.md`](./data_processing_guide.md)；规则速查见 [`tianchi_pipeline.md`](./tianchi_pipeline.md)；字段清单见 [`data_dictionary.md`](./data_dictionary.md)；排错见 [`troubleshooting.md`](./troubleshooting.md)。

---

## 第零章 导读：为何配置被拆成「五段 + 实验入口」

本项目的核心流水线是：**原始表 → 预处理物化张量 → 时间/服务器切分 → 训练 Grid CNN → 推理 → 评估**。其中「原始表路径、时间窗、网格形状、是否把 inventory 拼进通道」等决策，若在单一巨型 YAML 里混杂「运行目录名、随机种子、batch 大小」，会导致三类问题：**难以复现实验**（改了一处忘了另一处）、**难以在「同一数据底稿」上跑多套模型/训练策略**、**难以区分「数据定义」与「运行产物布局」**。

因此仓库采用固定结构：

1. **五段片段配置**（fragment）：`configs/base.yaml`、`configs/data.yaml`、`configs/model.yaml`、`configs/train.yaml`、`configs/infer.yaml`。每段只承载一类决策，便于版本管理与 Code Review。  
2. **实验入口 YAML**（experiment）：`configs/experiments/*.yaml`。它不重复抄写全部字段，而是通过 `base:` / `data:` 等键**指向**上述片段，再用 `overrides:` **仅覆盖**本次实验需要改动的子树。

这种拆法的**直接好处**是：`exp_baseline.yaml` 可以在**不修改** `data.yaml` 默认全表语义的前提下，用 `overrides.data` 强行加上「只读 500 行索引、只扫 800000 行 mcelog」等调试约束（二者须与 `index_build_presets.smoke` 一致）；而 `exp_full_best_ddp.yaml` 则可以把 `max_index_rows` / `mcelog_max_rows` 置为 `null`，表示**取消调试截断**，但仍与其它实验**共用**同一份 `data.yaml` 中的索引路径、窗口秒数、网格表示等——除非你在该实验的 `overrides` 里显式改掉它们。

下文将按「从抽象到具体」的顺序展开：先说明**合并规则与路径解析**（否则后面读每个字段会困惑），再**逐文件、逐字段**解释含义与改法，最后给出**场景化操作清单**与**与源码模块的映射**，便于你自行追溯实现。

---

## 第一章 配置分层与职责边界

### 1.1 五段片段各自回答什么问题

| 片段键 | 文件路径（默认） | 回答的核心问题 |
|--------|------------------|----------------|
| `base` | `configs/base.yaml` | 这次实验在人类可读层面的「名字」是什么？产物写到 `runs/` 下哪棵目录树？是否按时间戳分子目录避免覆盖？设备用 CPU 还是 GPU？ |
| `data` | `configs/data.yaml` | 原始 CSV 在哪里？索引表路径？特征窗口多长？标签回溯多长？网格是 spatial 热图还是 legacy？预处理并行度？训练/验证/测试如何切分？ |
| `model` | `configs/model.yaml` | 网络输入通道数、空间分辨率、是否融合 tabular 向量、Dropout、骨干通道列表等**与 `X_*.npy` 形状强绑定**的超参。 |
| `train` | `configs/train.yaml` | 优化器、batch、epoch、早停指标、类别不平衡（权重、Focal、过采样）、DDP、阈值搜索范围等。 |
| `infer` | `configs/infer.yaml` | 推理 batch、模式（当前仓库要求 `full`）。 |

**职责边界的设计原因**：`data` 与 `model` 必须一致，否则会在训练启动时报错或静默错位。常见错误是：改了 `grid_representation` 或 `inventory_as_grid_channels`，却忘记同步 `in_channels` / `tabular_dim` / `grid_height` / `grid_width`。因此 `validate_experiment`（训练前）会检查切分目录下的 `X_train.npy` 与 `model.yaml` 是否匹配；`validate_config`（加载配置时）会检查 `infer.mode`、`tabular_dim` 与 `TABULAR_DIM` 等硬约束。把 `model` 独立成段，是为了在文档与评审中明确：**模型段是「张量契约」**。

### 1.2 实验 YAML 与片段的关系

实验 YAML（例如 `configs/experiments/exp_full_best_ddp.yaml`）在磁盘上是**普通文本文件**，其角色是：

- **声明组合**：通过五个 `键: 相对路径` 告诉加载器去读哪些片段。  
- **声明差异**：通过 `overrides` 子树声明「相对默认片段，本次要改什么」。

**不是**「再发明一套配置语法」：最终程序里见到的仍是一个 Python `dict`，结构为 `cfg["base"]`、`cfg["data"]`、……，与「手写一个合并后的大 YAML」等价，只是物理上拆开了。

### 1.3 「真全量」与「取消行数上限」为何常被混为一谈

天池场景下，**样本条数由索引 CSV 的行数决定**。配置里有两类不同维度的「限制」：

1. **`index_features_path` 指向哪个文件**：若指向仓库内 `features_5m_index_demo.csv`，则无论 `max_index_rows` 是否为 `null`，**物理行数就只有演示规模**（约 **500 行**）。这是**数据源**层面的上限。  
2. **`max_index_rows` / `mcelog_max_rows`**：在**已选定文件**的前提下，再对「读多少行索引」「顺序扫描 mcelog 时最多累积多少行」做**调试截断**。这是**工程试跑**层面的上限。

`exp_full_best_ddp.yaml` 的 `overrides` 把第二类设为 `null`，语义是「**不要**在试跑阶段再砍索引行、不要提前结束 mcelog 扫描」。**第一类（索引文件路径）** 在仓库默认的 `exp_full_best_ddp.yaml` 中已通过 `overrides.data.index_features_path` 指向 `data/raw/features_index_h300s_step300s_full.csv`；baseline 仍用 `data.yaml` 的演示 stub。若自定义实验 yaml 未覆盖该键，或磁盘上仍是旧 preprocess 产物，则须 **`build_index.sh full` + 清空 `processed/`/`splits/` 后重跑 preprocess**。

这一区分是本项目文档中反复强调的**心智模型**：**改数据源路径 = 新一次数据构建；改 max_* = 同一张源表上的子集实验**。

---

## 第二章 物理路径、命名约定与运行目录

### 2.1 仓库根与「相对路径」的基准

所有流水线脚本设计为在**仓库根目录**执行（例如 `bash scripts/preprocess.sh ...`）。`load_experiment_config` 在合并后会调用 `_normalize_io_paths`：凡键名以 `_path` 或 `_dir` 结尾的字符串，若为相对路径，则**统一解析为「相对仓库根的绝对路径字符串」**。这样无论当前 shell 的 `cwd` 是否在 `notebooks/`，`Path(cfg["data"]["mcelog_path"])` 都稳定。

**配置原因**：数据科学与竞赛仓库中，常见失败模式是「在子目录跑脚本导致相对路径找不到文件」。统一绝对化后，日志里打印的路径也便于复制到文件管理器。

### 2.2 `runs/` 与 `data/` 的分工

| 树 | 内容性质 | 是否应进 Git（通常） |
|----|----------|----------------------|
| `data/raw/`、`data/index_stubs/` | 原始或演示数据 | 大文件常 gitignore，由用户自备 |
| `data/processed/`、`data/splits/` | 预处理与切分产物 | 通常 gitignore，由 preprocess 生成 |
| `runs/<run_name>/` | 训练 checkpoint、日志、推理预测、评估指标 | gitignore |

`base.run_name` 决定 `runs/` 下**哪一棵子树**归属本次「逻辑实验品牌」。`versioned_runs: true` 时，每次训练会再建**时间戳子目录**，并写 `.latest_train_session` 指针，供后续 `infer.sh` 默认找到最近一次权重。

**设计原因**：同一 `run_name` 下多次尝试共存，互不覆盖；Notebook 或 CI 可解析指针取「最新一次」。

---

## 第三章 配置加载的生命周期（与源码一致）

### 3.1 入口函数与调用方

核心函数为 `dram2cnn.utils.config.load_experiment_config(exp_path)`。典型调用方包括：

- `dram2cnn.pipelines.run_preprocess`  
- `dram2cnn.pipelines.run_train`  
- `dram2cnn.pipelines.run_infer`  
- 以及 `validate`、`evaluate` 等配套流水线。

**情况**：任何入口若传入的实验 yaml 路径错误，会在**最早阶段**失败，避免「预处理跑了半小时才发现 train 里 device 写错」。

### 3.2 解析 `base` / `data` 等 include 路径的规则

`_resolve_config_include(ref, exp_parent, root)` 的顺序为：

1. 若 `ref` 已是绝对路径，直接使用。  
2. 否则尝试「实验 yaml 所在目录 / ref」是否存在文件。  
3. 否则尝试「仓库根 / ref」是否存在文件。  

当前实验文件放在 `configs/experiments/`，片段放在 `configs/`，因此 `data: configs/data.yaml` 会由第三步在仓库根下找到。

**方法**：自定义实验时，你可以把片段放在 `configs/experiments/my_data.yaml` 并写 `data: my_data.yaml`，只要与实验 yaml 同目录即可被第二步找到——适合临时 fork 数据段而不污染主 `data.yaml`。

### 3.3 `deep_update` 的合并语义（必须掌握）

`deep_update(base, patch)` 的规则（见源码注释）：

- 若 `patch` 中某键的值是 `dict`，且 `base` 中对应键也是 `dict`，则**递归合并**。  
- 否则 **整键覆盖**（包括 list、标量、`null`）。

**配置原因**：这样可以在 `overrides.data` 里只写三行（如 `max_index_rows`），而**不**把 `data.yaml` 里其余几十行复制进实验文件。反过来也要注意：**列表型配置是整表替换**，例如你不能指望在 overrides 里只「追加」一个 `mcelog_error_type_filter` 元素而不重写整个列表——除非写成与父文件完全相同的列表再改一个元素（通常直接改 `data.yaml` 更清晰）。

### 3.4 `validate_config` 的硬约束（加载期）

在 `_normalize_io_paths` 之后会调用 `validate_config`，主要检查：

- `mcelog_path`、`tickets_path`、`index_features_path` 文件存在。  
- `inventory_path` 若为非空，文件必须存在；若为 `null`/省略，表示无 inventory（实现上 one-hot 全零，视模式而定）。  
- `window_seconds > 0`。  
- `split_strategy` 属于 `time`、`time_stratified` 或 `sid`。
- `normalization_method` 属于 `robust` 或 `zscore`。  
- `infer.mode` 必须为 `full`（与本仓库单头分类实现绑定）。  
- `model.tabular_dim` 若大于 0，必须等于 `TABULAR_DIM`（当前为 **15**，与 `inventory_encode` 中类别集合一致）。
- **`spatial_heatmap`**：`model.in_channels` 须等于 `num_bank_channels + (15 if inventory_as_grid_channels else 0)`；`inventory_as_grid_channels=true` 时 `tabular_dim` 须为 **0**，为 `false` 时 `tabular_dim` 须为 **15**。

**情况**：这些检查是「必要不充分」条件——通过加载校验不代表 `X_train` 与 `model` 的 C/H/W 已一致，那一步在 `validate_experiment(require_splits=True)` 中完成。

---

## 第四章 `configs/base.yaml` 逐项说明

### 4.1 `project_name`

字符串标识，便于日志或未来多项目共用仓库时区分。当前默认 `dram2cnn`，一般无需修改。

### 4.2 `seed`

随机种子基数。分布式训练时常见做法是对每个 rank 加上偏移（见 `run_train`），以降低各进程同一随机序列的概率。

### 4.3 `run_name`

**最关键的人类可读标识之一**，决定 `runs/<run_name>/` 目录名。`exp_baseline` 与 `exp_full_best_ddp` 通过 overrides 把它分别设为 `exp_tianchi_baseline` 与 `exp_tianchi_full_best_ddp`。

**原因**：同一台机器上并行保留 baseline 与 full 产物，互不混淆。

### 4.4 `versioned_runs`

- `true`（默认）：训练写入 `runs/<run_name>/<时间戳>/`；推理写入 `runs/<run_name>/infer_<时间戳>/`；并维护 `.latest_train_session`、`.latest_infer_session` 指针文件。  
- `false`：平铺到 `runs/<run_name>/`，后一次运行可能覆盖同名文件。

**适用情况**：正式多次对比实验强烈建议 `true`；仅本地临时调试且不需要按时间戳归档 run 时可临时 `false`。

### 4.5 `run_id`

当 `versioned_runs: true` 时，若 `run_id` 非 `null`，则训练直接写入 `runs/<run_name>/<run_id>/`，**不会**自动生成时间戳目录。用于可复现论文中「固定目录名」或 CI 中固定路径。

### 4.6 `checkpoint_run_id` / `infer_run_id` / `evaluate_run_id`

三者分别约束：推理时从哪个训练子目录加载权重、本次推理输出固定子目录名、评估时读哪次推理的 `predictions.csv`。均为 `null` 时走「指针或最新」的解析逻辑（见 `run_layout.py`）。

**方法**：写论文附录时，可把三次运行的子目录名抄进配置，保证审稿人复现时目录一致。

### 4.7 `device`

`auto` 表示检测 CUDA，不可用则回退 CPU。`train.sh` / `infer.sh` 启动前会读取 **`base.gpu` / `base.ddp`**（可被 `train` / `infer` 片段或实验 overrides 覆盖），结合可见 GPU 数决定是否 `torchrun` 及自动 batch 探测。

### 4.8 `gpu`（全 pipeline 默认）

定义于 `configs/base.yaml`，`train` / `infer` 可覆盖子键：

- `auto_batch_size`：启动前在单卡上探测 batch，使峰值显存 ≤ `vram_budget_frac`（默认 **true**）。  
- `vram_budget_frac`：默认 **0.8**（另预留 `vram_reserve_mib`，默认 512）。  
- `batch_size_min` / `batch_size_max`：探测上下界；推理可在 `infer.yaml` 提高 `batch_size_max`（仅 forward，通常可更大）。

### 4.9 `ddp`（全 pipeline 默认）

- `enabled`：`auto` 表示 ≥2 张可见 GPU 时由 `scripts/train.sh` / `scripts/infer.sh` 自动 `torchrun`；`false` 强制单进程。  
- `nproc_per_node`：`auto` 表示使用 `CUDA_VISIBLE_DEVICES` 内全部可见 GPU。

实验级可在 `overrides.train.ddp` / `overrides.infer.ddp` 单独关闭（如冒烟 `exp_baseline`、`exp_horizon_*`）。

---

## 第五章 `configs/data.yaml` 逐项说明（核心）

本章篇幅最长，因为 **`data` 段几乎完全定义了「样本是什么」**。

### 5.1 四类原始输入路径

#### 5.1.1 `mcelog_path`

可纠正错误日志，体量通常最大。预处理会顺序读取（分块大小 `mcelog_chunk_rows`），按 `sid` 建桶、排序，再在填网格阶段按索引行的时间窗切片。

**修改原因**：若数据放在 NAS 上，可改为绝对路径。**注意**：改路径后若行格式与列名不一致，需在 `tianchi_build` 或数据准备侧统一，而非仅改 YAML。

#### 5.1.2 `tickets_path`

故障工单表，用于打标。**不参与**网格数值填充，但决定 `y` 的语义。

#### 5.1.3 `inventory_path`

硬件台账。可为空（视实现是否允许 null）；若提供则用于 15 维 one-hot。spatial 模式下若 `inventory_as_grid_channels: true`，inventory 信息进入 `X` 的通道维，**不单独写** `tabular.npy`。

#### 5.1.4 `index_features_path`（再次强调与全量的关系）

每一行 `(sid, predict_time)` 定义一条样本。该路径是**样本数量的上界**（在 `max_index_rows` 为 `null` 时取文件全部行）。

**详细情况**：

- 演示路径 `data/index_stubs/features_5m_index_demo.csv`：约 500 行，适合 CI 与教学。  
- 天池全量：通常是你下载的 `features_5m_*.csv`，应放到你有权限读的路径，并改此键。

**方法**：修改配置后务必清空 `processed`/`splits` 并重跑 preprocess；否则训练会读取与当前配置不一致的 `X_train.npy`。

文件内长注释已说明「实验 yaml 里也可 `overrides.data.index_features_path`」——适用于「baseline 仍演示、full_best 指全量」的分叉需求。

### 5.2 `processed_dir` 与 `splits_dir`

预处理输出与切分输出根目录。默认 `data/processed`、`data/splits`。若磁盘分区紧张，可改到大盘，但需同步更新任何硬编码文档或外部脚本。

### 5.3 `grid_representation`

枚举：`spatial_heatmap`（默认，对齐天池 132973 报告推荐的空间分箱 + bank 通道）与 `legacy_rc_et`（原始 row/col + error_type 三通道，用于小网格调试）。

**配置原因**：两种表示下 `X` 的 `C/H/W` 完全不同。`exp_baseline` 使用 **`configs/model_smoke.yaml`**（与 `model.yaml` 同 C/H/W=31×128×32，但 backbone 更浅、参数量约 1/4）；`exp_full_best_ddp` / `exp_optuna_smoke` 使用完整 **`configs/model.yaml`**。切换 `grid_representation` 时须同步 `model.in_channels` / `grid_height` / `grid_width` 并重跑 preprocess。

**方法**：默认实验应使用 `spatial_heatmap`；切换 `grid_representation` 时**必须**重跑 preprocess 并同步 `model.yaml` 或实验 overrides 中的张量维。

### 5.4 `row_bin_size` / `col_bin_size` / `num_bank_channels`

spatial 模式下将逻辑 row/col 映射到 `grid_height × grid_width`，并把 `bankid` 映射到通道维。默认 `1024×32` 分箱与 `128×32` 网格、16 个 bank 通道对齐（详见 `tianchi_pipeline.md` 中的算术关系）。

**修改情况**：若你更换物理地址编码或赛题定义，可能需调整分箱；调整后 `model.grid_height/width/in_channels` 必须重算，错误组合会在 validate 阶段暴露。

### 5.5 `inventory_as_grid_channels`

- `true`：15 维 one-hot 以「通道复制到 H×W」方式拼入 `X`，此时 **`tabular_dim` 应为 0**（inventory 不作为独立向量分支）。  
- `false`：网格只含 CE 相关通道，inventory 写入 `tabular.npy`，模型侧 `tabular_dim` 必须为 15。

**设计原因**：竞赛报告中的「是否把静态硬件属性并入卷积早期层」是一种归纳偏置选择；仓库允许两种实现路径。

### 5.6 `mcelog_error_type_filter`

spatial 默认 `[1,2]` 表示只累计 Read CE 与 Scrub CE。若设为 `null`，代码路径使用默认 `[1,2]`（见 YAML 注释）。**改大集合**会改变通道语义与事件计数，需重新理解物理含义。

### 5.7 `window_seconds` 与 `label_horizon_seconds`

- **特征窗口** `[predict_ts - window_seconds, predict_ts)`（**半开**）：默认 **300 秒**；Horizon 实验常令 **W=H**（如 1800/3600），与负样本索引步长 **`sample_interval_seconds`** 可不同。
- **标签窗口** `[predict_ts, predict_ts + label_horizon_seconds]`（**闭**）：默认 **300 秒**。字段 **`label_backtrace_seconds`** 与之同义。

`exp_baseline` 使用 **`model_smoke.yaml`**（轻量 GridCNN，形状同 31×128×32）；`exp_optuna_smoke` / `exp_full_best_ddp` 使用完整 **`model.yaml`**。Horizon 实验 `exp_horizon_30m` / `exp_horizon_1h` 默认使用 `data/index_stubs/` 下 horizon 冒烟索引（`build_horizon_indexes.sh smoke`）；全量对比需 `build_horizon_indexes.sh full` 并改 `index_features_path`。

### 5.7b 索引生成（`index_build` / `scripts/build_index.py`）

| 字段 | 含义 |
|------|------|
| `sample_interval_seconds` | 负样本 / 近故障区锚点步长（秒） |
| `positive_sample_interval_seconds` | 正样本 label 窗口 `[ft−H, ft]` 内步长（默认 60；H=300 时约 5 点/故障） |
| `positive_prefault_seconds` | 近故障 hard-neg 区宽度（默认 1h；full preset 24h） |
| `target_positive_ratio` | 目标正样本比例（基值 0.25；**smoke preset 0.08**、**full preset 0.15**） |
| `neg_sample_strategy` | `uniform`（全量，正常时段均匀采样）或 `random`（冒烟） |
| `min_neg_per_ticket_sid` | 每个工单 sid 至少采样的负样本数 |
| `min_positives` / `min_total_rows` | 写入前校验（smoke preset 默认 5 / 50；full 默认 50 / 500） |
| `max_positive_candidates` | 可选正样本行数上限；为 `null` 且设 `max_rows` 时自动取 `max_rows×ratio/(1−ratio)`（smoke 约 43/500） |

**`window_seconds` / `label_horizon_seconds`** 在 **`index_build` 段不重复配置**，由 `data.yaml` 顶层字段（或 CLI `--label-horizon-seconds` / `--window-seconds`）决定，避免与 preprocess 不一致。

预设 **`smoke`** / **`full`** 见 `configs/data.yaml` 中 `index_build_presets`。

### 5.8 `label_failure_types`

列表型过滤：仅这些 `failure_type` 的工单参与 `y=1` 判定。默认 `[1,2,3]`。若赛题变更类型编码，这里必须同步。

### 5.9 `grid_max_row` / `grid_max_col`

仅 **legacy** 模式用于裁剪原始坐标；spatial 模式下由分箱参数决定网格范围，这两项在填网格阶段可忽略，但仍保留在 YAML 中以免 legacy 实验缺键。

### 5.10 `preprocess_workers` 与 `preprocess_parallel_frac`

在 Linux 上，预处理的部分阶段可用多进程（fork + 写时复制）。`preprocess_workers: null` 表示「自动按 CPU 核数 × 比例推算预算」。`preprocess_parallel_frac: 0.6` 表示大约使用 **60% 逻辑核** 作为进程预算上限（112 核 ≈ 67 worker）。

**全量实验 `exp_full_best_ddp`**：继承 **`ram_budget_frac: 0.5`**（系统内存峰值 ≤50% MemTotal），全量 ~66k 行时通常压至 **worker=1**；日志打印 RAM 预算与估算峰值。

**适用情况**：共享集群上为避免占满机器，可降至 `ram_budget_frac: 0.3` 或设 `preprocess_workers: 1`。

**原因**：顺序读 mcelog 仍是单线程流式；并行主要体现在「按 sid 排序合并」「填网格」等 CPU 密集段，详见 `preprocess_manifest.json` 中布尔标记。

### 5.10.1 `ram_total_gib` 与 `ram_budget_frac`（系统内存，非显存）

由 `dram2cnn.utils.preprocess_parallelism` 在已知样本数与网格形状后估算 preprocess 峰值 RSS，取 **`min(CPU 预算 worker, RAM 预算 worker)`**。

- `ram_total_gib`：`null` 时读 `/proc/meminfo` 的 MemTotal。
- `ram_budget_frac`：默认 **`0.5`**（峰值目标 ≤ 总内存的 **50%**；64GB → **32 GiB**）。设为 `null` 可关闭 RAM 限制。
- `preprocess_ignore_ram_budget: true`：跳过 RAM 限制，仅用 CPU 比例（高内存专用机）。

manifest 记录 `ram_budget_gib`、`preprocess_estimated_peak_gib`、`preprocess_ram_budget_exceeded`，以及 split/归一化阶段的 `split_normalize_estimated_peak_gib`、`split_normalize_ram_budget_exceeded`。split 阶段默认 **memmap 读 X + 分块写出 + 按通道归一化**（`normalization_stats.json` 中 `low_memory_channelwise: true`）。

**与显存区别**：训练/推理 batch 探测使用 `base.gpu.vram_budget_frac`（默认 **0.8**，即 GPU 显存 80%），与上述系统 RAM 预算是两套独立机制。

### 5.11 `mcelog_chunk_rows`

`read_csv` 分块行数。过大易 OOM，过小则系统调用与 Python 循环开销上升。默认 **`1000000`**（见 `configs/data.yaml`）。

### 5.12 `max_index_rows` / `mcelog_max_rows`

- `max_index_rows`：只读索引表前 N 行（`null` 表示全表）。  
- `mcelog_max_rows`：顺序扫描 mcelog 时最多读入多少行用于建桶（`null` 表示直到文件结束）。

**与 full_best 的关系**：`exp_full_best_ddp` 将二者设为 `null`，并在 `overrides.data.index_features_path` 指向全量 CSV（默认 `data/raw/features_index_h300s_step300s_full.csv`），表示在**全量索引 + 完整 mcelog** 上不做调试截断。

### 5.13 `split_strategy` 与 `time_split`

- `time_stratified`（**默认**）：分层时间切分，保证各 split 至少 `min_pos_per_split` 个正例。  
- `time`：按 `predict_ts` 排序后按比例切 train/val/test。  
- `sid`：按唯一 `sid` 分层切分（约 **64% / 16% / 20%** train/val/test）。

`time_split` 下三个比例默认 `0.7/0.15/0.15`，和为 1；`min_pos_per_split` 默认 1。**`exp_baseline` 冒烟实验覆盖为 `0.6/0.2/0.2`**（略增大 val/test 以便早停与指标更稳）；`exp_horizon_*` / `exp_full_best_ddp` 继承 data.yaml 默认 0.7/0.15/0.15。`normalization_method` 默认 `robust`；`normalization_clip` 默认 **20**（null 不裁剪）；`drop_empty_mcelog_windows` 默认 true（仅丢弃负样本空窗）。

---

## 第六章 `configs/model.yaml` 逐项说明

### 6.1 `arch`

当前固定为 `grid_cnn`，与工厂类构建逻辑绑定。

### 6.2 `in_channels` / `grid_height` / `grid_width`

必须与 `X_train.npy` 的 `[N,C,H,W]` 一致。spatial + `inventory_as_grid_channels` 时默认 **31 = 16 bank + 15 inventory**。legacy 调试模式可覆盖为 `3×256×256` + `tabular_dim=15`（见 `tests/test_data_pipeline.py`）。

### 6.3 `tabular_dim` 与 `tabular_fuse_dim`

- `tabular_dim`：额外向量分支输入维；为 0 表示模型前向不接收 `tabular` 张量。  
- `tabular_fuse_dim`：融合 MLP 隐藏宽度（仅当 `tabular_dim > 0` 时生效；默认 spatial 配置 `tabular_dim=0`，inventory 已拼入 `X`，此项不参与前向）。

### 6.4 GridCNN-Deep 容量与正则

| 字段 | 含义 |
|------|------|
| `stem_channels` | Stem 首层输出通道（默认 64） |
| `backbone_channels` | 各 stage 目标通道列表，长度 = stage 数（默认 4 段） |
| `blocks_per_stage` | 每个 stage 内卷积块数量（默认 2；越大越深） |
| `use_residual` | stage 内是否使用残差块（默认 true） |
| `conv_kernel_size` | 3 或 5 |
| `use_batch_norm` | 是否使用 BatchNorm |
| `head_hidden` / `dropout` | 分类 MLP 宽度与 Dropout |

默认 spatial 配置约 **19 个 Conv2d、~5.9M 参数**（`C=31`）。**修改 backbone 结构后须重新训练**（checkpoint 须与结构一致）。

调参时应在固定数据与训练策略下单变量改动，便于归因。

### 6.5 Optuna 可搜结构超参

见 `configs/hpo.yaml`：`backbone_depth`（即 `backbone_channels` 列表，**3～4 stage**）、`blocks_per_stage`、`use_residual`、`conv_kernel_size`、`head_hidden`、`dropout`、`lr` 等。**不含** `tabular_fuse_dim`（默认 spatial 下 `tabular_dim=0`，该维无效）。

---

## 第七章 `configs/train.yaml` 逐项说明

### 7.1 优化与数据加载

- `batch_size`：默认 256。DDP 下为**每进程 per-GPU batch**（总有效 batch ≈ `batch_size × world_size`）。  
- `epochs` / `lr` / `weight_decay`：标准监督学习超参。  
- `num_workers`：DataLoader worker；默认 `0`（冒烟/调试）。**全量 `exp_full_best_ddp` 默认 2**，配合 `prefetch_factor: 2`（64GB 机器省内存）。  
- `max_grad_norm`：梯度裁剪；在强 class weight 下抑制爆炸步长。  
- `cudnn_benchmark`：以可复现性换吞吐；论文终稿阶段若需严格比特一致可关。

### 7.2 早停：`patience` 与 `early_stop_metric`

- `patience`：验证指标连续多少轮无提升则停。默认 `50`。  
- `early_stop_metric`：默认 `avg_precision`（PR 曲线下面积的一种实现，即 sklearn 风格 average precision）。也可 `best_f1`、`roc_auc`、`f1_at_0p5`。

**关键心智模型**（避免误读日志）：

- 日志中的 **`best_f1` 列**始终表示「在验证集上对概率做阈值搜索后的最优 F1」，**不是**早停指标本身，除非你把 `early_stop_metric` 设为 `best_f1`。  
- `f1@0.5` 是固定 0.5 阈值下的 F1。  
- 当 `early_stop_metric: avg_precision` 时，若 PR-AUC 很早到达平台，会连续 `patience` 轮无提升而触发早停——这在**小验证集**上是正常现象，**不能**仅凭「很早停」推断数据泄漏。

### 7.3 `imbalance` 子树

- `use_class_weight`：是否在损失里使用类别权重。  
- `weight_scheme`：`inverse_freq` 或 `sqrt` 等，决定正类权重如何从 `neg/pos` 推导，并受 `positive_weight_cap` 限制。  
- `focal_gamma`：默认 **`0.5`**；设为 `0` 或 `null` 时退回普通加权 CE。  
- `positive_oversample_factor`：默认 **2**（`exp_baseline` 覆盖为 **1**）；将每个正样本索引重复扩展，**放大** `train_virtual_len`。  
- `use_weighted_sampler`：单机可选；DDP 下通常关闭（见 trainer 日志说明）。  
- `sampler_positive_boost`：若启用加权采样器时的额外正类倾向。

**配置原因**：天池式 DRAM 预测极度不平衡；仅靠 `batch_size` 采样不足以稳定学习正类，因此组合「重加权 + 可选 Focal + 过采样」是工程上常见解。副作用是：**验证集指标对阈值敏感**，因此单独提供阈值搜索与 `best_f1` 日志列。

### 7.4 `threshold_search`

验证阶段在 `[min,max]` 上以 `step` 扫描阈值，写入 **`best_threshold.json`**；推理阶段读取该文件作为默认阈值（缺失则用 0.5；**JSON 损坏会报错**，不会静默回退）。

### 7.5 GPU / DDP

默认见 **`configs/base.yaml`** 的 `gpu` / `ddp`（`train.sh` / `infer.sh` 通过 `dram2cnn.utils.gpu_launch` 解析）。本文件或实验 `overrides.train.*` 可覆盖子键，例如 `num_workers` 仍仅在本文件。

### 7.6 Optuna 超参搜索（`configs/hpo.yaml`）

- 入口：`bash scripts/optuna.sh [实验yaml] [hpo.yaml]`，默认 `exp_optuna_smoke.yaml`。  
- **前置**：目标实验已完成 preprocess（存在 `data/splits/`）。  
- 每个 trial 写入 `config_snapshot.json` 后，通过 **subprocess + `torchrun`（若 DDP 启用）** 调用 `run_train`，与 `train.sh` 共用 GPU 启动逻辑；study 汇总见 `runs/optuna/<study_name>/best_trial.json`。  
- 搜索空间：`backbone_channels`（3～4 stage）、`blocks_per_stage`、`use_residual`、`conv_kernel_size`、`head_hidden`、`dropout`、`lr` 等（见 `configs/hpo.yaml`；不含 `tabular_fuse_dim`，默认 spatial 下无效）。

---

## 第八章 `configs/infer.yaml` 逐项说明

- `batch_size`：推理通常可略大于训练（无反向）。默认 `512`；`auto_batch_size=true` 时由启动脚本探测覆盖。  
- `mode`：必须为 `full`，与 `validate_config` 一致。  
- `gpu` / `ddp`：默认继承 `base.yaml`；本文件可覆盖（如 `gpu.batch_size_max: 2048`）。

---

## 第九章 内置实验对照：`exp_baseline` vs `exp_full_best_ddp`

| 维度 | exp_baseline | exp_full_best_ddp |
|------|--------------|-------------------|
| `run_name` | `exp_tianchi_baseline` | `exp_tianchi_full_best_ddp` |
| 索引/mcelog 截断 | `max_index_rows=500`，`mcelog_max_rows=800000`（须与 `index_build_presets.smoke.mcelog_scan_rows` 一致） | 二者 `null`（不截断） |
| 网格 / horizon | 继承 `data.yaml`：**spatial_heatmap** + **5min** 窗/标签 | 同左 |
| 模型片段 | **`model_smoke.yaml`**（轻量 GridCNN） | **`model.yaml`**（完整 GridCNN-Deep） |
| 模型张量 | **`31×128×32`**，`tabular_dim=0` | 同左 |
| 训练预算 | `epochs=20`，`patience=6`；train/infer **关闭 DDP 与 auto_batch** | 继承 `train.yaml`（`epochs=300`）；**`num_workers=2`**；**gpu/ddp 继承 base（auto 多卡 + 80% 显存）** |
| 典型用途 | 冒烟、CI、`autotune.sh` | 全量 preprocess/train；**须** `build_index.sh full`；仓库默认已在 overrides 指向全量索引 |

**方法**：新人应先跑通 baseline，再切 full_best；切忌以为「full_best 文件名」就自动下载了天池数据。

---

## 第十章 场景化操作规程（建议打印）

### 场景 A：第一次克隆仓库，想确认环境

1. 保持默认 `index_features_path`（演示表）。  
2. `bash scripts/preprocess.sh configs/experiments/exp_baseline.yaml`。  
3. `bash scripts/train.sh configs/experiments/exp_baseline.yaml`。  
观察 `runs/exp_tianchi_baseline/` 是否出现权重与日志。

### 场景 B：接入全量索引与全量 mcelog

1. 将大文件放入 `data/raw/`（或任意可读路径）。  
2. **`bash scripts/build_index.sh full`**（或 `--label-horizon-seconds` 定制）生成 `data/raw/features_index_*.csv`。  
3. 修改 `index_features_path`（或实验 overrides）指向该 CSV。  
4. **删除** `data/processed`、`data/splits`（或清理 marker）。  
5. 使用 `exp_full_best_ddp.yaml` 跑 preprocess（含 validate）→ train。  
6. 打开训练日志，确认「数据摘要」中 `N_train`、`index_features_path` 符合预期，且无 stub WARNING。

### 场景 C：只想调早停，不动数据

改 `configs/train.yaml` 的 `patience` 或 `early_stop_metric`；**不需要** preprocess。但若同时改 `batch_size`/`imbalance`，建议记录实验 ID 以便对比。

### 场景 D：多机共享集群限 CPU

在实验 `overrides.data` 中设置 `preprocess_workers: 1` 或降低 `preprocess_parallel_frac`。仅影响预处理阶段。

---

## 第十一章 与源码模块的映射表（追溯用）

| 配置段 | 主要消费模块 |
|--------|----------------|
| `data.*` | `dram2cnn.data.tianchi_build`、`dram2cnn.data.index_builder`、`dram2cnn.pipelines.run_preprocess`、`dram2cnn.data.splitter` |
| `model.*` | `dram2cnn.models.factory`、`dram2cnn.engine.trainer`（形状校验） |
| `train.*` | `dram2cnn.engine.trainer`、`dram2cnn.engine.imbalance` |
| `infer.*` | `dram2cnn.pipelines.run_infer` |
| `base.*` | `dram2cnn.utils.run_layout`、`dram2cnn.utils.logger.get_logger` |

---

## 第十二章 文档化治理建议（团队）

1. **任何**改变 `X` 形状的改动，必须在 PR 描述中写明「已重跑 preprocess + validate + 一次短训」。  
2. 长注释优先写在 `data.yaml` 与 `train.yaml` 顶部，避免只存在于聊天或 issue。  
3. 对新成员：阅读顺序建议为：本手册第零～三章 → `data_processing_guide` 第 1～3 章 → 再读 `data.yaml` 注释。

---

## 附录 A 字段速查（片段默认值一览）

> 下列值为撰写本文时仓库默认值；若你本地已修改，以磁盘为准。

**base**：`versioned_runs=true`，`device=auto`，`run_id` 与各 `*_run_id` 为 `null`。  
**data**：spatial 默认 **`window_seconds=label_horizon_seconds=300`（5 分钟）**；`max_index_rows`/`mcelog_max_rows` 默认 `null`；索引由 `scripts/build_index.py` 生成，演示表见 `index_stubs/`。  
**model**：`31×128×32`，`tabular_dim=0`；GridCNN-Deep 默认 `stem_channels=64`，`backbone_channels=[64,128,256,256]`，`blocks_per_stage=2`，`use_residual=true`。  
**train**：`batch_size=256`，`epochs=300`，`patience=50`，`num_workers=0`，`focal_gamma=0.5`，`positive_weight_cap=50`，`early_stop_metric=avg_precision`，`imbalance.positive_oversample_factor=2`；**gpu/ddp** 默认见 `base.yaml`（`exp_baseline` 用轻量 `model_smoke.yaml` 且过采样=1；`exp_full_best_ddp` 继承 base 多卡默认）。  
**infer**：`batch_size=512`，`mode=full`。

---

## 附录 B 词汇表

- **片段配置**：`configs/*.yaml` 中被实验 include 的五段之一。  
- **实验 YAML**：`configs/experiments/*.yaml`。  
- **合并**：`deep_update` 递归覆盖生成最终 `cfg`。  
- **物化**：preprocess 写出 `npy`/`csv`/`json` 落盘。  
- **契约**：`X_train.npy` 形状与 `model` 段一致的不变式。

---

## 附录 C GPU / DDP 配置层级

**当前设计**：`configs/base.yaml` 定义全 pipeline 默认的 `gpu`（显存预算、自动 batch）与 `ddp`（`enabled: auto`、`nproc_per_node: auto`）。`train.yaml` / `infer.yaml` 或实验 `overrides` 可覆盖子键——例如冒烟实验关闭 DDP、推理提高 `batch_size_max`。

**原因**：训练与推理共用「可见 GPU 数 + 80% 显存上限」策略，避免两处重复维护；仍允许按 pipeline 微调（推理无 backward，batch 可更大）。

---

## 附录 D 版本与维护

本手册随仓库演进应更新：当新增 `configs/experiments`、新增 `data` 键或更改 `validate_config` 规则时，请同步修订附录与第九章对照表。若你发现默认值与代码不一致，以 `load_experiment_config` 与 YAML 文件为准提交修正。

---

## 第十三章 深入：早停、验证指标与日志列的语义辨析

本章专门回应训练日志中常见的**误读**，其根源在于「**列名**」与「**早停所用的标量**」不是同一概念。

### 13.1 验证阶段实际计算了什么

在每个 epoch 末尾（rank0），训练器会在验证集上收集所有样本的真实标签 `y_true`、模型对正类的预测概率 `y_prob`、以及 argmax 的离散预测 `y_pred`（用于 0.5 阈值下的分类指标）。随后：

1. 用**固定 0.5 阈值**从 `y_prob` 导出二值预测，计算 precision、recall、F1、accuracy 等，记为 **`f1@0.5` 一类指标**。  
2. 在配置给定的 `[min,max]` 上以 `step` 做阈值扫描，找到使 F1 最大的阈值，记为 **`best_t`**，并在该阈值下再算一套指标，其中的 F1 即日志里的 **`best_f1` 列**。  
3. ROC-AUC、PR-AUC（average precision）等**排序型指标**通常由 `y_prob` 与 `y_true` 直接计算，对固定 0.5 阈值不敏感，但在**极小验证集**或**正负比例极端**时会出现离散化平台（连续多个 epoch 数值完全相同）。

### 13.2 `early_stop_metric` 如何选取「那一标量」

`train.early_stop_metric` 决定「哪一个标量与训练过程中迄今最优的 `best_score` 比较」。实现上，该标量越大越好；无效时退回可比较的哨兵值，以避免首轮无法保存 checkpoint（详见 `engine/trainer.py` 与 `troubleshooting.md`）。

**默认 `avg_precision` 的配置原因**：在极度不平衡任务中，ROC-AUC 有时对「全负预测」仍不够敏感；PR 曲线更关注正类检索质量。赛题若更关心「少漏报」或「正类精度与召回的折中」，也可改为 `best_f1`，但要意识到此时早停直接受**阈值搜索**影响，与部署时是否使用同一阈值策略需一致。

### 13.3 为何 `best_f1` 与 `f1@0.5` 可能差距很大

当模型输出的正类概率整体偏高或偏低时，0.5 阈值可能落在「概率分布平坦区」之外：例如大部分正类样本概率在 0.3～0.45，负类在 0.05～0.2，则 0.5 阈值会导致召回极低，从而 **`f1@0.5` 很低**；但若在 0.35 附近存在更优阈值，则 **`best_f1` 可以明显更高**。这不一定是 bug，而是**阈值与校准**问题。反过来，若 **`best_f1` 极高而 `f1@0.5` 长期一般**，部署时就不能默认 0.5，而应在验证集或独立校准集上固定阈值并写入推理配置。

### 13.4 小验证集与早停的耦合关系

当 `N_val` 只有十几条时，AUC/AP 的估计量方差极大，且排序关系在有限样本上呈阶梯状：模型参数轻微变化不一定改变「正类分数的相对次序」，从而 AUC/AP **多 epoch 不变**。若此时 `early_stop_metric` 为 `avg_precision` 且 `patience` 较大，会出现「训练似乎没进步却跑了很多轮」或「很快触发早停」两种极端，取决于首轮是否偶然达到该平台。**这不等价于数据泄漏**；泄漏应通过「训练样本是否误入验证」「时间切分是否跨越泄漏」「重复 sid」等数据审计验证。仓库在训练开始打印的**数据摘要**与 stub 路径告警，是为了把「数据规模异常小」与「指标异常好/坏」区分开来。

---

## 第十四章 深入：`imbalance` 各子项如何共同作用

### 14.1 类别权重：`inverse_freq` 与 `cap`

设训练集（**原始** `y_train`，未过采样前）正样本数为 `P`、负样本数为 `N`。`inverse_freq` 方案下正类相对权重与 `N/P` 成正比，再经 `positive_weight_cap` 截断，避免 `P` 极小时权重爆炸导致优化发散。该权重作为 `CrossEntropyLoss` 的 `weight` 参数（或等价路径）进入损失。

**为何还要 Focal**：在难例与易例梯度比例仍不理想时，Focal 通过 `focal_gamma` 压低「易分类样本」的贡献，使优化更关注难分样本。`focal_gamma: 0` 或空值表示退回普通加权交叉熵，便于做 ablation。

### 14.2 正样本过采样：`positive_oversample_factor`

实现语义（简化理解）：对每个原本的正样本索引，在训练集的逻辑长度上**重复出现**多次，使每个 epoch 的「有效样本索引序列」变长，记日志为 `train_virtual_len`。注意：**类权重仍按原始 `y_train` 计数计算**（见 `train.yaml` 注释），以避免「过采样后再按频次算权重」的双重放大错误——若你修改源码，需保持这一不变量。

**与 DDP 的关系**：过采样通过构造索引映射与 `DistributedSampler` 协同；加权随机采样器在 DDP 下默认关闭，因为多进程下需保证各 rank 数据划分与梯度同步一致，简单 per-batch 加权采样易引入额外方差与实现复杂度。

### 14.3 调参建议（工程向，非保证最优）

1. 若训练 loss 下降但验证 PR 长期不动，先检查**是否小验证集**与**是否标签噪声**，再考虑调模型容量。  
2. 若梯度爆炸，优先检查 `max_grad_norm` 与 `positive_weight_cap`，而不是盲目加深网络。  
3. 若正类 recall 过低，可适度增大 `positive_oversample_factor` 或放宽 `positive_weight_cap`，但二者同时极大时常导致过拟合正噪声。

---

## 第十五章 深入：`split_strategy` 的业务含义

### 15.1 分层时间切分（默认 `time_stratified`）

负样本按 `predict_ts` 排序后依次填入 train → val → test；正样本按时间排序后保证每档至少 `min_pos_per_split` 个，再按剩余比例分配。兼顾「大致时间先后」与「验证集可计算 AUC/PR」。全量索引正样本仍不足 `3×min_pos_per_split` 时会报错，需重建索引或改用 `sid`。

### 15.2 纯时间切分（`time`）

按 `predict_ts` 排序后按比例切开，不做标签分层；正例极少时 train/val 可能无正样本。

### 15.3 按服务器切分（`sid`）

当同一 `sid` 的样本在时间上强相关、且你不希望「同一机器同时出现在 train 与 test」时，`sid` 切分更保守，通常使测试分数**低于**时间切分（更难）。选择哪种应在实验记录中写明，以便与论文或答辩材料对齐。

---

## 第十六章 深入：`inventory` 十五维 one-hot 与 `TABULAR_DIM` 校验

`dram2cnn.data.inventory_encode` 将 `inventory.csv` 中三类离散字段编码为固定 **15** 维：`DRAM_MODEL_ORDER`（7 类）+ `DIMM_ORDER`（4 档）+ `MANUFACTURER_ORDER`（4 家）。**校验原因**：若 `model.tabular_dim` 自行设为 16 或 14，全连接层权重形状与数据管道不一致，会在矩阵乘法处失败或（更危险）若某处错误广播导致静默错误。`validate_config` 强制 `tabular_dim>0` 时必须等于 15，是把竞赛侧类别集合**钉死**为单一真相来源（single source of truth）。

当 `inventory_as_grid_channels: true` 时，这 15 维被复制到空间网格上，成为 `X` 的一部分通道，此时 `tabular_dim` 应为 0，表示「不单独走 tabular 分支」。若配置为 `tabular_dim=0` 但 splits 仍含 `tabular_*.npy`，可能在切分或训练阶段出现维度不一致；splitter 会在 preprocess 时清理多余的 tabular 切分文件，但**配置层面仍须理解两种 inventory 接入方式的语义**。

---

## 第十七章 脚本、指针文件与「可复现目录名」的联合使用

### 17.1 典型 shell 调用链

`scripts/preprocess.sh`、`scripts/train.sh`、`scripts/train_ddp.sh`、`scripts/infer.sh`、`scripts/evaluate.sh` 均接收**同一个实验 yaml 路径**，从而保证「数据定义—模型—训练—推理」读取的是**同一份合并后配置**。若你在 preprocess 用 baseline、在 train 用 full_best，只要二者在 `data` 段不一致，就会导致**切分与特征版本错配**——这是新人高频错误，应避免。

### 17.2 `.latest_train_session` 与 `.latest_infer_session`

二者为 `runs/<run_name>/` 下的**单行文本文件**，内容为相对子目录名。设计目的是：后续 `infer` 默认找到「最近一次成功训练」的权重子目录，而无需手工拷贝路径。若你并行跑多次训练，务必确认 infer 读到的指针是否指向你**意图**的那次；必要时使用 `base.checkpoint_run_id` 或 `evaluate_run_id` 固定。

### 17.3 `run_id` / `infer_run_id` 的论文级用途

当你需要在补充材料中给出「固定目录名」以便审稿人复现，可将 `run_id` 设为例如 `paper_main_run`，则训练产物稳定落在 `runs/<run_name>/paper_main_run/`。注意：若目录已存在，需自行处理是否覆盖或与 `versioned_runs` 语义冲突——阅读 `run_layout.resolve_train_run_dir` 的分支行为。

---

## 第十八章 FAQ：只读本章也能解决一半问题

**问：我只改了 `model.yaml` 的 `grid_height`，为什么 train 还报错？**  
答：因为磁盘上的 `X_train.npy` 形状与当前 `model.yaml` 不一致。必须重跑 preprocess 与切分，并确认实验未在 overrides 里覆盖成不一致的 model 段。

**问：`exp_full_best_ddp` 已经写了全量，为什么 manifest 里 n 还是约 500？**  
答：「全量」指 `max_index_rows`/`mcelog_max_rows` 为 `null`，且 **`index_features_path` 指向全量 CSV**（`exp_full_best_ddp.yaml` 默认已覆盖为 `data/raw/features_index_h300s_step300s_full.csv`）。若 manifest 仍约 500 行，常见原因是 **未 `build_index.sh full`、未清空 `processed/`/`splits/` 重跑 preprocess**，或仍用 baseline 实验。若路径仍含 stub，**`assert_full_best_index_not_stub` 会直接报错**。

**问：能否让 baseline 与 full_best 同时存在两个不同索引？**  
答：可以。复制两份实验 yaml，在各自 `overrides.data.index_features_path` 写不同路径；或复制两份 `data` 片段文件并在实验中 `data: my_data_a.yaml` / `my_data_b.yaml` 指向它们。

**问：`inventory_path` 能否设为空？**  
答：取决于实现是否接受 null；`validate_config` 在路径非空时才检查存在性。若赛题无 inventory，应使用官方认可的「全零特征」语义并在文档中记录。

**问：为什么默认 `num_workers: 0`？**  
答：为兼容多环境（含 NFS、Windows 子系统、调试器 attach）。在 Linux 本地 SSD 上可尝试增大以加速 DataLoader，但若出现「too many open files」或死锁，应回退。

---

## 第十九章 推荐阅读顺序（培训大纲，约 2～4 学时）

1. 本文第零章～第三章：建立「五段 + 实验 + 合并 + 校验」心智模型。  
2. `data_processing_guide` 第一至第三节：理解样本、窗口、标签。  
3. 本文第五章：对照自己的 `data.yaml`，逐项打勾确认。  
4. `tianchi_pipeline` 速查表：记住 spatial 默认形状推导。  
5. 本文第七、十三章：训练与早停调参。  
6. `troubleshooting`：遇到 nan、全负标签、时间解析错误时查阅。

---

## 第二十章 术语中英对照（便于读代码）

| 中文 | 英文/代码键 |
|------|-------------|
| 索引表 | `index_features_path` |
| 半开时间窗 | half-open window `[t_end-W, t_end)` |
| 回溯 / horizon 打标 | `label_horizon_seconds`（`label_backtrace_seconds` 同义） |
| 空间热图 | `spatial_heatmap` |
| 分布式数据并行 | DDP / `base.ddp`（train/infer 可覆盖） |
| 平均精度 | average precision / `avg_precision` |
| 过采样因子 | `positive_oversample_factor` |

---

## 第二十一章 完整 Walkthrough：从「空仓库」到「可提交的一次全量实验」

本章用**叙事体**把配置变更串成一条时间线，便于你照抄检查清单。假设你已在 `data/raw/` 放好天池提供的 `mcelog.csv`、`trouble_tickets.csv`、`inventory.csv`，并已执行 **`bash scripts/build_index.sh full`** 生成全量索引（例如 `data/raw/features_index_h300s_step300s_full.csv`）。

**第 1 步：选择实验入口**  
决定本次产物写入 `runs/exp_tianchi_full_best_ddp/`（或你自定义的 `run_name`）。若沿用仓库文件，则实验 yaml 为 `configs/experiments/exp_full_best_ddp.yaml`。若你希望保留原 `run_name` 语义但改目录名，可在 `overrides.base.run_name` 写新字符串——注意这会改变 `runs/` 下文件夹名，所有后续脚本必须仍指向**同一份**实验 yaml。

**第 2 步：确认全量索引路径**  
仓库默认的 `exp_full_best_ddp.yaml` 已在 `overrides.data.index_features_path` 指向 `data/raw/features_index_h300s_step300s_full.csv`（须先 **`bash scripts/build_index.sh full`** 生成该文件）。若使用自定义实验 yaml，请显式设置该键；baseline 仍指向演示 stub 以便并行冒烟。

**第 3 步：确认截断开关**  
在 full_best 实验中，`max_index_rows` 与 `mcelog_max_rows` 应为 `null`。若你误把 baseline 的 `500`/`800000` 复制过来，会在全表上仍只读 500 行索引——这是典型的「复制粘贴 overrides」错误。

**第 4 步：清理预处理产物**  
删除 `data/processed` 与 `data/splits` 下全部生成文件，或删除流水线 marker。否则 preprocess 可能被跳过，训练会读取与当前配置不一致的 `npy`。

**第 5 步：预处理与校验**  
执行 `bash scripts/preprocess.sh configs/experiments/exp_full_best_ddp.yaml`，完成后检查 `preprocess_manifest.json` 中的样本数是否与索引行数（在截断为 null 时）一致、`pos_rate` 是否合理。随后 `bash scripts/validate.sh ...` 确认 `X_train` 形状与 `model.yaml` 一致。

**第 6 步：训练并阅读 rank0 日志**  
使用 `scripts/train_ddp.sh` 或单卡 `train.sh`。日志开头应出现**数据摘要**：若仍提示 stub 路径，说明合并后的 `index_features_path` 未改成功，应回到第 2 步排查是否改错文件或改错实验 yaml。训练过程中观察 `early_stop_metric` 打印列是否与你的预期一致。

**第 7 步：推理与评估**  
不改实验 yaml 的前提下跑 `infer` 与 `evaluate`，确认 `predictions.csv` 与 `metrics.json` 落盘路径。若需对比多次推理，依赖 `versioned_runs` 与指针文件，而不是手工移动 csv。

**第 8 步：实验记录（强烈建议）**  
在实验笔记中记录：**合并后** `index_features_path` 的绝对路径、`preprocess_manifest` 的哈希或关键字段、`git commit`、训练 `run` 子目录名、以及 `train.early_stop_metric`。这八步能把「可复现」从口号落实为可操作习惯。

---

## 第二十二章 与天池 DRAM 赛题叙事的逐项对齐（为何默认如此）

天池 132973 类报告通常强调：DRAM 可纠正错误在空间上具有结构性（row/col/bank），且与机型、容量档、厂商等静态属性共同影响故障风险。本仓库默认 `spatial_heatmap` 与 `inventory_as_grid_channels: true` 的组合，是在**归纳偏置**层面把「空间累积」与「静态属性」同时交给卷积早期层去建模，而不是手工构造上百列统计特征后再喂入 MLP。对应地，默认 **`window_seconds` / `label_horizon_seconds` 均为 5 分钟**，便于固定步长索引与短 horizon 故障预测；若改为 6 小时特征窗或 7 天标签窗，应在报告中显式披露。

工单类型 `label_failure_types` 默认包含三类，是把「多种故障形态」纳入正类定义；若业务只关心某一子类，应缩小列表并重新评估 `pos_rate` 变化。inventory 十五维 one-hot 与赛题提供的类别表对齐，若主办方更新类别集合，你必须同时更新 `inventory_encode` 与 `TABULAR_DIM` 及所有文档——这是少数「不能只改 YAML」的情形，因为维度是代码常量。

---

## 第二十三章 反模式清单：我们明确不推荐的用法

1. **在多个实验 yaml 之间来回改 `run_name` 指向同一目录并手动删权重**：容易误删他人产物；应使用 `versioned_runs` 或不同 `run_name`。  
2. **只增大 `positive_oversample_factor` 而不监控验证集 PR**：可能严重过拟合少数正样本噪声。  
3. **把 baseline 的短窗配置复制到全量 spatial 实验却不写进实验记录**：会导致结果与天池基线不可比。  
4. **在 Notebook 里硬编码路径绕过 `load_experiment_config`**：短期省事，长期不可复现；应通过 yaml 或环境变量注入。  
5. **假设 `ddp.enabled: auto` 就一定多卡**：实际仍由可见 GPU 数决定；仅 1 卡时退化为单进程。

---

## 第二十四章 与项目其它文档的交叉索引

| 主题 | 首选文档 |
|------|----------|
| 从 CSV 到 X/y 的逐步推导 | `docs/data_processing_guide.md` |
| 规则速查与并行度 manifest 含义 | `docs/tianchi_pipeline.md` |
| 列名与原始表字典 | `docs/data_dictionary.md` |
| nan、时间解析、tabular 维度混用 | `docs/troubleshooting.md` |
| Notebook 流水线与 marker | `notebooks/README.md`、`04_run_pipelines.ipynb` 说明格 |

当本手册与上述文档出现表述冲突时，以**源码行为**与**当前仓库 YAML 默认值**为准，并欢迎提交文档修正。

---

## 第二十五章 结语之后：维护者与审阅者检查表（可贴进 PR 模板）

- [ ] 若改 `data.grid_representation` 或 `inventory_as_grid_channels`，是否同步 `model` 段与 preprocess？  
- [ ] 若改 `index_features_path`，是否声明清理 processed/splits？  
- [ ] 若改 `train.early_stop_metric`，是否更新团队 wiki 中对日志列的解释？  
- [ ] 若新增实验 yaml，是否在 `configs/experiments/` 内添加注释说明适用场景？  
- [ ] 若调 `imbalance`，是否在 PR 描述中给出验证集 PR/F1 前后对比而非仅 loss？

以上检查表与第二十一章 Walkthrough 一起，可作为「配置相关变更」的最低审阅标准；它们不保证模型效果提升，但显著降低「以为全量其实演示表」「以为早停看 F1 其实看 AP」等沟通成本。

---

**结语**：配置的本质是把「我们对赛题与世界的不变量」编码成可执行约束。把**数据源**（索引路径）、**特征定义**（窗长、网格模式）、**模型契约**（C/H/W）、**优化与不平衡策略**（损失与采样）、**运行布局**（`run_name` 与版本目录）拆开，是为了让每一次实验差异都可被 **git diff** 阅读。希望本文能帮助你在不读完全部源码的前提下，仍能对「改这里会牵动哪里」有稳定直觉；当你需要下钻实现细节时，再以第十一章的模块映射为索引进入代码即可。

---

## 第二十六章 扩展阅读：把「配置」放进软件工程流程

在工业界竞赛或交付型项目中，配置管理往往还要对接：**数据版本（DVC/湖）、密钥与路径（环境变量）、算力队列（Slurm job yaml）**。本仓库刻意保持「纯 YAML + 脚本」的最小闭环，以便初学者不被工具链淹没。当你团队规模扩大时，建议在不变更合并语义的前提下，增加一层「环境 profile」：例如 `configs/profiles/cluster_a.yaml` 只放 `preprocess_parallel_frac` 与绝对路径，再在实验 yaml 最底部引用——注意当前 `load_experiment_config` **未**内置 profile 机制，任何此类扩展需同步修改加载器与文档，避免静默忽略文件。

---

## 第二十七章 模拟一次「配置评审会」对话（帮助自检）

**评审者**：你为什么把 `window_seconds` 改成 1800？  
**答辩者**：我们在做 30 分钟 horizon 对比实验，已用 `build_index.py` 生成匹配索引并重跑 preprocess。  
**评审者**：默认 5 分钟设定与天池 6 小时报告不同，请在报告中说明 horizon 与特征窗选择。  

**评审者**：你的 `pos_rate` 为何从 0.05 变成 0.45？  
**答辩者**：我们改了 `label_failure_types` 或修正了 `predict_time` 解析。  
**评审者**：请附 `preprocess_manifest.json` 片段与索引前若干行样例。  

这类对话能迫使团队把「配置改动」与「数据语义改动」绑定说明，减少口头传递误差。

---

## 第二十八章 维护声明

后续若 `configs/train.yaml` 增加新键（例如混合精度开关），应在第七章增补小节，并在附录 A 更新默认值表；若 `validate_config` 增删硬约束，应在第三章与第十八章同步更新，避免文档与校验逻辑不一致。

---

## 第二十九章 培训大纲（扩展版，约 4～6 学时）

第 1 学时：第零～三章，并阅读 `dram2cnn/utils/config.py` 中 `load_experiment_config` 源码。  
第 2 学时：第五章全章，对照本地 `data.yaml` 与 `preprocess_manifest.json`。  
第 3 学时：第六～七章与第十三章（早停与日志列）。  
第 4 学时：第九章对照表与第二十一章 Walkthrough 实操。  
第 5 学时：`data_processing_guide.md` 后半与 `tianchi_build` 关键函数略读。  
第 6 学时：第二十七章模拟评审，并将第二十五章检查表定稿为团队 PR 模板。

---

**文档说明**：正文以中文技术叙述为主，并保留 YAML 键名、路径与必要英文术语以便与仓库一致。若你只关心「最快改对全量索引」，可优先阅读第一章第 1.3 节、第五章第 5.1.4 节、第九章与第十八章 FAQ；若关心训练日志误读，可读第十三章。全文篇幅较长，可按第二十九章分次阅读。

---

## 第三十章 环境与安全：配置之外的「隐变量」

即便 YAML 完全正确，下列**环境层**因素仍会改变可复现性与性能，应在实验记录中一并记载。

**操作系统与文件系统**：预处理在 Linux 上可对部分阶段启用多进程；在 Windows 或某些网络文件系统上可能自动退回单进程或出现锁竞争。此时你即使把 `preprocess_parallel_frac` 调到 0.9，也可能看不到线性加速。

**CUDA 驱动与可见设备**：`CUDA_VISIBLE_DEVICES` 会改变「可见 GPU 个数」，从而影响 `train.sh` 是否走 DDP。若你在 shell 里导出该变量却在 Jupyter 里启动训练，二者环境可能不一致，导致「终端能跑、Notebook 不能跑」的假象——根因不是 YAML，而是进程环境。

**随机性与确定性**：`seed` 只控制部分随机源；cuDNN benchmark、多进程数据打乱顺序、浮点归约顺序仍可能带来微小差异。若论文要求严格可复现，应在方法部分声明「近似复现」并固定 `run_id` 与容器镜像版本。

**敏感路径**：生产环境常把大 CSV 放在权限受限目录，YAML 中写绝对路径时注意不要将含内网路径的配置提交到公开仓库；可用环境变量替换方案（需自行扩展加载器，本仓库默认未实现）。

**并发训练**：两个进程同时对同一 `run_name` 且 `versioned_runs: false` 写入，会互相覆盖。团队应约定「每人一个 run_name 前缀」或强制 `versioned_runs: true`。

以上各条并非「配错 YAML」类错误，却常在排错现场占用大量时间；把它们写进第二十一章的实验笔记扩展栏，可显著降低集体心智负载。

此外，若你在容器内挂载只读数据卷，务必确认预处理写入目录 `processed_dir` / `splits_dir` 仍指向可写卷；否则会在运行中途才暴露权限错误，浪费已扫描的大量 mcelog 时间。若赛方更新数据字典列名，应先在 `tianchi_build` 或读取层做列映射，再讨论是否值得为此新增一层「列别名配置」，以免 YAML 膨胀成不可维护的万能文件。最后提醒：本手册任何示例路径均须替换为你环境中的真实路径，切勿照抄竞赛内网路径到公开仓库。

---

## 第三十一章 小结：把配置当作「接口契约」而非「注释堆砌」

你可以把五段片段看成五组**接口**：`data` 段定义上游数据契约，`model` 段定义张量契约，`train` 段定义优化与采样契约，`infer` 段定义服务化推理契约，`base` 段定义产物与设备契约。实验 yaml 则是「某次集成的版本号」。当团队开始用「接口契约」的语言讨论改动时，Review 会自然聚焦在：是否破坏契约、是否需要迁移数据、是否需要同步文档——而不是纠结在「这个键名好不好听」的表层问题。至此，本手册从动机、机制、字段、流程、反模式、培训、环境七个维度完成了对项目配置体系的展开；若你仍觉得某一段落抽象，请以具体一次 PR 的 `git diff` 为材料，对照第五章逐行标注「这一行改动会影响预处理还是仅影响训练」，反复几次后即可内化为直觉。
