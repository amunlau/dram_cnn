# GridCNN 剪枝对比实验报告

- checkpoint: `/root/workspace/runs/exp_tianchi_full_f1_ddp/20260729_170226_369953/checkpoints/grid_cnn.pt`
- primary threshold: **val-retuned F1**（各模型独立；与训练 threshold_search 一致）
- baseline deploy τ（对照）: **0.84**
- device: `cuda:0`
- finetune_epochs: **5**
- sparsities / prune ratios: [0.3, 0.5, 0.7]

## Baseline (FP32)

| F1 | Precision | Recall | ROC-AUC | params |
|----|-----------|--------|---------|--------|
| 0.7528 | 0.8597 | 0.6696 | 0.8940 | 3184450 |

## Results

| method | ratio | valτ | F1 | ΔF1 | P | R | params_after | param_red | theo_GMACs | ok |
|--------|-------|------|----|-----|---|---|--------------|-----------|------------|----|
| unstructured | 0.30 | 0.72 | 0.7587 | +0.0058 | 0.8672 | 0.6743 | 2230306 | 30.0% | 0.8440 | ok |
| unstructured | 0.50 | 0.58 | 0.7453 | -0.0075 | 0.8415 | 0.6689 | 1594210 | 49.9% | 0.6028 | ok |
| unstructured | 0.70 | 0.70 | 0.7522 | -0.0006 | 0.8705 | 0.6622 | 958114 | 69.9% | 0.3617 | ok |
| bn_channel | 0.30 | 0.64 | 0.7413 | -0.0115 | 0.8135 | 0.6810 | 1735862 | 45.5% | 0.6703 | ok |
| bn_channel | 0.50 | 0.62 | 0.7530 | +0.0002 | 0.8351 | 0.6857 | 1017154 | 68.1% | 0.4035 | ok |
| bn_channel | 0.70 | 0.62 | 0.7519 | -0.0009 | 0.8415 | 0.6796 | 496230 | 84.4% | 0.2134 | ok |
| filter | 0.30 | 0.72 | 0.7549 | +0.0021 | 0.8629 | 0.6709 | 1735862 | 45.5% | 0.6703 | ok |
| filter | 0.50 | 0.66 | 0.7579 | +0.0051 | 0.8825 | 0.6642 | 1017154 | 68.1% | 0.4035 | ok |
| filter | 0.70 | 0.66 | 0.7530 | +0.0002 | 0.8381 | 0.6836 | 496230 | 84.4% | 0.2134 | ok |

## 方法说明

### 1. unstructured（非结构化 L1）

- 对 Conv/Linear 权重做全局 L1 magnitude 剪枝（`torch.nn.utils.prune`）。
- 不改网络结构；稀疏加速需专用内核，报告中 MACs 为理论值。

### 2. bn_channel（BN-γ 通道剪枝）

- Network Slimming **剪枝阶段**：按 `|γ|` **逐层**淘汰通道（可用 `--global-ranking`），切片 Conv/BN 并同步后续 `in_channels` 与 `head.0`。
- **注意**：全量训练未对 γ 施加 L1；此处用现成 γ 作重要性；默认短时微调。

### 3. filter（Filter L1 结构化）

- 按每个输出卷积核权重 L1-norm 剪通道；通道衔接与 bn_channel 相同。

## 注意事项

- **主指标**：各模型在 val 上重搜 F1 阈值后再评 time-stratified 全量 test。
- 默认**短时微调**（`--finetune-epochs`，默认 5）；设为 0 可对比「纯剪枝」掉点。
- 结构化剪枝每层至少保留 `min_keep_channels` 个通道；默认保护 stem / 末层卷积。
- 分类末层 `head.3` 输出维不剪；stem 输入通道（31）不剪。
- 全量训练未对 BN γ 做 L1；bn_channel 仅用现成 γ 作重要性。
- 若强制固定基线部署 τ，量化/剪枝后可能出现 P 虚高、R 偏低；见 `metrics_val_retuned.json`。

