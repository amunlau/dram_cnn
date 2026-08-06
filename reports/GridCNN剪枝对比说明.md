# GridCNN 三种粒度剪枝对比说明

**关联实验**：`exp_full_f1_ddp` / 训练会话 `20260729_170226_369953`  
**实测汇总**：[`GridCNN剪枝对比实验报告.md`](GridCNN剪枝对比实验报告.md)（由 `prune_compare_*/summary.md` 归档）  
**代码**：`src/dram2cnn/engine/prune/`、`pipelines/run_prune.py`、`scripts/prune.sh`  
**串联量化**：见 [`GridCNN先剪枝后量化说明.md`](GridCNN先剪枝后量化说明.md)

---

## 1. 实验设定

| 项 | 值 |
|----|-----|
| 基线权重 | `checkpoints/grid_cnn.pt`（基线 valτ=0.84 时 F1≈0.7528） |
| 评估数据 | `data/splits`，**time_stratified 全量**（train/val/test=24958/5348/5348；非 smoke） |
| 主指标阈值 | 各剪枝模型在 **val** 上按 `threshold_search(objective=f1)` 重搜（valτ），再评 **test** |
| 对照 | `metrics_val_retuned.json` / `*_at_deploy_tau`：强制基线部署 τ=0.84 |
| 比例 | 0.3 / 0.5 / 0.7（非结构化=权重稀疏率；结构化=通道剪除比例） |
| 微调 | 默认 5 epoch，Adam lr=1e-4，全量 `X_train` |
| 结构化默认 | 逐层同比例；保护 stem 与末层卷积；`min_keep_channels=8` |

产物目录：

```
<train_dir>/prune_compare_<timestamp>/
  prune.log
  summary.md / summary.json
  baseline/{metrics.json, metrics_val_retuned.json, stats.json}
  unstructured/sparsity_XX/{pruned_model.pt, metrics.json, metrics_val_retuned.json, stats.json}
  bn_channel/sparsity_XX/{..., channel_masks.json}
  filter/sparsity_XX/{...}
```

---

## 2. 方法一：非结构化权重稀疏（Unstructured L1）

### 思路

对所有 `Conv2d` / `Linear` 的 `.weight` 做 **全局 L1 magnitude** 剪枝：按 `|w|` 排序，将全局最小的约 `sparsity` 比例权重置零。实现使用 `torch.nn.utils.prune.global_unstructured` + `prune.remove` 固化为零，便于存盘。

### 注意事项

- **不改变张量形状**，参数量名义不变，有效参数看非零个数。
- 实测加速依赖稀疏内核；报告中的 MACs 为 `dense_macs × (1−sparsity)` 理论值。
- 短时微调有助于高稀疏率恢复；读表请以 **valτ 主指标** 为准。

---

## 3. 方法二：基于 BN 的通道剪枝（Network Slimming 剪枝阶段）

### 思路

1. 遍历有序 `_ConvBNReLU`，取 `BatchNorm2d.weight`（γ）的绝对值作通道分数。  
2. **逐层**按 `prune_ratio` 淘汰低分通道（可选 `--global-ranking` 跨层全局淘汰）。  
3. 切片当前层 `conv`/`bn` 的输出通道，并同步 **下一层输入通道** 与 `head.0` 的 `in_features`。

### 与完整 Network Slimming 的差异

经典流程在训练时对 γ 加 L1，使不重要通道的 γ→0。本仓库全量模型**未**做该正则；本实验仅用**现成 γ** 做重要性，属于 Slimming 的「剪枝阶段」迁移。

### 注意事项

- 需要 `use_batch_norm=true`。  
- 通道剪 30% 时，中间层参数近似按 `0.7×0.7` 缩放，**参数量下降常大于 30%**。  
- **无微调**时精度极易崩塌；务必微调或降低剪除比例。  
- 默认保护 stem / 末层卷积，减轻崩塌。

---

## 4. 方法三：Filter / 卷积核结构化剪枝（L1-norm）

### 思路

对每个输出 filter 计算 `sum(|W[c,:,:,:]|)`，按分数淘汰输出通道；**通道衔接与方法二共用** `structured_utils.apply_channel_keeps`。

### 注意事项

- 与 BN 法剪除比例相同时，通道数与理论 MACs 变化量级相近，但保留集合不同，精度曲线可有差异。  
- 同样强烈依赖短时微调。  
- 不剪 `head.3` 的类别维；不改 stem 输入 31 通道。

---

## 5. 复现命令

```bash
export PYTHONPATH=src
bash scripts/prune.sh configs/experiments/exp_full_f1_ddp.yaml \
  --train-dir runs/exp_tianchi_full_f1_ddp/20260729_170226_369953 \
  --sparsities 0.3,0.5,0.7 \
  --finetune-epochs 5
```

纯剪枝（无微调）对比：追加 `--finetune-epochs 0`。  
先剪枝后量化：`bash scripts/prune_quantize.sh configs/experiments/exp_full_f1_ddp.yaml`。

---

## 6. 实测结论摘要（微调 5 epoch，主指标=valτ）

相对 baseline F1≈0.7528（valτ=0.84）：

| 方法 | 0.3 | 0.5 | 0.7 | 备注 |
|------|-----|-----|-----|------|
| unstructured | 0.759 | 0.745 | 0.752 | valτ 约 0.58–0.72 |
| bn_channel | 0.741 | 0.753 | 0.752 | 参数/MACs 降幅大于名义通道比 |
| filter | 0.755 | 0.758 | 0.753 | 与 bn 同结构宽度，保留集合不同 |

详细数字与路径以 `summary.md` / `metrics_val_retuned.json` / `prune.log` 为准。若只看固定 τ=0.84，P 会偏高、R/F1 偏低，勿与主表混读。
