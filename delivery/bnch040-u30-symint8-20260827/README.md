# 交付说明（BN-γ + 非结构化 + 整网对称 INT8）

目录：`delivery/bnch040-u30-symint8-20260827/`

相对此前仅非结构化剪枝的交付，本次在融合前增加 **BN-γ 通道剪枝（目标 40%）**，并保留 **非结构化 L1 30%**、**融合无偏置**、**整网逐层对称 signed INT8**。

验收（测试集、部署阈值）：**精确率 > 0.8，召回率 > 0.6** — 本包 **已达标**。

---

## 1. 关键结果

| 项 | 值 |
|----|-----|
| BN-γ 目标比例 | **0.40**（实际通道剪除约 33%） |
| 非结构化稀疏度 | **0.30** |
| 融合 | Conv+BN+ReLU，强制无偏置 |
| 量化 | 整网逐层对称 signed INT8（zp=0） |
| 部署阈值 | **τ = 0.5** |
| INT8 F1 / P / R | **0.7559 / 0.9244 / 0.6394** |
| packed 体积 | 约 **1.30 MB** |

### BN-γ 扫参结果（选用依据）

流程均为：**BN-γ → 融合无偏置 → 非结构化 30% → 整网 INT8**。  
验收：**精确率 > 0.8 且召回率 > 0.6**。

| BN-γ | INT8 F1 | 精确率 | 召回率 | τ | packed | 达标 |
|------|---------|--------|--------|---|--------|------|
| 0.10 | 0.7504 | 0.9535 | 0.6186 | 0.56 | 2.53 MB | 是 |
| 0.20 | 0.7504 | 0.9535 | 0.6186 | 0.56 | 2.08 MB | 是 |
| 0.30 | 0.7553 | 0.9184 | 0.6414 | 0.58 | 1.67 MB | 是 |
| **0.40（选用）** | **0.7559** | **0.9244** | **0.6394** | **0.50** | **1.30 MB** | **是** |
| 0.50 | 0.7504 | 0.9535 | 0.6186 | 0.64 | 0.98 MB | 是 |

五档均达标；选用 **0.40**：INT8 F1 最高，召回仍稳，体积相对仅非结构化包（≈3.04 MB）明显更小。

完整说明与原始 CSV：[`evidence/notes/BN通道剪枝比例对比说明.md`](evidence/notes/BN通道剪枝比例对比说明.md)、[`evidence/sweep/compare.csv`](evidence/sweep/compare.csv)。流水线已支持 `--bn-channel-ratio`。

### 阶段指标（测试集）

| 阶段 | F1 | 精确率 | 召回率 | 备注 |
|------|-----|--------|--------|------|
| FP32 训练权重 | 0.7670 | 0.8216 | 0.7192 | τ=0.70 |
| BN-γ 0.40 + 微调 | ≈0.76 | — | — | 验证重搜 τ |
| 融合丢偏置 | 明显下降 | — | — | 属预期 |
| +非结构化 30% + 微调 | 0.7560 量级 | — | — | 验证重搜得 τ=0.5 |
| **INT8 交付** | **0.7559** | **0.9244** | **0.6394** | 达标 |

---

## 2. 目录结构

```text
delivery/bnch040-u30-symint8-20260827/
├── README.md
├── 量化方法与输出后处理说明.md
├── manifest.json
├── model/          # packed、阈值、layer_quant_specs、channel_masks、bias_audit
├── data/           # X/y 与 X_*_int8
├── evidence/       # 训练 / 融合剪枝量化 / 比例扫参 / INT8 复核
└── metrics/
```

**主模型**：`model/grid_cnn_w8a8_packed.pt`（含 `channel_masks`，加载时需按 masks 重建通道拓扑）。

---

## 3. 使用提示

```bash
export PYTHONPATH=src
python3 -m dram2cnn.pipelines.run_infer_int8 \
  configs/experiments/exp_full_f1_nobias_apstop_allint8_ddp.yaml \
  --pack delivery/bnch040-u30-symint8-20260827/model/grid_cnn_w8a8_packed.pt \
  --threshold 0.5
```

自定义实现时：

1. 按 `channel_masks.json` 重建各层输出通道与 `head.0` 输入维  
2. 全层无偏置；权重/激活为逐层对称 INT8  
3. Softmax 在图外；正类概率与 **τ=0.5** 比较  

输入量化 scale ≈ `0.0642202557541254`（见 `input_quant_params.json`）。

---

## 4. 自检

- [ ] `bias_audit.json`：`ok_no_bias` 与 `all_layers_int8` 为真  
- [ ] 存在 `channel_masks.json` 且与 packed 一致  
- [ ] 部署 τ=0.5；测试 P>0.8、R>0.6  
- [ ] 已阅读比例对比说明，知为何选 0.40  

## 5. 复现

```bash
bash scripts/fuse_prune_quantize.sh \
  configs/experiments/exp_full_f1_nobias_apstop_allint8_ddp.yaml \
  --train-dir runs/exp_tianchi_full_f1_nobias_apstop_allint8_ddp/20260827_092834_346804 \
  --sparsity 0.3 --bn-channel-ratio 0.4 --finetune-epochs 5
```
