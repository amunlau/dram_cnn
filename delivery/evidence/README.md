# 过程证据目录（训练 / 推理 / 量化）

本目录存放与交付模型对应的**可追溯过程材料**。  
**部署仍以 `../model/` 与 `../data/` 为准**；本目录内容不必加载进推理链路。

本次交付要点：**整网全 INT8 + 逐层对称 signed INT8**（主干首层与分类末层均量化；zp=0；非 uint8、非逐通道）。

---

## 目录结构

```text
evidence/
├── README.md
├── train/                             # ① 无偏置三卡训练
│   ├── session_meta.txt
│   ├── config_snapshot.json
│   ├── train_*.log
│   ├── epoch_metrics.csv
│   ├── best_val_metrics.json          # 最优轮次（第 2 轮）
│   ├── best_threshold.json            # 训练阶段验证集最优阈值（τ=0.76；非最终部署阈值）
│   ├── val_predictions.csv
│   └── figures/
├── infer_eval/                        # ② 单精度测试集推理与评估
│   ├── metrics.json                   # 部署口径主指标（F1≈0.7679）
│   ├── confusion_matrix.json
│   ├── predictions.csv / pr_curve.csv
│   └── figures/
├── fuse_prune_quant/                  # ③ 融合无偏置 → 剪枝 → 整网逐层对称 INT8
│   ├── fuse_prune_quantize.log
│   ├── summary.md / summary.json
│   ├── fused_nobias_fp32.pt
│   ├── fused_nobias_pruned_fp32.pt
│   └── w8a8/
│       ├── quantization_report.json
│       ├── test_predictions_compare.csv
│       ├── fp32_grid_cnn.pt / grid_cnn_w8a8_ptq.pt
│       └── grid_cnn_w8a8_packed.pt    # 与 ../../model/ 同内容（硬链接）
├── notes/
│   ├── 训练早停与第2轮最优说明.md
│   └── metrics_snapshot.json
└── logs_archive/                      # 控制台完整归档
```

---

## 建议阅读顺序

1. `train/session_meta.txt` 与 `train/best_val_metrics.json` → 最优在第 2 轮、早停在第 52 轮（patience=50）  
2. `train/figures/val_best_f1_curve.png` 与 `epoch_metrics.csv`  
3. `notes/训练早停与第2轮最优说明.md`  
4. `infer_eval/metrics.json` → 单精度测试集 F1≈0.7679  
5. `fuse_prune_quant/summary.md` → 各阶段指标与 **整网逐层对称 INT8** 交付结果  
6. `fuse_prune_quant/w8a8/quantization_report.json` → `protect_layers.stem/head.3` 均为 `false`，`scheme=W8A8_per_layer_symmetric_int8`  
7. `infer_int8/` → 对交付 packed 模型的独立 INT8 参考推理复核（F1 应与量化报告一致）

---

## 关键数字速查

| 阶段 | 位置 | 要点 |
|------|------|------|
| 训练最优 | `train/best_val_metrics.json` | 第 **2** 轮，验证搜阈 F1≈**0.760**，τ=0.76；patience=**50**，第 **52** 轮才停。**不是只训了 2 轮**，见 `notes/训练早停与第2轮最优说明.md` |
| 单精度测试 | `infer_eval/metrics.json` | 部署 F1≈**0.7679** |
| 融合并去掉偏置 | `fuse_prune_quant/summary.md` | 未微调时 F1 会明显下降（属预期） |
| 剪枝 + 微调 | 同上 | 验证集重新搜索阈值后 F1≈**0.768**，τ 调整为 **0.56** |
| 整网 INT8 交付 | `w8a8/quantization_report.json` 与 `../../model/` | F1≈**0.7488**，打包约 3.04MB，**无偏置、无 FP32 保护层、对称 signed INT8** |

---

## 注意事项

- 中间权重 `*.pt` 多为与仓库 `runs/` 的**硬链接**。  
- 部署权重为 **`../model/grid_cnn_w8a8_packed.pt`**。  
- 核对整网 INT8：打包文件中全部 `packed_weights.*.qweight.dtype == int8`，`protect_stem=false`、`protect_head_out=false`，`quant_scheme.not_uint8=true`。  
- **第 2 轮最优属预期**：训练损失继续下降 ≠ 验证更好；`f1@0.5` 后期升高 ≠ 应改用后期权重。完整说明见 `notes/训练早停与第2轮最优说明.md`。
