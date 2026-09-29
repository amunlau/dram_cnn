# 过程证据目录（训练 / 推理 / 量化）

本目录存放与交付模型对应的**可追溯过程材料**。  
**部署仍以 `../model/` 与 `../data/` 为准**；本目录内容不必加载进推理链路。

本次交付要点：**整网全 INT8 + 逐层对称 signed INT8**；训练早停用验证 **PR-AUC**，最优在第 **3** 轮。

---

## 目录结构

```text
evidence/
├── README.md
├── train/                             # ① 无偏置三卡训练
├── infer_eval/                        # ② 单精度测试集推理与评估
├── fuse_prune_quant/                  # ③ 融合无偏置 → 剪枝 → 整网逐层对称 INT8
├── infer_int8/                        # ④ packed INT8 参考推理复核（τ=0.52）
├── notes/
│   ├── 训练早停与第3轮最优说明.md
│   └── metrics_snapshot.json
└── logs_archive/
```

---

## 建议阅读顺序

1. `train/session_meta.txt` 与 `train/best_val_metrics.json` → 最优在第 3 轮（PR-AUC）  
2. `train/figures/val_avg_precision_curve.png` 与 `epoch_metrics.csv`  
3. `notes/训练早停与第3轮最优说明.md`  
4. `infer_eval/metrics.json` → 单精度测试集 F1≈0.7670（τ=0.70）  
5. `fuse_prune_quant/summary.md` → 各阶段指标与整网 INT8 结果  
6. `infer_int8/metrics_int8.json` → 与量化报告一致的 F1≈0.7570（τ=0.52）

---

## 关键数字速查

| 阶段 | 位置 | 要点 |
|------|------|------|
| 训练最优 | `train/best_val_metrics.json` | 第 **3** 轮，验证 PR-AUC≈**0.849**；训练阶段搜阈 τ=**0.70**；patience=50，第 **53** 轮停 |
| 单精度测试 | `infer_eval/metrics.json` | 部署 F1≈**0.7670** |
| 融合并去掉偏置 | `fuse_prune_quant/summary.md` | 未微调时 F1 会明显下降（属预期） |
| 剪枝 + 微调 | 同上 | 验证集重搜阈值后 F1≈**0.763**，τ=**0.52** |
| 整网 INT8 交付 | `w8a8/quantization_report.json` 与 `../../model/` | F1≈**0.7570**，打包约 3.04MB，**无偏置、无 FP32 保护层、对称 signed INT8** |

---

## 注意事项

- 中间权重 `*.pt` 多为与仓库 `runs/` 的硬链接。  
- 部署权重为 **`../model/grid_cnn_w8a8_packed.pt`**。  
- 核对整网 INT8：`protect_stem=false`、`protect_head_out=false`，`quant_scheme.not_uint8=true`。  
- 「第 3 轮最优」不是只训 3 轮；见 `notes/训练早停与第3轮最优说明.md`。
