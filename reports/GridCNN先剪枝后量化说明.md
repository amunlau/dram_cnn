# GridCNN 先剪枝后量化说明

**流水线**：FP32 训练 → 剪枝（±微调）→ W8A8 PTQ → packed INT8 参考推理。  
**不包含**：先量化后剪枝。

## 数据与评估口径（强制）

- **切分**：实验 YAML `split_strategy: time_stratified`，目录 `data/splits`（全量索引 `features_index_h300s_step300s_full.csv`）。
- **规模**：train/val/test = **24958 / 5348 / 5348**（正样本 test=1492）；非 `splits_smoke`。
- **主指标**：各模型在 **验证集** 上按训练同款 `threshold_search(objective=f1)` 重搜阈值（valτ），再在 **全量测试集** 上报告 F1/P/R。
- **对照**：表中「固定τ下P/R」强制使用基线部署 τ（本实验 **0.84**）。量化后概率更保守时，固定高阈值下可出现 **P=1.0 且 R 偏低**（FP=0），这是阈值错配，不是算错或数据泄漏。

## 入口

```bash
# 1) 剪枝对比（若尚未跑过）
bash scripts/prune.sh configs/experiments/exp_full_f1_ddp.yaml --finetune-epochs 5

# 2) 对上述产物逐个量化
bash scripts/prune_quantize.sh configs/experiments/exp_full_f1_ddp.yaml
```

产物：`<train_dir>/prune_then_quant_<ts>/`

| 文件 | 内容 |
|------|------|
| `summary.md` | 完整对比大表（主指标=valτ） |
| `comparison_table.csv` | 同表 CSV（含 `*_at_deploy_tau`） |
| `summary.json` | 机器可读（含 `data_audit`） |
| `*/metrics_val_retuned.json` | val/test 双口径混淆矩阵 |
| `prune_quantize.log` | 运行日志 |

## 实现要点

- 结构化剪枝模型通过 `channel_masks.json` 重建拓扑后再 PTQ。  
- packed INT8 写入 `channel_masks`，`infer_int8` 加载时可还原通道。  
- 剪枝与先剪枝后量化流水线默认均以 **val 重搜 τ** 作为主表指标。

归档报告见同目录 `GridCNN先剪枝后量化对比报告.md`。
