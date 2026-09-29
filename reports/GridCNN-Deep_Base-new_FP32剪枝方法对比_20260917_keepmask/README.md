# GridCNN-Deep「Base-new」FP32 剪枝对比（稀疏保持修订版 · 2026-09-17）

本目录为修正「非结构化微调训没稀疏」后的正式重跑结果。

| 文件 | 说明 |
|------|------|
| [`报告.md`](报告.md) | 正式中文报告 |
| [`compare_metrics.csv`](compare_metrics.csv) | 对比表 |
| [`summary_metrics.json`](summary_metrics.json) | 机器可读汇总 |
| [`experiment_summary.md`](experiment_summary.md) | 流水线简表 |

实验目录：`runs/exp_tianchi_full_f1_ddp/20260917_153434_808743/prune_compare_fp32_bn_filter_uL1_uL2_r010-080_ft10_keepmask/`

上一轮无 mask 的旧结果（非结构化无效）见同会话下 `prune_compare_fp32_bn_filter_uL1_uL2_r010-080_ft10/`，请勿再作稀疏结论依据。
