# 融合无偏置 → 剪枝 → W8A8 摘要

- 流程：`fuse_nobias_prune_w8a8`
- 目标稀疏度：**0.3**
- 剪枝后微调轮数：**5**
- 强制无偏置：**True**
- 保护主干首层：**False**（False=该层 INT8）
- 保护分类末层：**False**（False=该层 INT8）
- 整网全 INT8：**True**
- 量化：逐层对称 signed INT8（zp=0；非逐通道、非 uint8）
- 耗时（秒）：178.8

## 各阶段测试指标

| 阶段 | F1@deployτ | F1@valτ | P@valτ | R@valτ |
|------|------------|---------|--------|--------|
| FP32 训练权重（验证搜阈 F1 第 2 轮最优；patience=50，第 52 轮才停） | 0.7679 | 0.7679 | 0.8291 | 0.7151 |
| 融合+无 bias | 0.2345 | 0.4363 | 0.2790 | 1.0000 |
| 剪枝(+微调) | 0.7236 | 0.7676 | 0.8654 | 0.6897 |
| INT8 packed | 0.7488 | — | 0.9375 | 0.6233 |

- INT8 部署阈值 τ=0.56
- packed: `/root/workspace/runs/exp_tianchi_full_f1_nobias_allint8_ddp/20260818_193250_910883/fuse_prune_quant_20260818_194524_730260/w8a8/grid_cnn_w8a8_packed.pt`
