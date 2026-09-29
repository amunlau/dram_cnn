# 融合无偏置 → 剪枝 → W8A8 摘要

- 流程：`fuse_nobias_prune_w8a8`
- 目标稀疏度：**0.3**
- 剪枝后微调轮数：**5**
- 强制无偏置：**True**
- 保护主干首层：**False**（False=该层 INT8）
- 保护分类末层：**False**（False=该层 INT8）
- 整网全 INT8：**True**
- 量化：逐层对称 signed INT8（zp=0；非逐通道、非 uint8）
- 耗时（秒）：176.3

## 各阶段测试指标

| 阶段 | F1@deployτ | F1@valτ | P@valτ | R@valτ |
|------|------------|---------|--------|--------|
| FP32 训练权重 | 0.7670 | 0.7670 | 0.8216 | 0.7192 |
| 融合+无 bias | 0.2526 | 0.4365 | 0.2792 | 1.0000 |
| 剪枝(+微调) | 0.7369 | 0.7629 | 0.8368 | 0.7011 |
| INT8 packed | 0.7570 | — | 0.9263 | 0.6401 |

- INT8 部署阈值 τ=0.52
- packed: `/root/workspace/runs/exp_tianchi_full_f1_nobias_apstop_allint8_ddp/20260827_092834_346804/fuse_prune_quant_20260827_093937_648424/w8a8/grid_cnn_w8a8_packed.pt`
