# 融合无偏置 → 剪枝 → W8A8 摘要

- 流程：`bn_channel_fuse_nobias_unstructured_w8a8`
- BN-γ 通道剪枝比例：**0.4**
- 非结构化稀疏度：**0.3**
- BN-γ 后微调轮数：**5**
- 非结构化后微调轮数：**5**
- 强制无偏置：**True**
- 保护主干首层：**False**（False=该层 INT8）
- 保护分类末层：**False**（False=该层 INT8）
- 整网全 INT8：**True**
- 量化：逐层对称 signed INT8（zp=0；非逐通道、非 uint8）
- 耗时（秒）：308.8

## 各阶段测试指标

| 阶段 | F1@deployτ | F1@valτ | P@valτ | R@valτ |
|------|------------|---------|--------|--------|
| FP32 训练权重 | 0.7670 | 0.7670 | 0.8216 | 0.7192 |
| BN-γ 通道剪枝(+微调) | 0.7606 | 0.7595 | 0.8305 | 0.6997 |
| 融合+无 bias | 0.2520 | 0.4363 | 0.2790 | 1.0000 |
| 非结构化剪枝(+微调) | 0.7586 | 0.7638 | 0.8516 | 0.6924 |
| INT8 packed | 0.7559 | — | 0.9244 | 0.6394 |

- INT8 部署阈值 τ=0.5
- packed: `/root/workspace/runs/exp_tianchi_full_f1_nobias_apstop_allint8_ddp/20260827_092834_346804/fuse_prune_quant_bnch_bn040_u30/w8a8/grid_cnn_w8a8_packed.pt`
