# 过程证据目录

部署以 `../model/` 与 `../data/` 为准。

本次交付：**BN-γ=0.40 + 非结构化 30% + 融合无偏置 + 整网对称 INT8**。  
INT8 测试：F1≈0.7559，P≈0.9244，R≈0.6394，τ=0.5。

## 建议阅读

1. **`notes/BN通道剪枝比例对比说明.md`** → 五档扫参表与选用 0.40 的理由（必读）  
2. `sweep/compare.csv` → 扫参机器可读汇总  
3. `fuse_prune_quant/summary.md` → 选用流水线各阶段  
4. `infer_int8/metrics_int8.json` → 独立 INT8 复核  

上级入口 [`../README.md`](../README.md) 与 [`../量化方法与输出后处理说明.md`](../量化方法与输出后处理说明.md) 已内嵌同一张扫参表。
