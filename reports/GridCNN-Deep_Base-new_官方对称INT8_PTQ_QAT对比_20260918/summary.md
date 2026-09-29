# 官方对称 INT8（BN 折叠 + FakeQuant）：FP32 vs PTQ vs QAT

- API：`torch.ao.quantization eager FakeQuantize, symmetric signed INT8 W/A`
- 拓扑：`BN-fold (fuse_modules) then FakeQuant; stem/head_out FP32`
- 保护层：stem / head_out = FP32
- QAT：20 epoch，lr=0.0001，DDP=True×3
- ckpt：`/root/workspace/runs/exp_tianchi_full_f1_ddp/20260917_153434_808743/checkpoints/grid_cnn.pt`

> 指标为 **BN 已折叠** 的对称 signed INT8 FakeQuant 仿真；未 convert 到 FBGEMM 整型算子。 ONNX 为无 BN 的 QDQ 图；packed 权重与 FakeQuant 的 ``round(w/scale)`` 一致。

## 对比表（test @ val 重搜 τ）

| 模型 | F1 | P | R | τ | params | ckpt MB | packed 实测 MB | packed 理论 MB | QDQ ONNX | BN folded |
|------|----|---|---|---|--------|---------|----------------|----------------|----------|-----------|
| FP32 | 0.7679 | 0.8460 | 0.7031 | 0.78 | 3184450 | 12.186 | — | — | None | True |
| PTQ | 0.7639 | 0.8353 | 0.7038 | 0.78 | 3182594 | — | 3.106 | 3.086 | True | True |
| QAT | 0.7525 | 0.8232 | 0.6930 | 0.88 | 3182594 | — | 3.106 | 3.086 | True | True |

## 产物

- `ptq/` / `qat/`：FakeQuant 状态、无 BN 的 QDQ ONNX、审计通过的 packed INT8
