# 交付说明（整网逐层对称 INT8 部署）

本文档对应目录 `delivery/nobias-symint8-apstop-20260827/`，用于加载**无偏置、整网 W8A8** 的整数推理模型，完成推理与性能测试。

关键约束：**主干首层（stem）与分类末层（head.3）不再保持单精度，全部卷积/全连接层均为 INT8。**  
量化方法：**逐层（per-tensor）对称量化到 signed INT8**（zero-point=0，整数域 `[-128, 127]`）。**不是 uint8，不是逐通道量化。**

与旧 `delivery/nobias-symint8-20260818/` 的主要差别：训练早停改用验证 **PR-AUC**（最优第 **3** 轮），融合剪枝后部署阈值 **τ=0.52**，INT8 测试 F1≈**0.7570**。

---

## 1. 交付内容一览

```text
delivery/nobias-symint8-apstop-20260827/
├── README.md                          # 本说明（部署入口）
├── 量化方法与输出后处理说明.md         # ★ 量化语义、整数前向、Softmax 后处理
├── manifest.json                      # 路径、形状、阈值与证据索引
├── model/                             # ★ 部署用模型与量化参数
│   ├── grid_cnn_w8a8_packed.pt
│   ├── 模型结构说明.md
│   ├── deploy_threshold.json
│   ├── input_quant_params.json
│   ├── layer_quant_specs.json
│   ├── quantization_report.json
│   ├── bias_audit.json
│   ├── config_snapshot.json
│   └── train_best_threshold.json
├── data/                              # ★ 可直接输入模型的验证集/测试集张量
├── evidence/                          # ★ 训练/推理/量化过程证据
└── metrics/                           # 兼容跳转至 evidence/
```

**主交付模型**：`model/grid_cnn_w8a8_packed.pt`（非结构化剪枝约 30% + **整网逐层对称 signed INT8**；融合后强制无偏置）。

> **量化与输出后处理（部署必读）**：[`量化方法与输出后处理说明.md`](量化方法与输出后处理说明.md)  
> **模型结构**：[`model/模型结构说明.md`](model/模型结构说明.md)  
> **过程证据**：[`evidence/README.md`](evidence/README.md)  
> **训练补充**（为何验证最优在第 3 轮）：[`evidence/notes/训练早停与第3轮最优说明.md`](evidence/notes/训练早停与第3轮最优说明.md)

> 说明：本次未重新导出 ONNX。结构与历史整网 INT8 交付同构；**上线以本目录 packed INT8 与 τ=0.52 为准**。

---

## 2. 模型要点

| 项 | 内容 |
|----|------|
| 结构 | GridCNN-Deep（主干卷积 + 4 个下采样阶段 + 全局平均池化 + 两层全连接头） |
| 输入 | 形状 `[样本数, 31, 128, 32]`，已完成预处理与归一化 |
| 输出 | 二分类得分 → 经 softmax 得到正类概率 |
| 无偏置 | 卷积层与全连接层**均无偏置**（见 `bias_audit.json`） |
| 融合 | 训练后将「卷积 + 批归一化 + ReLU」融合，并**丢弃**批归一化折合出的偏置项 |
| 剪枝 | 非结构化 L1 剪枝，目标稀疏度约 **30%** |
| 量化 | 训练后量化（W8A8）：**全部层 INT8**（含 stem / head.3）；权值与激活均为**逐层对称 signed int8**（zp=0） |
| 打包格式 | `gridcnn_w8a8_packed_v3` |
| 部署阈值 | **τ = 0.52**（见 `deploy_threshold.json`） |
| 体积 | 打包后约 **3.04 MB** |

### 关键指标（测试集）

| 阶段 | F1 | 精确率 | 召回率 | 备注 |
|------|-----|--------|--------|------|
| 单精度、无偏置训练权重 | **0.7670** | 0.8216 | 0.7192 | τ=0.70（训练阶段验证搜阈；**最优权重来自第 3 轮，早停指标=PR-AUC**） |
| 融合后立即去掉偏置（未微调） | ≈0.4365 | — | — | 验证集重新搜索阈值后；指标明显下降属预期 |
| 剪枝 30% + 微调 5 轮 | 0.7629 | 0.8368 | 0.7011 | 验证集重新搜索阈值 τ=0.52 |
| **整网 INT8 打包模型（交付）** | **0.7570** | **0.9263** | **0.6401** | 部署阈值 **τ=0.52**；stem/head 均为 INT8 |

训练会话：`runs/exp_tianchi_full_f1_nobias_apstop_allint8_ddp/20260827_092834_346804`  
融合—剪枝—量化：`.../fuse_prune_quant_20260827_093937_648424`

---

## 3. 数据说明（可直接输入模型）

### 3.1 形状与含义

| 文件 | 数据类型 | 形状 | 说明 |
|------|----------|------|------|
| `X_{test,val}.npy` | float32 | `(5348, 31, 128, 32)` | 已归一化，**可直接作为模型输入** |
| `X_{test,val}_int8.npy` | **int8** | 同上 | 按 `input_quant_params.json` **对称量化**后的激活 |
| `y_{test,val}.npy` | int64 | `(5348,)` | 标签：`0` 负类 / `1` 正类 |

**不要**再对 `X_*.npy` 做额外标准化。

### 3.2 读取示例

```python
import numpy as np
X = np.load("data/X_test.npy")            # float32, (5348, 31, 128, 32)
y = np.load("data/y_test.npy")            # int64
X8 = np.load("data/X_test_int8.npy")      # int8（signed，不是 uint8）
```

### 3.3 输入量化公式

见 `model/input_quant_params.json`（stem 层激活参数，`zero_point=0`）：

```text
q = clip(round(x_fp32 / scale), -128, 127)   # int8
x_hat = scale * q
```

当前 `scale = 0.0642202557541254`。

---

## 4. 推荐使用方法

### 4.1 用本仓库参考 INT8 推理（校验数值）

```bash
export PYTHONPATH=src
python3 -m dram2cnn.pipelines.run_infer_int8 \
  configs/experiments/exp_full_f1_nobias_apstop_allint8_ddp.yaml \
  --pack delivery/nobias-symint8-apstop-20260827/model/grid_cnn_w8a8_packed.pt \
  --threshold 0.52
```

> 注意：若不传 `--threshold`，脚本可能优先读取训练会话的 `best_threshold.json`（本包为 0.70）。**部署请显式传 0.52**，或与 packed 内阈值一致。

### 4.2 自定义推理实现（最小步骤）

1. 加载 `grid_cnn_w8a8_packed.pt`（`torch.load(..., weights_only=False)`）  
2. 读取打包权重（**全部层** int8 权重 + **逐层标量**缩放，`zero_point=0`）与各层量化说明  
3. **确认无偏置**；**确认无 `keep_fp32` 层**（`bias_audit.json`）  
4. 按层前向：激活对称量化到 INT8 → 整型卷积/全连接再乘 `act_scale * weight_scale` → 融合单元内的 ReLU  
5. Softmax 取正类概率，与 **τ=0.52** 比较得到 0/1 预测  
6. 用 `data/y_test.npy` 计算精确率、召回率与 F1

### 4.3 验证集

联调可用 `data/X_val.npy` 与 `y_val.npy`。  
**正式验收请以测试集 + 固定 τ=0.52 为准**。

---

## 5. 注意事项

1. **无偏置是硬性约束**：请勿再人为加回偏置。  
2. **整网全 INT8**：请按 `layer_quant_specs.json` 对 **stem / head.3** 同样做 INT8。  
3. **对称 signed INT8**：不要按 uint8 / 非对称 / 逐通道实现。  
4. **阈值**：交付部署阈值为 **0.52**；训练阶段 τ=0.70 仅作 FP32 对照。  
5. **随机失活**：推理阶段必须关闭。  
6. **数据切分**：复用既有时间分层全量 `data/splits`（与旧交付同 inode）。  
7. **过程证据**：见 `evidence/`。  
8. **训练第 3 轮即为最优（属预期）**：早停盯验证 PR-AUC，`patience=50`，实际到第 **53** 轮才停。完整说明见 [`evidence/notes/训练早停与第3轮最优说明.md`](evidence/notes/训练早停与第3轮最优说明.md)。

---

## 6. 快速自检清单

- [ ] 已阅读 `量化方法与输出后处理说明.md`  
- [ ] 已阅读 `model/模型结构说明.md`  
- [ ] `bias_audit.json` 中 `ok_no_bias` 与 `all_layers_int8` 均为真  
- [ ] `layer_quant_specs.json` 中 `stem` / `head.3` 的 `keep_fp32` 均为 `false`，且 `act_in_zp=0`  
- [ ] packed `format` 为 `gridcnn_w8a8_packed_v3`，`qweight.dtype` 为 **int8**  
- [ ] 使用 `deploy_threshold.json` 中的 τ=0.52  
- [ ] 测试集 F1 应接近 **0.757**（允许实现差异）  
- [ ] 已了解「第 3 轮最优」说明  

---

## 7. 复现入口（训练与量化）

| 步骤 | 命令 / 路径 |
|------|-------------|
| 实验配置 | `configs/experiments/exp_full_f1_nobias_apstop_allint8_ddp.yaml` |
| 三卡训练 | `bash scripts/train_ddp.sh configs/experiments/exp_full_f1_nobias_apstop_allint8_ddp.yaml` |
| 推理评估 | `bash scripts/infer.sh` / `bash scripts/evaluate.sh`（同上配置） |
| 融合剪枝量化（默认整网 INT8） | `bash scripts/fuse_prune_quantize.sh ... --sparsity 0.3`（勿加 `--protect-ends`） |

若数值与本文明显不符，请先核对：是否为本目录 `grid_cnn_w8a8_packed.pt`、阈值是否为 **0.52**、输入是否为交付的 `X_*.npy`、是否误用了 uint8/逐通道或旧 `delivery/nobias-symint8-20260818/` 权重。
