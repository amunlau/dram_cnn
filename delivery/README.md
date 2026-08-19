# 交付说明（整网逐层对称 INT8 部署）

本文档对应本目录 `delivery/`，用于加载**无偏置、整网 W8A8** 的整数推理模型，完成推理与性能测试。

关键约束：**主干首层（stem）与分类末层（head.3）不再保持单精度，全部卷积/全连接层均为 INT8。**  
量化方法：**逐层（per-tensor）对称量化到 signed INT8**（zero-point=0，整数域 `[-128, 127]`）。**不是 uint8，不是逐通道量化。**

---

## 1. 交付内容一览

```text
delivery/
├── README.md                          # 本说明（部署入口）
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
│   ├── X_{test,val}.npy / X_*_int8.npy / y_*.npy
├── evidence/                          # ★ 训练/推理/量化过程证据
│   ├── README.md
│   ├── train/
│   ├── infer_eval/
│   ├── fuse_prune_quant/
│   ├── notes/
│   └── logs_archive/
└── metrics/                           # 兼容跳转至 evidence/
```

**主交付模型**：`model/grid_cnn_w8a8_packed.pt`（非结构化剪枝约 30% + **整网逐层对称 signed INT8**；融合后强制无偏置）。

> **模型结构**：[`model/模型结构说明.md`](model/模型结构说明.md)  
> **过程证据**：[`evidence/README.md`](evidence/README.md)  
> **ONNX 导出**（可选）：[`onnx/`](onnx/) — `grid_cnn_w8a8_pruned_allint8.onnx` 与 [`onnx/模型结构详细说明.md`](onnx/模型结构详细说明.md)
> **训练补充**（为何验证最优出现在第 2 轮）：  
> [`evidence/notes/训练早停与第2轮最优说明.md`](evidence/notes/训练早停与第2轮最优说明.md)。

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
| 部署阈值 | **τ = 0.56**（见 `deploy_threshold.json`） |
| 体积 | 打包后约 **3.04 MB** |

### 关键指标（测试集）

| 阶段 | F1 | 精确率 | 召回率 | 备注 |
|------|-----|--------|--------|------|
| 单精度、无偏置训练权重 | **0.7679** | 0.8291 | 0.7151 | τ=0.76（验证集按 F1 搜索；**最优权重来自第 2 轮**） |
| 融合后立即去掉偏置（未微调） | 0.4363 | — | — | 验证集重新搜索阈值后；指标明显下降属预期 |
| 剪枝 30% + 微调 5 轮 | 0.7676 | 0.8654 | 0.6897 | 验证集重新搜索阈值 τ=0.56 |
| **整网 INT8 打包模型（交付）** | **0.7488** | **0.9375** | **0.6233** | 部署阈值 **τ=0.56**；stem/head 均为 INT8 |

训练会话：`runs/exp_tianchi_full_f1_nobias_allint8_ddp/20260818_193250_910883`  
融合—剪枝—量化：`.../fuse_prune_quant_20260818_194524_730260`

---

## 3. 数据说明（可直接输入模型）

### 3.1 形状与含义

| 文件 | 数据类型 | 形状 | 说明 |
|------|----------|------|------|
| `X_{test,val}.npy` | float32 | `(5348, 31, 128, 32)` | 已归一化，**可直接作为模型输入** |
| `X_{test,val}_int8.npy` | **int8** | 同上 | 按 `input_quant_params.json` **对称量化**后的激活（供整数通路联调） |
| `y_{test,val}.npy` | int64 | `(5348,)` | 标签：`0` 负类 / `1` 正类 |

31 个通道 = 16 个 bank 热图通道 + 15 个硬件库存属性通道；空间网格为 128×32。

**不要**再对 `X_*.npy` 做额外标准化。

### 3.2 为何使用 `.npy`，以及怎么处理

1. **与训练流程一致**：交付文件即已切分好的验证集/测试集张量。  
2. **保留多维数组信息**：形状 `(样本数, 通道, 高, 宽)` 自带于 `.npy` 头。  
3. **体积更小、读取更快**；可用内存映射。  
4. **便于跨语言对接**：可按 NumPy 规范解析，或导出裸二进制。

```python
import numpy as np
X = np.load("data/X_test.npy")            # float32, (5348, 31, 128, 32)
y = np.load("data/y_test.npy")            # int64
X8 = np.load("data/X_test_int8.npy")      # int8（signed，不是 uint8）
```

### 3.3 输入量化公式（对称 signed INT8）

见 `model/input_quant_params.json`（stem 层激活参数，`zero_point=0`）：

```text
q = clip(round(x_fp32 / scale), -128, 127)   # int8
x_hat = scale * q
```

当前 `scale = 0.0642202557541254`。

说明：本次 **stem 已量化为 INT8**，整数通路应从输入激活量化开始进入主干首层。参考推理仍可用单精度 `X_*.npy`（实现内部再量化）。

---

## 4. 推荐使用方法

### 4.1 用本仓库参考 INT8 推理（校验数值）

```bash
export PYTHONPATH=src
python3 -m dram2cnn.pipelines.run_infer_int8   configs/experiments/exp_full_f1_nobias_allint8_ddp.yaml   --pack delivery/model/grid_cnn_w8a8_packed.pt   --threshold 0.56
```

### 4.2 自定义推理实现（最小步骤）

1. 加载 `grid_cnn_w8a8_packed.pt`（`torch.load(..., weights_only=False)`）  
2. 读取打包权重（**全部层** int8 权重 + **逐层标量**缩放，`zero_point=0`）与各层量化说明  
3. **确认无偏置**；**确认无 `keep_fp32` 层**（`bias_audit.json` 中 `all_layers_int8` 为真）  
4. 按层前向：激活对称量化到 INT8 → 整型卷积/全连接再乘 `act_scale * weight_scale` → 融合单元内的 ReLU  
5. 得到正类概率后，与 **τ=0.56** 比较得到 0/1 预测  
6. 用 `data/y_test.npy` 计算精确率、召回率与 F1

### 4.3 验证集

联调或在验证集上搜索阈值可用 `data/X_val.npy` 与 `y_val.npy`。  
**正式验收请以测试集 + 固定 τ=0.56 为准**。

---

## 5. 注意事项

1. **无偏置是硬性约束**：请勿再人为加回偏置。  
2. **整网全 INT8**：与「保护首尾层为单精度」的旧交付不同；请按 `layer_quant_specs.json` 对 **stem / head.3** 同样做 INT8。  
3. **对称 signed INT8**：不要按 uint8 / 非对称 / 逐通道实现；`act_in_zp` 全部为 0。  
4. **阈值**：交付部署阈值为 **0.56**；训练阶段 τ=0.76 仅作对照（对应第 2 轮 FP32 权重）。  
5. **随机失活（Dropout）**：推理阶段必须关闭。  
6. **数据切分**：时间分层全量切分，与既有全量实验一致。  
7. **硬链接**：`X_*.npy` / `y_*.npy` 可能与仓库 `data/splits/` 同 inode。  
8. **过程证据**：见 `evidence/`。  
9. **性能对照**：本仓库 INT8 参考推理保证数值语义，不能当作专用加速引擎的耗时基准。  
10. **训练第 2 轮即为最优（属预期，不是只训了 2 轮）**：早停盯验证集搜阈 F1，`patience=50`，实际训练到第 **52** 轮才停；磁盘权重是历史最优（第 2 轮）。训练损失后来仍下降，但验证 AUC / 搜阈 F1 回落，属过拟合。独立测试集部署 F1≈**0.7679**，与历史全量一致。完整论证见 [`evidence/notes/训练早停与第2轮最优说明.md`](evidence/notes/训练早停与第2轮最优说明.md)。

---

## 6. 快速自检清单

- [ ] 已阅读 `model/模型结构说明.md`  
- [ ] `bias_audit.json` 中 `ok_no_bias` 与 `all_layers_int8` 均为真  
- [ ] `layer_quant_specs.json` 中 `stem` / `head.3` 的 `keep_fp32` 均为 `false`，且 `act_in_zp=0`、`weight_scale` 长度为 1  
- [ ] packed `format` 为 `gridcnn_w8a8_packed_v3`，`qweight.dtype` 为 **int8**（不是 uint8）  
- [ ] `X_test.npy` 形状为 `(5348, 31, 128, 32)`，`X_test_int8.npy` dtype 为 **int8**  
- [ ] 使用 `deploy_threshold.json` 中的 τ=0.56  
- [ ] 测试集 F1 应接近 **0.75**（允许实现差异）  
- [ ] 已了解「第 2 轮最优」：不是只训 2 轮；见 `evidence/notes/训练早停与第2轮最优说明.md`  
- [ ] 输入未再次做均值/方差归一化  
- [ ] 推理关闭随机失活；按融合后的「卷积 + ReLU」实现  

---

## 7. 复现入口（训练与量化）

| 步骤 | 命令 / 路径 |
|------|-------------|
| 实验配置 | `configs/experiments/exp_full_f1_nobias_allint8_ddp.yaml` |
| 三卡训练 | `bash scripts/train_ddp.sh configs/experiments/exp_full_f1_nobias_allint8_ddp.yaml` |
| 推理评估 | `bash scripts/infer.sh` / `bash scripts/evaluate.sh`（同上配置） |
| 融合剪枝量化（默认整网 INT8） | `bash scripts/fuse_prune_quantize.sh ... --sparsity 0.3`（勿加 `--protect-ends`） |

若数值与本文明显不符，请先核对：模型是否为本次 `grid_cnn_w8a8_packed.pt`、阈值是否为 0.56、输入是否为交付的 `X_*.npy`、是否误用了「首尾保护」或 **uint8/逐通道** 的旧打包模型。
