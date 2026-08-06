# GridCNN-Deep W8A8 量化报告

---

**文档编号**：DRAM2CNN-QUANT-W8A8-2026-001  
**项目名称**：天池 DRAM 故障预测 · 空间热图 2D-CNN（dram2cnn）  
**版本**：V1.0  
**编制日期**：2026年7月29日  
**关联实验**：时间分层 F1 主实验 `exp_full_f1_ddp`  
**仓库路径**：`/root/workspace`

---

## 摘要

本报告记录对已验收通过的 **GridCNN-Deep** 全精度模型所做的 **W8A8 Post-Training Quantization（PTQ）** 结果。量化对象为 2026-07-29 时间分层全量训练会话权重；方案为 **BatchNorm Folding + AdaRound 风格逐层权重重构 + 验证集激活校准**，**不使用 QAT**。

在测试集、固定部署阈值 τ=**0.84** 下：

| 指标 | FP32 | W8A8 | 绝对变化 |
|------|------|------|----------|
| **F1** | **0.7528** | **0.7501** | **-0.0027** |
| Precision | 0.8597 | 0.9525 | +0.0928 |
| Recall | 0.6696 | 0.6186 | -0.0509 |
| ROC-AUC | 0.8940 | 0.8739 | -0.0201 |

主目标「部署阶段 F1 绝对掉点 < 0.01」**已达成**。INT8 打包体积约 **3.13 MB**，相对 FP32 权重+偏置约 **3.91×** 压缩。Precision 上升来自固定阈值下误报（FP）大幅减少，属可解释的 PR 权衡，已用混淆矩阵与逐样本预测 CSV 独立复核。

> **工程更新（2026-08）**：仓库已提供 `scripts/infer_int8.sh` / `dram2cnn.engine.infer_int8`，可加载 `grid_cnn_w8a8_packed.pt` 做 **INT8 参考推理**（权重以 int8 驻留；非 TensorRT 内核）。PTQ 流程含 AdaRound 后二次激活校准；复现见 `docs/user_guide.md` §9.1。若重新跑 `quantize.sh`，报告中的数值可能因校准顺序优化而略有变化。
>
> **与剪枝串联实验的口径差异**：本专题报告的主数字为 **固定基线部署 τ=0.84**（与训练 `best_threshold.json` 一致）。「先剪枝后量化」总对比表（`reports/GridCNN先剪枝后量化对比报告.md`）的**主指标**改为各模型 **val 重搜 τ**，并另附固定 τ 对照列；量化后固定高 τ 下可能出现 Precision=1.0（FP=0），属阈值错配对照，见该说明文档。二者勿混读。

**关键词**：W8A8；PTQ；AdaRound；BatchNorm Folding；GridCNN-Deep；部署阶段 F1；INT8 参考推理

---

## 目录

1. [量化背景与目标](#1-量化背景与目标)
2. [量化对象与基线](#2-量化对象与基线)
3. [量化方案设计](#3-量化方案设计)
4. [工程实现与复现](#4-工程实现与复现)
5. [实验结果](#5-实验结果)
6. [Precision 上升原因分析](#6-precision-上升原因分析)
7. [资源与运行环境](#7-资源与运行环境)
8. [结论与建议](#8-结论与建议)
9. [附录：产物路径](#9-附录产物路径)

---

## 1 量化背景与目标

### 1.1 任务与评价口径

本项目将 DRAM 可纠正错误（CE）聚合为空间热图，用二维 CNN 预测未来短时窗内是否出现工单级故障。主验收指标为测试集 **部署阶段 F1**：阈值在验证集上按 F1 搜索得到后，**固定**用于测试集；禁止用测试集事后搜阈作为验收结论。

量化评估沿用同一口径，确保与全精度主实验可比。

### 1.2 量化目标

| 项 | 要求 |
|----|------|
| 方案 | 仅高质量 PTQ；禁止端到端 QAT |
| 精度格式 | W8A8（权重 8-bit + 激活 8-bit） |
| 权重粒度 | Per-channel 对称量化 |
| 激活粒度 | Per-tensor 量化（本实现为非对称，含零点） |
| 前置 | 必须 BatchNorm Folding |
| 精度目标 | 部署阶段 F1 绝对掉点 < 0.01（相对本次 FP32 基线） |
| 特殊层 | Stem 首层、分类末层 Linear 可保持更高精度 |

### 1.3 为何做量化

在保持部署 F1 基本不掉的前提下，降低权重存储体积，为边缘/高吞吐推理或后续 TensorRT/ORT INT8 部署提供可复现的 PTQ 基线。

---

## 2 量化对象与基线

### 2.1 网络结构

| 项 | 值 |
|----|-----|
| 架构 | GridCNN-Deep（类 VGG 纯卷积，**无残差**） |
| 定义 | `src/dram2cnn/models/grid_cnn.py` |
| 配置 | `configs/model.yaml` |
| 输入 | `[N, 31, 128, 32]` |
| Stem | 64 通道 |
| 骨干通道 | `[64, 128, 256, 256]`，`blocks_per_stage=2` |
| Conv2d 层数 | **11**（含 Stem 与通道投影） |
| 可训练参数 | **3,184,450** |
| BatchNorm | 有（`use_batch_norm=true`） |
| 分类头 | GAP → Linear(256) → ReLU → Dropout → Linear(2) |

### 2.2 全精度权重来源

| 项 | 路径 / 值 |
|----|-----------|
| 实验 | `configs/experiments/exp_full_f1_ddp.yaml` |
| 切分 | `time_stratified`（时间分层） |
| 训练会话 | `runs/exp_tianchi_full_f1_ddp/20260729_170226_369953` |
| 权重 | `…/checkpoints/grid_cnn.pt` |
| 部署阈值 | `best_threshold.json` → **τ=0.84** |
| 数据 | `data/splits/`（与 F1 主路径预处理一致，N≈35654） |

### 2.3 FP32 部署基线（本次量化前独立复测）

测试集、τ=0.84：

| 指标 | 数值 |
|------|------|
| F1 | 0.752826 |
| Precision | 0.859725 |
| Recall | 0.669571 |
| ROC-AUC | 0.893995 |
| Accuracy | 0.877337 |
| 混淆矩阵 | TP=999, FP=163, FN=493, TN=3693 |

该基线与同日全量训练后的 `infer_*/metrics.json` 部署段一致（F1≈0.7528）。

---

## 3 量化方案设计

### 3.1 总体流程

```text
FP32 GridCNN（含 BN）
    → BatchNorm Folding（Conv+BN → Conv+bias，再接 ReLU）
    → 验证集采样校准激活范围（Per-tensor）
    → 逐层 AdaRound 风格权重重构（Per-channel INT8）
    → Fake-quant 前向（W8A8 仿真）
    → 测试集部署评价（固定 τ=0.84）
    → 导出 PTQ 状态 / INT8 打包权重 / 报告与日志
```

**不做 QAT**；AdaRound 仅在校准子集上对单层做短时重构优化（layer-wise），不属于全网重新训练。

### 3.2 BatchNorm Folding

训练结构为 `Conv → BN → ReLU`。折叠将 BN 的 scale/shift 并入卷积权重与偏置，得到 `Conv(bias) → ReLU`。折叠后 FP32 指标与折叠前一致（F1 同为 0.752826），验证折叠正确。

### 3.3 权重量化（W8）

- 位宽：INT8  
- 粒度：按输出通道 **Per-channel**  
- 对称性：**对称**（zero_point=0）  
- 舍入：AdaRound 风格 soft→hard；每层 400 step，目标为层输出 MSE + 退火正则  

### 3.4 激活量化（A8）

- 位宽：INT8  
- 粒度：**Per-tensor**  
- 范围：验证集激活 99.9% 分位，并强制包含 0（适配 ReLU 后非负特征）  
- 对称性：**非对称**（含 zero_point）  
- 校准样本：验证集无放回随机 **1024** 条（seed=42）  

### 3.5 敏感层保护

| 层 | 处理 |
|----|------|
| Stem（首层卷积） | 保持 **FP32** |
| `head.3`（末层分类 Linear） | 保持 **FP32** |
| 其余卷积 / `head.0` | W8A8 |

首层直接接触稀疏热图与 inventory 通道，末层决定 logits；保护二者可显著降低掉点，是 PTQ 常规做法。

### 3.6 评价协议

| 项 | 设定 |
|----|------|
| 测试集 | `data/splits/X_test.npy` 等 |
| 阈值 | 验证集 F1 最优 τ=**0.84**（不重搜） |
| 主指标 | 部署阶段 F1 |
| 辅指标 | Precision、Recall、ROC-AUC、混淆矩阵 |
| 复核 | `test_predictions_compare.csv` 手工按计数重算 P/R/F1 |

---

## 4 工程实现与复现

### 4.1 代码布局（与训练/推理同构）

| 角色 | 路径 |
|------|------|
| PTQ 核心 | `src/dram2cnn/engine/quantize_ptq.py` |
| INT8 参考推理 | `src/dram2cnn/engine/infer_int8.py` |
| 流水线入口 | `src/dram2cnn/pipelines/run_quantize.py`、`run_infer_int8.py` |
| Shell 入口 | `scripts/quantize.sh`、`scripts/infer_int8.sh` |
| 日志工具 | `dram2cnn.utils.logger.get_logger` → `quantize.log` / `infer_int8.log` |

不再将大段算法堆在 `scripts/*.py`；`scripts/` 仅保留薄封装，与 `train.sh` / `infer.sh` / `evaluate.sh` 一致。

### 4.2 复现命令

```bash
cd /root/workspace
export PYTHONPATH=src

bash scripts/quantize.sh configs/experiments/exp_full_f1_ddp.yaml \
  --train-dir runs/exp_tianchi_full_f1_ddp/20260729_170226_369953 \
  --device cuda:0 \
  --calib-size 1024 \
  --adaround-steps 400

# 加载 packed INT8 做参考推理（默认读 .latest_quant_session）
bash scripts/infer_int8.sh configs/experiments/exp_full_f1_ddp.yaml
```

产物写入训练会话下时间戳目录 `quant_w8a8_<timestamp>/`，并通过 `.latest_quant_session` 指针定位最近一次量化；INT8 推理写入 `runs/<run_name>/infer_int8_<timestamp>/`。

### 4.3 本次正式报告所依据的运行

| 项 | 值 |
|----|-----|
| 量化会话 | `quant_w8a8_20260729_173414_971956` |
| 日志 | `…/quantize.log` |
| 机器时间 | 2026-07-29 17:34–17:36（约 2 分钟完成校准+AdaRound+评估） |

---

## 5 实验结果

### 5.1 模型体积对比

| 项 | FP32 | W8A8 |
|----|------|------|
| 参数个数 | 3,184,450 | 同左（存储位宽变化） |
| state_dict 文件 | ≈12.19 MB | 导出含 float 仿真态 ≈12.17 MB |
| **INT8 打包文件** | — | **≈3.13 MB** |
| 理论打包体积 | ≈12.14 MB（权重+偏置） | **≈3.10 MB** |
| **理论压缩比** | 1.0× | **≈3.91×** |

说明：`grid_cnn_w8a8_ptq.pt` 为便于 fake-quant 复现的导出格式；真正体积收益与 INT8 推理入口均以 **`grid_cnn_w8a8_packed.pt`**（INT8 权重 + scale，format=`gridcnn_w8a8_packed_v2`）为准，由 `scripts/infer_int8.sh` 加载。

### 5.2 部署阶段精度对比（τ=0.84）

| 指标 | FP32 | BN-Fold FP32 | W8A8 | 相对 FP32 绝对变化 |
|------|------|--------------|------|-------------------|
| F1 | 0.752826 | 0.752826 | **0.750102** | **-0.002724** |
| Precision | 0.859725 | 0.859725 | **0.952528** | **+0.092804** |
| Recall | 0.669571 | 0.669571 | 0.618633 | -0.050938 |
| ROC-AUC | 0.893995 | 0.893995 | 0.873876 | -0.020119 |
| Accuracy | 0.877337 | 0.877337 | 0.885004 | +0.007666 |
| Avg Precision | 0.847204 | 0.847211 | 0.829395 | -0.017809 |

- F1 相对掉点约 **-0.36%**，绝对掉点 **0.0027 < 0.01**，满足预设目标。  
- BN Folding 不改变部署指标，折叠可信。

### 5.3 混淆矩阵

| | TP | FP | FN | TN |
|--|----|----|----|-----|
| FP32 | 999 | 163 | 493 | 3693 |
| W8A8 | 923 | **46** | 569 | 3810 |
| 变化 | -76 | **-117** | +76 | +117 |

由计数直接计算：

- FP32：P = 999/(999+163) = **0.859725**  
- W8A8：P = 923/(923+46) = **0.952528**  

与 sklearn 指标及 CSV 重算完全一致。

### 5.4 概率分布摘要（测试集）

| 统计量 | FP32 | W8A8 |
|--------|------|------|
| 全体平均正类概率 | 0.4488 | 0.3877 |
| 正样本平均概率 | 0.8146 | 0.7537 |
| 负样本平均概率 | 0.3073 | 0.2461 |
| 预测为正比例（≥0.84） | 0.2173 | 0.1812 |

量化后整体概率下移，在固定高阈值下更“保守”，与 FP↓、召回↓、精确率↑ 一致。

### 5.5 前向耗时（参考）

| 模式 | 测试集前向（含数据加载，秒） |
|------|------------------------------|
| FP32 | ≈4.25 |
| W8A8 fake-quant | ≈3.96 |

注：当前为 CUDA **fake-quant 仿真**，不是 TensorRT/ONNX Runtime 真实 INT8 kernel；时延仅作相对参考，体积压缩比更具工程意义。

---

## 6 Precision 上升原因分析

### 6.1 是否算错？

**否。** 已通过三条路径交叉验证：

1. `classification_metrics`（sklearn）  
2. 混淆矩阵计数公式 `TP/(TP+FP)`、`TP/(TP+FN)`  
3. `test_predictions_compare.csv` 逐样本独立重算  

三条路径给出相同的 P/R/F1。

### 6.2 机制解释

在**固定阈值** τ=0.84 下：

1. 量化引入的误差使正类 softmax 概率整体略降；  
2. 越过阈值的预测正例变少（21.7% → 18.1%）；  
3. 其中 **误报 FP 从 163 降至 46**（降幅大），真阳 TP 从 999 降至 923（降幅相对小）；  
4. 故 Precision 上升、Recall 下降，二者对 F1 近乎抵消 → F1 仅微降。

这是二分类在固定阈值下的常见 PR 权衡，**不代表指标实现错误**，也不意味着模型“变强”；若业务更重视召回，应对量化模型在验证集上**单独重搜阈值**（本报告主口径仍使用原 FP32 部署阈值，以保持与主验收一致）。

---

## 7 资源与运行环境

### 7.1 GPU 使用说明

| 项 | 说明 |
|----|------|
| 可见 GPU | 3× NVIDIA GeForce RTX 4090 |
| 本次量化设备 | **`cuda:0`（单卡）** |
| 是否异常 | **否** |

全量训练使用 DDP 多卡合理；PTQ（尤其 AdaRound 逐层重构）为**单进程、层间串行**优化，数据并行收益极低。业界 PTQ 工具链亦多为单卡校准。换卡示例：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/quantize.sh configs/experiments/exp_full_f1_ddp.yaml \
  --train-dir runs/exp_tianchi_full_f1_ddp/20260729_170226_369953
```

### 7.2 软件栈

- PyTorch（本环境为 CUDA 构建版本）  
- 项目内自研 PTQ（未依赖 MQBench / AIMET 等外部量化套件）  
- 评估：项目既有 `classification_metrics` + 计数复核  

---

## 8 结论与建议

### 8.1 结论

1. 已对时间分层 F1 主实验 GridCNN-Deep 完成 **W8A8 PTQ**（BN Fold + AdaRound 风格），**无 QAT**。  
2. 部署阶段 **F1：0.7528 → 0.7501**（掉点 0.0027），满足 **< 0.01** 目标，且仍远高于验收线 F1≥0.6。  
3. INT8 打包约 **3.13 MB**，理论压缩约 **3.91×**。  
4. Precision 上升、Recall 下降已用混淆矩阵解释并独立复核，属固定阈值下的合理现象。  
5. 工程入口已与训练/推理对齐：`engine` + `pipelines` + `scripts/quantize.sh`，日志写入 `quantize.log`。

### 8.2 后续建议

| 方向 | 说明 |
|------|------|
| 业务阈值 | 若上线更重召回，对 W8A8 模型在验证集重搜 τ 后再报部署指标 |
| 真 INT8 推理 | 将 `grid_cnn_w8a8_packed.pt` 对接 TensorRT / ONNX Runtime 测真实吞吐 |
| 全层 W8A8 | 可用 `--no-protect` 对比掉点，评估取消首末层保护的代价 |
| 校准规模 | 可试更大 val 校准子集或 MSE 观察器，观察 AUC/召回是否回升 |

---

## 9 附录：产物路径

### 9.1 本报告对应量化会话

```text
/root/workspace/runs/exp_tianchi_full_f1_ddp/20260729_170226_369953/quant_w8a8_20260729_173414_971956/
├── quantize.log                      # 完整运行日志
├── quantization_report.json          # 机器可读明细
├── quantization_report.md            # 会话内短报告
├── fp32_grid_cnn.pt
├── grid_cnn_w8a8_ptq.pt              # fake-quant 可加载导出
├── grid_cnn_w8a8_packed.pt           # INT8 打包权重
└── test_predictions_compare.csv      # FP32 vs W8A8 逐样本对比
```

指针文件：`…/20260729_170226_369953/.latest_quant_session`

### 9.2 关联文档

| 文档 | 说明 |
|------|------|
| `reports/基于CNN的内存故障预测算法设计报告.md` | 算法主报告 |
| `reports/算法开发与实验结果.md` | 实验与验收指标说明 |
| `reports/algorithm_scripts_index.md` | 含 `quantize.sh` 入口索引 |
| 本文件 | W8A8 量化专题报告 |

---

**编制说明**：本报告数值均来自 `quant_w8a8_20260729_173414_971956` 独立重跑结果；评价口径为测试集部署阶段（验证集阈值固定），与项目主验收定义一致。
