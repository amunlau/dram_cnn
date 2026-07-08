# 算法脚本与流水线入口索引

与《基于CNN的内存故障预测算法设计报告》第 6 章对应。

## 1. Shell 入口脚本

| 脚本 | 用途 | 默认/典型实验 YAML |
|------|------|-------------------|
| `scripts/build_index.sh smoke\|full` | 生成索引 CSV | — |
| `scripts/preprocess.sh [exp]` | 物化 processed + 切分 splits | `exp_baseline.yaml` / `exp_full_f1_ddp.yaml` |
| `scripts/resplit.sh [exp]` | 仅重切分（复用 processed） | `exp_full_f1_sid.yaml` |
| `scripts/validate.sh [exp]` | 配置与张量形状校验 | 同实验 yaml |
| `scripts/train.sh [exp]` | 单卡训练 | `exp_baseline.yaml` |
| `scripts/train_ddp.sh [exp]` | 多卡 DDP 训练 | `exp_full_f1_ddp.yaml` |
| `scripts/infer.sh [exp]` | 测试集推理 | 同训练实验 |
| `scripts/evaluate.sh [exp]` | 指标与图表（deployment KPI） | 同训练实验 |
| `scripts/autotune.sh` | 冒烟一键 preprocess→train→infer→evaluate | 内置 baseline |

## 2. Python 流水线（`src/dram2cnn/pipelines/`）

| 模块 | 功能 |
|------|------|
| `run_build_index.py` | 索引表生成 |
| `run_preprocess.py` | 物化 `processed/`、切分 `splits/` |
| `run_validate.py` | 实验级校验 |
| `run_train.py` | 训练 GridCNN，写 `best_threshold.json` |
| `run_infer.py` | 加载权重，写 `predictions.csv` |
| `run_evaluate.py` | 读 predictions，写 `metrics.json` |

## 3. 冒烟复现

```bash
cd /path/to/workspace
export PYTHONPATH=src

bash scripts/build_index.sh smoke
bash scripts/preprocess.sh configs/experiments/exp_baseline.yaml
bash scripts/train.sh configs/experiments/exp_baseline.yaml
bash scripts/infer.sh configs/experiments/exp_baseline.yaml
bash scripts/evaluate.sh configs/experiments/exp_baseline.yaml
```

## 4. 全量 F1 主 KPI（验收 F1≥0.6，已通过）

主实验 deployment F1=**0.761**（`exp_tianchi_full_f1_ddp`）；sid 泛化对照 F1=**0.714**。指标见 `runs/exp_tianchi_full_f1_ddp/infer_20260616_224655_159358/metrics.json`。

```bash
export PYTHONPATH=src

bash scripts/build_index.sh full
rm -rf data/processed/* data/splits/*

bash scripts/preprocess.sh configs/experiments/exp_full_f1_ddp.yaml
bash scripts/validate.sh configs/experiments/exp_full_f1_ddp.yaml

CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_ddp.sh configs/experiments/exp_full_f1_ddp.yaml
bash scripts/infer.sh configs/experiments/exp_full_f1_ddp.yaml
bash scripts/evaluate.sh configs/experiments/exp_full_f1_ddp.yaml
```

sid 泛化对照（复用同一 preprocess）：`bash scripts/resplit.sh configs/experiments/exp_full_f1_sid.yaml` 后 train/infer/evaluate 同实验 yaml。
