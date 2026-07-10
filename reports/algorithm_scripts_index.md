# 算法脚本与流水线入口索引

与《基于CNN的内存故障预测算法设计报告》第 6 章对应。

## 1. 脚本入口

| 脚本 | 用途 | 默认/典型实验 YAML |
|------|------|-------------------|
| `scripts/build_index.sh smoke\|full` | 生成索引 CSV | — |
| `scripts/preprocess.sh [exp]` | 生成 processed 并切分 splits | `exp_baseline.yaml` / `exp_full_f1_ddp.yaml` |
| `scripts/resplit.sh [exp]` | 仅重切分（复用 processed） | `exp_full_f1_sid.yaml` |
| `scripts/validate.sh [exp]` | 配置与张量形状校验 | 同实验 yaml |
| `scripts/train.sh [exp]` | 单卡训练 | `exp_baseline.yaml` |
| `scripts/train_ddp.sh [exp]` | 多卡分布式训练 | `exp_full_f1_ddp.yaml` |
| `scripts/infer.sh [exp]` | 测试集推理 | 同训练实验 |
| `scripts/evaluate.sh [exp]` | 指标与图表（部署阶段评价指标） | 同训练实验 |
| `scripts/autotune.sh` | 冒烟自动化 预处理→训练→推理→评估 | 内置 baseline |
| `scripts/run_full_notebook_batch.sh` | 全量：先工程实验 → 再 F1 六组 | 见脚本头注释 |
| `scripts/run_full_pipeline_cli.sh` | 全量：先 F1 → 再工程实验 → notebooks | 见脚本头注释 |
| `scripts/run_full_all_notebooks.sh` | 仅执行全部 notebooks（全量开关开启） | — |

## 2. Python 流水线模块（`src/dram2cnn/pipelines/`）

| 模块 | 功能 |
|------|------|
| `run_build_index.py` | 索引表生成 |
| `run_preprocess.py` | 生成 `processed/`、切分 `splits/` |
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

## 4. 全量 F1 主评价指标（验收目标 F1≥0.6）

主实验配置为 `exp_full_f1_ddp`（时间分层 + 多卡）；服务器划分对照为 `exp_full_f1_sid`。正式指标见 `runs/exp_tianchi_full_f1_ddp/infer_*/metrics.json` 部署阶段评价段及 `outputs/tables/full_all_strategies_deployment_kpi.csv`。权重：`runs/<run_name>/<训练会话>/checkpoints/grid_cnn.pt`（可用 `latest_train`）。

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

sid 泛化对照（复用同一预处理输出）：`bash scripts/resplit.sh configs/experiments/exp_full_f1_sid.yaml` 后按同实验配置执行训练、推理与评估。
