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
| `scripts/quantize.sh [exp] [--train-dir …]` | GridCNN W8A8 PTQ（BN Fold + AdaRound；导出 packed INT8） | `exp_full_f1_ddp.yaml` |
| `scripts/infer_int8.sh [exp] [--pack …]` | 加载 packed INT8 参考推理 | `exp_full_f1_ddp.yaml` |
| `scripts/prune.sh [exp] [--sparsities …]` | 三种粒度剪枝对比（主指标=val 重搜 τ → test） | `exp_full_f1_ddp.yaml` |
| `scripts/prune_quantize.sh [exp] [--prune-dir …]` | 先剪枝后 W8A8 量化总对比（含固定 τ 对照列） | `exp_full_f1_ddp.yaml` |
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
| `run_quantize.py` | W8A8 PTQ，写 `quant_w8a8_*/`（含 `grid_cnn_w8a8_packed.pt`） |
| `run_infer_int8.py` | 加载 packed INT8，写 `infer_int8_*/predictions.csv` |
| `run_prune.py` | 三种粒度剪枝对比，写 `prune_compare_*/`（主指标 valτ） |
| `run_prune_quantize.py` | 先剪枝后 W8A8 量化，写 `prune_then_quant_*/`（结束后自动 val 重搜覆盖主表） |

## 2.1 剪枝 / 量化评估口径

| 项 | 说明 |
|----|------|
| 数据 | 实验 YAML 的 `splits_dir`；全量 F1 主实验为 `data/splits`（`time_stratified`，train/val/test≈24958/5348/5348） |
| 剪枝 / 先剪枝后量化主指标 | 各模型 **val** 上 `threshold_search(objective=f1)` → **test** 上 F1/P/R |
| 固定基线 τ 对照 | CSV / 表中 `*_at_deploy_tau`；量化后固定高 τ 可能 P=1.0（FP=0），非主验收 |
| 单独 `quantize.sh` | 默认仍用训练 `best_threshold.json` 做部署阶段对比（与 W8A8 专题报告一致） |
| 顺序 | **仅**先剪枝后量化；不支持先量化后剪枝 |

归档：`reports/GridCNN剪枝对比*.md`、`reports/GridCNN先剪枝后量化*.md`、`reports/GridCNN-Deep_W8A8量化报告.md`。

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

主实验配置为 `exp_full_f1_ddp`（时间分层 + 多卡）；服务器划分对照为 `exp_full_f1_sid`。正式指标见 `runs/exp_tianchi_full_f1_ddp/infer_*/metrics.json` 部署阶段评价段及 `outputs/tables/full_all_strategies_deployment_kpi.csv`。权重：`runs/<run_name>/<训练会话>/checkpoints/grid_cnn.pt`（可用 `latest_train`）。checkpoint 是验证搜阈 F1 的历史最优，不是最后一轮；全量 `patience=50` 时高峰常在前几轮（当前交付第 2 轮、第 52 轮早停）。见 `delivery/evidence/notes/训练早停与第2轮最优说明.md`。

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

## 5. 可选：量化 / 剪枝 / 先剪枝后量化

```bash
export PYTHONPATH=src
# 需已有全量训练会话（例：runs/exp_tianchi_full_f1_ddp/20260729_170226_369953）

bash scripts/quantize.sh configs/experiments/exp_full_f1_ddp.yaml
bash scripts/infer_int8.sh configs/experiments/exp_full_f1_ddp.yaml

bash scripts/prune.sh configs/experiments/exp_full_f1_ddp.yaml --finetune-epochs 5
bash scripts/prune_quantize.sh configs/experiments/exp_full_f1_ddp.yaml
# 产物：prune_then_quant_*/summary.md（主指标 valτ；含固定基线 τ 对照）
```
