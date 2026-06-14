#!/usr/bin/env bash
# 全量 Notebook 实验批量：召回 / 精度 / F1（time + sid）
# 用法：bash scripts/run_full_notebook_batch.sh
# 日志：runs/pipeline-logs/full_notebook_batch.log
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
LOG="${PROJECT_ROOT}/runs/pipeline-logs/full_notebook_batch.log"
mkdir -p "${PROJECT_ROOT}/runs/pipeline-logs"
exec > >(tee -a "$LOG") 2>&1

step() { echo "=== $(date '+%F %T') $* ==="; }

run() {
  step "RUN: $*"
  "$@"
}

# ---------- Phase A: 工程向（复用现有 processed + splits）----------
step "Phase A: recall time"
run bash scripts/train_ddp.sh configs/experiments/exp_full_recall_ddp.yaml
run bash scripts/infer.sh configs/experiments/exp_full_recall_ddp.yaml
run bash scripts/evaluate.sh configs/experiments/exp_full_recall_ddp.yaml

step "Phase A: recall sid"
run bash scripts/resplit.sh configs/experiments/exp_full_recall_sid.yaml
run bash scripts/train_ddp.sh configs/experiments/exp_full_recall_sid.yaml
run bash scripts/infer.sh configs/experiments/exp_full_recall_sid.yaml
run bash scripts/evaluate.sh configs/experiments/exp_full_recall_sid.yaml

step "Phase A: precision time"
run bash scripts/train_ddp.sh configs/experiments/exp_full_precision_ddp.yaml
run bash scripts/infer.sh configs/experiments/exp_full_precision_ddp.yaml
run bash scripts/evaluate.sh configs/experiments/exp_full_precision_ddp.yaml

step "Phase A: precision sid"
run bash scripts/resplit.sh configs/experiments/exp_full_precision_sid.yaml
run bash scripts/train_ddp.sh configs/experiments/exp_full_precision_sid.yaml
run bash scripts/infer.sh configs/experiments/exp_full_precision_sid.yaml
run bash scripts/evaluate.sh configs/experiments/exp_full_precision_sid.yaml

# ---------- Phase B: F1 最大化（需重 preprocess）----------
step "Phase B: F1 preprocess (drop_empty_positive=false)"
rm -rf "${PROJECT_ROOT}/data/processed"/* "${PROJECT_ROOT}/data/splits"/* "${PROJECT_ROOT}/data/splits_sid"/* 2>/dev/null || true
run bash scripts/preprocess.sh configs/experiments/exp_full_f1_ddp.yaml

step "Phase B: F1 time"
run bash scripts/train_ddp.sh configs/experiments/exp_full_f1_ddp.yaml
run bash scripts/infer.sh configs/experiments/exp_full_f1_ddp.yaml
run bash scripts/evaluate.sh configs/experiments/exp_full_f1_ddp.yaml

step "Phase B: F1 sid"
run bash scripts/resplit.sh configs/experiments/exp_full_f1_sid.yaml
run bash scripts/train_ddp.sh configs/experiments/exp_full_f1_sid.yaml
run bash scripts/infer.sh configs/experiments/exp_full_f1_sid.yaml
run bash scripts/evaluate.sh configs/experiments/exp_full_f1_sid.yaml

step "ALL DONE $(date)"

step "Aggregate deployment KPI tables"
python3 -m dram2cnn.utils.deployment_kpi
