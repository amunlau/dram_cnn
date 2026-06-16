#!/usr/bin/env bash
# 全量 CLI 批量：工程向（召回/精度 time+sid）→ F1 最大化（time+sid）
#
# 前置（Phase A 工程向）：
#   bash scripts/build_index.sh full
#   bash scripts/preprocess.sh configs/experiments/exp_full_recall_ddp.yaml
#   （工程向 preprocess：drop_empty_positive_windows=true，与 F1 不可混用）
#
# Phase B 会清空 data/processed、splits、splits_sid 并重跑 F1 preprocess。
#
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

PROC_MAN="${PROJECT_ROOT}/data/processed/preprocess_manifest.json"
if [[ ! -f "${PROC_MAN}" ]]; then
  echo "错误：缺少 ${PROC_MAN}" >&2
  echo "请先：bash scripts/build_index.sh full" >&2
  echo "      bash scripts/preprocess.sh configs/experiments/exp_full_recall_ddp.yaml" >&2
  exit 1
fi

# ---------- Phase A: 工程向（复用工程向 preprocess 产物）----------
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
