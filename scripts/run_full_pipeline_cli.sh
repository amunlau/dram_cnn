#!/usr/bin/env bash
# 全量 CLI 流水线（单实例）：F1 time+sid → 工程实验预处理+召回/精度 → notebooks → KPI
# 日志：notebooks_run_logs/full_pipeline_cli.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export DRAM2CNN_RUN_FULL_NOTEBOOKS=1

mkdir -p notebooks_run_logs runs/pipeline-logs/markers data/processed data/splits data/splits_sid
LOCK="${ROOT}/runs/pipeline-logs/full_pipeline_cli.lock"
LOG="${ROOT}/notebooks_run_logs/full_pipeline_cli.log"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "另一全量流水线已在运行（$LOCK）；退出。" >&2
  exit 1
fi

# 后续输出同时写入日志
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "======== $(date '+%F %T') $* ========"; }

mark() {
  local exp="$1" stepn="$2"
  mkdir -p runs/pipeline-logs/markers
  date +%Y%m%d_%H%M%S > "runs/pipeline-logs/markers/${exp}__${stepn}.done"
}

run_marked() {
  local exp="$1" stepn="$2"; shift 2
  local m="runs/pipeline-logs/markers/${exp}__${stepn}.done"
  if [[ -f "$m" ]]; then
    echo "[SKIP] $exp/$stepn"
    return 0
  fi
  echo "[RUN ] $exp/$stepn: $*"
  "$@"
  mark "$exp" "$stepn"
  echo "[DONE] $exp/$stepn"
}

clear_processed_splits() {
  rm -rf data/processed/* data/splits/* data/splits_sid/*
  mkdir -p data/processed data/splits data/splits_sid
  touch data/processed/.gitkeep data/splits/.gitkeep
}

step "START full CLI pipeline pid=$$"
echo "DRAM2CNN_RUN_FULL_NOTEBOOKS=$DRAM2CNN_RUN_FULL_NOTEBOOKS"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
free -h | head -2

# 全量索引已存在则打 marker，避免重复 build
if [[ -f data/raw/features_index_h300s_step300s_full.csv ]]; then
  [[ -f runs/pipeline-logs/markers/full_f1__build_index.done ]] || mark full_f1 build_index
  [[ -f runs/pipeline-logs/markers/full_eng__build_index.done ]] || mark full_eng build_index
else
  run_marked full_f1 build_index bash scripts/build_index.sh full
  mark full_eng build_index
fi

# ---- Phase F1（drop_empty_positive=false）----
step "F1 preprocess"
if [[ ! -f runs/pipeline-logs/markers/full_f1__preprocess.done ]]; then
  clear_processed_splits
fi
run_marked full_f1 preprocess bash scripts/preprocess.sh configs/experiments/exp_full_f1_ddp.yaml
run_marked full_f1 inspect_data bash scripts/inspect_data.sh configs/experiments/exp_full_f1_ddp.yaml

step "F1 time train/infer/evaluate"
run_marked full_f1 train bash scripts/train_ddp.sh configs/experiments/exp_full_f1_ddp.yaml
run_marked full_f1 infer bash scripts/infer.sh configs/experiments/exp_full_f1_ddp.yaml
run_marked full_f1 evaluate bash scripts/evaluate.sh configs/experiments/exp_full_f1_ddp.yaml

step "F1 sid"
run_marked full_f1_sid resplit bash scripts/resplit.sh configs/experiments/exp_full_f1_sid.yaml
run_marked full_f1_sid train bash scripts/train_ddp.sh configs/experiments/exp_full_f1_sid.yaml
run_marked full_f1_sid infer bash scripts/infer.sh configs/experiments/exp_full_f1_sid.yaml
run_marked full_f1_sid evaluate bash scripts/evaluate.sh configs/experiments/exp_full_f1_sid.yaml

# ---- Phase Eng（drop_empty_positive=true；须重 preprocess）----
step "Eng preprocess (clear F1 processed)"
if [[ ! -f runs/pipeline-logs/markers/full_eng__preprocess.done ]]; then
  clear_processed_splits
  rm -f runs/pipeline-logs/markers/full_eng__inspect_data.done
fi
run_marked full_eng preprocess bash scripts/preprocess.sh configs/experiments/exp_full_recall_ddp.yaml
run_marked full_eng inspect_data bash scripts/inspect_data.sh configs/experiments/exp_full_recall_ddp.yaml

step "Recall time/sid"
run_marked full_recall train bash scripts/train_ddp.sh configs/experiments/exp_full_recall_ddp.yaml
run_marked full_recall infer bash scripts/infer.sh configs/experiments/exp_full_recall_ddp.yaml
run_marked full_recall evaluate bash scripts/evaluate.sh configs/experiments/exp_full_recall_ddp.yaml
run_marked full_recall_sid resplit bash scripts/resplit.sh configs/experiments/exp_full_recall_sid.yaml
run_marked full_recall_sid train bash scripts/train_ddp.sh configs/experiments/exp_full_recall_sid.yaml
run_marked full_recall_sid infer bash scripts/infer.sh configs/experiments/exp_full_recall_sid.yaml
run_marked full_recall_sid evaluate bash scripts/evaluate.sh configs/experiments/exp_full_recall_sid.yaml

step "Precision time/sid"
run_marked full_precision train bash scripts/train_ddp.sh configs/experiments/exp_full_precision_ddp.yaml
run_marked full_precision infer bash scripts/infer.sh configs/experiments/exp_full_precision_ddp.yaml
run_marked full_precision evaluate bash scripts/evaluate.sh configs/experiments/exp_full_precision_ddp.yaml
run_marked full_precision_sid resplit bash scripts/resplit.sh configs/experiments/exp_full_precision_sid.yaml
run_marked full_precision_sid train bash scripts/train_ddp.sh configs/experiments/exp_full_precision_sid.yaml
run_marked full_precision_sid infer bash scripts/infer.sh configs/experiments/exp_full_precision_sid.yaml
run_marked full_precision_sid evaluate bash scripts/evaluate.sh configs/experiments/exp_full_precision_sid.yaml

step "Aggregate KPI"
python3 -m dram2cnn.utils.deployment_kpi || true

# 冒烟输出供 01–03 默认路径；全量已由 marker 跳过重训
step "Ensure smoke processed for notebooks 01-03"
if [[ ! -f data/processed_smoke/X.npy ]]; then
  bash scripts/build_index.sh smoke || true
  bash scripts/preprocess.sh configs/experiments/exp_baseline.yaml || true
fi

step "Run all notebooks (FULL=1; heavy steps SKIP via markers when done)"
python3 scripts/run_all_notebooks.py

step "ALL DONE $(date '+%F %T')"
