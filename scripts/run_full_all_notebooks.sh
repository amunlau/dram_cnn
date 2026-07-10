#!/usr/bin/env bash
# 全量执行 notebooks/ 下全部笔记（含 04/05 全量训练单元）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export DRAM2CNN_RUN_FULL_NOTEBOOKS=1
export PYTHONUNBUFFERED=1
mkdir -p notebooks_run_logs runs/pipeline-logs/markers

LOG="notebooks_run_logs/full_runner_stdout.log"
echo "=== $(date '+%F %T') START full notebook run ===" | tee "$LOG"
echo "DRAM2CNN_RUN_FULL_NOTEBOOKS=$DRAM2CNN_RUN_FULL_NOTEBOOKS" | tee -a "$LOG"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)" | tee -a "$LOG"

# Prefer existing full index (already marked); do not wipe those markers
python3 scripts/run_all_notebooks.py 2>&1 | tee -a "$LOG"
EC=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') notebooks exit=$EC ===" | tee -a "$LOG"

# Post: error analysis on full F1 if predictions exist
if [[ -f runs/exp_tianchi_full_f1_ddp/.latest_infer_session ]] || ls runs/exp_tianchi_full_f1_ddp/infer_*/predictions.csv >/dev/null 2>&1; then
  echo "=== $(date '+%F %T') Post: 03-style error analysis on full F1 ===" | tee -a "$LOG"
  python3 << 'PY' 2>&1 | tee -a notebooks_run_logs/full_f1_error_analysis.log
from pathlib import Path
import json, os, sys
import pandas as pd
import numpy as np
ROOT = Path('/root/workspace')
sys.path.insert(0, str(ROOT/'src'))
os.chdir(ROOT)
from dram2cnn.utils.config import load_experiment_config
from dram2cnn.utils.run_layout import read_pointer_dir
from dram2cnn.utils.pipeline_helpers import splits_dir_for_exp

exp = ROOT/'configs/experiments/exp_full_f1_ddp.yaml'
cfg = load_experiment_config(str(exp))
run_name = cfg['base']['run_name']
run_root = ROOT/'runs'/run_name
infer_dir = read_pointer_dir(run_root, '.latest_infer_session', 'latest_infer')
if infer_dir is None:
    cands = sorted(run_root.glob('infer_*/predictions.csv'))
    infer_dir = cands[-1].parent if cands else None
assert infer_dir and (infer_dir/'predictions.csv').is_file(), f'no predictions under {run_root}'
df = pd.read_csv(infer_dir/'predictions.csv')
yt, yp = df['y_true'].astype(int), df['y_pred'].astype(int)
summary = pd.DataFrame([
    {'name':'TP','count':int(((yt==1)&(yp==1)).sum())},
    {'name':'FP','count':int(((yt==0)&(yp==1)).sum())},
    {'name':'FN','count':int(((yt==1)&(yp==0)).sum())},
    {'name':'TN','count':int(((yt==0)&(yp==0)).sum())},
])
out = ROOT/'outputs/tables'
out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out/'error_confusion_counts_full_f1.csv', index=False)
print(summary.to_string(index=False))
mp = infer_dir/'metrics.json'
if mp.is_file():
    m = json.loads(mp.read_text())
    print('deployment f1:', (m.get('deployment') or {}).get('metrics', m).get('f1'))
    print('metrics:', mp)
print('infer_dir:', infer_dir)
PY
fi

# Aggregate KPI tables
echo "=== $(date '+%F %T') Aggregate deployment KPI ===" | tee -a "$LOG"
python3 -m dram2cnn.utils.deployment_kpi 2>&1 | tee -a "$LOG" || true

echo "=== $(date '+%F %T') ALL DONE exit=$EC ===" | tee -a "$LOG"
exit $EC
