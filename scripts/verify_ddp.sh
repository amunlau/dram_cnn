#!/usr/bin/env bash
# 验证 train / infer / optuna 在 ≥2 GPU 时确实走 torchrun 多卡（读取 launch plan 与训练日志）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXP="${1:-configs/experiments/exp_ddp_verify.yaml}"
HPO="${2:-configs/hpo/hpo_verify.yaml}"

CUDA_COUNT="$(python3 -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
echo "=== DDP verify: visible GPUs=${CUDA_COUNT} ==="

if [[ "${CUDA_COUNT}" -lt 2 ]]; then
  echo "SKIP: need >=2 GPUs for DDP verification (current=${CUDA_COUNT})"
  exit 0
fi

check_launch() {
  local mode="$1"
  local ddp enabled nproc all_gpus auto_bs
  readarray -t _info < <(python3 -m dram2cnn.utils.gpu_launch "${mode}" "${EXP}")
  ddp="${_info[1]}"
  nproc="${_info[2]}"
  all_gpus="${_info[5]}"
  auto_bs="${_info[7]}"
  echo "[launch:${mode}] ddp_enabled=${ddp} nproc=${nproc} using_all_gpus=${all_gpus} auto_batch=${auto_bs}"
  [[ "${ddp}" == "1" ]] || { echo "FAIL: ${mode} ddp_enabled expected 1"; return 1; }
  [[ "${nproc}" -ge 2 ]] || { echo "FAIL: ${mode} nproc expected >=2"; return 1; }
  [[ "${all_gpus}" == "1" ]] || { echo "FAIL: ${mode} not using all visible GPUs"; return 1; }
  [[ "${auto_bs}" == "1" ]] || { echo "FAIL: ${mode} auto_batch_size expected 1"; return 1; }
}

check_launch train
check_launch infer

echo "=== train.sh (expect torchrun + world_size=${CUDA_COUNT}) ==="
TRAIN_LOG="$(mktemp)"
bash scripts/train.sh "${EXP}" 2>&1 | tee "${TRAIN_LOG}"
grep -q "torchrun" "${TRAIN_LOG}" || { echo "FAIL: train.sh did not invoke torchrun"; exit 1; }
grep -q "world_size=${CUDA_COUNT}" "${TRAIN_LOG}" || { echo "FAIL: train log missing world_size=${CUDA_COUNT}"; exit 1; }
grep -q "ddp=True" "${TRAIN_LOG}" || { echo "FAIL: train log missing ddp=True"; exit 1; }

echo "=== infer.sh (expect torchrun + DDP manifest world_size) ==="
INFER_LOG="$(mktemp)"
bash scripts/infer.sh "${EXP}" 2>&1 | tee "${INFER_LOG}"
grep -q "torchrun" "${INFER_LOG}" || { echo "FAIL: infer.sh did not invoke torchrun"; exit 1; }

RUN_ROOT="$(python3 -c "
from dram2cnn.utils.config import load_experiment_config, project_root
cfg = load_experiment_config('${EXP}')
print(project_root() / 'runs' / cfg['base']['run_name'])
")"
INFER_PTR="$(cat "${RUN_ROOT}/.latest_infer_session")"
LATEST_INFER="${RUN_ROOT}/${INFER_PTR}"
WS="$(python3 -c "import json; print(json.load(open('${LATEST_INFER}/predictions_manifest.json'))['world_size'])")"
[[ "${WS}" == "${CUDA_COUNT}" ]] || { echo "FAIL: infer manifest world_size=${WS}, expected ${CUDA_COUNT}"; exit 1; }
echo "[infer] predictions_manifest world_size=${WS} OK"

echo "=== optuna 1-trial (expect torchrun + world_size in trial log) ==="
OPTUNA_LOG="$(mktemp)"
set +e
python3 -m dram2cnn.pipelines.run_optuna configs/experiments/exp_optuna_smoke.yaml "${HPO}" 2>&1 | tee "${OPTUNA_LOG}"
OPTUNA_RC=$?
set -e
TRIAL_LOG="$(find "${PROJECT_ROOT}/runs/optuna" -path '*/grid_cnn_hpo_verify/trial*/*train*.log' 2>/dev/null | sort | tail -1)"
if [[ -z "${TRIAL_LOG}" ]] || ! grep -q "world_size=${CUDA_COUNT}" "${TRIAL_LOG}"; then
  echo "FAIL: optuna trial missing world_size=${CUDA_COUNT} in ${TRIAL_LOG:-<no log>}"
  exit 1
fi
if ! grep -q "ddp=True" "${TRIAL_LOG}"; then
  echo "FAIL: optuna trial log missing ddp=True"
  exit 1
fi
echo "[optuna] trial DDP OK: ${TRIAL_LOG} (study exit=${OPTUNA_RC}, prune on smoke data is expected)"

echo "=== ALL DDP CHECKS PASSED (GPUs=${CUDA_COUNT}) ==="
