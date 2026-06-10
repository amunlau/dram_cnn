#!/usr/bin/env bash
# =============================================================================
# dram2cnn — Optuna GridCNN 超参搜索
#
# 前置：已对 EXP 完成 preprocess（末尾已 validate，存在 data/splits/）。
# 每个 trial 经 subprocess 调用 run_train，与 train.sh 共用 base.yaml 的 DDP + 80% 显存 batch 策略。
# 用法：bash scripts/optuna.sh [实验yaml] [hpo.yaml]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXP="${1:-configs/experiments/exp_optuna_smoke.yaml}"
HPO="${2:-configs/hpo.yaml}"

python3 -m dram2cnn.pipelines.run_optuna "${EXP}" "${HPO}"
