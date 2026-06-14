#!/usr/bin/env bash
# 在已有 data/processed/ 上仅重跑切分（不重复扫描 mcelog）。
# 典型：全量 preprocess 完成后，对比 time_stratified vs sid 泛化。
#
#   bash scripts/resplit.sh configs/experiments/exp_full_best_sid.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
python3 -m dram2cnn.pipelines.run_preprocess --split-only "${1:-configs/experiments/exp_full_best_sid.yaml}"
