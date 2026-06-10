#!/usr/bin/env bash
# 从 data/raw 生成索引表（smoke | full）
# 用法：bash scripts/build_index.sh smoke
#       bash scripts/build_index.sh full
#       bash scripts/build_index.sh full 1800 600   # horizon=1800s, interval=600s
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODE="${1:-smoke}"
HORIZON="${2:-}"
INTERVAL="${3:-}"

if [[ -n "${INTERVAL}" && -z "${HORIZON}" ]]; then
  echo "ERROR: 指定采样步长（第 3 参数）时必须同时指定 horizon（第 2 参数）。" >&2
  exit 1
fi

ARGS=(--mode "${MODE}")
if [[ -n "${HORIZON}" ]]; then
  ARGS+=(--label-horizon-seconds "${HORIZON}")
  ARGS+=(--window-seconds "${HORIZON}")
  if [[ -z "${INTERVAL}" ]]; then
    ARGS+=(--sample-interval-seconds "${HORIZON}")
    echo "[build_index.sh] 未指定第 3 参数，采样步长默认与 horizon 相同（${HORIZON}s）。30min/1h 对比请用 bash scripts/build_horizon_indexes.sh ${MODE}" >&2
  fi
fi
if [[ -n "${INTERVAL}" ]]; then
  ARGS+=(--sample-interval-seconds "${INTERVAL}")
fi
if [[ "${MODE}" == "smoke" && -z "${HORIZON}" && -z "${INTERVAL}" ]]; then
  ARGS+=(--out data/index_stubs/features_5m_index_demo.csv)
fi

python3 scripts/build_index.py "${ARGS[@]}"
