#!/usr/bin/env bash
# 为 horizon 对比实验批量生成索引（5min / 30min / 1h）
# 默认生成冒烟 stub（data/index_stubs/）；传参 full 则全量扫描写入 data/raw/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODE="${1:-smoke}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "用法: bash scripts/build_horizon_indexes.sh [smoke|full]" >&2
  exit 1
fi

if [[ "${MODE}" == "smoke" ]]; then
  OUT_DIR="data/index_stubs"
  SUFFIX="smoke"
else
  OUT_DIR="data/raw"
  SUFFIX="full"
fi

echo "=== 5 分钟 horizon（步长 5 分钟） mode=${MODE} ==="
python3 scripts/build_index.py --mode "${MODE}" \
  --label-horizon-seconds 300 --sample-interval-seconds 300 \
  --out "${OUT_DIR}/features_index_h300s_step300s_${SUFFIX}.csv"

echo "=== 30 分钟 horizon（步长 10 分钟） mode=${MODE} ==="
python3 scripts/build_index.py --mode "${MODE}" \
  --label-horizon-seconds 1800 --sample-interval-seconds 600 --window-seconds 1800 \
  --out "${OUT_DIR}/features_index_h1800s_step600s_${SUFFIX}.csv"

echo "=== 1 小时 horizon（步长 30 分钟） mode=${MODE} ==="
python3 scripts/build_index.py --mode "${MODE}" \
  --label-horizon-seconds 3600 --sample-interval-seconds 1800 --window-seconds 3600 \
  --out "${OUT_DIR}/features_index_h3600s_step1800s_${SUFFIX}.csv"

echo "Done (${MODE}). 请在 exp_horizon_*.yaml 或 data.yaml 中设置 index_features_path 后重跑 preprocess。"
