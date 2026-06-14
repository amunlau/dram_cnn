#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 配置与数据校验（validate）
#
# 【做什么】
#   调用 run_validate：检查配置项、张量形状是否与模型一致等。
#   尚未跑 preprocess、没有 splits 时，请附加 --no-require-splits，只校验配置与原始路径。
#
# 【怎么用】
#   bash scripts/validate.sh
#   bash scripts/validate.sh configs/experiments/exp_full_f1_ddp.yaml
#   bash scripts/validate.sh configs/experiments/exp_baseline.yaml --no-require-splits
#   bash scripts/validate.sh --no-require-splits configs/experiments/exp_horizon_30m.yaml
#
# 【参数说明】
#   实验 YAML 与 --no-require-splits 可任意顺序；未指定 YAML 时默认 exp_baseline.yaml。
#
# -----------------------------------------------------------------------------
# set -euo pipefail / PYTHONPATH / cd：同 preprocess.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 解析参数：允许多种顺序（YAML 在前/在后；--no-require-splits 任意位置），但只允许一个 YAML
EXP_CONFIG=""
REQUIRE_SPLITS=1
for arg in "$@"; do
  case "$arg" in
    --no-require-splits)
      REQUIRE_SPLITS=0
      ;;
    -*)
      echo "[validate.sh] 未知选项: ${arg}" >&2
      exit 1
      ;;
    *)
      if [ -n "${EXP_CONFIG}" ]; then
        echo "[validate.sh] 只能指定一个实验 YAML，已见: ${EXP_CONFIG}，多余: ${arg}" >&2
        exit 1
      fi
      EXP_CONFIG="${arg}"
      ;;
  esac
done
EXP_CONFIG="${EXP_CONFIG:-configs/experiments/exp_baseline.yaml}"

PY_ARGS=( "${EXP_CONFIG}" )
if [ "${REQUIRE_SPLITS}" -eq 0 ]; then
  PY_ARGS+=( --no-require-splits )
fi
python3 -m dram2cnn.pipelines.run_validate "${PY_ARGS[@]}"
