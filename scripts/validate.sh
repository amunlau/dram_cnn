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
#   bash scripts/validate.sh configs/experiments/exp_full_best_ddp.yaml
#   bash scripts/validate.sh configs/experiments/exp_baseline.yaml --no-require-splits
#   bash scripts/validate.sh --no-require-splits
#
# 【参数说明】
#   第一个参数通常是实验 YAML；若以 -- 开头（例如只有 --no-require-splits），
#   脚本会自动补上默认 exp_baseline.yaml，再把后续标志传给 Python。
#
# -----------------------------------------------------------------------------
# set -euo pipefail / PYTHONPATH / cd：同 preprocess.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 无参数 → 默认 YAML；第一个参数是选项 → 插入默认 YAML 后再展开原有参数
if [ "$#" -eq 0 ]; then
  set -- configs/experiments/exp_baseline.yaml
elif [[ "${1:-}" == -* ]]; then
  set -- configs/experiments/exp_baseline.yaml "$@"
fi

python3 -m dram2cnn.pipelines.run_validate "$@"
