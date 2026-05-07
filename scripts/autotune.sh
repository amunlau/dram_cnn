#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 一键冒烟流水线（autotune）
#
# 【做什么】
#   顺序执行 preprocess → validate → train → infer → evaluate（具体实验列表在
#   dram2cnn.pipelines.run_autotune 内配置，如 EXPS）。
#   用于 CI 或本地快速验证「整条链路能跑通」。
#
# 【怎么用】
#   bash scripts/autotune.sh
#
# 【注意】
#   耗时与数据规模、训练轮数有关；大数据集上请谨慎在非交互环境超时设置。
#
# -----------------------------------------------------------------------------
# set -euo pipefail / PYTHONPATH / cd：同 preprocess.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python3 -m dram2cnn.pipelines.run_autotune
