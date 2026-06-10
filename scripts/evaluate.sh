#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 评估（evaluate）
#
# 【做什么】
#   读取推理阶段生成的 predictions.csv（路径由配置解析），计算 metrics.json，
#   并可生成 PR 曲线、混淆矩阵等（见 dram2cnn.engine.evaluator）。
#
# 【怎么用】
#   bash scripts/evaluate.sh
#   bash scripts/evaluate.sh configs/experiments/exp_baseline.yaml
#
# 【前置条件】
#   已有对应 run 的 infer 输出；否则会在加载预测文件阶段报错（ evaluator 内亦有路径检查）。
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

python3 -m dram2cnn.pipelines.run_evaluate "${1:-configs/experiments/exp_baseline.yaml}"
