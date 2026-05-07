#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 测试集推理（infer）
#
# 【做什么】
#   加载训练得到的权重（最近 run 或配置指定的 checkpoint），对 data/splits/X_test.npy 前向，
#   写出 predictions.csv，并更新本次推理会话标记（如 .latest_infer_session，见代码）。
#
# 【怎么用】
#   bash scripts/infer.sh
#   bash scripts/infer.sh configs/experiments/exp_baseline.yaml
#
# 【前置条件】
#   已完成训练且切分中的张量形状与模型一致；路径由实验 YAML / base 配置解析。
#
# -----------------------------------------------------------------------------
# set -euo pipefail / PYTHONPATH / cd 到仓库根：同 preprocess.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python3 -m dram2cnn.pipelines.run_infer "${1:-configs/experiments/exp_baseline.yaml}"
