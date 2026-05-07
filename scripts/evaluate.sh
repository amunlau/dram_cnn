#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 评估：读取 infer 输出目录中的 predictions.csv，计算 metrics.json、
# 可选 PR 曲线与混淆矩阵（见 engine.evaluator）。
#
# 用法: bash scripts/evaluate.sh [实验yaml]
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
python3 -m dram2cnn.pipelines.run_evaluate ${1:-configs/experiments/exp_baseline.yaml}
