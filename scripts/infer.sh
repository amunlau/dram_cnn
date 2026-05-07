#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 测试集推理：加载最近一次训练（或 base.checkpoint_run_id 指定）的 grid_cnn.pt，
# 对 data/splits/X_test.npy 前向，写出 predictions.csv 并更新 .latest_infer_session。
#
# 前置条件：已完成 train，且 splits 与训练时模型输入形状一致。
# 用法: bash scripts/infer.sh [实验yaml]
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
python3 -m dram2cnn.pipelines.run_infer ${1:-configs/experiments/exp_baseline.yaml}
