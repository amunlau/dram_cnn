#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 多 GPU 训练入口：内部调用 **scripts/train.sh**；后者根据 `train.ddp` 与可见 GPU
# 数量自动选择 `torchrun` 或单进程。默认实验为 **exp_full_best_ddp**。
#
# 用法: bash scripts/train_ddp.sh [实验yaml]
# 限定显卡: CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ddp.sh
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}

EXP_CONFIG=${1:-configs/experiments/exp_full_best_ddp.yaml}

echo "[train_ddp.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not-set, use all visible GPUs>}"
echo "[train_ddp.sh] exp_config=${EXP_CONFIG}"

bash scripts/train.sh "${EXP_CONFIG}"
