#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 训练（单进程或 DDP 多卡）
#
# 【做什么】
#   校验切分 → 训练 GridCNN → 写入 checkpoints（如 checkpoints/grid_cnn.pt，具体见配置）。
#   checkpoint 是验证集 early_stop_metric（默认 best_f1）的历史最优，不是最后一轮。
#   全量 patience=50；搜阈 F1 常在前几轮见顶（当前交付第 2 轮、第 52 轮才停），属预期。
#   见 delivery/nobias-symint8-20260818/evidence/notes/训练早停与第2轮最优说明.md 与 docs/troubleshooting.md。
#   启动前自动：检测可见 GPU 数量与显存；base.ddp.enabled=auto 且 ≥2 卡时 torchrun 占用全部可见 GPU；
#   base.gpu.auto_batch_size=true 时在单卡上探测 batch_size，使峰值显存 ≤ vram_budget_frac（默认 80%）。
#
# 【怎么用】
#   bash scripts/train.sh
#   bash scripts/train.sh configs/experiments/exp_baseline.yaml
#
# 【多卡】若配置启用 DDP 且 nproc>1，本脚本会调用 torchrun；单机多卡前可设置：
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh configs/experiments/exp_xxx.yaml
#
# 【注意】传入的实验 YAML 为「位置参数」（不是 --config=），与 python -m dram2cnn.pipelines.run_train 一致。
#
# -----------------------------------------------------------------------------
# set -euo pipefail：见 scripts/preprocess.sh 顶部说明。
# PYTHONPATH：仓库 src/，见 preprocess.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
# shellcheck source=lib/gpu_launch.sh
source "${SCRIPT_DIR}/lib/gpu_launch.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXP_CONFIG="${1:-configs/experiments/exp_baseline.yaml}"

gpu_launch_run train "${EXP_CONFIG}" dram2cnn.pipelines.run_train
