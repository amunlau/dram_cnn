#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 多 GPU 训练入口（薄封装）
#
# 【做什么】
#   内部调用同目录下的 scripts/train.sh；后者根据 YAML 里 train.ddp 与可见 GPU 数量，
#   自动选择 torchrun 或单进程。默认实验配置偏向「全量最佳 + DDP」：exp_full_best_ddp.yaml。
#
# 【怎么用】
#   bash scripts/train_ddp.sh
#   bash scripts/train_ddp.sh configs/experiments/exp_full_best_ddp.yaml
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ddp.sh   # 只用部分显卡
#
# 【与 train.sh 的关系】
#   本脚本只做默认 YAML 与日志提示，真正启动逻辑在 train.sh。
#
# -----------------------------------------------------------------------------
# set -euo pipefail / PYTHONPATH / PROJECT_ROOT：同 preprocess.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXP_CONFIG="${1:-configs/experiments/exp_full_best_ddp.yaml}"

echo "[train_ddp.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not-set, use all visible GPUs>}"
echo "[train_ddp.sh] exp_config=${EXP_CONFIG}"

# 必须用 SCRIPT_DIR 拼出 train.sh 路径：否则在用户「不在仓库根目录」执行时会找不到 scripts/train.sh
bash "${SCRIPT_DIR}/train.sh" "${EXP_CONFIG}"
