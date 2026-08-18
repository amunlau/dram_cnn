#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 融合(无 bias) → 非结构化剪枝 → W8A8 量化
#
# 【做什么】
#   对已训练 checkpoint：
#     1) Conv+BN+ReLU 融合，并强制丢弃 bias（硬件无偏置）
#     2) 非结构化 L1 剪枝（默认 30%）+ 可选短时微调
#     3) W8A8 PTQ，导出 packed INT8
#   写入 <train_dir>/fuse_prune_quant_<timestamp>/（含 fuse_prune_quantize.log）
#
# 【怎么用】
#   bash scripts/fuse_prune_quantize.sh configs/experiments/exp_full_f1_nobias_allint8_ddp.yaml
#   bash scripts/fuse_prune_quantize.sh configs/experiments/exp_full_f1_nobias_allint8_ddp.yaml \
#       --sparsity 0.3 --finetune-epochs 5
#   # 若需恢复旧行为（首尾层保持单精度）：追加 --protect-ends
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXP_CONFIG="${1:-configs/experiments/exp_full_f1_nobias_allint8_ddp.yaml}"
shift || true

python3 -m dram2cnn.pipelines.run_fuse_prune_quantize "${EXP_CONFIG}" "$@"
