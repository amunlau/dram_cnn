#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 先剪枝后 W8A8 量化对比
#
# 【做什么】
#   读取 prune_compare_* 中的剪枝模型，逐个做 W8A8 PTQ（packed INT8），
#   合并 baseline / 仅剪枝 FP32 / 剪枝+量化 的完整对比表。
#   结束后自动按各模型 val 重搜阈值覆盖主表指标（与训练 threshold_search 一致）；
#   CSV 另含 *_at_deploy_tau（固定基线部署 τ；量化后可能出现 P=1.0 对照）。
#   仅支持「先剪枝后量化」，不考虑先量化后剪枝。
#
# 【怎么用】
#   bash scripts/prune_quantize.sh configs/experiments/exp_full_f1_ddp.yaml
#   bash scripts/prune_quantize.sh ... --prune-dir <train>/prune_compare_<ts>
#
# 【前置】
#   已跑 prune.sh；默认读 .latest_prune_session。数据为实验 YAML 的全量 time_stratified splits。
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXP_CONFIG="${1:-configs/experiments/exp_full_f1_ddp.yaml}"
shift || true

python3 -m dram2cnn.pipelines.run_prune_quantize "${EXP_CONFIG}" "$@"
