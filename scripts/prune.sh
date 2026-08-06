#!/usr/bin/env bash
# =============================================================================
# dram2cnn — GridCNN 三种粒度剪枝对比实验
#
# 【做什么】
#   对已训练 checkpoint 分别做：
#     1) unstructured L1 非结构化权重稀疏
#     2) bn_channel（按 BN γ 的通道剪枝，Slimming 剪枝阶段）
#     3) filter L1 结构化卷积核剪枝
#   在多个稀疏率/剪除比例下评估，写入
#   <train_dir>/prune_compare_<timestamp>/（含 prune.log、summary.*）。
#
# 【评估口径】
#   使用实验 YAML 的 time_stratified 全量 splits；主指标为各模型在 val 上
#   threshold_search(objective=f1) 重搜阈值后再评 test。另保留固定基线部署 τ 对照。
#
# 【怎么用】
#   bash scripts/prune.sh configs/experiments/exp_full_f1_ddp.yaml
#   bash scripts/prune.sh configs/experiments/exp_full_f1_ddp.yaml \
#       --sparsities 0.3,0.5,0.7 \
#       --methods unstructured,bn_channel,filter \
#       --finetune-epochs 5
#
# 【前置】
#   已完成 train；默认读 .latest_train_session 与 best_threshold.json。
#   默认短时微调 5 epoch（结构化剪枝强烈建议）；纯剪枝对比加 --finetune-epochs 0。
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

python3 -m dram2cnn.pipelines.run_prune "${EXP_CONFIG}" "$@"
