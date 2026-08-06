#!/usr/bin/env bash
# =============================================================================
# dram2cnn — W8A8 PTQ 量化（GridCNN-Deep）
#
# 【做什么】
#   对已训练 checkpoint 做 BN Folding + AdaRound 风格 PTQ（W8A8），
#   AdaRound 后二次激活校准；导出 fake-quant 评估模型与 grid_cnn_w8a8_packed.pt
#   （权重以 int8 驻留）。在测试集上按验证集阈值评估部署阶段 F1/P/R，
#   并额外跑 packed INT8 参考推理校验。产物写入训练会话下
#   quant_w8a8_<timestamp>/（含 quantize.log、报告与模型）。
#   注：单独 quantize.sh 默认用训练 best_threshold.json（固定部署 τ）。
#   「先剪枝后量化」总对比表另用各模型 val 重搜 τ 作主指标，见 prune_quantize.sh。
#
# 【怎么用】
#   bash scripts/quantize.sh configs/experiments/exp_full_f1_ddp.yaml
#   bash scripts/quantize.sh configs/experiments/exp_full_f1_ddp.yaml \
#       --train-dir runs/exp_tianchi_full_f1_ddp/20260729_170226_369953
#
# 【后续 INT8 推理】
#   bash scripts/infer_int8.sh configs/experiments/exp_full_f1_ddp.yaml
#
# 【GPU】
#   PTQ 为单卡逐层流程（非 DDP）。默认使用 cuda:0；可用：
#     CUDA_VISIBLE_DEVICES=1 bash scripts/quantize.sh ...
#     或 python -m ... --device cuda:0
#
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

python3 -m dram2cnn.pipelines.run_quantize "${EXP_CONFIG}" "$@"
