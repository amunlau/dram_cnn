#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 官方对称 signed INT8（torch.ao FakeQuant）PTQ + QAT + QDQ ONNX
#
# 【做什么】
#   在已训练 FP32 checkpoint 上：
#     1) 评估 FP32 基线（val 重搜 τ → test F1/P/R）
#     2) Eager FakeQuant PTQ（W/A 对称 qint8；stem / 末层 Linear 保持 FP32）
#     3) QAT 短训（默认 20 epoch，较小 lr；多卡时 torchrun DDP）
#     4) PTQ / QAT 各导出一份 QDQ ONNX + packed INT8 体积统计
#   产物：<train_dir>/quant_official_symint8_ptq_qat_<tag>/
#
# 【怎么用】
#   bash scripts/quantize_official.sh configs/experiments/exp_full_f1_ddp.yaml \
#     --train-dir runs/exp_tianchi_full_f1_ddp/20260917_153434_808743
#
# 【GPU】
#   默认尽量多卡 DDP（与 train_ddp.sh 相同的本机 IPv4 rendezvous）。
#   单卡：NPROC=1 bash scripts/quantize_official.sh ...
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXP_CONFIG="${1:-configs/experiments/exp_full_f1_ddp.yaml}"
shift || true

NPROC="${NPROC:-}"
if [[ -z "${NPROC}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    NPROC=1
  fi
fi
if [[ "${NPROC}" -lt 1 ]]; then
  NPROC=1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29571}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

echo "[quantize_official.sh] nproc=${NPROC} exp=${EXP_CONFIG} $*"

if [[ "${NPROC}" -gt 1 ]]; then
  torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m dram2cnn.pipelines.run_quantize_official \
    "${EXP_CONFIG}" \
    "$@"
else
  python3 -m dram2cnn.pipelines.run_quantize_official "${EXP_CONFIG}" "$@"
fi
