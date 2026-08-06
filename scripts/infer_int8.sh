#!/usr/bin/env bash
# =============================================================================
# dram2cnn — W8A8 packed INT8 参考推理
#
# 【做什么】
#   加载 quantize.sh 导出的 grid_cnn_w8a8_packed.pt（权重以 int8 驻留），
#   对 splits 测试集做 INT8 参考前向，写出 predictions.csv 与 metrics_int8.json。
#
# 【怎么用】
#   bash scripts/infer_int8.sh configs/experiments/exp_full_f1_ddp.yaml
#   bash scripts/infer_int8.sh configs/experiments/exp_full_f1_ddp.yaml \
#       --train-dir runs/exp_tianchi_full_f1_ddp/<session> \
#       --quant-dir quant_w8a8_<ts>
#
# 【前置】
#   已完成 train + quantize；默认读 .latest_train_session / .latest_quant_session。
#
# 【说明】
#   单卡参考实现（非 DDP、非 TensorRT）；数值语义与 PTQ 报告中
#   w8a8_int8_packed_test_deployment 一致。
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

python3 -m dram2cnn.pipelines.run_infer_int8 "${EXP_CONFIG}" "$@"
