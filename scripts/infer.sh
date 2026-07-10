#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 测试集推理（infer）
#
# 【做什么】
#   加载训练得到的权重（最近 run 或配置指定的 checkpoint），对实验 YAML 的 splits_dir/X_test.npy 前向，
#   写出 predictions.csv，并更新本次推理会话标记（如 .latest_infer_session，见代码）。
#   启动策略与 train.sh 相同：默认 DDP 使用全部可见 GPU，并按 80% 显存上限自动探测 batch_size。
#
# 【怎么用】
#   bash scripts/infer.sh
#   bash scripts/infer.sh configs/experiments/exp_baseline.yaml
#
# 【前置条件】
#   已完成训练且切分中的张量形状与模型一致；路径由实验 YAML / base 配置解析。
#
# -----------------------------------------------------------------------------
# set -euo pipefail / PYTHONPATH / cd 到仓库根：同 preprocess.sh。
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

gpu_launch_run infer "${EXP_CONFIG}" dram2cnn.pipelines.run_infer
