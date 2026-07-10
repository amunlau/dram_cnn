#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 端到端链路检查流水线（autotune）
#
# 【做什么】
#   顺序执行 preprocess（含末尾 validate）→ train → infer → evaluate（实验列表见
#   dram2cnn.pipelines.run_autotune 内 EXPS）。
#   用于 CI 或本地快速验证「整条链路可完整执行」。
#
# 【怎么用】
#   bash scripts/autotune.sh
#
# 【注意】
#   耗时与数据规模、训练轮数有关；大数据集上请谨慎在非交互环境超时设置。
#   autotune 通过 scripts/train.sh / infer.sh 启动，继承 base.yaml 的 gpu/ddp 默认；
#   exp_baseline 实验内关闭了 DDP 与 auto_batch_size 以保持冒烟速度。
#
# -----------------------------------------------------------------------------
# set -euo pipefail / PYTHONPATH / cd：同 preprocess.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python3 -m dram2cnn.pipelines.run_autotune
