#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 一键冒烟：顺序执行 preprocess → validate → train → infer → evaluate。
# 实验列表在 pipelines/run_autotune.py 的 EXPS 中配置。
#
# 用法: bash scripts/autotune.sh
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
python3 -m dram2cnn.pipelines.run_autotune
