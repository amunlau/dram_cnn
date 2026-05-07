#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 预处理 + 切分：读取实验 YAML，调用 materialize_dataset 生成 data/processed/
# （X.npy / y.npy / meta.csv / preprocess_manifest.json；tabular.npy 视配置），
# 再按 time 或 sid 策略写入 data/splits/ 并对训练集拟合标准化。
#
# 前置条件：configs 中 mcelog、tickets、index、inventory 等路径可读。
# 用法: bash scripts/preprocess.sh [实验yaml]
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
python3 -m dram2cnn.pipelines.run_preprocess ${1:-configs/experiments/exp_baseline.yaml}
