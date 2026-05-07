#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 校验配置与张量形状；无 splits 时使用 **--no-require-splits**。
# -----------------------------------------------------------------------------
# 校验配置与数据形状。用法示例：
#   bash scripts/validate.sh
#   bash scripts/validate.sh configs/experiments/exp_full_best_ddp.yaml
#   bash scripts/validate.sh configs/experiments/exp_baseline.yaml --no-require-splits
#   bash scripts/validate.sh --no-require-splits
# （尚未 preprocess 时请加 --no-require-splits，仅校验配置与原始路径）
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
if [ "$#" -eq 0 ]; then
  set -- configs/experiments/exp_baseline.yaml
elif [[ "${1:-}" == -* ]]; then
  set -- configs/experiments/exp_baseline.yaml "$@"
fi
python3 -m dram2cnn.pipelines.run_validate "$@"
