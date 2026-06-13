#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 多 GPU 训练入口（薄封装）
#
# 【做什么】
#   内部调用 scripts/train.sh；后者读取 base.yaml 的 gpu/ddp（可被实验 overrides 覆盖），
#   自动选择 torchrun 或单进程，并按 80% 显存上限探测 batch_size。
#
# 【怎么用】
#   bash scripts/train_ddp.sh
#   bash scripts/train_ddp.sh configs/experiments/exp_full_best_ddp.yaml
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ddp.sh   # 只用部分显卡
#
# 【与 train.sh 的关系】
#   本脚本只做默认 YAML 与日志提示，真正启动逻辑在 train.sh。
#
# -----------------------------------------------------------------------------
# set -euo pipefail / PYTHONPATH / PROJECT_ROOT：同 preprocess.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXP_CONFIG="${1:-configs/experiments/exp_full_best_ddp.yaml}"

echo "[train_ddp.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not-set, use all visible GPUs>}"
echo "[train_ddp.sh] exp_config=${EXP_CONFIG}"

# 全量 DDP 实验不应仍指向演示 stub 索引（须 index_features_path 指向 full CSV，见 exp_full_best_ddp.yaml）
python3 - "${EXP_CONFIG}" <<'PY' || exit 1
import sys
from dram2cnn.utils.config import load_experiment_config
from dram2cnn.utils.experiment_guards import assert_full_best_index_not_stub

try:
    cfg = load_experiment_config(sys.argv[1])
    assert_full_best_index_not_stub(cfg, context="train_ddp.sh")
    print(f"[train_ddp.sh] index_features_path OK: {cfg['data'].get('index_features_path')}")
except RuntimeError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1) from exc
PY

# 必须用 SCRIPT_DIR 拼出 train.sh 路径：否则在用户「不在仓库根目录」执行时会找不到 scripts/train.sh
bash "${SCRIPT_DIR}/train.sh" "${EXP_CONFIG}"
