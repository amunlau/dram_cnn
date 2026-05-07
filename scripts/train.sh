#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 训练（单进程或 DDP 多卡）
#
# 【做什么】
#   校验切分 → 训练 GridCNN → 写入 checkpoints（如 checkpoints/grid_cnn.pt，具体见配置）。
#   是否使用 torchrun 多进程：由实验 YAML 里 train.ddp、device、GPU 数量等决定（下方嵌入的 Python 小段只负责解析配置并打印启动计划）。
#
# 【怎么用】
#   bash scripts/train.sh
#   bash scripts/train.sh configs/experiments/exp_baseline.yaml
#
# 【多卡】若配置启用 DDP 且 nproc>1，本脚本会调用 torchrun；单机多卡前可设置：
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh configs/experiments/exp_xxx.yaml
#
# 【注意】传入的实验 YAML 为「位置参数」（不是 --config=），与 python -m dram2cnn.pipelines.run_train 一致。
#
# -----------------------------------------------------------------------------
# set -euo pipefail：见 scripts/preprocess.sh 顶部说明。
# PYTHONPATH：仓库 src/，见 preprocess.sh。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXP_CONFIG="${1:-configs/experiments/exp_baseline.yaml}"

# 从 YAML 解析 device / DDP / nproc，输出多行供 bash 读取（避免在 bash 里手写复杂判断）
readarray -t _launch_info < <(
python3 - "$EXP_CONFIG" <<'PY'
from dram2cnn.utils.config import load_experiment_config
import torch
import sys

cfg = load_experiment_config(sys.argv[1])
raw_device = str(cfg.get("base", {}).get("device", "auto")).strip().lower()
if raw_device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = raw_device
ddp_cfg = cfg.get("train", {}).get("ddp", {})
ddp_enabled = bool(ddp_cfg.get("enabled", False))
nproc_cfg = ddp_cfg.get("nproc_per_node", "auto")

if str(nproc_cfg).strip().lower() == "auto":
    nproc = torch.cuda.device_count()
else:
    nproc = int(nproc_cfg)
cuda_count = torch.cuda.device_count()
using_all_visible_gpus = int(device.startswith("cuda") and ddp_enabled and nproc > 1 and nproc == cuda_count)

print(device)
print("1" if ddp_enabled else "0")
print(max(1, nproc))
print(raw_device)
print(cuda_count)
print(using_all_visible_gpus)
PY
)

device="${_launch_info[0]}"
ddp_enabled="${_launch_info[1]}"
nproc_per_node="${_launch_info[2]}"
configured_device="${_launch_info[3]}"
cuda_count="${_launch_info[4]}"
using_all_visible_gpus="${_launch_info[5]}"

echo "[train.sh] device configured=${configured_device}, resolved=${device}, cuda_count=${cuda_count}, ddp_enabled=${ddp_enabled}, nproc_per_node=${nproc_per_node}"
echo "[train.sh] launch_plan using_all_visible_gpus=${using_all_visible_gpus} (1=yes,0=no)"

if [[ "$device" == cuda* ]] && [[ "$ddp_enabled" == "1" ]] && [[ "$nproc_per_node" -gt 1 ]]; then
  echo "[train.sh] DDP enabled: torchrun --nproc_per_node=${nproc_per_node} -m dram2cnn.pipelines.run_train ${EXP_CONFIG}"
  torchrun --standalone --nproc_per_node="${nproc_per_node}" -m dram2cnn.pipelines.run_train "${EXP_CONFIG}"
else
  echo "[train.sh] single process: python3 -m dram2cnn.pipelines.run_train ${EXP_CONFIG}"
  python3 -m dram2cnn.pipelines.run_train "${EXP_CONFIG}"
fi
