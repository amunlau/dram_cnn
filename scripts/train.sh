#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 单进程或多进程训练（由 train.ddp 与 GPU 数量决定，逻辑见下方 Python 嵌入段）。
# 默认调用 run_train：校验 splits → 训练 GridCNN → 写入 checkpoints/grid_cnn.pt。
#
# 用法: bash scripts/train.sh [实验yaml]
# 多卡且启用 DDP 时，本脚本会通过 torchrun 拉起多进程（需 PyTorch NCCL 环境正常）。
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}

EXP_CONFIG=${1:-configs/experiments/exp_baseline.yaml}

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
