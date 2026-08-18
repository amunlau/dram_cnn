#!/usr/bin/env bash
# =============================================================================
# 解析 GPU 启动计划并执行 torchrun 或单进程 Python 模块。
#
# 用法（由 train.sh / infer.sh source）：
#   gpu_launch_run train configs/experiments/exp.yaml dram2cnn.pipelines.run_train
#   gpu_launch_run infer configs/experiments/exp.yaml dram2cnn.pipelines.run_infer
# =============================================================================
gpu_launch_run() {
  local MODE="$1"
  local EXP_CONFIG="$2"
  local PY_MODULE="$3"

  readarray -t _launch_info < <(
    python3 -m dram2cnn.utils.gpu_launch "${MODE}" "${EXP_CONFIG}"
  )

  local device="${_launch_info[0]}"
  local ddp_enabled="${_launch_info[1]}"
  local nproc_per_node="${_launch_info[2]}"
  local configured_device="${_launch_info[3]}"
  local cuda_count="${_launch_info[4]}"
  local using_all_visible_gpus="${_launch_info[5]}"
  local resolved_batch_size="${_launch_info[6]}"
  local batch_size_auto="${_launch_info[7]}"

  if [[ "$batch_size_auto" == "1" ]]; then
    export DRAM2CNN_BATCH_SIZE="${resolved_batch_size}"
  else
    unset DRAM2CNN_BATCH_SIZE 2>/dev/null || true
  fi

  echo "[${MODE}.sh] device configured=${configured_device}, resolved=${device}, cuda_count=${cuda_count}, ddp_enabled=${ddp_enabled}, nproc_per_node=${nproc_per_node}"
  echo "[${MODE}.sh] batch_size=${resolved_batch_size} (auto=${batch_size_auto})"
  echo "[${MODE}.sh] launch_plan using_all_visible_gpus=${using_all_visible_gpus} (1=yes,0=no)"

  if [[ "$device" == cuda* ]] && [[ "$ddp_enabled" == "1" ]] && [[ "$nproc_per_node" -gt 1 ]]; then
    # 容器内 hostname 常无法解析 IPv6，导致 c10d 长时间卡住、GPU 无负载。
    # 强制走本机 IPv4 rendezvous，并尽量补全 /etc/hosts 中的 hostname 映射。
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    _hn="$(hostname 2>/dev/null || true)"
    if [[ -n "${_hn}" ]] && [[ -w /etc/hosts ]]; then
      grep -qE "^127\\.0\\.0\\.1[[:space:]].*${_hn}([[:space:]]|$)" /etc/hosts || echo "127.0.0.1 ${_hn}" >> /etc/hosts || true
      grep -qE "^::1[[:space:]].*${_hn}([[:space:]]|$)" /etc/hosts || echo "::1 ${_hn}" >> /etc/hosts || true
    fi
    echo "[${MODE}.sh] DDP: torchrun --nnodes=1 --nproc_per_node=${nproc_per_node} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} -m ${PY_MODULE} ${EXP_CONFIG}"
    torchrun \
      --nnodes=1 \
      --nproc_per_node="${nproc_per_node}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      -m "${PY_MODULE}" "${EXP_CONFIG}"
  else
    echo "[${MODE}.sh] single process: python3 -m ${PY_MODULE} ${EXP_CONFIG}"
    python3 -m "${PY_MODULE}" "${EXP_CONFIG}"
  fi
}
