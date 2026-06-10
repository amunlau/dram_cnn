#!/usr/bin/env bash
# 高性能 Linux 服务器推荐环境变量（被 preprocess/train 等脚本 source）。
# 避免 fork 子进程与 OpenMP/BLAS 多线程叠加导致 CPU 过载或死锁。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# 多卡 DDP：异步错误便于定位 NCCL 问题
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
