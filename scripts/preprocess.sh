#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 预处理 + 数据集切分
#
# 【做什么】
#   读取实验 YAML，运行与 `python3 -m dram2cnn.pipelines.run_preprocess` 相同的逻辑：
#   生成 data/processed/（X.npy、y.npy、meta.csv、preprocess_manifest.json；tabular.npy 视配置）
#   再写入 data/splits/，并对训练集拟合标准化（若配置启用），最后自动 validate（与 autotune 一致）。
#
# 【怎么用】（可在任意当前目录执行；脚本会自动 cd 到仓库根目录）
#   bash scripts/preprocess.sh
#   bash scripts/preprocess.sh configs/experiments/exp_baseline.yaml
#   第一个参数缺省时，默认使用 configs/experiments/exp_baseline.yaml
#
# 【前置条件】
#   configs/data.yaml（及实验 YAML）里指向的 mcelog、tickets、index、inventory 等路径可读。
#   索引表可先：bash scripts/build_index.sh smoke|full
#
# 【实现说明】
#   - PYTHONPATH：必须包含仓库下的 src/，Python 才能 import dram2cnn（下面用绝对路径，避免依赖「当前目录」）。
#   - 相对路径 configs/、data/：均以「仓库根目录」为基准，故脚本会先 cd 到 PROJECT_ROOT。
#
# -----------------------------------------------------------------------------
# Bash「严格模式」set -euo pipefail（常见最佳实践）
#   -e  errexit   ：任意命令返回非 0（失败）时，立即退出整个脚本，避免后续在错误状态下继续跑。
#   -u  nounset   ：引用未定义变量时报错退出，避免变量名拼写错误被当成空字符串。
#   -o pipefail   ：管道 a | b | c 中任一环节失败，整条管道视为失败；否则默认可能只看最后一个命令的退出码。
# =============================================================================
set -euo pipefail

# 脚本所在目录 → 仓库根（scripts/ 的上一级），保证从任意 cwd 调用都能找到 src 与 configs/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# 模块方式运行：${1:-默认yaml} 表示「若用户未传第 1 个参数，则用默认路径」
python3 -m dram2cnn.pipelines.run_preprocess "${1:-configs/experiments/exp_baseline.yaml}"
