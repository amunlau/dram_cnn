#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 快速查看预处理产物概览（仅诊断用）
#
# 【做什么】
#   用 mmap 读取 data/processed 下 X.npy / y.npy，打印形状、正负比例、meta 前几行等。
#   不占满内存；输出中会提示：训练实际使用的是 data/splits 里「按训练集标准化后」的 X_*，
#   此处为预处理全量池的原始特征尺度。
#
# 【怎么用】
#   bash scripts/inspect_data.sh
#
# 【前置条件】
#   已执行 bash scripts/preprocess.sh（或等价预处理），存在 data/processed/X.npy、y.npy、meta.csv。
#
# -----------------------------------------------------------------------------
# set -euo pipefail / PYTHONPATH / cd：同 preprocess.sh。
# 下方内嵌 Python 使用相对路径 data/processed，依赖「当前已在 PROJECT_ROOT」。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python3 - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

p = Path("data/processed")
x_npy = p / "X.npy"
y_npy = p / "y.npy"
meta_csv = p / "meta.csv"
x_preview = p / "X_preview.csv"
y_csv = p / "y.csv"

missing = [str(fp) for fp in [x_npy, y_npy, meta_csv] if not fp.exists()]
if missing:
    raise FileNotFoundError("缺少预处理产物，请先执行 `bash scripts/preprocess.sh`:\n" + "\n".join(missing))

X = np.load(x_npy, mmap_mode="r")
y = np.load(y_npy, mmap_mode="r")
pos = int((y == 1).sum())
neg = int((y == 0).sum())
total = int(len(y))

print("=== 数据概览（data/processed，原始特征尺度）===")
print("提示：训练/验证/测试实际使用 data/splits/X_*.npy（切分后按训练集 mean/std 归一化）。此处为预处理全量池。")
print(f"X shape: {tuple(X.shape)}")
print(f"y shape: {tuple(y.shape)}")
print(f"样本总数: {total}")
print(f"正样本: {pos} ({pos / total:.6f})")
print(f"负样本: {neg} ({neg / total:.6f})")
print(f"meta.csv: {meta_csv}")

if y_csv.exists():
    ydf = pd.read_csv(y_csv)
    print(f"y.csv: {y_csv} (rows={len(ydf)})")

meta = pd.read_csv(meta_csv)
print("\n=== meta 前3行 ===")
mcols = [c for c in ("sid", "predict_time", "label") if c in meta.columns]
print(meta[mcols].head(3).to_string(index=False))

if x_preview.exists():
    pv = pd.read_csv(x_preview)
    print(f"\n（可选）X_preview.csv: {x_preview} (rows={len(pv)})")
else:
    print("\n未找到 X_preview.csv（可选；若需要可自行从 X.npy 导出切片）。")
PY
