#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 快速查看 data/processed 中 X/y 形状、正负比例与 meta 样例（mmap 读大数组，不占满内存）。
# 注意：训练使用的是 data/splits 下 **已标准化** 的 X_train 等；此处为预处理全量池原始尺度。
#
# 前置条件：已运行 preprocess。
# -----------------------------------------------------------------------------
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}

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
