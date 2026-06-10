#!/usr/bin/env bash
# =============================================================================
# dram2cnn — 快速查看预处理产物概览（仅诊断用）
#
# 【怎么用】
#   bash scripts/inspect_data.sh
#   bash scripts/inspect_data.sh configs/experiments/exp_horizon_30m.yaml
#   bash scripts/inspect_data.sh --processed-dir data/processed
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=lib/env_perf.sh
source "${SCRIPT_DIR}/lib/env_perf.sh"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODE="config"
EXP_CONFIG="configs/experiments/exp_baseline.yaml"
PROCESSED_DIR=""

if [[ "${1:-}" == "--processed-dir" ]]; then
  MODE="processed"
  PROCESSED_DIR="${2:-data/processed}"
elif [[ -n "${1:-}" ]]; then
  EXP_CONFIG="${1}"
fi

python3 - "${MODE}" "${PROCESSED_DIR}" "${EXP_CONFIG}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dram2cnn.utils.config import load_experiment_config, project_root

mode, processed_arg, exp_arg = sys.argv[1], sys.argv[2], sys.argv[3]
root = project_root()

if mode == "processed":
    p = Path(processed_arg)
    if not p.is_absolute():
        p = (root / p).resolve()
    print("模式: --processed-dir（不加载实验 YAML）")
else:
    cfg = load_experiment_config(exp_arg)
    processed_raw = Path(cfg["data"]["processed_dir"])
    p = processed_raw if processed_raw.is_absolute() else root / processed_raw
    print(f"实验配置: {exp_arg}")

print(f"processed_dir: {p}")

x_npy = p / "X.npy"
y_npy = p / "y.npy"
meta_csv = p / "meta.csv"
x_preview = p / "X_preview.csv"
y_csv = p / "y.csv"

missing = [str(fp) for fp in [x_npy, y_npy, meta_csv] if not fp.exists()]
if missing:
    raise FileNotFoundError(
        "缺少预处理产物，请先执行 `bash scripts/preprocess.sh <实验yaml>`:\n" + "\n".join(missing)
    )

X = np.load(x_npy, mmap_mode="r")
y = np.load(y_npy, mmap_mode="r")
pos = int((y == 1).sum())
neg = int((y == 0).sum())
total = int(len(y))

print("=== 数据概览（processed，原始特征尺度）===")
print("提示：训练/验证/测试实际使用 splits/X_*.npy（切分后按训练集 mean/std 归一化）。此处为预处理全量池。")
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

man_p = p / "preprocess_manifest.json"
if man_p.is_file():
    import json

    man = json.loads(man_p.read_text(encoding="utf-8"))
    print("\n=== preprocess_manifest（摘要）===")
    for k in (
        "index_features_path",
        "time_axis",
        "time_epoch_offset_absolute",
        "grid_shape",
        "grid_representation",
        "window_seconds",
        "label_horizon_seconds",
        "n_samples",
        "pos_rate",
        "n_empty_grids",
    ):
        if k in man:
            print(f"  {k}: {man[k]}")

if x_preview.exists():
    pv = pd.read_csv(x_preview)
    print(f"\n（可选）X_preview.csv: {x_preview} (rows={len(pv)})")
else:
    print("\n未找到 X_preview.csv（可选；若需要可自行从 X.npy 导出切片）。")
PY
