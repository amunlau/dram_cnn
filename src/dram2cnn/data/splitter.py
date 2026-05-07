from __future__ import annotations

"""train/val/test 切分与 **仅用训练集** 的归一化（网格 `[N,C,H,W]` + 可选 tabular `[N,F]`）。

默认 **`split_by_time`**：按 `predict_ts` 排序后按比例切段，减轻未来信息泄漏。
**`split_by_sid`**：按服务器 ID 分层划分（需样本量足够）。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_split_features_by_train(splits_dir: str | Path) -> None:
    """用 **X_train**（及若存在的 **tabular_train**）估 mean/std，写回各 split 的 `X_*.npy` / `tabular_*.npy`。"""
    out = Path(splits_dir)
    train_path = out / "X_train.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}; cannot fit normalization.")
    X_train = np.load(train_path).astype(np.float64, copy=False)
    if X_train.ndim != 4:
        raise ValueError(f"Expected X_train [N,C,H,W], got {X_train.shape}")
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    for name in ("train", "val", "test"):
        p = out / f"X_{name}.npy"
        X = np.load(p).astype(np.float64, copy=False)
        if X.ndim != 4 or X.shape[1:] != X_train.shape[1:]:
            raise ValueError(f"Shape mismatch {p.name}: {X.shape} vs train {X_train.shape}")
        Xn = ((X - mean) / std).astype(np.float32)
        np.save(p, Xn)

    stats: dict = {
        "fit_split": "train",
        "axes": [0],
        "mean_shape": list(mean.squeeze(axis=0).shape),
        "std_shape": list(std.squeeze(axis=0).shape),
        "epsilon": 1e-6,
        "note": "Per (C,H,W) mean/std from X_train only.",
    }

    tab_train_path = out / "tabular_train.npy"
    if tab_train_path.exists():
        T_train = np.load(tab_train_path).astype(np.float64, copy=False)
        if T_train.ndim != 2:
            raise ValueError(f"Expected tabular_train [N,F], got {T_train.shape}")
        tm = T_train.mean(axis=0, keepdims=True)
        ts = T_train.std(axis=0, keepdims=True) + 1e-6
        for name in ("train", "val", "test"):
            tp = out / f"tabular_{name}.npy"
            T = np.load(tp).astype(np.float64, copy=False)
            if T.ndim != 2 or T.shape[1] != T_train.shape[1]:
                raise ValueError(f"tabular shape mismatch {tp.name}: {T.shape} vs train {T_train.shape}")
            Tn = ((T - tm) / ts).astype(np.float32)
            np.save(tp, Tn)
        stats["tabular"] = {
            "mean_shape": list(tm.squeeze(axis=0).shape),
            "std_shape": list(ts.squeeze(axis=0).shape),
            "epsilon": 1e-6,
            "note": "Per-feature mean/std from tabular_train only (one-hot friendly).",
        }

    (out / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _save_split(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    idx: np.ndarray,
    out: Path,
    name: str,
    tab: np.ndarray | None = None,
) -> None:
    np.save(out / f"X_{name}.npy", X[idx])
    np.save(out / f"y_{name}.npy", y[idx])
    if tab is not None:
        np.save(out / f"tabular_{name}.npy", tab[idx])
    meta_out = meta.loc[idx].reset_index(drop=True)
    meta_out.to_csv(out / f"meta_{name}.csv", index=False)


def split_by_sid(processed_dir: str, splits_dir: str, random_state: int = 42) -> None:
    """按 **sid** 划分 train/val/test，再归一化（同 sid 的样本只落在一个 split）。"""
    pdir = Path(processed_dir)
    X = np.load(pdir / "X.npy")
    y = np.load(pdir / "y.npy")
    meta = pd.read_csv(pdir / "meta.csv")
    tab_path = pdir / "tabular.npy"
    tab = np.load(tab_path) if tab_path.is_file() else None

    sids = np.asarray(meta["sid"].unique().tolist(), dtype=object)
    train_sid, test_sid = train_test_split(sids, test_size=0.2, random_state=random_state)
    train_sid, val_sid = train_test_split(train_sid, test_size=0.2, random_state=random_state)

    def idx_of(sid_set):
        return meta.index[meta["sid"].isin(set(sid_set))].to_numpy()

    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_split(X, y, meta, idx_of(train_sid), out, "train", tab)
    _save_split(X, y, meta, idx_of(val_sid), out, "val", tab)
    _save_split(X, y, meta, idx_of(test_sid), out, "test", tab)
    normalize_split_features_by_train(out)


def split_by_time(processed_dir: str, splits_dir: str, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """按 **`predict_ts` 升序** 后按时间比例切分，再调用 `normalize_split_features_by_train`。"""
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")

    pdir = Path(processed_dir)
    X = np.load(pdir / "X.npy")
    y = np.load(pdir / "y.npy")
    meta = pd.read_csv(pdir / "meta.csv")
    tab_path = pdir / "tabular.npy"
    tab = np.load(tab_path) if tab_path.is_file() else None
    if "predict_ts" not in meta.columns:
        raise ValueError("meta.csv 需含 predict_ts 列（由 preprocess 写入）")

    order = np.argsort(meta["predict_ts"].to_numpy(), kind="mergesort")
    meta_o = meta.iloc[order].reset_index(drop=True)
    X_o = X[order]
    y_o = y[order]
    tab_o = tab[order] if tab is not None else None
    n = len(meta_o)
    if n < 3:
        raise ValueError("样本过少，无法按时间切分")

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError("时间切分比例导致某一档为空，请调整 ratio")

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)

    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_split(X_o, y_o, meta_o, train_idx, out, "train", tab_o)
    _save_split(X_o, y_o, meta_o, val_idx, out, "val", tab_o)
    _save_split(X_o, y_o, meta_o, test_idx, out, "test", tab_o)
    normalize_split_features_by_train(out)
