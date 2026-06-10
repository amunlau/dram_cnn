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


def _remove_orphan_tabular_splits(splits_dir: Path) -> None:
    """本次预处理未写出 `processed/tabular.npy` 时，删除 splits 下多余的 `tabular_*.npy`。

    当 `tabular_dim=0`（inventory 已拼入 X 通道）时，splits 目录不应保留 tabular 切分文件，
    否则 validate 会因维度不一致报错。
    """
    for name in ("train", "val", "test"):
        p = splits_dir / f"tabular_{name}.npy"
        if p.is_file():
            p.unlink()


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
        is_one_hot = bool(np.all((T_train == 0) | (T_train == 1)))
        if is_one_hot:
            stats["tabular"] = {
                "normalized": False,
                "note": "Skipped per-feature normalization (one-hot tabular).",
            }
        else:
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
                "normalized": True,
                "note": "Per-feature mean/std from tabular_train only.",
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


def _load_processed_arrays(pdir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray | None]:
    """加载 processed 产物；缺文件时给出明确路径提示。"""
    for name in ("X.npy", "y.npy", "meta.csv"):
        p = pdir / name
        if not p.is_file():
            raise FileNotFoundError(f"缺少预处理产物 {p}；请先运行 preprocess。")
    X = np.load(pdir / "X.npy")
    y = np.load(pdir / "y.npy")
    meta = pd.read_csv(pdir / "meta.csv")
    tab_path = pdir / "tabular.npy"
    tab = np.load(tab_path) if tab_path.is_file() else None
    return X, y, meta, tab


def split_by_sid(processed_dir: str, splits_dir: str, random_state: int = 42) -> None:
    """按 **sid** 划分 train/val/test（固定比例约 64% / 16% / 20%），再归一化。

    先对唯一 sid 做 80/20 分出 test，再对剩余 sid 做 80/20 分出 val，等价于 sid 级 64/16/20。
    同一 sid 的全部样本只落在一个 split 中。
    """
    pdir = Path(processed_dir)
    X, y, meta, tab = _load_processed_arrays(pdir)

    sids = np.asarray(meta["sid"].unique().tolist(), dtype=object)
    if len(sids) < 5:
        raise ValueError(
            f"split_by_sid 需要至少 5 个唯一 sid（当前 {len(sids)} 个）。"
            "样本过少时请改用 configs 中 split_strategy: time。"
        )
    train_sid, test_sid = train_test_split(sids, test_size=0.2, random_state=random_state)
    train_sid, val_sid = train_test_split(train_sid, test_size=0.2, random_state=random_state)

    def idx_of(sid_set):
        return meta.index[meta["sid"].isin(set(sid_set))].to_numpy()

    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    if tab is None:
        _remove_orphan_tabular_splits(out)
    _save_split(X, y, meta, idx_of(train_sid), out, "train", tab)
    _save_split(X, y, meta, idx_of(val_sid), out, "val", tab)
    _save_split(X, y, meta, idx_of(test_sid), out, "test", tab)
    normalize_split_features_by_train(out)


def split_by_time(processed_dir: str, splits_dir: str, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """按 **`predict_ts` 升序** 后按时间比例切分，再调用 `normalize_split_features_by_train`。"""
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")

    pdir = Path(processed_dir)
    X, y, meta, tab = _load_processed_arrays(pdir)
    if "predict_ts" not in meta.columns:
        raise ValueError("meta.csv 需含 predict_ts 列（由 preprocess 写入）")

    order = np.argsort(meta["predict_ts"].to_numpy(), kind="mergesort")
    meta_o = meta.iloc[order].reset_index(drop=True)
    X_o = X[order]
    y_o = y[order]
    tab_o = tab[order] if tab is not None else None
    n = len(meta_o)
    if n < 3:
        raise ValueError("样本过少，无法按时间切分（至少需要 3 条）")

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = n - n_train - n_val
    if n_test < 1:
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        n_test = n - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"时间切分比例导致某一档为空（n={n}，ratio={train_ratio}/{val_ratio}/{test_ratio}）；"
            "请增大 max_index_rows 或调整 time_split。"
        )

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)

    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    if tab_o is None:
        _remove_orphan_tabular_splits(out)
    _save_split(X_o, y_o, meta_o, train_idx, out, "train", tab_o)
    _save_split(X_o, y_o, meta_o, val_idx, out, "val", tab_o)
    _save_split(X_o, y_o, meta_o, test_idx, out, "test", tab_o)
    normalize_split_features_by_train(out)
