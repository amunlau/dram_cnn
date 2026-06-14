from __future__ import annotations

"""train/val/test 切分与 **仅用训练集** 的归一化（网格 `[N,C,H,W]` + 可选 tabular `[N,F]`）。

默认 **`split_by_time_stratified`**：按时间排序后分配样本，并保证各 split 至少含 `min_pos_per_split` 个正例。
**`split_by_time`**：纯时间序切分（无标签分层）。
**`split_by_sid`**：按唯一 sid 切分（比例继承 `time_split`）；同一 sid 不跨 split；默认按 sid 级正例分层。

归一化默认 **`robust`**（median / IQR，对 log1p 后长尾更稳）；可选 `zscore`。
大数组路径使用 memmap + 按通道拟合/应用，降低 split/normalize 峰值内存。
"""

import gc
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NormMethod = Literal["zscore", "robust"]

# 写 split / 归一化时每次处理的样本行数（控制 X[idx] 临时块大小）
_SPLIT_SAVE_CHUNK_ROWS = 256


def _remove_orphan_tabular_splits(splits_dir: Path) -> None:
    """本次预处理未写出 `processed/tabular.npy` 时，删除 splits 下多余的 `tabular_*.npy`。

    当 `tabular_dim=0`（inventory 已拼入 X 通道）时，splits 目录不应保留 tabular 切分文件，
    否则 validate 会因维度不一致报错。
    """
    for name in ("train", "val", "test"):
        p = splits_dir / f"tabular_{name}.npy"
        if p.is_file():
            p.unlink()


def _fit_plane_normalizer(X: np.ndarray, method: NormMethod) -> tuple[np.ndarray, np.ndarray, dict]:
    """在 X [N,H,W] 上拟合 per-(H,W) 归一化 center/scale。"""
    if method == "robust":
        center = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        mad = np.median(np.abs(X - center), axis=0)
        std = X.std(axis=0) + 1e-6
        scale = np.where(iqr > 1e-6, iqr, np.where(mad > 1e-6, mad, std))
        note = (
            "Per (C,H,W) median/IQR from X_train; IQR≈0 时回退 MAD，再回退 std，"
            "缓解稀疏 voxel 在 val/test 上爆炸。"
        )
    elif method == "zscore":
        center = X.mean(axis=0)
        scale = X.std(axis=0) + 1e-6
        note = "Per (C,H,W) mean/std from X_train only."
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    return center, scale + 1e-6, {"method": method, "note": note}


def _fit_grid_normalizer(X_train: np.ndarray, method: NormMethod) -> tuple[np.ndarray, np.ndarray, dict]:
    """在 X_train [N,C,H,W] 或 tabular 伪网格 [N,F,1,1] 上拟合 per-(C,H,W) 归一化。"""
    if X_train.ndim == 4:
        c = int(X_train.shape[1])
        centers = []
        scales = []
        norm_meta: dict | None = None
        for ch in range(c):
            center_ch, scale_ch, meta = _fit_plane_normalizer(
                np.asarray(X_train[:, ch], dtype=np.float64), method
            )
            centers.append(center_ch)
            scales.append(scale_ch)
            if norm_meta is None:
                norm_meta = meta
        center = np.stack(centers, axis=0)
        scale = np.stack(scales, axis=0)
        assert norm_meta is not None
        return center, scale, norm_meta
    raise ValueError(f"Expected X_train [N,C,H,W], got {X_train.shape}")


def _save_indexed_rows(
    arr: np.ndarray,
    idx: np.ndarray,
    path: Path,
    *,
    chunk_rows: int = _SPLIT_SAVE_CHUNK_ROWS,
) -> None:
    """将 ``arr[idx]`` 写入 ``path``；4D 网格按行分块，避免一次性 materialize 整个 split。"""
    n_out = int(len(idx))
    if n_out == 0:
        np.save(path, arr[idx])
        return
    if arr.ndim == 1:
        out = np.empty(n_out, dtype=arr.dtype)
        for start in range(0, n_out, chunk_rows):
            sl = idx[start : start + chunk_rows]
            out[start : start + len(sl)] = arr[sl]
        np.save(path, out)
        return
    if arr.ndim == 2:
        out = np.empty((n_out, arr.shape[1]), dtype=arr.dtype)
        for start in range(0, n_out, chunk_rows):
            sl = idx[start : start + chunk_rows]
            out[start : start + len(sl)] = arr[sl]
        np.save(path, out)
        return
    if arr.ndim != 4:
        np.save(path, arr[idx])
        return

    shape = (n_out,) + tuple(int(x) for x in arr.shape[1:])
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.is_file():
        tmp.unlink()
    out_mm = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=arr.dtype, shape=shape)
    try:
        for start in range(0, n_out, chunk_rows):
            sl = idx[start : start + chunk_rows]
            block = np.asarray(arr[sl])
            end = start + block.shape[0]
            out_mm[start:end] = block
            del block
    finally:
        del out_mm
    os.replace(tmp, path)


def _apply_grid_normalization_channelwise(
    splits_dir: Path,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    clip: float | None,
) -> None:
    """按通道读写各 split 的 X_*.npy，峰值约为单通道 float64 + 输出 memmap。"""
    out = Path(splits_dir)
    c = int(center.shape[0])
    clip_val = float(clip) if clip is not None and float(clip) > 0 else None
    for name in ("train", "val", "test"):
        p = out / f"X_{name}.npy"
        X = np.load(p, mmap_mode="r")
        if X.ndim != 4 or X.shape[1:] != center.shape:
            raise ValueError(f"Shape mismatch {p.name}: {X.shape} vs center {center.shape}")
        tmp = p.with_suffix(p.suffix + ".normtmp")
        if tmp.is_file():
            tmp.unlink()
        Xout = np.lib.format.open_memmap(
            str(tmp), mode="w+", dtype=np.float32, shape=X.shape
        )
        try:
            for ch in range(c):
                X_ch = np.asarray(X[:, ch], dtype=np.float64)
                Xn = ((X_ch - center[ch]) / scale[ch]).astype(np.float32, copy=False)
                if clip_val is not None:
                    np.clip(Xn, -clip_val, clip_val, out=Xn)
                Xout[:, ch] = Xn
                del X_ch, Xn
        finally:
            del X, Xout
        os.replace(tmp, p)


def normalize_split_features_by_train(
    splits_dir: str | Path,
    *,
    method: NormMethod = "robust",
    clip: float | None = 20.0,
) -> None:
    """用 **X_train**（及若存在的 **tabular_train**）估归一化统计量，写回各 split 的 `X_*.npy` / `tabular_*.npy`。"""
    out = Path(splits_dir)
    stats_path = out / "normalization_stats.json"
    if stats_path.is_file():
        try:
            prev = json.loads(stats_path.read_text(encoding="utf-8"))
            if prev.get("normalized") is True:
                raise ValueError(
                    f"{stats_path} 表明 splits 已归一化；请勿二次 normalize。"
                    "请清空 splits 并从 processed/ 重跑 preprocess。"
                )
        except json.JSONDecodeError:
            pass
    train_path = out / "X_train.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}; cannot fit normalization.")
    X_train_mm = np.load(train_path, mmap_mode="r")
    if X_train_mm.ndim != 4:
        raise ValueError(f"Expected X_train [N,C,H,W], got {X_train_mm.shape}")

    c, h, w = (int(X_train_mm.shape[1]), int(X_train_mm.shape[2]), int(X_train_mm.shape[3]))
    center = np.empty((c, h, w), dtype=np.float64)
    scale = np.empty((c, h, w), dtype=np.float64)
    norm_meta: dict | None = None
    for ch in range(c):
        center[ch], scale[ch], meta = _fit_plane_normalizer(
            np.asarray(X_train_mm[:, ch], dtype=np.float64), method
        )
        if norm_meta is None:
            norm_meta = meta
    del X_train_mm
    gc.collect()
    assert norm_meta is not None

    _apply_grid_normalization_channelwise(out, center, scale, clip=clip)

    stats: dict = {
        "fit_split": "train",
        "normalized": True,
        "axes": [0],
        "method": norm_meta["method"],
        "center_shape": list(center.shape),
        "scale_shape": list(scale.shape),
        "epsilon": 1e-6,
        "clip": (float(clip) if clip is not None and float(clip) > 0 else None),
        "note": norm_meta["note"],
        "low_memory_channelwise": True,
    }
    if method == "zscore":
        stats["mean_shape"] = stats["center_shape"]
        stats["std_shape"] = stats["scale_shape"]

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
            tm, ts, tab_meta = _fit_grid_normalizer(T_train[:, :, np.newaxis, np.newaxis], method)
            tm = tm.reshape(1, -1)
            ts = ts.reshape(1, -1)
            for name in ("train", "val", "test"):
                tp = out / f"tabular_{name}.npy"
                T = np.load(tp).astype(np.float64, copy=False)
                if T.ndim != 2 or T.shape[1] != T_train.shape[1]:
                    raise ValueError(f"tabular shape mismatch {tp.name}: {T.shape} vs train {T_train.shape}")
                Tn = ((T - tm) / ts).astype(np.float32)
                np.save(tp, Tn)
            stats["tabular"] = {
                "center_shape": list(tm.squeeze(axis=0).shape),
                "scale_shape": list(ts.squeeze(axis=0).shape),
                "method": tab_meta["method"],
                "epsilon": 1e-6,
                "normalized": True,
                "note": tab_meta["note"],
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
    _save_indexed_rows(X, idx, out / f"X_{name}.npy")
    _save_indexed_rows(y, idx, out / f"y_{name}.npy")
    if tab is not None:
        _save_indexed_rows(tab, idx, out / f"tabular_{name}.npy")
    meta_out = meta.loc[idx].reset_index(drop=True)
    meta_out.to_csv(out / f"meta_{name}.csv", index=False)


def _load_processed_arrays(pdir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray | None]:
    """加载 processed 产物；X 用 memmap 只读，避免 split 阶段再占一份整表 RSS。"""
    for name in ("X.npy", "y.npy", "meta.csv"):
        p = pdir / name
        if not p.is_file():
            raise FileNotFoundError(f"缺少预处理产物 {p}；请先运行 preprocess。")
    X = np.load(pdir / "X.npy", mmap_mode="r")
    y = np.load(pdir / "y.npy")
    meta = pd.read_csv(pdir / "meta.csv")
    tab_path = pdir / "tabular.npy"
    tab = np.load(tab_path, mmap_mode="r") if tab_path.is_file() else None
    return X, y, meta, tab


def _resolve_norm_method(method: str | None) -> NormMethod:
    m = str(method or "robust").lower().strip()
    if m not in {"zscore", "robust"}:
        raise ValueError(f"normalization_method 须为 zscore 或 robust，当前: {method}")
    return m  # type: ignore[return-value]


def _compute_time_split_sizes(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """按比例计算 train/val/test 样本数，保证每档至少 1。"""
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
            "请增大索引样本量或调整 time_split。"
        )
    return n_train, n_val, n_test


def _resolve_norm_clip(data_cfg: dict | None) -> float | None:
    if data_cfg is None:
        return 20.0
    raw = data_cfg.get("normalization_clip")
    if raw in (None, "null", False):
        return None
    return float(raw)


def _write_splits_and_normalize(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    tab: np.ndarray | None,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    out: Path,
    *,
    normalization_method: str | None = "robust",
    normalization_clip: float | None = 20.0,
) -> None:
    """写入 train/val/test 数组并调用 ``normalize_split_features_by_train``（单一出口，避免三处重复）。"""
    out.mkdir(parents=True, exist_ok=True)
    if tab is None:
        _remove_orphan_tabular_splits(out)
    stats_path = out / "normalization_stats.json"
    if stats_path.is_file():
        stats_path.unlink()
    _save_split(X, y, meta, train_idx, out, "train", tab)
    _save_split(X, y, meta, val_idx, out, "val", tab)
    _save_split(X, y, meta, test_idx, out, "test", tab)
    del X, y, tab
    gc.collect()
    normalize_split_features_by_train(
        out,
        method=_resolve_norm_method(normalization_method),
        clip=normalization_clip,
    )


def split_by_sid(
    processed_dir: str,
    splits_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    *,
    stratify_sid_label: bool = True,
    normalization_method: str | None = "robust",
    normalization_clip: float | None = 20.0,
) -> None:
    """按 **sid** 划分 train/val/test，再归一化。

    在**唯一 sid** 集合上按 ``train_ratio/val_ratio/test_ratio`` 切分（默认与 ``time_split`` 一致 70/15/15），
    同一 sid 的全部样本只落在一个 split 中。默认按 sid 是否曾出现正样本（``label.max``）分层，
    使各 split 的正例服务器占比接近，便于与 ``time_stratified`` 公平对比。
    """
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")

    pdir = Path(processed_dir)
    X, y, meta, tab = _load_processed_arrays(pdir)

    sids = np.asarray(meta["sid"].unique().tolist(), dtype=object)
    if len(sids) < 5:
        raise ValueError(
            f"split_by_sid 需要至少 5 个唯一 sid（当前 {len(sids)} 个）。"
            "样本过少时请改用 configs 中 split_strategy: time_stratified。"
        )

    sid_has_pos = meta.groupby("sid")["label"].max()

    def _sid_stratify(sid_arr: np.ndarray) -> np.ndarray | None:
        if not stratify_sid_label:
            return None
        labels = sid_has_pos.reindex(sid_arr).fillna(0).astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            return None
        return labels

    def _split_sid_groups(
        sid_arr: np.ndarray, test_size: float, rs: int
    ) -> tuple[np.ndarray, np.ndarray]:
        strat = _sid_stratify(sid_arr)
        if strat is not None:
            try:
                a, b = train_test_split(
                    sid_arr, test_size=test_size, random_state=rs, stratify=strat
                )
                return np.asarray(a, dtype=object), np.asarray(b, dtype=object)
            except ValueError:
                pass
        a, b = train_test_split(sid_arr, test_size=test_size, random_state=rs)
        return np.asarray(a, dtype=object), np.asarray(b, dtype=object)

    train_val_sid, test_sid = _split_sid_groups(sids, test_ratio, random_state)
    val_rel = val_ratio / (train_ratio + val_ratio)
    train_sid, val_sid = _split_sid_groups(train_val_sid, val_rel, random_state)

    def idx_of(sid_set):
        return meta.index[meta["sid"].isin(set(sid_set))].to_numpy()

    out = Path(splits_dir)
    _write_splits_and_normalize(
        X, y, meta, tab,
        idx_of(train_sid), idx_of(val_sid), idx_of(test_sid),
        out,
        normalization_method=normalization_method,
        normalization_clip=normalization_clip,
    )


def split_by_time(
    processed_dir: str,
    splits_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    normalization_method: str | None = "robust",
    normalization_clip: float | None = 20.0,
) -> None:
    """按 **`predict_ts` 升序** 后按时间比例切分，再调用 `normalize_split_features_by_train`。"""
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")

    pdir = Path(processed_dir)
    X, y, meta, tab = _load_processed_arrays(pdir)
    if "predict_ts" not in meta.columns:
        raise ValueError("meta.csv 需含 predict_ts 列（由 preprocess 写入）")

    order = np.argsort(meta["predict_ts"].to_numpy(), kind="mergesort")
    n = len(meta)
    if n < 3:
        raise ValueError("样本过少，无法按时间切分（至少需要 3 条）")

    n_train, n_val, n_test = _compute_time_split_sizes(n, train_ratio, val_ratio, test_ratio)

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    out = Path(splits_dir)
    _write_splits_and_normalize(
        X, y, meta, tab,
        train_idx, val_idx, test_idx,
        out,
        normalization_method=normalization_method,
        normalization_clip=normalization_clip,
    )


def split_by_time_stratified(
    processed_dir: str,
    splits_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    min_pos_per_split: int = 1,
    normalization_method: str | None = "robust",
    normalization_clip: float | None = 20.0,
) -> None:
    """按时间分层切分：负样本按时间序填充各 split，正样本保证每档至少 `min_pos_per_split` 个。

    正样本按 `predict_ts` 排序后依次划入 train/val/test（保留时间先后，同时避免正例全落 test）。
    负样本在剩余容量内按时间序依次填入 train → val → test。
    """
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")
    min_pos = max(0, int(min_pos_per_split))

    pdir = Path(processed_dir)
    X, y, meta, tab = _load_processed_arrays(pdir)
    if "predict_ts" not in meta.columns:
        raise ValueError("meta.csv 需含 predict_ts 列（由 preprocess 写入）")

    n = len(y)
    if n < 3:
        raise ValueError("样本过少，无法切分（至少需要 3 条）")

    ts = meta["predict_ts"].to_numpy(dtype=np.float64)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_pos = int(len(pos_idx))

    if min_pos > 0 and n_pos < 3 * min_pos:
        raise ValueError(
            f"time_stratified 切分需要至少 {3 * min_pos} 个正样本（min_pos_per_split={min_pos}×3），"
            f"当前仅 {n_pos} 个。请重建索引（提高 min_positives / positive_sample_interval_seconds）"
            "或改用 split_strategy: sid。"
        )

    n_train, n_val, n_test = _compute_time_split_sizes(n, train_ratio, val_ratio, test_ratio)

    pos_sorted = pos_idx[np.argsort(ts[pos_idx], kind="mergesort")]
    neg_sorted = neg_idx[np.argsort(ts[neg_idx], kind="mergesort")]

    if n_pos > 0 and min_pos > 0:
        base = min_pos
        rem = n_pos - 3 * base
        if rem < 0:
            raise ValueError(
                f"正样本 {n_pos} 不足以满足 min_pos_per_split={min_pos}（需 >= {3 * min_pos}）。"
            )
        extra_tr = int(round(rem * train_ratio))
        extra_va = int(round(rem * val_ratio))
        extra_te = rem - extra_tr - extra_va
        n_pos_tr = base + extra_tr
        n_pos_va = base + extra_va
        n_pos_te = base + extra_te
        train_pos = pos_sorted[:n_pos_tr]
        val_pos = pos_sorted[n_pos_tr : n_pos_tr + n_pos_va]
        test_pos = pos_sorted[n_pos_tr + n_pos_va :]
    else:
        train_pos = np.array([], dtype=np.int64)
        val_pos = np.array([], dtype=np.int64)
        test_pos = np.array([], dtype=np.int64)

    cap_tr = max(0, n_train - len(train_pos))
    cap_va = max(0, n_val - len(val_pos))
    cap_te = max(0, n_test - len(test_pos))

    ptr = 0
    train_neg = neg_sorted[ptr : ptr + cap_tr]
    ptr += len(train_neg)
    val_neg = neg_sorted[ptr : ptr + cap_va]
    ptr += len(val_neg)
    test_neg = neg_sorted[ptr : ptr + cap_te]

    train_idx = np.sort(np.concatenate([train_pos, train_neg]))
    val_idx = np.sort(np.concatenate([val_pos, val_neg]))
    test_idx = np.sort(np.concatenate([test_pos, test_neg]))

    split_total = len(train_idx) + len(val_idx) + len(test_idx)
    if split_total != n:
        raise ValueError(
            f"time_stratified 切分样本数不守恒：train+val+test={split_total} != processed n={n}。"
            "请检查 time_split 比例与 min_pos_per_split。"
        )

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"分层时间切分失败：train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}。"
            "请检查样本量与 time_split 比例。"
        )

    out = Path(splits_dir)
    _write_splits_and_normalize(
        X, y, meta, tab,
        train_idx, val_idx, test_idx,
        out,
        normalization_method=normalization_method,
        normalization_clip=normalization_clip,
    )
