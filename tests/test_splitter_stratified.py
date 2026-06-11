"""分层时间切分、robust 归一化相关测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dram2cnn.data.splitter import normalize_split_features_by_train, split_by_time_stratified


def _write_processed(
    tmp_path: Path,
    *,
    n: int,
    pos_indices: list[int],
) -> Path:
    proc = tmp_path / "processed"
    proc.mkdir(parents=True)
    y = np.zeros(n, dtype=np.int64)
    for i in pos_indices:
        y[i] = 1
    X = np.random.RandomState(0).rand(n, 2, 2, 2).astype(np.float32) + 1.0
    meta = pd.DataFrame(
        {
            "sid": [f"S{i % 5}" for i in range(n)],
            "predict_time": [f"2000-01-01 {i:02d}:00:00" for i in range(n)],
            "label": y,
            "predict_ts": np.arange(n, dtype=np.float64) * 300.0,
        }
    )
    np.save(proc / "X.npy", X)
    np.save(proc / "y.npy", y)
    meta.to_csv(proc / "meta.csv", index=False)
    return proc


def test_split_by_time_stratified_min_pos_per_split(tmp_path: Path) -> None:
    proc = _write_processed(tmp_path, n=30, pos_indices=[5, 10, 15, 20, 25, 28])
    spl = tmp_path / "splits"
    split_by_time_stratified(str(proc), str(spl), 0.6, 0.2, 0.2, min_pos_per_split=1)
    for name in ("train", "val", "test"):
        y = np.load(spl / f"y_{name}.npy")
        assert y.sum() >= 1, name
    stats = json.loads((spl / "normalization_stats.json").read_text(encoding="utf-8"))
    assert stats["method"] == "robust"


def test_split_by_time_stratified_rejects_too_few_positives(tmp_path: Path) -> None:
    proc = _write_processed(tmp_path, n=20, pos_indices=[3, 8])
    spl = tmp_path / "splits"
    with pytest.raises(ValueError, match="至少 3 个正样本"):
        split_by_time_stratified(str(proc), str(spl), 0.7, 0.15, 0.15, min_pos_per_split=1)


def test_robust_normalization_channelwise_matches_batched(tmp_path: Path) -> None:
    """分通道低内存归一化应与整表 float64 路径数值一致（小数组参考实现）。"""
    rng = np.random.RandomState(42)
    n, c, h, w = 48, 3, 4, 4
    X = (rng.rand(n, c, h, w).astype(np.float32) + 0.5) * rng.randint(1, 5, (1, c, 1, 1))
    n_train, n_val = 30, 9
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)
    spl = tmp_path / "splits"
    spl.mkdir()
    np.save(spl / "X_train.npy", X[train_idx])
    np.save(spl / "X_val.npy", X[val_idx])
    np.save(spl / "X_test.npy", X[test_idx])

    normalize_split_features_by_train(spl, method="robust", clip=20.0)
    X_train_norm = np.load(spl / "X_train.npy")

    ref_src = X[train_idx].astype(np.float64)
    center = np.median(ref_src, axis=0, keepdims=True)
    q75 = np.percentile(ref_src, 75, axis=0, keepdims=True)
    q25 = np.percentile(ref_src, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    mad = np.median(np.abs(ref_src - center), axis=0, keepdims=True)
    std = ref_src.std(axis=0, keepdims=True) + 1e-6
    scale = np.where(iqr > 1e-6, iqr, np.where(mad > 1e-6, mad, std)) + 1e-6
    ref = np.clip(((ref_src - center) / scale).astype(np.float32), -20.0, 20.0)
    np.testing.assert_allclose(X_train_norm, ref, rtol=1e-5, atol=1e-5)


def test_robust_normalization_rejects_double_normalize(tmp_path: Path) -> None:
    proc = _write_processed(tmp_path, n=12, pos_indices=[2, 7, 9])
    spl = tmp_path / "splits"
    split_by_time_stratified(str(proc), str(spl), 0.5, 0.25, 0.25, min_pos_per_split=1)
    with pytest.raises(ValueError, match="已归一化"):
        normalize_split_features_by_train(spl, method="robust")


def test_split_rerun_replaces_stale_normalization_stats(tmp_path: Path) -> None:
    """重跑 split 阶段时不应因旧 normalization_stats.json 失败（preprocess 可重复执行）。"""
    proc = _write_processed(tmp_path, n=12, pos_indices=[2, 7, 9])
    spl = tmp_path / "splits"
    split_by_time_stratified(str(proc), str(spl), 0.5, 0.25, 0.25, min_pos_per_split=1)
    split_by_time_stratified(str(proc), str(spl), 0.5, 0.25, 0.25, min_pos_per_split=1)
    stats = json.loads((spl / "normalization_stats.json").read_text(encoding="utf-8"))
    assert stats.get("normalized") is True
