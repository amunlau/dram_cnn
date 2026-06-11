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


def test_robust_normalization_rejects_double_normalize(tmp_path: Path) -> None:
    proc = _write_processed(tmp_path, n=12, pos_indices=[2, 7, 9])
    spl = tmp_path / "splits"
    split_by_time_stratified(str(proc), str(spl), 0.5, 0.25, 0.25, min_pos_per_split=1)
    with pytest.raises(ValueError, match="已归一化"):
        normalize_split_features_by_train(spl, method="robust")
