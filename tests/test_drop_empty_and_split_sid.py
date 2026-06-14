"""drop_empty_mcelog_windows 与 split_by_sid 行为测试。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dram2cnn.data.splitter import split_by_sid
from dram2cnn.data.tianchi_build import materialize_dataset


def _write_minimal_raw(raw: Path) -> None:
    pd.DataFrame(
        [
            ["S1", 1, 1, 1, 0, 0, 1, "2000-01-01 00:04:00"],
            ["S2", 1, 1, 1, 1, 1, 1, "2000-01-01 00:05:00"],
        ],
        columns=["sid", "memoryid", "rankid", "bankid", "row", "col", "error_type", "error_time"],
    ).to_csv(raw / "mcelog.csv", index=False)
    pd.DataFrame(
        [["S1", 1, "2000-01-01 00:10:00"]],
        columns=["sid", "failure_type", "failed_time"],
    ).to_csv(raw / "tickets.csv", index=False)
    pd.DataFrame(
        [
            ["S1", "2000-01-01 00:05:00"],
            ["S2", "2000-01-01 00:06:00"],
            ["S3", "2000-01-01 00:07:00"],
            ["S4", "2000-01-01 00:08:00"],
            ["S5", "2000-01-01 00:09:00"],
            ["S6", "2000-01-01 00:10:00"],
        ],
        columns=["sid", "predict_time"],
    ).to_csv(raw / "index.csv", index=False)


def test_drop_empty_mcelog_windows_keeps_positive(tmp_path: Path) -> None:
    """空 mcelog 窗：丢弃 y=0，保留 y=1；manifest 字段与样本数一致。"""
    raw = tmp_path / "raw"
    raw.mkdir()
    proc = tmp_path / "processed"
    _write_minimal_raw(raw)

    data_cfg = {
        "mcelog_path": str(raw / "mcelog.csv"),
        "tickets_path": str(raw / "tickets.csv"),
        "index_features_path": str(raw / "index.csv"),
        "inventory_path": None,
        "grid_representation": "legacy_rc_et",
        "window_seconds": 300,
        "label_horizon_seconds": 3600,
        "label_failure_types": [1],
        "grid_max_row": 16,
        "grid_max_col": 16,
        "mcelog_chunk_rows": 1000,
        "drop_empty_mcelog_windows": True,
    }
    model_cfg = {
        "in_channels": 3,
        "grid_height": 8,
        "grid_width": 8,
        "num_classes": 2,
        "backbone_channels": [8, 16],
        "head_hidden": 16,
        "tabular_dim": 0,
        "tabular_fuse_dim": 8,
        "dropout": 0.0,
    }
    manifest = materialize_dataset(data_cfg, model_cfg, proc)
    y = np.load(proc / "y.npy")
    meta = pd.read_csv(proc / "meta.csv")
    assert len(y) < 6
    assert int(y.sum()) >= 1
    assert "empty_mcelog_window" in meta.columns
    assert manifest["drop_empty_mcelog_windows"] is True
    assert int(manifest["n_samples_before_drop"]) == 6
    assert int(manifest["n_empty_grids_dropped"]) >= 1
    assert int(manifest["n_empty_grids_kept_positive"]) >= 0


def test_split_by_sid_no_sid_leakage(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    proc = tmp_path / "processed"
    spl = tmp_path / "splits"
    _write_minimal_raw(raw)

    n = 30
    y = np.zeros(n, dtype=np.int64)
    y[5] = 1
    X = np.random.RandomState(1).rand(n, 2, 4, 4).astype(np.float32)
    sids = [f"S{i % 6}" for i in range(n)]
    meta = pd.DataFrame(
        {
            "sid": sids,
            "predict_time": [f"2000-01-01 {i:02d}:00:00" for i in range(n)],
            "label": y,
            "predict_ts": np.arange(n, dtype=np.float64) * 60.0,
        }
    )
    proc.mkdir()
    np.save(proc / "X.npy", X)
    np.save(proc / "y.npy", y)
    meta.to_csv(proc / "meta.csv", index=False)

    split_by_sid(str(proc), str(spl), random_state=0, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_s = set(pd.read_csv(spl / "meta_train.csv")["sid"])
    val_s = set(pd.read_csv(spl / "meta_val.csv")["sid"])
    test_s = set(pd.read_csv(spl / "meta_test.csv")["sid"])
    assert train_s.isdisjoint(val_s)
    assert train_s.isdisjoint(test_s)
    assert val_s.isdisjoint(test_s)
    n_total = len(train_s) + len(val_s) + len(test_s)
    assert abs(len(test_s) / n_total - 0.15) < 0.08
    stats = json.loads((spl / "normalization_stats.json").read_text(encoding="utf-8"))
    assert stats.get("normalized") is True
