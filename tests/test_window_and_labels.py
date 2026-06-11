"""特征窗 / 标签窗 / 相对时间轴边界单测。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dram2cnn.data.index_builder import IndexBuildConfig, _positive_candidates, _valid_predict_bounds
from dram2cnn.data.time_axis import (
    index_file_uses_relative_display,
    index_time_axis_from_sidecar,
    predict_time_series_looks_relative,
    resolve_index_predict_ts,
)
from dram2cnn.data.tianchi_build import _validate_index_sidecar_against_config, merge_ticket_labels, window_slice


def test_window_slice_half_open() -> None:
    """半开 [t_end-W, t_end)：恰在 t_end 的事件 excluded，恰在 t_end-W included。"""
    events = {
        "S1": {
            "t": np.array([0.0, 50.0, 100.0, 150.0], dtype=np.float64),
            "row": np.array([0, 0, 0, 0], dtype=np.int32),
            "col": np.array([0, 0, 0, 0], dtype=np.int32),
            "etype": np.array([1, 1, 1, 1], dtype=np.int8),
            "bank": np.array([0, 0, 0, 0], dtype=np.int8),
        }
    }
    sl = window_slice(events, "S1", t_end=100.0, window_sec=100.0)
    assert sl is not None
    rows, _, _, t_slice, _ = sl
    assert t_slice.tolist() == [0.0, 50.0]
    assert len(rows) == 2

    sl_end = window_slice(events, "S1", t_end=100.0, window_sec=50.0)
    assert sl_end is not None
    _, _, _, t_end_slice, _ = sl_end
    assert t_end_slice.tolist() == [50.0]


def test_merge_ticket_labels_closed_interval(tmp_path: Path) -> None:
    """闭区间 [predict_ts, predict_ts+H]：边界 failed_ts 应打标。"""
    t0 = pd.Timestamp("2000-01-01 00:00:00")
    offset = float((t0 - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1))

    tickets = tmp_path / "tickets.csv"
    pd.DataFrame(
        [
            ["S1", 1, "2000-01-01 00:10:00"],
            ["S2", 1, "2000-01-01 00:15:00"],
        ],
        columns=["sid", "failure_type", "failed_time"],
    ).to_csv(tickets, index=False)

    idx = pd.DataFrame(
        [
            ["S1", 600.0],
            ["S2", 900.0],
        ],
        columns=["sid", "predict_ts"],
    )
    y = merge_ticket_labels(idx, tickets, [1], label_horizon_seconds=300.0, time_epoch_offset=offset)
    assert y.tolist() == [1, 1]


def test_positive_candidates_skip_sid_without_mcelog() -> None:
    sid_bounds = {"S1": (0.0, 3600.0)}
    by_sid_fail = {"S1": np.array([1800.0]), "S_no_mcelog": np.array([900.0])}
    cfg = IndexBuildConfig(
        mcelog_path=Path("dummy.csv"),
        tickets_path=Path("dummy.csv"),
        sample_interval_seconds=300,
        label_horizon_seconds=300,
        window_seconds=300,
        positive_prefault_seconds=600,
    )
    pos = _positive_candidates(by_sid_fail, cfg, sid_bounds)
    assert set(pos["sid"].unique()) == {"S1"}
    assert "S_no_mcelog" not in set(pos["sid"])


def test_predict_time_series_looks_relative_legacy_vs_build() -> None:
    legacy_2000 = pd.Series(["2000-01-01 00:05:00", "2000-01-01 00:06:00"])
    legacy_abs = np.array([946685100.0, 946685400.0])
    assert not predict_time_series_looks_relative(legacy_2000, legacy_abs)

    legacy_1985 = pd.Series(["1985-06-14 10:00:00"])
    legacy_1985_abs = np.array([488_000_000.0])
    assert not predict_time_series_looks_relative(legacy_1985, legacy_1985_abs)

    relative = pd.Series(["1970-06-02 21:13:06", "1970-02-24 21:17:56"])
    relative_abs = np.array([13_920_000.0, 5_000_000.0])
    assert predict_time_series_looks_relative(relative, relative_abs)


def test_index_sidecar_and_resolve(tmp_path: Path) -> None:
    idx = tmp_path / "index.csv"
    pd.DataFrame([["S1", "1970-01-01 00:10:00"]], columns=["sid", "predict_time"]).to_csv(idx, index=False)
    sidecar = tmp_path / "index.csv.stats.json"
    sidecar.write_text(
        '{"time_axis": "relative_seconds_since_global_min", "time_epoch_offset_absolute": 0.0}',
        encoding="utf-8",
    )
    assert index_time_axis_from_sidecar(idx) is True
    assert index_file_uses_relative_display(idx) is True
    rel = resolve_index_predict_ts(pd.Series(["1970-01-01 00:10:00"]), time_epoch_offset=None, index_path=idx)
    assert abs(float(rel[0]) - 600.0) < 1e-3


def test_validate_index_sidecar_horizon_mismatch() -> None:
    idx_stats = {
        "label_horizon_seconds": 300,
        "window_seconds": 300,
        "time_epoch_offset_absolute": 0.0,
    }
    data_cfg = {"label_horizon_seconds": 1800, "window_seconds": 1800}
    with pytest.raises(ValueError, match="label_horizon_seconds"):
        _validate_index_sidecar_against_config(
            idx_stats, data_cfg, mcelog_cap=None, time_epoch_offset=0.0
        )

    idx_stats["label_horizon_seconds"] = 1800
    with pytest.raises(ValueError, match="window_seconds"):
        _validate_index_sidecar_against_config(
            idx_stats, data_cfg, mcelog_cap=None, time_epoch_offset=0.0
        )


def test_valid_predict_bounds_none_when_window_too_long() -> None:
    vb = _valid_predict_bounds("S1", {"S1": (0.0, 500.0)}, win=300.0, horizon=300.0)
    assert vb is None
