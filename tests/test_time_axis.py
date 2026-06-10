"""相对时间轴：全局 min 平移与索引展示格式。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dram2cnn.data.time_axis import (
    compute_global_time_epoch_offset,
    index_file_uses_relative_display,
    index_timestamps_look_relative,
    predict_time_series_looks_relative,
    relative_seconds_to_predict_time_str,
    resolve_index_predict_ts,
    series_to_absolute_seconds,
)


def test_relative_display_roundtrip() -> None:
    rel = 12345.0
    s = relative_seconds_to_predict_time_str(rel)
    out = float(series_to_absolute_seconds(pd.Series([s]))[0])
    assert abs(out - rel) < 1e-6


def test_index_timestamps_look_relative() -> None:
    legacy = np.array([-6.2e10, -6.1e10])
    modern = np.array([0.0, 3600.0, 7200.0])
    calendar_2000 = np.array([946684800.0, 946685100.0])
    assert not index_timestamps_look_relative(legacy)
    assert index_timestamps_look_relative(modern)
    assert not index_timestamps_look_relative(calendar_2000)


def test_legacy_1985_not_relative() -> None:
    s = pd.Series(["1985-06-14 10:00:00"])
    abs_ts = series_to_absolute_seconds(s)
    assert not predict_time_series_looks_relative(s, abs_ts)
    assert not index_timestamps_look_relative(abs_ts, predict_time_strs=s)


def test_compute_global_time_epoch_offset(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    t0 = pd.Timestamp("2000-01-01 00:00:00")
    t1 = t0 + pd.Timedelta(hours=1)
    pd.DataFrame(
        {
            "sid": ["S1", "S1"],
            "error_time": [t0.isoformat(sep=" ", timespec="seconds"), t1.isoformat(sep=" ", timespec="seconds")],
        }
    ).to_csv(raw / "mcelog.csv", index=False)
    pd.DataFrame(
        {
            "sid": ["S1"],
            "failure_type": [2],
            "failed_time": [(t0 + pd.Timedelta(minutes=30)).isoformat(sep=" ", timespec="seconds")],
        }
    ).to_csv(raw / "tickets.csv", index=False)
    offset = compute_global_time_epoch_offset(raw / "mcelog.csv", raw / "tickets.csv")
    expected = float(series_to_absolute_seconds(pd.Series([t0.isoformat(sep=" ", timespec="seconds")]))[0])
    assert abs(offset - expected) < 1e-3


def test_resolve_legacy_index_with_offset(tmp_path: Path) -> None:
    t0 = pd.Timestamp("2000-01-01 00:00:00")
    offset = float(series_to_absolute_seconds(pd.Series([t0.isoformat(sep=" ", timespec="seconds")]))[0])
    rel = resolve_index_predict_ts(
        pd.Series(["2000-01-01 00:05:00"]),
        time_epoch_offset=offset,
        index_path=None,
    )
    assert abs(float(rel[0]) - 300.0) < 1e-3


def test_index_file_uses_relative_display_with_sidecar(tmp_path: Path) -> None:
    idx = tmp_path / "demo.csv"
    pd.DataFrame([["S1", "1970-01-02 00:00:00"]], columns=["sid", "predict_time"]).to_csv(idx, index=False)
    assert index_file_uses_relative_display(idx) is True
    (tmp_path / "demo.csv.stats.json").write_text('{"time_axis": "other"}', encoding="utf-8")
    assert index_file_uses_relative_display(idx) is False
