"""索引 bounds 扫描与 preprocess 事件过滤一致性。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dram2cnn.data.index_builder import scan_mcelog_sid_bounds


def test_scan_bounds_ignores_non_ce_events_for_spatial(tmp_path: Path) -> None:
    """spatial 默认 bounds 仅统计 error_type∈{1,2}，避免 CE 外事件撑大时间轴导致空特征窗。"""
    mcelog = tmp_path / "mcelog.csv"
    rows = [
        # CE 事件集中在 0–600s
        ["S1", 1, 1, 1, 0, 0, 1, "2000-01-01 00:05:00"],
        ["S1", 1, 1, 1, 0, 0, 2, "2000-01-01 00:10:00"],
        # 非 CE（type=3）在更晚时刻——不应撑大 bounds 上界
        ["S1", 1, 1, 1, 0, 0, 3, "2000-01-01 02:00:00"],
    ]
    pd.DataFrame(
        rows,
        columns=["sid", "memoryid", "rankid", "bankid", "row", "col", "error_type", "error_time"],
    ).to_csv(mcelog, index=False)

    t0 = float((pd.Timestamp("2000-01-01 00:05:00") - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1))
    bounds_ce = scan_mcelog_sid_bounds(
        mcelog,
        chunk_rows=100,
        time_epoch_offset=t0,
        mcelog_error_type_filter=[1, 2],
        legacy_error_type_lt=None,
    )
    bounds_all = scan_mcelog_sid_bounds(
        mcelog,
        chunk_rows=100,
        time_epoch_offset=t0,
        mcelog_error_type_filter=None,
        legacy_error_type_lt=None,
    )
    assert "S1" in bounds_ce
    assert bounds_ce["S1"][1] < 4000.0  # ~300s 相对上界
    assert bounds_all["S1"][1] > 6500.0  # type=3 事件撑大上界（约 6900s）
