"""索引表生成器单元测试（合成 mcelog / tickets）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dram2cnn.data.index_builder import IndexBuildConfig, build_index_to_csv


def _write_synthetic_raw(tmp: Path) -> None:
    mcelog = tmp / "mcelog.csv"
    tickets = tmp / "tickets.csv"
    rows = []
    base = "0001-06-14 10:00:00"
    for i in range(60):
        t = pd.Timestamp(base) + pd.Timedelta(minutes=i * 5)
        rows.append(
            {
                "sid": "Server_A",
                "memoryid": 1,
                "rankid": 1,
                "bankid": 0,
                "row": 100,
                "col": 8,
                "error_type": 1,
                "error_time": t.isoformat(sep=" ", timespec="seconds"),
            }
        )
    for i in range(40):
        t = pd.Timestamp(base) + pd.Timedelta(minutes=i * 5)
        rows.append(
            {
                "sid": "Server_B",
                "memoryid": 2,
                "rankid": 1,
                "bankid": 1,
                "row": 200,
                "col": 16,
                "error_type": 2,
                "error_time": t.isoformat(sep=" ", timespec="seconds"),
            }
        )
    pd.DataFrame(rows).to_csv(mcelog, index=False)
    fail_t = pd.Timestamp(base) + pd.Timedelta(hours=4)
    pd.DataFrame(
        [
            {"sid": "Server_A", "failure_type": 2, "failed_time": fail_t.isoformat(sep=" ", timespec="seconds")},
            {"sid": "Server_A", "failure_type": 2, "failed_time": (fail_t + pd.Timedelta(hours=2)).isoformat(sep=" ", timespec="seconds")},
        ]
    ).to_csv(tickets, index=False)


def test_build_index_smoke_synthetic(tmp_path: Path) -> None:
    _write_synthetic_raw(tmp_path)
    cfg = IndexBuildConfig(
        mcelog_path=tmp_path / "mcelog.csv",
        tickets_path=tmp_path / "tickets.csv",
        sample_interval_seconds=300,
        label_horizon_seconds=300,
        window_seconds=300,
        positive_prefault_seconds=3600,
        positive_sample_interval_seconds=60,
        target_positive_ratio=0.3,
        seed=0,
        max_rows=200,
        mcelog_scan_rows=None,
        min_positives=1,
        min_total_rows=5,
        neg_sample_strategy="random",
    )
    out = tmp_path / "index.csv"
    summary = build_index_to_csv(cfg, out, write_stats_json=False)
    assert out.is_file()
    assert summary["n_pos"] >= 1
    assert summary["n_rows"] >= 5
    assert summary["pos_rate"] <= 0.95

    df = pd.read_csv(out)
    assert list(df.columns) == ["sid", "predict_time"]


def test_positive_zone_allows_failure_after_mcelog_tmax(tmp_path: Path) -> None:
    """故障时刻晚于 mcelog 扫描上界时，正样本区仍应在 [ft−H, ft] 生成锚点。"""
    mcelog = tmp_path / "mcelog.csv"
    tickets = tmp_path / "tickets.csv"
    base = pd.Timestamp("2000-01-01 00:00:00")
    rows = []
    for i in range(20):
        rows.append(
            {
                "sid": "Server_A",
                "memoryid": 1,
                "rankid": 1,
                "bankid": 0,
                "row": 1,
                "col": 1,
                "error_type": 1,
                "error_time": (base + pd.Timedelta(minutes=i * 30)).isoformat(sep=" ", timespec="seconds"),
            }
        )
    pd.DataFrame(
        rows,
        columns=["sid", "memoryid", "rankid", "bankid", "row", "col", "error_type", "error_time"],
    ).to_csv(mcelog, index=False)
    fail_t = base + pd.Timedelta(days=300)
    pd.DataFrame(
        [{"sid": "Server_A", "failure_type": 1, "failed_time": fail_t.isoformat(sep=" ", timespec="seconds")}],
        columns=["sid", "failure_type", "failed_time"],
    ).to_csv(tickets, index=False)

    cfg = IndexBuildConfig(
        mcelog_path=mcelog,
        tickets_path=tickets,
        sample_interval_seconds=300,
        label_horizon_seconds=300,
        window_seconds=300,
        positive_prefault_seconds=600,
        positive_sample_interval_seconds=60,
        target_positive_ratio=0.5,
        seed=0,
        max_rows=50,
        mcelog_scan_rows=None,
        min_positives=1,
        min_total_rows=2,
        neg_sample_strategy="random",
    )
    summary = build_index_to_csv(cfg, tmp_path / "late_fail.csv", write_stats_json=False)
    assert summary["n_pos"] >= 1

    """正样本区细步长应比统一步长 300s 产生更多 y=1 锚点。"""
    _write_synthetic_raw(tmp_path)
    base = dict(
        mcelog_path=tmp_path / "mcelog.csv",
        tickets_path=tmp_path / "tickets.csv",
        sample_interval_seconds=300,
        label_horizon_seconds=300,
        window_seconds=300,
        positive_prefault_seconds=3600,
        target_positive_ratio=0.5,
        seed=0,
        mcelog_scan_rows=None,
        min_positives=1,
        min_total_rows=5,
        neg_sample_strategy="random",
        max_rows=500,
    )
    coarse = IndexBuildConfig(**base, positive_sample_interval_seconds=300)
    fine = IndexBuildConfig(**base, positive_sample_interval_seconds=60)
    coarse_sum = build_index_to_csv(coarse, tmp_path / "coarse.csv", write_stats_json=False)
    fine_sum = build_index_to_csv(fine, tmp_path / "fine.csv", write_stats_json=False)
    assert fine_sum["n_pos"] >= coarse_sum["n_pos"]
    assert fine_sum["n_pos"] > coarse_sum["n_pos"]
