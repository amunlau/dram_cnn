"""deployment_kpi 合并逻辑测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dram2cnn.utils.deployment_kpi import KPI_COLUMNS, merge_all_strategies_kpi, write_kpi_csv


def test_write_kpi_csv_and_merge(tmp_path: Path) -> None:
    f1_rows = [
        {
            "strategy": "F1/time",
            "run_name": "exp_tianchi_full_f1_ddp",
            "f1": 0.7,
            "precision": 0.8,
            "recall": 0.6,
            "roc_auc": 0.9,
            "threshold": 0.5,
            "metrics_path": "/tmp/m1.json",
        }
    ]
    eng_rows = [
        {
            "strategy": "recall/time",
            "run_name": "exp_tianchi_full_recall_ddp",
            "f1": 0.4,
            "precision": 0.3,
            "recall": 0.8,
            "roc_auc": 0.75,
            "threshold": 0.05,
            "metrics_path": "/tmp/m2.json",
        }
    ]
    td = tmp_path / "tables"
    write_kpi_csv(f1_rows, td / "full_f1_time_vs_sid_deployment_kpi.csv")
    write_kpi_csv(eng_rows, td / "full_recall_vs_precision_deployment_kpi.csv")
    out = merge_all_strategies_kpi(tables_dir=td)
    assert out is not None
    df = pd.read_csv(out)
    assert list(df.columns) == list(KPI_COLUMNS)
    assert len(df) == 2
