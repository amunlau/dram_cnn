import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dram2cnn.data.splitter import split_by_time
from dram2cnn.data.tianchi_build import materialize_dataset


def test_tianchi_materialize_and_split(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    proc = tmp_path / "processed"
    spl = tmp_path / "splits"

    pd.DataFrame(
        [
            ["S1", 1, 1, 1, 0, 0, 0, "2000-01-01 00:04:00"],
            ["S1", 1, 1, 1, 0, 0, 1, "2000-01-01 00:04:15"],
            ["S1", 1, 1, 1, 0, 0, 2, "2000-01-01 00:04:30"],
            ["S2", 1, 1, 1, 1, 1, 1, "2000-01-01 00:05:00"],
        ],
        columns=["sid", "memoryid", "rankid", "bankid", "row", "col", "error_type", "error_time"],
    ).to_csv(raw / "mcelog.csv", index=False)

    pd.DataFrame(
        [
            ["S1", 1, "2000-01-01 00:10:00"],
        ],
        columns=["sid", "failure_type", "failed_time"],
    ).to_csv(raw / "tickets.csv", index=False)

    pd.DataFrame(
        [
            ["S1", "2000-01-01 00:05:00"],
            ["S2", "2000-01-01 00:06:00"],
            ["S2", "2000-01-01 00:07:00"],
        ],
        columns=["sid", "predict_time"],
    ).to_csv(raw / "index.csv", index=False)

    pd.DataFrame(
        [
            ["S1", "M1", "A1", 8],
            ["S2", "M2", "B1", 12],
        ],
        columns=["sid", "server_manufacturer", "DRAM_model", "DIMM_number"],
    ).to_csv(raw / "inventory.csv", index=False)

    data_cfg = {
        "mcelog_path": str(raw / "mcelog.csv"),
        "tickets_path": str(raw / "tickets.csv"),
        "index_features_path": str(raw / "index.csv"),
        "inventory_path": str(raw / "inventory.csv"),
        "grid_representation": "legacy_rc_et",
        "window_seconds": 300,
        "label_backtrace_seconds": 3600,
        "label_failure_types": [1],
        "grid_max_row": 16,
        "grid_max_col": 16,
        "mcelog_chunk_rows": 1000,
        "max_index_rows": None,
    }
    model_cfg = {
        "in_channels": 3,
        "grid_height": 8,
        "grid_width": 8,
        "num_classes": 2,
        "backbone_channels": [8, 16],
        "head_hidden": 16,
        "tabular_dim": 15,
        "tabular_fuse_dim": 8,
        "dropout": 0.0,
    }
    materialize_dataset(data_cfg, model_cfg, proc)
    assert (proc / "X.npy").is_file()
    assert (proc / "tabular.npy").is_file()
    X = np.load(proc / "X.npy")
    T = np.load(proc / "tabular.npy")
    assert X.shape[1:] == (3, 8, 8)
    assert T.shape == (3, 15)
    # S1 窗口内 read+scrub+type0 均在 (0,0)：三通道均 >0
    assert (X[0, 0, 0, 0] > 0) and (X[0, 1, 0, 0] > 0) and (X[0, 2, 0, 0] > 0)

    split_by_time(str(proc), str(spl), train_ratio=0.34, val_ratio=0.33, test_ratio=0.33)
    assert (spl / "X_train.npy").is_file()
    assert (spl / "tabular_train.npy").is_file()
    assert (spl / "y_test.npy").is_file()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="grid fill parallel uses fork+COW on Linux only")
def test_materialize_parallel_matches_serial(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        [
            ["S1", 1, 1, 1, 0, 0, 0, "2000-01-01 00:04:00"],
            ["S1", 1, 1, 1, 0, 0, 1, "2000-01-01 00:04:15"],
            ["S1", 1, 1, 1, 0, 0, 2, "2000-01-01 00:04:30"],
            ["S2", 1, 1, 1, 1, 1, 1, "2000-01-01 00:05:00"],
        ],
        columns=["sid", "memoryid", "rankid", "bankid", "row", "col", "error_type", "error_time"],
    ).to_csv(raw / "mcelog.csv", index=False)

    pd.DataFrame(
        [["S1", 1, "2000-01-01 00:10:00"]],
        columns=["sid", "failure_type", "failed_time"],
    ).to_csv(raw / "tickets.csv", index=False)

    base = pd.Timestamp("2000-01-01 00:06:00")
    idx_rows = [["S1", (base + pd.Timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")] for i in range(40)]
    pd.DataFrame(idx_rows, columns=["sid", "predict_time"]).to_csv(raw / "index.csv", index=False)

    pd.DataFrame(
        [["S1", "M1", "A1", 8], ["S2", "M2", "B1", 12]],
        columns=["sid", "server_manufacturer", "DRAM_model", "DIMM_number"],
    ).to_csv(raw / "inventory.csv", index=False)

    data_base = {
        "mcelog_path": str(raw / "mcelog.csv"),
        "tickets_path": str(raw / "tickets.csv"),
        "index_features_path": str(raw / "index.csv"),
        "inventory_path": str(raw / "inventory.csv"),
        "grid_representation": "legacy_rc_et",
        "window_seconds": 300,
        "label_backtrace_seconds": 3600,
        "label_failure_types": [1],
        "grid_max_row": 16,
        "grid_max_col": 16,
        "mcelog_chunk_rows": 1000,
        "max_index_rows": None,
        "preprocess_parallel_frac": 0.6,
    }
    model_cfg = {
        "in_channels": 3,
        "grid_height": 8,
        "grid_width": 8,
        "num_classes": 2,
        "backbone_channels": [8, 16],
        "head_hidden": 16,
        "tabular_dim": 15,
        "tabular_fuse_dim": 8,
        "dropout": 0.0,
    }
    proc_a = tmp_path / "pa"
    proc_b = tmp_path / "pb"
    cfg_serial = {**data_base, "preprocess_workers": 1}
    cfg_par = {**data_base, "preprocess_workers": 8}
    materialize_dataset(cfg_serial, model_cfg, proc_a)
    materialize_dataset(cfg_par, model_cfg, proc_b)
    Xa = np.load(proc_a / "X.npy")
    Xb = np.load(proc_b / "X.npy")
    np.testing.assert_array_equal(Xa, Xb)
    man_b = json.loads((proc_b / "preprocess_manifest.json").read_text(encoding="utf-8"))
    assert man_b.get("preprocess_grid_fill_parallel") is True


def test_spatial_heatmap_binned_banks(tmp_path: Path):
    """spatial_heatmap：row/col 分箱 × bank 通道；inventory 走独立 tabular。"""
    raw = tmp_path / "raw"
    raw.mkdir()
    proc = tmp_path / "processed"

    pd.DataFrame(
        [["S1", 2, 2500, 64, 1, "2000-01-01 12:05:00"]],
        columns=["sid", "bankid", "row", "col", "error_type", "error_time"],
    ).to_csv(raw / "mcelog.csv", index=False)

    pd.DataFrame(columns=["sid", "failure_type", "failed_time"]).to_csv(raw / "tickets.csv", index=False)

    pd.DataFrame(
        [["S1", "2000-01-01 12:10:00"]],
        columns=["sid", "predict_time"],
    ).to_csv(raw / "index.csv", index=False)

    data_cfg = {
        "mcelog_path": str(raw / "mcelog.csv"),
        "tickets_path": str(raw / "tickets.csv"),
        "index_features_path": str(raw / "index.csv"),
        "inventory_path": None,
        "grid_representation": "spatial_heatmap",
        "row_bin_size": 1000,
        "col_bin_size": 32,
        "num_bank_channels": 16,
        "inventory_as_grid_channels": False,
        "window_seconds": 600,
        "label_backtrace_seconds": 3600,
        "label_failure_types": [1],
        "mcelog_chunk_rows": 1000,
        "max_index_rows": None,
        "mcelog_error_type_filter": [1, 2],
    }
    model_cfg = {
        "in_channels": 16,
        "grid_height": 4,
        "grid_width": 4,
        "num_classes": 2,
        "backbone_channels": [8, 16],
        "head_hidden": 16,
        "tabular_dim": 15,
        "tabular_fuse_dim": 8,
        "dropout": 0.0,
    }
    materialize_dataset(data_cfg, model_cfg, proc)
    X = np.load(proc / "X.npy")
    assert X.shape == (1, 16, 4, 4)
    # bank=2, row_bin=2, col_bin=2
    assert X[0, 2, 2, 2] > 0
    man = json.loads((proc / "preprocess_manifest.json").read_text(encoding="utf-8"))
    assert man.get("schema") == "tianchi_spatial_heatmap"
    assert man.get("tabular_dim") == 15
