import json
from pathlib import Path

import numpy as np
import pytest

from dram2cnn.engine.validator import check_preprocess_manifest


def test_preprocess_manifest_grid_mismatch_raises(tmp_path: Path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        json.dumps({"grid_shape": [2, 8, 8], "tabular_dim": 15}),
        encoding="utf-8",
    )
    model = {"in_channels": 3, "grid_height": 8, "grid_width": 8, "tabular_dim": 15}
    with pytest.raises(ValueError, match="grid_shape"):
        check_preprocess_manifest(proc, model, (3, 8, 8))


def test_preprocess_manifest_x_chw_mismatch_raises(tmp_path: Path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        json.dumps({"grid_shape": [3, 8, 8], "tabular_dim": 15}),
        encoding="utf-8",
    )
    np.save(proc / "X.npy", np.zeros((1, 3, 4, 4), dtype=np.float32))
    model = {"in_channels": 3, "grid_height": 8, "grid_width": 8, "tabular_dim": 15}
    with pytest.raises(ValueError, match="X.npy"):
        check_preprocess_manifest(proc, model, (3, 8, 8))


def test_preprocess_manifest_tabular_mismatch_raises(tmp_path: Path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        json.dumps({"grid_shape": [3, 8, 8], "tabular_dim": 10}),
        encoding="utf-8",
    )
    model = {"in_channels": 3, "grid_height": 8, "grid_width": 8, "tabular_dim": 15}
    with pytest.raises(ValueError, match="tabular_dim"):
        check_preprocess_manifest(proc, model, (3, 8, 8))


def test_preprocess_manifest_horizon_mismatch_raises(tmp_path: Path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        json.dumps({"grid_shape": [3, 8, 8], "window_seconds": 300.0, "label_horizon_seconds": 300.0}),
        encoding="utf-8",
    )
    model = {"in_channels": 3, "grid_height": 8, "grid_width": 8, "tabular_dim": 0}
    data_cfg = {"window_seconds": 1800, "label_horizon_seconds": 1800}
    with pytest.raises(ValueError, match="window_seconds|label_horizon_seconds"):
        check_preprocess_manifest(proc, model, (3, 8, 8), data_cfg=data_cfg, project_root_path=tmp_path)


def test_preprocess_manifest_mcelog_cap_mismatch_raises(tmp_path: Path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        json.dumps({"grid_shape": [3, 8, 8], "mcelog_max_rows": 800000}),
        encoding="utf-8",
    )
    model = {"in_channels": 3, "grid_height": 8, "grid_width": 8, "tabular_dim": 0}
    data_cfg = {"mcelog_max_rows": None}
    with pytest.raises(ValueError, match="mcelog_max_rows"):
        check_preprocess_manifest(proc, model, (3, 8, 8), data_cfg=data_cfg, project_root_path=tmp_path)


def test_preprocess_manifest_drop_empty_positive_mismatch_raises(tmp_path: Path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        json.dumps(
            {
                "grid_shape": [3, 8, 8],
                "drop_empty_positive_windows": True,
            }
        ),
        encoding="utf-8",
    )
    model = {"in_channels": 3, "grid_height": 8, "grid_width": 8, "tabular_dim": 0}
    data_cfg = {"drop_empty_positive_windows": False}
    with pytest.raises(ValueError, match="drop_empty_positive_windows"):
        check_preprocess_manifest(proc, model, (3, 8, 8), data_cfg=data_cfg, project_root_path=tmp_path)
