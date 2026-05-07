import json
from pathlib import Path

import numpy as np
import pytest

from dram2cnn.engine.validator import _check_preprocess_manifest


def test_preprocess_manifest_grid_mismatch_raises(tmp_path: Path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        json.dumps({"grid_shape": [2, 8, 8], "tabular_dim": 15}),
        encoding="utf-8",
    )
    model = {"in_channels": 3, "grid_height": 8, "grid_width": 8, "tabular_dim": 15}
    with pytest.raises(ValueError, match="grid_shape"):
        _check_preprocess_manifest(proc, model, (3, 8, 8))


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
        _check_preprocess_manifest(proc, model, (3, 8, 8))


def test_preprocess_manifest_tabular_mismatch_raises(tmp_path: Path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        json.dumps({"grid_shape": [3, 8, 8], "tabular_dim": 10}),
        encoding="utf-8",
    )
    model = {"in_channels": 3, "grid_height": 8, "grid_width": 8, "tabular_dim": 15}
    with pytest.raises(ValueError, match="tabular_dim"):
        _check_preprocess_manifest(proc, model, (3, 8, 8))
