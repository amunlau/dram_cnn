"""pipeline_helpers 路径解析测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dram2cnn.utils.config import project_root
from dram2cnn.utils.pipeline_helpers import (
    have_feature_data_for_exp,
    have_processed_for_exp,
    processed_dir_for_exp,
    require_preprocess_manifest,
    resolve_feature_xy_paths,
    splits_dir_for_exp,
)


def test_reload_dram2cnn_modules_clears_cache() -> None:
    import dram2cnn.utils.pipeline_helpers as ph

    n = ph.reload_dram2cnn_modules()
    assert n >= 1
    from dram2cnn.utils.experiment_guards import assert_full_scale_index_not_stub

    assert callable(assert_full_scale_index_not_stub)


def test_baseline_exp_dirs_match_smoke_layout() -> None:
    root = project_root()
    exp = root / "configs/experiments/exp_baseline.yaml"
    assert processed_dir_for_exp(exp, root) == root / "data/processed_smoke"
    assert splits_dir_for_exp(exp, root) == root / "data/splits_smoke"


def test_resolve_feature_xy_paths_for_baseline_when_data_exists() -> None:
    root = project_root()
    exp = root / "configs/experiments/exp_baseline.yaml"
    if not have_feature_data_for_exp(exp, root):
        return
    x_path, y_path, label = resolve_feature_xy_paths(exp, root=root)
    assert x_path.is_file() and y_path.is_file()
    assert "processed_smoke" in str(x_path) or "splits_smoke" in str(x_path)
    assert "processed_smoke" in label or "splits_smoke" in label


def test_resolve_prefers_processed_over_splits(tmp_path: Path) -> None:
    proc = tmp_path / "data/processed_smoke"
    spl = tmp_path / "data/splits_smoke"
    proc.mkdir(parents=True)
    spl.mkdir(parents=True)
    np.save(proc / "X.npy", np.zeros((2, 2, 4, 4), dtype=np.float32))
    np.save(proc / "y.npy", np.array([0, 1]))
    np.save(spl / "X_train.npy", np.zeros((1, 2, 4, 4), dtype=np.float32))
    np.save(spl / "y_train.npy", np.array([1]))

    exp = tmp_path / "configs/experiments/exp_baseline.yaml"
    exp.parent.mkdir(parents=True)
    exp.write_text(
        (project_root() / "configs/experiments/exp_baseline.yaml").read_text(),
        encoding="utf-8",
    )

    x_path, y_path, label = resolve_feature_xy_paths(exp, root=tmp_path)
    assert x_path.name == "X.npy"
    assert y_path.name == "y.npy"
    assert "processed_smoke" in str(x_path)
    assert "processed_smoke" in label


def test_require_preprocess_manifest_baseline() -> None:
    root = project_root()
    exp = root / "configs/experiments/exp_baseline.yaml"
    proc = processed_dir_for_exp(exp, root)
    man = proc / "preprocess_manifest.json"
    if not man.is_file():
        return
    assert require_preprocess_manifest(exp, root=root) == man
