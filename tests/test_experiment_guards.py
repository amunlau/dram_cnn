from pathlib import Path

import json

import pytest
import torch

from dram2cnn.engine.validator import validate_experiment
from dram2cnn.utils.config import load_experiment_config
from dram2cnn.utils.experiment_guards import (
    assert_checkpoint_matches_experiment,
    assert_full_best_index_not_stub,
    assert_predictions_match_experiment,
    assert_processed_matches_experiment,
    is_stub_index_path,
    splits_match_processed,
    state_dict_has_non_finite,
)


def test_is_stub_index_path() -> None:
    assert is_stub_index_path("data/index_stubs/features_5m_index_demo.csv")
    assert not is_stub_index_path("data/raw/features_index_h300s_step300s_full.csv")


def test_assert_full_best_index_not_stub_raises() -> None:
    cfg = {
        "base": {"run_name": "exp_tianchi_full_best_ddp"},
        "data": {"index_features_path": "data/index_stubs/features_5m_index_demo.csv"},
    }
    with pytest.raises(RuntimeError, match="演示索引"):
        assert_full_best_index_not_stub(cfg)


def test_assert_predictions_match_experiment_mismatch(tmp_path: Path) -> None:
    art = tmp_path / "infer"
    art.mkdir()
    (art / "predictions_manifest.json").write_text(
        '{"splits_dir": "/tmp/other/splits"}',
        encoding="utf-8",
    )
    cfg = {"data": {"splits_dir": str(tmp_path / "splits")}}
    (tmp_path / "splits").mkdir()
    with pytest.raises(ValueError, match="splits_dir"):
        assert_predictions_match_experiment(cfg, art, tmp_path)


def test_assert_predictions_match_requires_manifest_when_csv_exists(tmp_path: Path) -> None:
    art = tmp_path / "infer"
    art.mkdir()
    (art / "predictions.csv").write_text("y_true,y_pred\n0,0\n", encoding="utf-8")
    cfg = {"data": {"splits_dir": str(tmp_path / "splits")}}
    with pytest.raises(ValueError, match="predictions_manifest"):
        assert_predictions_match_experiment(cfg, art, tmp_path)


def test_preprocess_full_best_stub_raises() -> None:
    cfg = {
        "base": {"run_name": "exp_tianchi_full_best_ddp"},
        "data": {"index_features_path": "data/index_stubs/features_5m_index_demo.csv"},
    }
    with pytest.raises(RuntimeError, match="run_preprocess"):
        assert_full_best_index_not_stub(cfg, context="run_preprocess")


def test_validate_horizon_config_only_with_no_require_splits(tmp_path: Path) -> None:
    """有 processed 产物时，horizon 实验仍可做配置级 validate（不依赖仓库磁盘状态）。"""
    import numpy as np

    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        '{"grid_shape": [31, 128, 32], "window_seconds": 1800, "label_horizon_seconds": 1800}',
        encoding="utf-8",
    )
    np.save(proc / "X.npy", np.zeros((4, 31, 128, 32), dtype=np.float32))
    np.save(proc / "y.npy", np.zeros(4, dtype=np.int64))

    cfg = load_experiment_config("configs/experiments/exp_horizon_30m.yaml")
    cfg["data"] = {**cfg["data"], "processed_dir": str(proc)}
    report = validate_experiment(cfg, require_splits=False)
    assert "expected_shape" in report
    assert report["checks"]["processed"]["exists"] is True


def test_assert_processed_matches_experiment_horizon_mismatch(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        '{"grid_shape": [31, 128, 32], "window_seconds": 300, "label_horizon_seconds": 300}',
        encoding="utf-8",
    )
    cfg = {
        "data": {
            "processed_dir": str(proc),
            "window_seconds": 1800,
            "label_horizon_seconds": 1800,
            "index_features_path": str(tmp_path / "idx.csv"),
        },
        "model": {"in_channels": 31, "grid_height": 128, "grid_width": 32, "tabular_dim": 0},
    }
    with pytest.raises(ValueError, match="window_seconds|label_horizon"):
        assert_processed_matches_experiment(cfg, tmp_path)


def test_splits_match_processed_row_count(tmp_path: Path) -> None:
    import numpy as np

    proc = tmp_path / "processed"
    spl = tmp_path / "splits"
    proc.mkdir()
    spl.mkdir()
    (proc / "preprocess_manifest.json").write_text(
        '{"grid_shape": [31, 128, 32], "n_samples": 4, "window_seconds": 300, "label_horizon_seconds": 300}',
        encoding="utf-8",
    )
    for tag, n in (("train", 2), ("val", 1), ("test", 1)):
        np.save(spl / f"X_{tag}.npy", np.zeros((n, 31, 128, 32), dtype=np.float32))
        np.save(spl / f"y_{tag}.npy", np.zeros(n, dtype=np.int64))
    (spl / "normalization_stats.json").write_text("{}", encoding="utf-8")
    cfg = {
        "data": {
            "processed_dir": str(proc),
            "splits_dir": str(spl),
            "window_seconds": 300,
            "label_horizon_seconds": 300,
        },
        "model": {"in_channels": 31, "grid_height": 128, "grid_width": 32, "tabular_dim": 0},
    }
    assert splits_match_processed(cfg, tmp_path) is True
    np.save(spl / "y_test.npy", np.zeros(2, dtype=np.int64))
    assert splits_match_processed(cfg, tmp_path) is False


def test_state_dict_has_non_finite() -> None:
    ok = {"w": torch.tensor([1.0, 2.0])}
    bad = {"w": torch.tensor([1.0, float("nan")])}
    assert state_dict_has_non_finite(ok) is False
    assert state_dict_has_non_finite(bad) is True


def test_assert_checkpoint_matches_experiment_ok(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = {"arch": "grid_cnn", "in_channels": 31, "grid_height": 128, "grid_width": 32}
    (run_dir / "config_snapshot.json").write_text(
        json.dumps({"model": model, "base": {"run_name": "exp_a"}}),
        encoding="utf-8",
    )
    cfg = {"model": model, "base": {"run_name": "exp_a"}}
    assert_checkpoint_matches_experiment(cfg, run_dir, tmp_path)


def test_assert_checkpoint_matches_experiment_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.json").write_text(
        json.dumps({"model": {"in_channels": 31}, "base": {"run_name": "exp_a"}}),
        encoding="utf-8",
    )
    cfg = {"model": {"in_channels": 16}, "base": {"run_name": "exp_b"}}
    with pytest.raises(RuntimeError, match="model 配置不一致"):
        assert_checkpoint_matches_experiment(cfg, run_dir, tmp_path)
