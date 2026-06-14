import pytest

from dram2cnn.data.inventory_encode import TABULAR_DIM
from dram2cnn.utils.config import validate_config


def test_validate_config_rejects_wrong_tabular_dim():
    cfg = {
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": __file__,
            "window_seconds": 300,
            "label_horizon_seconds": 300,
        },
        "model": {"tabular_dim": TABULAR_DIM + 1},
        "infer": {"mode": "full"},
    }
    with pytest.raises(ValueError, match="tabular_dim"):
        validate_config(cfg)


def test_validate_config_accepts_matching_tabular_dim():
    cfg = {
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": __file__,
            "window_seconds": 300,
            "label_horizon_seconds": 300,
            "inventory_as_grid_channels": False,
        },
        "model": {"in_channels": 16, "tabular_dim": TABULAR_DIM},
        "infer": {"mode": "full"},
    }
    validate_config(cfg)


def test_validate_config_rejects_spatial_in_channels_mismatch():
    cfg = {
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": __file__,
            "window_seconds": 300,
            "label_horizon_seconds": 300,
            "grid_representation": "spatial_heatmap",
            "inventory_as_grid_channels": True,
        },
        "model": {"in_channels": 16, "tabular_dim": 0},
        "infer": {"mode": "full"},
    }
    with pytest.raises(ValueError, match="in_channels"):
        validate_config(cfg)


def test_validate_config_rejects_bad_early_stop_metric():
    cfg = {
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": __file__,
            "window_seconds": 300,
            "label_horizon_seconds": 300,
        },
        "model": {"in_channels": 31, "tabular_dim": 0},
        "train": {"early_stop_metric": "not_a_metric"},
        "infer": {"mode": "full"},
    }
    with pytest.raises(ValueError, match="early_stop_metric"):
        validate_config(cfg)


def test_validate_config_rejects_bad_weight_scheme():
    cfg = {
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": __file__,
            "window_seconds": 300,
            "label_horizon_seconds": 300,
        },
        "model": {"in_channels": 31, "tabular_dim": 0},
        "train": {"imbalance": {"weight_scheme": "invalid"}},
        "infer": {"mode": "full"},
    }
    with pytest.raises(ValueError, match="weight_scheme"):
        validate_config(cfg)


def test_validate_config_rejects_bad_time_split_ratios():
    cfg = {
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": __file__,
            "window_seconds": 300,
            "label_horizon_seconds": 300,
            "split_strategy": "time",
            "time_split": {"train_ratio": 0.5, "val_ratio": 0.3, "test_ratio": 0.3},
        },
        "model": {"tabular_dim": 0},
        "infer": {"mode": "full"},
    }
    with pytest.raises(ValueError, match="time_split"):
        validate_config(cfg)


def test_validate_config_rejects_bad_ram_budget_frac():
    cfg = {
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": __file__,
            "window_seconds": 300,
            "label_horizon_seconds": 300,
            "ram_budget_frac": 1.5,
        },
        "model": {"in_channels": 31, "tabular_dim": 0},
        "train": {},
        "infer": {"mode": "full"},
    }
    with pytest.raises(ValueError, match="ram_budget_frac"):
        validate_config(cfg)


def test_validate_config_rejects_bad_merged_vram_frac():
    cfg = {
        "base": {"gpu": {"vram_budget_frac": 1.5}},
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": __file__,
            "window_seconds": 300,
            "label_horizon_seconds": 300,
        },
        "model": {"in_channels": 31, "tabular_dim": 0},
        "train": {},
        "infer": {"mode": "full"},
    }
    with pytest.raises(ValueError, match="vram_budget_frac"):
        validate_config(cfg)


def test_validate_config_allows_missing_regenerable_full_index():
    cfg = {
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": "/nonexistent/data/raw/features_index_h300s_step300s_full.csv",
            "window_seconds": 300,
            "label_horizon_seconds": 300,
        },
        "model": {"in_channels": 31, "tabular_dim": 0},
        "infer": {"mode": "full"},
    }
    validate_config(cfg)


def test_validate_config_requires_full_index_when_requested():
    cfg = {
        "data": {
            "mcelog_path": __file__,
            "tickets_path": __file__,
            "index_features_path": "/nonexistent/data/raw/features_index_h300s_step300s_full.csv",
            "window_seconds": 300,
            "label_horizon_seconds": 300,
        },
        "model": {"in_channels": 31, "tabular_dim": 0},
        "infer": {"mode": "full"},
    }
    with pytest.raises(FileNotFoundError, match="index_features_path"):
        validate_config(cfg, require_full_index=True)
