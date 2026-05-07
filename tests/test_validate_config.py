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
        },
        "model": {"tabular_dim": TABULAR_DIM},
        "infer": {"mode": "full"},
    }
    validate_config(cfg)
