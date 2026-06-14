"""Trainer / evaluator 数值守卫回归测试。"""

import numpy as np
import pandas as pd
import pytest
import torch

from dram2cnn.engine.trainer import (
    _early_stop_score,
    _is_valid_early_stop_score,
)
from dram2cnn.engine.evaluator import evaluate
from dram2cnn.utils.experiment_guards import (
    is_deep_grid_cnn,
    state_dict_has_non_finite,
    warn_unstable_deep_focal_combo,
)


def test_is_valid_early_stop_score_rejects_sentinel():
    assert _is_valid_early_stop_score(-1.0) is False
    assert _is_valid_early_stop_score(0.0) is True
    assert _is_valid_early_stop_score(0.15) is True


def test_early_stop_score_returns_sentinel_when_auc_missing():
    m = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.5}
    assert _early_stop_score("avg_precision", m, m, None, None) == -1.0


def test_state_dict_has_non_finite():
    sd = {"w": torch.tensor([1.0, 2.0])}
    assert state_dict_has_non_finite(sd) is False
    sd_bad = {"w": torch.tensor([1.0, float("nan")])}
    assert state_dict_has_non_finite(sd_bad) is True


def test_is_deep_grid_cnn():
    assert is_deep_grid_cnn({"blocks_per_stage": 2, "stem_channels": 64, "backbone_channels": [64, 128, 256, 256]})
    assert not is_deep_grid_cnn({"blocks_per_stage": 1, "stem_channels": 32, "backbone_channels": [32, 64, 128, 128]})


def test_warn_unstable_deep_focal_combo(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    cfg = {
        "model": {"blocks_per_stage": 2, "stem_channels": 64, "backbone_channels": [64, 128, 256, 256]},
        "train": {"imbalance": {"focal_gamma": 0.5}},
    }

    class _Log:
        def warning(self, msg, *args):
            import logging

            logging.warning(msg, *args)

    warn_unstable_deep_focal_combo(cfg, _Log())
    assert "focal_gamma" in caplog.text


def test_evaluate_rejects_nan_y_prob(tmp_path):
    pred = tmp_path / "predictions.csv"
    pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 0], "y_prob": [0.1, float("nan")]}).to_csv(pred, index=False)
    with pytest.raises(ValueError, match="y_prob"):
        evaluate(str(pred), str(tmp_path / "metrics.json"))
