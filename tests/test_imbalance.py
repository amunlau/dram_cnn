import numpy as np
import pytest
import torch

from dram2cnn.data.dataset import IndexMappedDataset
from dram2cnn.engine.imbalance import (
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_POSITIVE_OVERSAMPLE_FACTOR,
    DEFAULT_POSITIVE_WEIGHT_CAP,
    build_positive_oversample_index_map,
    compute_class_weights_tensor,
    compute_per_sample_weights,
    focal_gamma_from_cfg,
    positive_oversample_factor_from_cfg,
)


def test_compute_class_weights_sqrt_and_cap():
    y = np.array([0] * 900 + [1] * 100, dtype=np.int64)
    w = compute_class_weights_tensor(
        y,
        {"weight_scheme": "sqrt", "positive_weight_cap": 50.0},
    )
    assert w.shape == (2,)
    # ratio=9, sqrt=3
    assert torch.isclose(w[1], torch.tensor(3.0))


def test_compute_class_weights_inverse_freq_capped():
    y = np.array([0] * 990 + [1] * 10, dtype=np.int64)
    w = compute_class_weights_tensor(
        y,
        {"weight_scheme": "inverse_freq", "positive_weight_cap": 20.0},
    )
    assert float(w[1]) == 20.0


def test_compute_class_weights_no_cap():
    y = np.array([0] * 100 + [1] * 10, dtype=np.int64)
    w = compute_class_weights_tensor(
        y,
        {"weight_scheme": "inverse_freq", "positive_weight_cap": None},
    )
    assert float(w[1]) == 10.0


def test_compute_class_weights_all_positive_no_neg():
    """无非正类样本时 inverse_freq 会得到 w_pos=0，应退回单位权重。"""
    y = np.ones(8, dtype=np.int64)
    w = compute_class_weights_tensor(
        y,
        {"weight_scheme": "inverse_freq", "positive_weight_cap": 200.0},
    )
    assert torch.allclose(w, torch.tensor([1.0, 1.0]))


def test_invalid_weight_scheme_raises():
    y = np.array([0, 1], dtype=np.int64)
    try:
        compute_class_weights_tensor(y, {"weight_scheme": "nope", "positive_weight_cap": 10.0})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "weight_scheme" in str(e).lower() or "inverse_freq" in str(e)


def test_build_positive_oversample_index_map_factor_one():
    y = np.array([0, 1, 0, 0, 1], dtype=np.int64)
    m = build_positive_oversample_index_map(y, 1)
    assert list(m) == [0, 1, 2, 3, 4]


def test_build_positive_oversample_index_map_repeats_positives():
    y = np.array([0, 1, 0, 0, 1], dtype=np.int64)
    m = build_positive_oversample_index_map(y, 3)
    assert m.shape[0] == 5 + 2 * 2  # base + 2*(factor-1) positives
    assert np.bincount(m, minlength=5)[1] == 3
    assert np.bincount(m, minlength=5)[4] == 3


def test_build_positive_oversample_no_positives():
    y = np.zeros(4, dtype=np.int64)
    m = build_positive_oversample_index_map(y, 10)
    assert list(m) == [0, 1, 2, 3]


def test_index_mapped_dataset_matches_base_rows():
    x = torch.randn(3, 2, 2, 2)
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    ds = IndexMappedDataset((x, y, None), np.array([2, 1, 1], dtype=np.int64))
    assert len(ds) == 3
    x0, y0 = ds[0]
    assert torch.equal(x0, x[2]) and int(y0) == 0


def test_per_sample_weights_shape_and_positive_heavier():
    y = np.array([0, 1, 0, 0, 1], dtype=np.int64)
    sw = compute_per_sample_weights(y, {"sampler_positive_boost": 1.0})
    assert sw.shape == (5,)
    assert sw[y == 0][0] == 1.0
    assert sw[y == 1][0] > 1.0


def test_imbalance_cfg_fallbacks_match_train_yaml_defaults():
    """空 imb_cfg 时 fallback 须与 configs/train.yaml 默认一致（sqrt + cap=30）。"""
    assert positive_oversample_factor_from_cfg({}) == DEFAULT_POSITIVE_OVERSAMPLE_FACTOR
    assert focal_gamma_from_cfg({}) == DEFAULT_FOCAL_GAMMA
    w = compute_class_weights_tensor(
        np.array([0, 0, 1], dtype=np.int64),
        {},
    )
    assert float(w[1]) == pytest.approx(min(float(np.sqrt(2.0)), DEFAULT_POSITIVE_WEIGHT_CAP))
