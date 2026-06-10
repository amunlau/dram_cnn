import numpy as np
import pytest

from dram2cnn.data.dataset import IndexMappedDataset, NpyTensorDataset
from dram2cnn.engine.imbalance import build_positive_oversample_index_map, compute_per_sample_weights


def test_oversample_expands_index_map() -> None:
    y = np.array([0, 1, 0, 0, 1], dtype=np.int64)
    m = build_positive_oversample_index_map(y, 3)
    assert m.shape[0] > len(y)


def test_weighted_sampler_weights_len_must_match_virtual_dataset() -> None:
    """文档化：WeightedRandomSampler 权重须与 train_ds 同长；过采样时二者不一致。"""
    y = np.array([0, 1, 0, 0, 1], dtype=np.int64)
    idx_map = build_positive_oversample_index_map(y, 3)
    sw = compute_per_sample_weights(y, {"sampler_positive_boost": 1.0})
    assert sw.shape[0] == len(y)
    assert idx_map.shape[0] > len(y)
    assert sw.shape[0] != idx_map.shape[0]
