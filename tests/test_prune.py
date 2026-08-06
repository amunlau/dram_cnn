"""剪枝模块单元测试（小模型、CPU）。"""
from __future__ import annotations

import torch

from dram2cnn.engine.prune.bn_channel import prune_bn_channels
from dram2cnn.engine.prune.common import clone_model, list_conv_bn_relu, weight_sparsity
from dram2cnn.engine.prune.filter_prune import prune_filters_l1
from dram2cnn.engine.prune.structured_utils import (
    apply_channel_keeps,
    select_keep_indices_per_layer,
    verify_forward,
)
from dram2cnn.engine.prune.unstructured import prune_unstructured_l1
from dram2cnn.models.grid_cnn import GridCNN


def _tiny_cfg(**overrides) -> dict:
    cfg = {
        "arch": "grid_cnn",
        "in_channels": 4,
        "grid_height": 16,
        "grid_width": 8,
        "num_classes": 2,
        "stem_channels": 8,
        "backbone_channels": [8, 16],
        "blocks_per_stage": 1,
        "head_hidden": 16,
        "dropout": 0.0,
        "tabular_dim": 0,
        "use_batch_norm": True,
        "conv_kernel_size": 3,
    }
    cfg.update(overrides)
    return cfg


def _trained_tiny() -> GridCNN:
    m = GridCNN(_tiny_cfg())
    with torch.no_grad():
        _ = m(torch.randn(4, 4, 16, 8))
    return m


def test_unstructured_sparsity_close_to_target():
    m = _trained_tiny()
    stats = prune_unstructured_l1(m, 0.5, input_size=(4, 16, 8))
    sp = weight_sparsity(m)
    assert 0.45 <= sp <= 0.55, sp
    assert abs(stats["achieved_weight_sparsity"] - sp) < 1e-6
    y = m(torch.randn(2, 4, 16, 8))
    assert y.shape == (2, 2)


def test_bn_channel_forward_and_shrink():
    m = _trained_tiny()
    n_before = sum(b.conv.out_channels for _, b in list_conv_bn_relu(m))
    stats = prune_bn_channels(m, 0.5, min_keep_channels=2, input_size=(4, 16, 8))
    n_after = sum(b.conv.out_channels for _, b in list_conv_bn_relu(m))
    assert n_after < n_before
    assert stats["n_channels_after"] == n_after
    assert m.head[0].in_features == list_conv_bn_relu(m)[-1][1].conv.out_channels
    verify_forward(m, (4, 16, 8))


def test_filter_prune_forward_and_shrink():
    m = _trained_tiny()
    n_before = sum(b.conv.out_channels for _, b in list_conv_bn_relu(m))
    stats = prune_filters_l1(m, 0.4, min_keep_channels=2, input_size=(4, 16, 8))
    n_after = sum(b.conv.out_channels for _, b in list_conv_bn_relu(m))
    assert n_after < n_before
    assert stats["channel_prune_ratio"] > 0
    verify_forward(m, (4, 16, 8))


def test_select_keep_respects_min_keep():
    scores = [torch.arange(10.0), torch.arange(6.0)]
    keeps = select_keep_indices_per_layer(scores, 0.9, min_keep_channels=3, global_ranking=True)
    assert all(int(k.numel()) >= 3 for k in keeps)


def test_apply_channel_keeps_roundtrip_shape():
    m = _trained_tiny()
    blocks = list_conv_bn_relu(m)
    # keep half channels per layer (at least 2)
    keeps = []
    for _, b in blocks:
        n = b.conv.out_channels
        k = max(2, n // 2)
        keeps.append(torch.arange(k))
    apply_channel_keeps(m, keeps)
    verify_forward(m, (4, 16, 8))
    assert m.head[0].in_features == keeps[-1].numel()


def test_clone_independent():
    m = _trained_tiny()
    m2 = clone_model(m)
    prune_unstructured_l1(m2, 0.7, input_size=(4, 16, 8))
    assert weight_sparsity(m) < 0.01
    assert weight_sparsity(m2) > 0.6
