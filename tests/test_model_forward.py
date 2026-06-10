import torch

from dram2cnn.models.factory import build_model
from dram2cnn.models.grid_cnn import GridCNN


def test_grid_cnn_forward_shape():
    cfg = {
        "arch": "grid_cnn",
        "in_channels": 3,
        "grid_height": 32,
        "grid_width": 48,
        "num_classes": 2,
        "backbone_channels": [8, 16],
        "head_hidden": 32,
        "dropout": 0.1,
        "tabular_dim": 0,
    }
    m = GridCNN(cfg)
    m.eval()
    x = torch.randn(4, 3, 32, 48)
    logits = m(x)
    assert logits.shape == (4, 2)


def test_grid_cnn_forward_with_tabular():
    cfg = {
        "arch": "grid_cnn",
        "in_channels": 3,
        "grid_height": 32,
        "grid_width": 48,
        "num_classes": 2,
        "backbone_channels": [8, 16],
        "head_hidden": 32,
        "dropout": 0.1,
        "tabular_dim": 15,
        "tabular_fuse_dim": 16,
    }
    m = GridCNN(cfg)
    x = torch.randn(4, 3, 32, 48)
    tab = torch.randn(4, 15)
    logits = m(x, tab)
    assert logits.shape == (4, 2)


def test_grid_cnn_kernel5_forward():
    cfg = {
        "arch": "grid_cnn",
        "in_channels": 3,
        "grid_height": 32,
        "grid_width": 48,
        "num_classes": 2,
        "backbone_channels": [8, 16],
        "conv_kernel_size": 5,
        "use_batch_norm": False,
        "head_hidden": 32,
        "dropout": 0.1,
        "tabular_dim": 0,
    }
    m = GridCNN(cfg)
    x = torch.randn(2, 3, 32, 48)
    assert m(x).shape == (2, 2)


def test_grid_cnn_default_spatial_shape():
    """默认 spatial 输入 31×128×32 与 configs/model.yaml 一致。"""
    cfg = {
        "arch": "grid_cnn",
        "in_channels": 31,
        "grid_height": 128,
        "grid_width": 32,
        "num_classes": 2,
        "stem_channels": 64,
        "backbone_channels": [64, 128, 256, 256],
        "blocks_per_stage": 2,
        "use_residual": True,
        "head_hidden": 256,
        "dropout": 0.25,
        "tabular_dim": 0,
    }
    m = GridCNN(cfg)
    assert m.count_conv_layers() >= 8
    x = torch.randn(2, 31, 128, 32)
    assert m(x).shape == (2, 2)


def test_build_model_factory():
    cfg = {
        "arch": "grid_cnn",
        "in_channels": 3,
        "grid_height": 16,
        "grid_width": 16,
        "num_classes": 2,
        "backbone_channels": [8, 8],
        "head_hidden": 16,
        "dropout": 0.0,
        "tabular_dim": 0,
    }
    m = build_model(cfg)
    assert isinstance(m, GridCNN)
