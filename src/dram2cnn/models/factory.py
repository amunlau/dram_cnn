"""模型工厂：根据 `model.arch` 构造 GridCNN-Deep。

当前仅支持 **`grid_cnn`**（见 `configs/model.yaml`）。
`in_channels` / `grid_height` / `grid_width` 须与 preprocess 写出的 `X.npy` 及 manifest 一致。
"""

from __future__ import annotations

import torch.nn as nn

from .grid_cnn import GridCNN


def build_model(model_cfg: dict) -> nn.Module:
    """构建 GridCNN-Deep 实例。

    Args:
        model_cfg: 合并后的 `cfg["model"]` 字典。

    Returns:
        未放置 device 的 `nn.Module`；由 trainer/predictor 负责 `.to(device)`。
    """
    arch = str(model_cfg.get("arch", "grid_cnn")).lower().strip()
    if arch != "grid_cnn":
        raise ValueError(f"Unknown model.arch={arch!r}; this project only supports grid_cnn.")
    return GridCNN(model_cfg)
