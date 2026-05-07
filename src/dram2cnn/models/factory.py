from __future__ import annotations

"""模型工厂：根据 `model.arch` 构造网络。

当前仅支持 **`grid_cnn`**（见 `configs/model.yaml`）。输入通道数须与预处理得到的 `X.npy` 一致。
"""

import torch.nn as nn

from .grid_cnn import GridCNN


def build_model(model_cfg: dict) -> nn.Module:
    arch = str(model_cfg.get("arch", "grid_cnn")).lower().strip()
    if arch != "grid_cnn":
        raise ValueError(f"Unknown model.arch={arch!r}; this project only supports grid_cnn.")
    return GridCNN(model_cfg)
