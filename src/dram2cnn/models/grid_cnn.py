from __future__ import annotations

"""固定尺寸 **[N,C,H,W]** 网格上的轻量 2D CNN。

**C** 可为 legacy（read/scrub/type0 三通道）或 spatial（bank + inventory 拼接通道，如 31）。Backbone：Conv+BN+ReLU → MaxPool(2)；末端 AdaptiveAvgPool2d(1,1)。若 **`tabular_dim>0`**，将 **[N,F]** one-hot 投影后与网格向量拼接再接 MLP；**`tabular_dim=0`** 时仅用网格支路。"""

import torch
import torch.nn as nn


class GridCNN(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        in_ch = int(model_cfg.get("in_channels", 3))
        num_classes = int(model_cfg.get("num_classes", 2))
        channels = list(model_cfg.get("backbone_channels", [32, 64, 128, 128]))
        head_h = int(model_cfg.get("head_hidden", 256))
        drop = float(model_cfg.get("dropout", 0.25))
        tab_dim = int(model_cfg.get("tabular_dim", 0))
        fuse_h = int(model_cfg.get("tabular_fuse_dim", 64))

        blocks: list[nn.Module] = []
        c = in_ch
        for i, oc in enumerate(channels):
            blocks.append(nn.Conv2d(c, int(oc), kernel_size=3, padding=1))
            blocks.append(nn.BatchNorm2d(int(oc)))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(2))
            c = int(oc)
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.tab_dim = tab_dim
        if tab_dim > 0:
            self.tab_proj = nn.Sequential(
                nn.Linear(tab_dim, fuse_h),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
            )
            head_in = c + fuse_h
        else:
            self.tab_proj = None
            head_in = c
        self.head = nn.Sequential(
            nn.Linear(head_in, head_h),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(head_h, num_classes),
        )

    def forward(self, x: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h).flatten(1)
        if self.tab_proj is not None:
            if tab is None:
                raise ValueError("模型配置了 tabular_dim>0，前向传播必须传入 tab")
            h = torch.cat([h, self.tab_proj(tab)], dim=1)
        return self.head(h)
