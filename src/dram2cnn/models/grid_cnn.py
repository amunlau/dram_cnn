from __future__ import annotations

"""固定尺寸 **[N,C,H,W]** 网格上的深度 2D CNN（GridCNN-Deep / model_smoke 共用此架构，深度由配置决定）。

## 设计动机

DRAM 可纠正错误（CE）在 **row/col/bank** 上呈现空间聚集性（参见天池 132973、M2-MFP 等）。
预处理将事件累积为 **128×32** 分箱热图（+ inventory 通道），等价于一张「故障密度图像」。
CE 在 row/col/bank 上具有 multi-scale 空间模式，因此采用深度 backbone：

- **Stem**：首层 Conv 将 31 通道映射到较宽特征空间；
- **4 个 Stage**：每 stage 含 `blocks_per_stage` 个卷积块 + MaxPool(2)；
- **Global Average Pooling + MLP**：将空间维压成向量后做二分类。

骨干采用 VGG 风格的纯卷积堆叠（Conv→BN→ReLU）。

## 默认张量流（spatial，默认配置见 `configs/model.yaml`，典型 C=31、H=128、W=32）

    [N,C,H,W]
      → Stem Conv
      → 默认 4×Stage（Optuna 可搜 3～4 stage；每 stage `blocks_per_stage` 个卷积块 + MaxPool(2)）
      → AdaptiveAvgPool2d(1,1)
      → Linear → [N,2]

配置见 `configs/model.yaml`；结构超参搜索见 `configs/hpo/hpo.yaml` 与 `run_optuna.py`。
"""

import torch
import torch.nn as nn


class _ConvBNReLU(nn.Module):
    """标准卷积单元：Conv2d → BatchNorm（可选）→ ReLU。

    使用 same padding（kernel//2）保持 H×W 不变，便于在 stage 内堆叠多层。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, use_bn: bool):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _Stage(nn.Module):
    """单个下采样 stage：升/降通道 → N 个卷积块 → 可选 Dropout2d → MaxPool(2)。

    参数语义：
    - `in_ch` / `out_ch`：进入 stage 与 stage 内目标通道；不等时先用 Conv 匹配通道。
    - `n_blocks`：对应配置 `blocks_per_stage`；≥2 时 stage 内「加深」。
    - MaxPool(2)：H、W 各减半，扩大感受野，与 VGG 式 stage 设计类似。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        kernel: int,
        use_bn: bool,
        drop: float,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("blocks_per_stage must be >= 1")
        layers: list[nn.Module] = []
        # 通道匹配（等价于 stride=1 的投影，此处用 3×3 Conv 保持实现统一）
        if in_ch != out_ch:
            layers.append(_ConvBNReLU(in_ch, out_ch, kernel, use_bn))
            in_ch = out_ch
        for i in range(n_blocks):
            layers.append(_ConvBNReLU(in_ch if i == 0 else out_ch, out_ch, kernel, use_bn))
            in_ch = out_ch
        # 每 stage 末块后施加空间 Dropout，正则化特征图
        if drop > 0:
            layers.append(nn.Dropout2d(p=drop))
        layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GridCNN(nn.Module):
    """Spatial 热图 **GridCNN-Deep**：Stem + 多 stage 卷积 backbone + 可选 tabular 融合 + MLP 头。

    `model_cfg` 关键字段（见 `configs/model.yaml`）：
    - `in_channels`, `grid_height`, `grid_width`：须与 预处理输出 `X.npy` 一致；
    - `stem_channels`, `backbone_channels`, `blocks_per_stage`：backbone 深度与宽度；
    - `tabular_dim`：>0 时从 `tabular.npy` 读入向量并在 GAP 后拼接（独立 tabular 融合）；
    - `head_hidden`, `dropout`：分类 MLP 容量与正则；
    - `head_bias`：分类头（及 tabular 投影）Linear 是否含 bias（默认 true；硬件无偏置交付可 false）。
    """

    def __init__(self, model_cfg: dict):
        super().__init__()
        in_ch = int(model_cfg.get("in_channels", 31))
        num_classes = int(model_cfg.get("num_classes", 2))
        channels = list(model_cfg.get("backbone_channels", [64, 128, 256, 256]))
        blocks_per_stage = int(model_cfg.get("blocks_per_stage", 2))
        stem_width = int(model_cfg.get("stem_channels", channels[0] if channels else 64))
        head_h = int(model_cfg.get("head_hidden", 256))
        drop = float(model_cfg.get("dropout", 0.25))
        tab_dim = int(model_cfg.get("tabular_dim", 0))
        fuse_h = int(model_cfg.get("tabular_fuse_dim", 64))
        kernel = int(model_cfg.get("conv_kernel_size", 3))
        use_bn = bool(model_cfg.get("use_batch_norm", True))
        # 分类头 Linear 是否含 bias；硬件无偏置交付时可设 false（默认 true 保持兼容）
        head_bias = bool(model_cfg.get("head_bias", True))
        if kernel not in (3, 5):
            raise ValueError(f"conv_kernel_size must be 3 or 5, got {kernel}")
        if not channels:
            raise ValueError("backbone_channels must be non-empty")

        # Stem：将原始 bank+inventory 通道映射到统一特征宽度
        # 注意：use_bn=True 时 Conv 本身已无 bias（见 _ConvBNReLU）
        self.stem = _ConvBNReLU(in_ch, stem_width, kernel, use_bn)
        stages: list[nn.Module] = []
        c = stem_width
        for oc in channels:
            stages.append(
                _Stage(
                    c,
                    int(oc),
                    blocks_per_stage,
                    kernel,
                    use_bn,
                    drop,
                )
            )
            c = int(oc)
        self.stages = nn.Sequential(*stages)
        # 对任意 H×W 输出做全局平均，得到固定长度向量（不依赖输入尺寸微调）
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = c

        self.tab_dim = tab_dim
        self.head_bias = head_bias
        if tab_dim > 0:
            self.tab_proj = nn.Sequential(
                nn.Linear(tab_dim, fuse_h, bias=head_bias),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
            )
            head_in = c + fuse_h
        else:
            self.tab_proj = None
            head_in = c
        self.head = nn.Sequential(
            nn.Linear(head_in, head_h, bias=head_bias),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(head_h, num_classes, bias=head_bias),
        )

    def forward(self, x: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播。

        Args:
            x: [N, C, H, W] 网格特征（已按训练集标准化）。
            tab: 可选 [N, F] inventory one-hot（`tabular_dim>0` 时必传）。

        Returns:
            logits: [N, num_classes]，未做 softmax；训练时由 CrossEntropy/Focal 消费。
        """
        h = self.stem(x)
        h = self.stages(h)
        h = self.pool(h).flatten(1)
        if self.tab_proj is not None:
            if tab is None:
                raise ValueError("模型配置了 tabular_dim>0，前向传播必须传入 tab")
            h = torch.cat([h, self.tab_proj(tab)], dim=1)
        return self.head(h)

    def count_conv_layers(self) -> int:
        """统计 backbone 中 Conv2d 层数（含 stem / stage 内所有 Conv）。"""
        return sum(1 for m in self.modules() if isinstance(m, nn.Conv2d))

    def count_parameters(self) -> int:
        """可训练参数量；rank0 训练日志可引用以便对比不同 trial。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
