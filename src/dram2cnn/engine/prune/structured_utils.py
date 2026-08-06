"""结构化通道剪枝：按 keep 索引切片 Conv/BN，并同步后续层 in_channels 与 head。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dram2cnn.engine.prune.common import list_conv_bn_relu
from dram2cnn.models.grid_cnn import GridCNN, _ConvBNReLU


def _replace_conv(old: nn.Conv2d, weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Conv2d:
    out_ch, in_ch = int(weight.shape[0]), int(weight.shape[1]) * int(old.groups)
    new = nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        new.weight.copy_(weight)
        if new.bias is not None and bias is not None:
            new.bias.copy_(bias)
    return new


def _replace_bn(old: nn.BatchNorm2d, keep: torch.Tensor) -> nn.BatchNorm2d:
    n = int(keep.numel())
    new = nn.BatchNorm2d(n, eps=old.eps, momentum=old.momentum, affine=old.affine, track_running_stats=old.track_running_stats)
    with torch.no_grad():
        if old.affine:
            new.weight.copy_(old.weight.detach()[keep])
            new.bias.copy_(old.bias.detach()[keep])
        if old.track_running_stats:
            new.running_mean.copy_(old.running_mean.detach()[keep])
            new.running_var.copy_(old.running_var.detach()[keep])
            new.num_batches_tracked.copy_(old.num_batches_tracked)
    return new


def select_keep_indices_per_layer(
    scores_per_layer: list[torch.Tensor],
    prune_ratio: float,
    *,
    min_keep_channels: int = 8,
    global_ranking: bool = True,
    protect_layer_indices: set[int] | None = None,
) -> list[torch.Tensor]:
    """根据每层通道分数选择保留索引。

    ``prune_ratio``：目标剪除比例（0.5 → 约保留 50% 通道）。
    ``global_ranking=True``：跨层合并分数，全局淘汰最低的 prune_ratio 比例通道
    （各层仍至少保留 min_keep_channels）。
    ``protect_layer_indices``：这些层的全部通道强制保留（如 stem / 末层）。
    """
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError(f"prune_ratio 应在 [0,1)，得到 {prune_ratio}")

    protect = protect_layer_indices or set()
    device = scores_per_layer[0].device
    keeps: list[torch.Tensor] = []

    if not global_ranking:
        for li, scores in enumerate(scores_per_layer):
            n = int(scores.numel())
            if li in protect:
                keeps.append(torch.arange(n, device=device, dtype=torch.long))
                continue
            n_keep = max(min_keep_channels, int(round(n * (1.0 - prune_ratio))))
            n_keep = min(n_keep, n)
            _, idx = torch.topk(scores, n_keep, largest=True, sorted=False)
            keeps.append(torch.sort(idx).values)
        return keeps

    # 全局：带层编号的 (score, layer_i, ch_j)
    tagged: list[tuple[float, int, int]] = []
    for li, scores in enumerate(scores_per_layer):
        if li in protect:
            continue
        for cj, s in enumerate(scores.detach().float().cpu().tolist()):
            tagged.append((float(s), li, cj))
    tagged.sort(key=lambda t: t[0])  # 升序：先淘汰低分
    n_total = len(tagged)
    n_prune = int(round(n_total * prune_ratio))
    layer_counts = [int(s.numel()) for s in scores_per_layer]
    keep_sets: list[set[int]] = [set(range(c)) for c in layer_counts]
    pruned = 0
    for _score, li, cj in tagged:
        if pruned >= n_prune:
            break
        if len(keep_sets[li]) <= min_keep_channels:
            continue
        if cj in keep_sets[li]:
            keep_sets[li].remove(cj)
            pruned += 1

    for li, ks in enumerate(keep_sets):
        if li in protect:
            idx = torch.arange(layer_counts[li], dtype=torch.long, device=device)
        else:
            idx = torch.tensor(sorted(ks), dtype=torch.long, device=device)
        keeps.append(idx)
    return keeps


def apply_channel_keeps(
    model: GridCNN,
    keep_out_per_layer: list[torch.Tensor],
) -> dict[str, Any]:
    """按有序 ConvBNReLU 的输出 keep 索引切片，并同步下一层输入与 head.0。

    返回每层 keep 的可序列化记录。
    """
    blocks = list_conv_bn_relu(model)
    if len(keep_out_per_layer) != len(blocks):
        raise ValueError(
            f"keep 列表长度 {len(keep_out_per_layer)} != ConvBNReLU 层数 {len(blocks)}"
        )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    record: dict[str, Any] = {"layers": []}
    prev_keep: torch.Tensor | None = None

    for i, ((name, block), keep_out) in enumerate(zip(blocks, keep_out_per_layer)):
        keep_out = keep_out.to(torch.long).cpu()
        conv = block.conv
        w = conv.weight.detach().cpu()
        # 输入通道：第一层不动 in；其后按上一层 keep
        if prev_keep is not None:
            w = w[:, prev_keep, :, :].contiguous()
        w = w[keep_out, :, :, :].contiguous()
        b = None
        if conv.bias is not None:
            b = conv.bias.detach().cpu()[keep_out].contiguous()

        new_conv = _replace_conv(conv, w, b).to(device=device, dtype=dtype)
        block.conv = new_conv

        if isinstance(block.bn, nn.BatchNorm2d):
            block.bn = _replace_bn(block.bn, keep_out).to(device=device, dtype=dtype)
        elif isinstance(block.bn, nn.Identity):
            pass
        else:
            raise TypeError(f"{name}: 不支持的 bn 类型 {type(block.bn)}")

        record["layers"].append(
            {
                "name": name,
                "keep_out": keep_out.tolist(),
                "out_channels": int(keep_out.numel()),
                "in_channels": int(new_conv.in_channels),
            }
        )
        prev_keep = keep_out

    # 同步 head.0：GAP 后特征维 = 末层 out_channels
    if prev_keep is None:
        raise RuntimeError("模型中无 ConvBNReLU")
    head0 = model.head[0]
    if not isinstance(head0, nn.Linear):
        raise TypeError("model.head[0] 应为 Linear")
    with torch.no_grad():
        new_in = int(prev_keep.numel())
        new_lin = nn.Linear(new_in, head0.out_features, bias=head0.bias is not None)
        new_lin.weight.copy_(head0.weight.detach().cpu()[:, prev_keep].contiguous())
        if head0.bias is not None and new_lin.bias is not None:
            new_lin.bias.copy_(head0.bias.detach().cpu())
        model.head[0] = new_lin.to(device=device, dtype=dtype)
    model.out_channels = int(prev_keep.numel())
    record["head0_in_features"] = int(prev_keep.numel())
    return record


def verify_forward(model: GridCNN, input_size: tuple[int, int, int] = (31, 128, 32)) -> None:
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    c, h, w = input_size
    x = torch.randn(2, c, h, w, device=device, dtype=dtype)
    with torch.no_grad():
        y = model(x)
    if y.shape != (2, int(getattr(model.head[3], "out_features", 2))):
        raise RuntimeError(f"前向 shape 异常: {tuple(y.shape)}")
