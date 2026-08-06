"""结构化 Filter / 卷积核剪枝：按输出通道权重 L1-norm。"""
from __future__ import annotations

from typing import Any

import torch

from dram2cnn.engine.prune.common import list_conv_bn_relu, model_stats
from dram2cnn.engine.prune.structured_utils import (
    apply_channel_keeps,
    select_keep_indices_per_layer,
    verify_forward,
)
from dram2cnn.models.grid_cnn import GridCNN


def _filter_l1_scores(model: GridCNN) -> list[torch.Tensor]:
    scores: list[torch.Tensor] = []
    for _name, block in list_conv_bn_relu(model):
        w = block.conv.weight.detach()
        # [C_out, C_in, kH, kW] → 每 filter 的 L1
        s = w.abs().sum(dim=(1, 2, 3)).float()
        scores.append(s)
    return scores


def prune_filters_l1(
    model: GridCNN,
    prune_ratio: float,
    *,
    min_keep_channels: int = 8,
    global_ranking: bool = False,
    protect_stem: bool = True,
    protect_last_conv: bool = True,
    input_size: tuple[int, int, int] = (31, 128, 32),
) -> dict[str, Any]:
    """按 filter L1-norm 剪除约 ``prune_ratio`` 比例的输出通道。

    默认逐层同比例；``global_ranking=True`` 为跨层全局淘汰。
    默认保护 stem 与末层卷积。
    """
    before = model_stats(model, input_size=input_size, sparsity_for_macs=0.0)
    scores = _filter_l1_scores(model)
    n_layers = len(scores)
    protect: set[int] = set()
    if protect_stem and n_layers > 0:
        protect.add(0)
    if protect_last_conv and n_layers > 1:
        protect.add(n_layers - 1)
    keeps = select_keep_indices_per_layer(
        scores,
        prune_ratio,
        min_keep_channels=min_keep_channels,
        global_ranking=global_ranking,
        protect_layer_indices=protect,
    )
    record = apply_channel_keeps(model, keeps)
    verify_forward(model, input_size=input_size)
    after = model_stats(model, input_size=input_size, sparsity_for_macs=0.0)

    n_ch_before = sum(int(s.numel()) for s in scores)
    n_ch_after = sum(len(r["keep_out"]) for r in record["layers"])
    return {
        "method": "filter_l1",
        "target_prune_ratio": float(prune_ratio),
        "channel_prune_ratio": 1.0 - (n_ch_after / max(n_ch_before, 1)),
        "n_channels_before": n_ch_before,
        "n_channels_after": n_ch_after,
        "min_keep_channels": min_keep_channels,
        "global_ranking": global_ranking,
        "protect_stem": protect_stem,
        "protect_last_conv": protect_last_conv,
        "channel_masks": record,
        "before": before,
        "after": after,
        "param_reduction_ratio": 1.0 - (after["n_params"] / max(before["n_params"], 1)),
        "theoretical_macs_reduction_ratio": 1.0
        - (after["dense_macs"] / max(before["dense_macs"], 1.0)),
        "notes": [
            "准则：每个输出 filter 的权重绝对值之和（L1-norm）。",
            "通道衔接与 BN-channel 共用 structured_utils。",
            "默认保护 stem / 末层卷积以缓解无微调时的精度崩塌。",
        ],
    }
