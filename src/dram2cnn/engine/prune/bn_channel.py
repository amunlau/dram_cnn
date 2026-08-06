"""基于 BN γ 的通道剪枝（Network Slimming 剪枝阶段）。

说明：经典 Network Slimming 在训练时对 γ 加 L1；本仓库全量模型未做该正则。
此处仅用现成 ``BatchNorm2d.weight``（γ）幅度作为通道重要性，执行剪枝与通道衔接。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dram2cnn.engine.prune.common import list_conv_bn_relu, model_stats
from dram2cnn.engine.prune.structured_utils import (
    apply_channel_keeps,
    select_keep_indices_per_layer,
    verify_forward,
)
from dram2cnn.models.grid_cnn import GridCNN


def _bn_gamma_scores(model: GridCNN) -> list[torch.Tensor]:
    scores: list[torch.Tensor] = []
    for name, block in list_conv_bn_relu(model):
        if not isinstance(block.bn, nn.BatchNorm2d):
            raise RuntimeError(
                f"{name}: BN-based 剪枝需要 BatchNorm2d，实际为 {type(block.bn)}。"
                "请确认 model.use_batch_norm=true。"
            )
        scores.append(block.bn.weight.detach().abs().float())
    return scores


def prune_bn_channels(
    model: GridCNN,
    prune_ratio: float,
    *,
    min_keep_channels: int = 8,
    global_ranking: bool = False,
    protect_stem: bool = True,
    protect_last_conv: bool = True,
    input_size: tuple[int, int, int] = (31, 128, 32),
) -> dict[str, Any]:
    """按 |γ| 剪除约 ``prune_ratio`` 比例的输出通道，并重建层间连接。

    默认 ``global_ranking=False``：逐层按相同比例淘汰（更稳）；
    ``True`` 时跨层全局淘汰低 |γ| 通道（Network Slimming 常见变体）。
    默认保护 stem 与末层卷积（无微调时显著减轻精度崩塌）。
    """
    before = model_stats(model, input_size=input_size, sparsity_for_macs=0.0)
    scores = _bn_gamma_scores(model)
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
        "method": "bn_channel",
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
            "重要性来自现成 BN γ，未在本实验对 γ 做 L1 再训练。",
            "结构化剪枝后 dense_macs 已按新通道数计算。",
            "默认保护 stem / 末层卷积以缓解无微调时的精度崩塌。",
        ],
    }
