"""非结构化权重稀疏：全局 L1 magnitude（torch.nn.utils.prune）。"""
from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.nn.utils.prune as prune

from dram2cnn.engine.prune.common import model_stats, weight_sparsity
from dram2cnn.models.grid_cnn import GridCNN


def _prunable_params(model: nn.Module) -> list[tuple[nn.Module, str]]:
    params: list[tuple[nn.Module, str]] = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight is not None:
            params.append((m, "weight"))
    return params


def prune_unstructured_l1(
    model: GridCNN,
    sparsity: float,
    *,
    input_size: tuple[int, int, int] = (31, 128, 32),
) -> dict[str, Any]:
    """对 Conv2d/Linear 权重做全局 L1 非结构化剪枝。

    ``sparsity``：目标零权重占比（例如 0.5 → 约 50% 权重置零）。
    使用 ``prune.global_unstructured`` + ``prune.remove`` 固化为零，便于保存。
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity 应在 [0,1)，得到 {sparsity}")

    before = model_stats(model, input_size=input_size, sparsity_for_macs=0.0)
    params = _prunable_params(model)
    if not params:
        raise RuntimeError("未找到可剪枝的 Conv2d/Linear 权重")

    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=float(sparsity),
    )
    for module, name in params:
        prune.remove(module, name)

    after = model_stats(model, input_size=input_size, sparsity_for_macs=None)
    return {
        "method": "unstructured_l1",
        "target_sparsity": float(sparsity),
        "achieved_weight_sparsity": weight_sparsity(model),
        "before": before,
        "after": after,
        "param_reduction_ratio": 1.0 - (after["n_params_nonzero"] / max(before["n_params"], 1)),
        "theoretical_macs_reduction_ratio": 1.0
        - (after["theoretical_macs"] / max(before["dense_macs"], 1.0)),
        "notes": [
            "非结构化剪枝不改变张量形状；加速依赖稀疏内核。",
            "theoretical_macs 按 (1-sparsity)*dense_macs 估算。",
        ],
    }
