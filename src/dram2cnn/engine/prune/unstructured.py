"""非结构化权重稀疏：全局 L1 / L2 magnitude（torch.nn.utils.prune）。"""
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


def finalize_unstructured_pruning(model: nn.Module) -> int:
    """将仍挂着的 unstructured prune mask 固化进权重（``prune.remove``）。

    微调期间应保留 mask，使前向始终乘 mask，稀疏不会被 Adam 训回；
    微调结束后再调用本函数固化，便于存盘与部署。
    返回完成 remove 的参数个数。
    """
    n = 0
    for module, name in _prunable_params(model):
        # 仅当该参数处于 re-param 剪枝状态时 remove
        if not prune.is_pruned(module):
            continue
        try:
            prune.remove(module, name)
            n += 1
        except ValueError:
            # 该 name 上无挂 mask（例如只剪了部分层）
            continue
    return n


def prune_unstructured_l1(
    model: GridCNN,
    sparsity: float,
    *,
    input_size: tuple[int, int, int] = (31, 128, 32),
    finalize: bool = True,
) -> dict[str, Any]:
    """对 Conv2d/Linear 权重做全局 L1 非结构化剪枝。

    ``sparsity``：目标零权重占比（例如 0.5 → 约 50% 权重置零）。

    ``finalize=True``（默认）：立即 ``prune.remove`` 固化为零。
    ``finalize=False``：保留 mask，供后续短时微调（前向仍稀疏）；微调后再调用
    ``finalize_unstructured_pruning``。若在 remove 之后直接微调，零权重会被优化器
    更新，稀疏会消失——对比实验务必在微调前保持 mask。
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
    if finalize:
        for module, name in params:
            prune.remove(module, name)

    after = model_stats(model, input_size=input_size, sparsity_for_macs=None)
    return {
        "method": "unstructured_l1",
        "target_sparsity": float(sparsity),
        "achieved_weight_sparsity": weight_sparsity(model),
        "finalize": bool(finalize),
        "before": before,
        "after": after,
        "param_reduction_ratio": 1.0 - (after["n_params_nonzero"] / max(before["n_params"], 1)),
        "theoretical_macs_reduction_ratio": 1.0
        - (after["theoretical_macs"] / max(before["dense_macs"], 1.0)),
        "notes": [
            "非结构化剪枝不改变张量形状；加速依赖稀疏内核。",
            "theoretical_macs 按 (1-sparsity)*dense_macs 估算。",
            "重要性为 |w|（L1 magnitude）。",
            "微调须在 finalize=False（保留 mask）下进行，结束后再 remove 固化。",
        ],
    }


def prune_unstructured_l2(
    model: GridCNN,
    sparsity: float,
    *,
    input_size: tuple[int, int, int] = (31, 128, 32),
    finalize: bool = True,
) -> dict[str, Any]:
    """对 Conv2d/Linear 权重做全局 L2 magnitude（w²）非结构化剪枝。

    向 ``global_unstructured`` 传入 ``importance_scores=w²``；L1Unstructured 对
    importance 取 abs（abs(w²)=w²）。对标量权重，w² 与 |w| 排序等价。

    ``finalize`` 语义同 ``prune_unstructured_l1``。
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity 应在 [0,1)，得到 {sparsity}")

    before = model_stats(model, input_size=input_size, sparsity_for_macs=0.0)
    params = _prunable_params(model)
    if not params:
        raise RuntimeError("未找到可剪枝的 Conv2d/Linear 权重")

    importance_scores = {
        (module, name): getattr(module, name).detach().pow(2) for module, name in params
    }
    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=float(sparsity),
        importance_scores=importance_scores,
    )
    if finalize:
        for module, name in params:
            prune.remove(module, name)

    after = model_stats(model, input_size=input_size, sparsity_for_macs=None)
    return {
        "method": "unstructured_l2",
        "target_sparsity": float(sparsity),
        "achieved_weight_sparsity": weight_sparsity(model),
        "finalize": bool(finalize),
        "before": before,
        "after": after,
        "param_reduction_ratio": 1.0 - (after["n_params_nonzero"] / max(before["n_params"], 1)),
        "theoretical_macs_reduction_ratio": 1.0
        - (after["theoretical_macs"] / max(before["dense_macs"], 1.0)),
        "notes": [
            "非结构化剪枝不改变张量形状；加速依赖稀疏内核。",
            "theoretical_macs 按 (1-sparsity)*dense_macs 估算。",
            "重要性为 w²（L2 magnitude）；对标量权重与 |w| 排序等价。",
            "微调须在 finalize=False（保留 mask）下进行，结束后再 remove 固化。",
        ],
    }


# 向后兼容
prune_unstructured = prune_unstructured_l1
