from __future__ import annotations

"""类别极不平衡时的训练侧辅助：类别权重与加权采样权重。

原理简述（与 ``classification_loss`` 加权 CE / 可选 Focal 配合）
----------------------------
在正负样本比例极端（如正类 <1%）时，batch 内梯度由多数类主导，模型易退化为「几乎总预测负类」，
此时验证集上 F1@0.5 常接近 0——这不代表「没有调阈值的空间」，而是 **损失与「检出稀有正类」目标不一致**。

本项目采用两类互补手段（均在 ``configs/train.yaml`` 的 ``imbalance`` 下配置）：

1. **Class weight（类别权重）**  
   计算 ``weight=[w_{\\text{neg}}, w_{\\text{pos}}]``（默认 ``w_{neg}=1``），供 ``classification_loss`` / ``F.cross_entropy`` 使用（可选 Focal γ，见 ``train.imbalance.focal_gamma``）。
   支持 ``inverse_freq``（与频数成反比）与 ``sqrt``（平方根压缩，较温和，配合较高 cap 时常更稳）。

2. **WeightedRandomSampler（可选，仅单机训练）**  
   按样本权重采样，使每个 epoch 内正样本被抽中的期望次数提高，与 class weight 不同：权重改的是「错分的代价」，
   采样改的是「每个 step 见到多少正例」。DDP 多进程下默认关闭采样器（各 rank 已用 ``DistributedSampler`` 划分数据，
   与全局加权采样叠加语义复杂），此时主要依赖 class weight；单机可把 ``use_weighted_sampler`` 设为 ``true``。

3. **正样本索引过采样（``positive_oversample_factor``，DDP 可用）**  
   在训练集上为每个正例额外拼接 ``factor-1`` 份索引，使虚拟数据集更长、每个 epoch 内 batch 更易含正例；
   **类别权重仍按原始 ``y_train`` 计数计算**，避免与过采样双重放大。见 ``build_positive_oversample_index_map``。

阈值搜索（``threshold_search``）与验证集 **Average Precision (PR-AUC)** 记录在 ``trainer`` / ``metrics`` 中，
用于极不平衡场景下比 ROC-AUC 更贴近「找正类」的排序质量评估。
"""

from typing import Any

import numpy as np
import torch

# 与 configs/train.yaml 默认一致；实验 yaml 未覆盖 imbalance 子键时作 fallback。
DEFAULT_POSITIVE_OVERSAMPLE_FACTOR = 2
DEFAULT_POSITIVE_WEIGHT_CAP = 30.0
DEFAULT_FOCAL_GAMMA = 0.0
DEFAULT_WEIGHT_SCHEME = "sqrt"


def positive_oversample_factor_from_cfg(imb_cfg: dict[str, Any]) -> int:
    """解析 ``train.imbalance.positive_oversample_factor``（≥1；None 视为 1）。"""
    raw = imb_cfg.get("positive_oversample_factor", DEFAULT_POSITIVE_OVERSAMPLE_FACTOR)
    return int(raw) if raw is not None else 1


def focal_gamma_from_cfg(imb_cfg: dict[str, Any]) -> float:
    """解析 ``train.imbalance.focal_gamma``（0 / null 表示普通加权 CE）。"""
    raw = imb_cfg.get("focal_gamma", DEFAULT_FOCAL_GAMMA)
    return 0.0 if raw is None else float(raw)


def build_positive_oversample_index_map(y_np: np.ndarray, factor: int) -> np.ndarray:
    """构造训练用索引序列：在全体样本各出现一次的基础上，将每个正样本索引再重复 ``factor-1`` 次。

    ``factor<=1`` 时返回 ``arange(N)``。无正样本时亦返回 ``arange(N)``。
    可与 ``DistributedSampler`` 及 ``DataLoader`` 的 ``shuffle`` 一起使用；**损失里的 class weight 仍须按原始 ``y_train`` 频数计算**（见 ``trainer``）。
    """
    y_flat = np.asarray(y_np).reshape(-1)
    n = int(len(y_flat))
    if n == 0:
        return np.arange(0, dtype=np.int64)
    k = int(factor)
    if k <= 1:
        return np.arange(n, dtype=np.int64)
    pos_idx = np.where(y_flat == 1)[0].astype(np.int64, copy=False)
    if pos_idx.size == 0:
        return np.arange(n, dtype=np.int64)
    base = np.arange(n, dtype=np.int64)
    extras = np.tile(pos_idx, k - 1)
    return np.concatenate([base, extras])


def compute_class_weights_tensor(y_np: np.ndarray, imb_cfg: dict[str, Any]) -> torch.Tensor:
    """根据训练标签与配置构造 ``weight=[w_neg, w_pos]``（CPU float32 Tensor），供 ``classification_loss`` / ``F.cross_entropy(..., weight=...)`` 使用。

    Parameters
    ----------
    y_np
        训练集标签，形状 ``(N,)``，元素为 0/1。
    imb_cfg
        ``train.imbalance`` 字典；关键键：

        - ``weight_scheme``: ``inverse_freq`` | ``sqrt`` — 由 ``neg/pos`` 推导 ``w_pos`` 的主模式；
        - ``positive_weight_cap``: 对 ``w_pos`` 的上限；``null`` 表示不裁剪（慎用，易不稳定）。默认 **30.0**（与 ``configs/train.yaml`` 一致）。
    """
    y_flat = np.asarray(y_np).reshape(-1)
    pos = float((y_flat == 1).sum())
    neg = float((y_flat == 0).sum())
    if pos <= 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    # 训练集若无非正类标签（neg==0），inverse_freq 会得到 w_pos=0，反向传播不稳定
    if neg <= 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    ratio = neg / pos
    scheme = str(imb_cfg.get("weight_scheme", DEFAULT_WEIGHT_SCHEME)).lower().strip()
    cap_raw = imb_cfg.get("positive_weight_cap", DEFAULT_POSITIVE_WEIGHT_CAP)
    cap = float(cap_raw) if cap_raw is not None else None

    if scheme == "inverse_freq":
        w_pos = float(ratio)
    elif scheme == "sqrt":
        w_pos = float(np.sqrt(ratio))
    else:
        raise ValueError(
            "train.imbalance.weight_scheme must be 'inverse_freq' or 'sqrt'; "
            f"got {scheme!r}. See configs/train.yaml → imbalance 与 engine/imbalance.py。"
        )

    if cap is not None and cap > 0:
        w_pos = min(w_pos, cap)

    return torch.tensor([1.0, w_pos], dtype=torch.float32)


def compute_per_sample_weights(y_np: np.ndarray, imb_cfg: dict[str, Any]) -> np.ndarray:
    """构造 ``WeightedRandomSampler`` 所需的逐样本正数权重（双精度 numpy）。

    负样本权重为 1；正样本权重为 ``(neg/pos) * sampler_positive_boost``，使在 ``replacement=True``、
    ``num_samples=len(dataset)`` 的典型设定下，每个 epoch 内正样本被抽中的期望次数相对提升。
    """
    y_flat = np.asarray(y_np).reshape(-1)
    pos = int((y_flat == 1).sum())
    neg = int((y_flat == 0).sum())
    if pos <= 0 or neg <= 0:
        return np.ones(len(y_flat), dtype=np.float64)

    boost = float(imb_cfg.get("sampler_positive_boost", 1.0))
    w_pos = (float(neg) / float(pos)) * max(boost, 1e-8)
    return np.where(y_flat == 1, w_pos, 1.0).astype(np.float64)
