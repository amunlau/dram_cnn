from __future__ import annotations

"""分类损失：多类交叉熵，可选 **类别权重**（不平衡）与 **Focal**（`focal_gamma>0`）。

权重张量通常由 `engine.imbalance.compute_class_weights_tensor` 生成；Focal 在 CE 基础上抑制易分类样本。
"""

import torch.nn.functional as F


def _focal_nll_from_logits(
    logits: torch.Tensor, target: torch.Tensor, class_weight: torch.Tensor | None, gamma: float
) -> torch.Tensor:
    if gamma is None or float(gamma) <= 0.0:
        return F.cross_entropy(logits, target, weight=class_weight)

    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    nll = -log_pt
    mod = (1.0 - pt) ** float(gamma)
    if class_weight is not None:
        w = class_weight.gather(0, target)
        return (mod * nll * w).mean()
    return (mod * nll).mean()


def classification_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_weight=None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    return _focal_nll_from_logits(logits, y, class_weight, focal_gamma)
