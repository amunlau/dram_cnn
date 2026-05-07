import torch

from dram2cnn.models.losses import classification_loss


def test_classification_loss_with_focal():
    logits = torch.randn(8, 2, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    cw = torch.tensor([1.0, 3.0])
    loss = classification_loss(logits, y, class_weight=cw, focal_gamma=2.0)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None
