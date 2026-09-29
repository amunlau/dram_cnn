"""剪枝后短时微调（可选）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dram2cnn.data.dataset import NpyTensorDataset
from dram2cnn.models.grid_cnn import GridCNN

LogFn = Callable[..., None]


def short_finetune(
    model: GridCNN,
    splits_dir: Path,
    device: torch.device,
    *,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_train_samples: int | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """在训练集上做短时 CE 微调，恢复结构化剪枝后的精度。

    若模型处于 ``torch.nn.utils.prune`` 未 finalize 状态（仍挂 weight_mask），
    前向会自动乘 mask，微调不会破坏有效稀疏；调用方应在微调结束后再
    ``finalize_unstructured_pruning``。
    """

    def _log(msg: str, *args: Any) -> None:
        if log is not None:
            log(msg, *args)

    if epochs <= 0:
        return {"epochs": 0, "skipped": True}

    x_path = splits_dir / "X_train.npy"
    y_path = splits_dir / "y_train.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"微调需要 {x_path} 与 {y_path}")

    ds = NpyTensorDataset(str(x_path), str(y_path))
    if max_train_samples is not None and max_train_samples < len(ds):
        from torch.utils.data import Subset
        import numpy as np

        rng = np.random.RandomState(42)
        idx = rng.choice(len(ds), size=int(max_train_samples), replace=False).tolist()
        ds = Subset(ds, idx)  # type: ignore[assignment]

    pin = device.type == "cuda"
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    history: list[float] = []
    for ep in range(epochs):
        total, n = 0.0, 0
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(y.shape[0])
            n += int(y.shape[0])
        avg = total / max(n, 1)
        history.append(avg)
        _log("  finetune epoch %d/%d loss=%.4f", ep + 1, epochs, avg)
    model.eval()
    return {"epochs": epochs, "lr": lr, "train_loss": history, "n_train": n, "skipped": False}
