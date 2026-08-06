"""剪枝公共工具：参数量、稀疏率、理论 MACs、split loader 与阈值评估。

评估约定：剪枝 / 先剪枝后量化流水线的主指标为「val 重搜 τ → test」；
``eval_at_threshold`` 本身只负责给定 τ 的指标计算。
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dram2cnn.data.dataset import NpyTensorDataset
from dram2cnn.engine.metrics import classification_metrics
from dram2cnn.models.grid_cnn import GridCNN, _ConvBNReLU


def load_checkpoint_state(path: Path, map_location) -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError, OSError, pickle.UnpicklingError):
        return torch.load(str(path), map_location=map_location, weights_only=False)


def count_parameters(model: nn.Module, *, nonzero_only: bool = False) -> int:
    total = 0
    for p in model.parameters():
        if not p.requires_grad and p.dim() == 0:
            continue
        if nonzero_only:
            total += int(torch.count_nonzero(p.detach()).item())
        else:
            total += int(p.numel())
    return total


def weight_sparsity(model: nn.Module) -> float:
    """可训练权重中零元素占比（含被 prune.remove 固化的零）。"""
    n_all = 0
    n_zero = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight is not None:
            w = m.weight.detach()
            n_all += int(w.numel())
            n_zero += int((w == 0).sum().item())
    if n_all == 0:
        return 0.0
    return float(n_zero) / float(n_all)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def list_conv_bn_relu(model: GridCNN) -> list[tuple[str, _ConvBNReLU]]:
    """按前向顺序返回 (name, module)。"""
    out: list[tuple[str, _ConvBNReLU]] = []
    if isinstance(model.stem, _ConvBNReLU):
        out.append(("stem", model.stem))
    for si, stage in enumerate(model.stages):
        for li, layer in enumerate(stage.net):  # type: ignore[attr-defined]
            if isinstance(layer, _ConvBNReLU):
                out.append((f"stages.{si}.net.{li}", layer))
    return out


def estimate_conv_macs(
    model: GridCNN,
    input_size: tuple[int, int, int] = (31, 128, 32),
) -> dict[str, Any]:
    """理论卷积 MACs（乘加次数，近似 FLOPs/2 习惯上报告为 MACs）。

    按模块实际 in/out 通道与特征图尺寸累计；非结构化稀疏时仍按稠密形状计，
    调用方可再乘 (1 - sparsity) 得到「理论稀疏 MACs」。
    """
    c_in, h, w = input_size
    macs = 0
    details: list[dict[str, Any]] = []

    def _add_conv(name: str, conv: nn.Conv2d, fh: int, fw: int) -> tuple[int, int]:
        nonlocal macs
        kh, kw = int(conv.kernel_size[0]), int(conv.kernel_size[1])
        cout = int(conv.out_channels)
        cin = int(conv.in_channels) // int(conv.groups)
        # output spatial (same padding assumed for GridCNN)
        oh, ow = fh, fw
        layer_macs = int(oh * ow * cout * cin * kh * kw)
        macs += layer_macs
        details.append({"name": name, "macs": layer_macs, "shape_out": [cout, oh, ow]})
        return oh, ow

    # stem
    if isinstance(model.stem, _ConvBNReLU):
        h, w = _add_conv("stem", model.stem.conv, h, w)
        c_in = int(model.stem.conv.out_channels)

    for si, stage in enumerate(model.stages):
        for li, layer in enumerate(stage.net):  # type: ignore[attr-defined]
            name = f"stages.{si}.net.{li}"
            if isinstance(layer, _ConvBNReLU):
                h, w = _add_conv(name, layer.conv, h, w)
                c_in = int(layer.conv.out_channels)
            elif isinstance(layer, nn.MaxPool2d):
                h = max(1, h // 2)
                w = max(1, w // 2)

    # head linear MACs (approx)
    head0 = model.head[0]
    if isinstance(head0, nn.Linear):
        lin_macs = int(head0.in_features * head0.out_features)
        macs += lin_macs
        details.append({"name": "head.0", "macs": lin_macs})
    head3 = model.head[3]
    if isinstance(head3, nn.Linear):
        lin_macs = int(head3.in_features * head3.out_features)
        macs += lin_macs
        details.append({"name": "head.3", "macs": lin_macs})

    return {"macs": macs, "gmacs": macs / 1e9, "details": details}


def model_stats(
    model: GridCNN,
    *,
    input_size: tuple[int, int, int] = (31, 128, 32),
    sparsity_for_macs: float | None = None,
) -> dict[str, Any]:
    n_params = count_parameters(model)
    n_nz = count_parameters(model, nonzero_only=True)
    sp = weight_sparsity(model)
    mac = estimate_conv_macs(model, input_size=input_size)
    dens_macs = float(mac["macs"])
    # 结构化：通道已变窄，macs 已反映；非结构化：乘 (1-sparsity)
    if sparsity_for_macs is not None:
        theo_macs = dens_macs * (1.0 - float(sparsity_for_macs))
    else:
        theo_macs = dens_macs * (1.0 - sp) if sp > 1e-8 else dens_macs
    return {
        "n_params": n_params,
        "n_params_nonzero": n_nz,
        "weight_sparsity": sp,
        "dense_macs": dens_macs,
        "theoretical_macs": theo_macs,
        "theoretical_gmacs": theo_macs / 1e9,
        "macs_note": "theoretical; unstructured sparse speedup needs sparse kernels",
    }


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        logits = model(x.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1]
        ys.append(y.numpy())
        ps.append(prob.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def eval_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    m = classification_metrics(y_true, y_pred, y_prob)
    m["threshold"] = float(threshold)
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    m["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return m


def make_split_loader(
    splits_dir: Path,
    split: str,
    batch_size: int,
    pin_memory: bool = False,
) -> DataLoader:
    """``split`` ∈ {train, val, test}，读取 ``X_{split}.npy`` / ``y_{split}.npy``。"""
    tag = str(split).strip().lower()
    if tag not in ("train", "val", "test"):
        raise ValueError(f"split 须为 train/val/test，得到 {split!r}")
    ds = NpyTensorDataset(str(splits_dir / f"X_{tag}.npy"), str(splits_dir / f"y_{tag}.npy"))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)


def make_test_loader(splits_dir: Path, batch_size: int, pin_memory: bool = False) -> DataLoader:
    return make_split_loader(splits_dir, "test", batch_size, pin_memory=pin_memory)


def sparsity_dirname(sparsity: float) -> str:
    pct = int(round(float(sparsity) * 100))
    return f"sparsity_{pct:02d}"
