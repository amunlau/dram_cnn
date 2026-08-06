"""GridCNN-Deep W8A8 PTQ：BN Folding、AdaRound 风格重构、激活校准、打包与评估。

仅 Post-Training Quantization（不做 QAT）。由 ``dram2cnn.pipelines.run_quantize`` 调用。
打包产物可由 ``dram2cnn.engine.infer_int8`` 做 **INT8 参考推理**（权重以 int8 驻留）。
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dram2cnn.data.dataset import NpyTensorDataset
from dram2cnn.engine.metrics import classification_metrics
from dram2cnn.models.factory import build_model
from dram2cnn.models.grid_cnn import GridCNN, _ConvBNReLU

LogFn = Callable[..., None]

PACKED_FORMAT = "gridcnn_w8a8_packed_v2"


def _load_checkpoint_state(path: Path, map_location) -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError, OSError, pickle.UnpicklingError):
        return torch.load(str(path), map_location=map_location, weights_only=False)


def make_loader(
    ds: NpyTensorDataset,
    batch_size: int,
    indices: list[int] | None = None,
    *,
    pin_memory: bool = False,
) -> DataLoader:
    subset = Subset(ds, indices) if indices is not None else ds
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
    )


# ---------------------------------------------------------------------------
# BN folding
# ---------------------------------------------------------------------------


class FoldedConvReLU(nn.Module):
    """BN 已折叠进 Conv 的 Conv(+bias)+ReLU 单元。"""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    if bn.running_mean is None or bn.running_var is None:
        raise RuntimeError("BatchNorm2d 缺少 running_mean/running_var，无法折叠")
    w = conv.weight.detach().clone()
    b = (
        conv.bias.detach().clone()
        if conv.bias is not None
        else torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
    )
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps
    gamma = bn.weight.detach() if bn.weight is not None else torch.ones_like(mean)
    beta = bn.bias.detach() if bn.bias is not None else torch.zeros_like(mean)
    std = torch.sqrt(var + eps)
    w_fold = w * (gamma / std).reshape(-1, 1, 1, 1)
    b_fold = beta + (b - mean) * (gamma / std)
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )
    fused.weight.data.copy_(w_fold)
    fused.bias.data.copy_(b_fold)
    return fused


def fold_batchnorm(model: GridCNN) -> GridCNN:
    """原地将所有 ``_ConvBNReLU`` 折叠为 ``FoldedConvReLU``。

    若 ``use_batch_norm=false``（``bn`` 为 ``Identity``），仅包装原 Conv（已含 bias）。
    """
    model.eval()

    def _replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, _ConvBNReLU):
                if isinstance(child.bn, nn.BatchNorm2d):
                    fused = _fuse_conv_bn(child.conv, child.bn)
                elif isinstance(child.bn, nn.Identity):
                    conv = child.conv
                    if conv.bias is None:
                        raise RuntimeError(
                            f"{name}: use_batch_norm=false 时期望 Conv 含 bias，实际 bias=None"
                        )
                    fused = conv
                else:
                    raise TypeError(f"{name}: 不支持的 bn 类型 {type(child.bn)}")
                setattr(module, name, FoldedConvReLU(fused))
            else:
                _replace(child)

    _replace(model)
    return model


# ---------------------------------------------------------------------------
# Fake-quant helpers + AdaRound
# ---------------------------------------------------------------------------


def qparams_per_channel_symmetric(w: torch.Tensor, n_bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 scale[C_out], zero_point[C_out]=0（对称）。"""
    qmax = 2 ** (n_bits - 1) - 1
    flat = w.detach().reshape(w.shape[0], -1)
    max_abs = flat.abs().amax(dim=1).clamp(min=1e-8)
    scale = max_abs / qmax
    zp = torch.zeros_like(scale, dtype=torch.int32)
    return scale, zp


def qparams_per_tensor_asymmetric(xmin: float, xmax: float, n_bits: int = 8) -> tuple[float, int]:
    qmin, qmax = 0, 2**n_bits - 1
    xmin = min(xmin, 0.0)
    xmax = max(xmax, 0.0)
    if xmax - xmin < 1e-8:
        xmax = xmin + 1e-8
    scale = (xmax - xmin) / float(qmax - qmin)
    zp = int(round(qmin - xmin / scale))
    zp = int(max(qmin, min(qmax, zp)))
    return float(scale), zp


def fake_quant_weight_per_channel(w: torch.Tensor, scale: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    qmin, qmax = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
    shape = [w.shape[0]] + [1] * (w.ndim - 1)
    s = scale.detach().reshape(shape)
    inv = 1.0 / s
    q = torch.clamp(torch.round(w * inv), qmin, qmax)
    return q * s


def fake_quant_act_per_tensor(x: torch.Tensor, scale: float, zp: int, n_bits: int = 8) -> torch.Tensor:
    qmin, qmax = 0, 2**n_bits - 1
    inv = 1.0 / scale
    q = torch.clamp(torch.round(x * inv + zp), qmin, qmax)
    return (q - zp) * scale


class AdaRoundWeight(nn.Module):
    """AdaRound：学习 soft 舍入，逼近层输出重构误差最小。"""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits
        self.register_buffer("scale", scale.detach().clone())
        shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
        s = self.scale.reshape(shape)
        self.register_buffer("w_floor", torch.floor(weight.detach() / s))
        rest = weight.detach() / s - self.w_floor
        init = -torch.log(1.0 / rest.clamp(1e-4, 1 - 1e-4) - 1.0)
        self.alpha = nn.Parameter(init)
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1

    def soft_weight(self, temp: float = 1.0) -> torch.Tensor:
        shape = [self.w_floor.shape[0]] + [1] * (self.w_floor.ndim - 1)
        s = self.scale.reshape(shape)
        h = torch.sigmoid(self.alpha / max(temp, 1e-3))
        w_int = torch.clamp(self.w_floor + h, self.qmin, self.qmax)
        return w_int * s

    def hard_weight(self) -> torch.Tensor:
        shape = [self.w_floor.shape[0]] + [1] * (self.w_floor.ndim - 1)
        s = self.scale.reshape(shape)
        h = (self.alpha >= 0).to(self.w_floor.dtype)
        w_int = torch.clamp(self.w_floor + h, self.qmin, self.qmax)
        return w_int * s


# ---------------------------------------------------------------------------
# Quantized forward model (folded + fake quant)
# ---------------------------------------------------------------------------


@dataclass
class LayerQuantSpec:
    name: str
    kind: str  # conv | linear
    weight_scale: list[float]
    act_in_scale: float
    act_in_zp: int
    keep_fp32: bool = False


class QuantGridCNN(nn.Module):
    """折叠 BN 后的 GridCNN，带输入/权重/层间激活 fake-quant（PTQ 精度评估用）。"""

    def __init__(self, folded: GridCNN):
        super().__init__()
        if getattr(folded, "tab_proj", None) is not None or int(getattr(folded, "tab_dim", 0) or 0) > 0:
            raise ValueError(
                "QuantGridCNN / W8A8 PTQ 当前要求 tabular_dim=0（inventory 拼入网格通道）。"
            )
        self.stem = folded.stem
        self.stages = folded.stages
        self.pool = folded.pool
        self.head = folded.head
        self.out_channels = folded.out_channels
        self.tab_proj = None
        self.layer_specs: dict[str, LayerQuantSpec] = {}
        self.adaround = nn.ModuleDict()
        self.use_hard_round = False
        self.enable_act_quant = True
        self.enable_weight_quant = True

    def set_spec(self, spec: LayerQuantSpec) -> None:
        self.layer_specs[spec.name] = spec

    def _qweight(self, name: str, w: torch.Tensor) -> torch.Tensor:
        if not self.enable_weight_quant:
            return w
        spec = self.layer_specs.get(name)
        if spec is None or spec.keep_fp32:
            return w
        ar_key = name.replace(".", "__")
        if ar_key in self.adaround:
            mod = self.adaround[ar_key]
            return mod.hard_weight() if self.use_hard_round else mod.soft_weight()
        scale = torch.tensor(spec.weight_scale, device=w.device, dtype=w.dtype)
        return fake_quant_weight_per_channel(w, scale)

    def _qact(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_act_quant:
            return x
        spec = self.layer_specs.get(name)
        if spec is None or spec.keep_fp32:
            return x
        return fake_quant_act_per_tensor(x, spec.act_in_scale, spec.act_in_zp)

    def _conv_relu(self, name: str, module: FoldedConvReLU, x: torch.Tensor) -> torch.Tensor:
        x = self._qact(name, x)
        w = self._qweight(name, module.conv.weight)
        b = module.conv.bias
        y = F.conv2d(
            x, w, b, module.conv.stride, module.conv.padding, module.conv.dilation, module.conv.groups
        )
        return module.relu(y)

    def _linear(self, name: str, lin: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        x = self._qact(name, x)
        w = self._qweight(name, lin.weight)
        return F.linear(x, w, lin.bias)

    def forward(self, x: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        if tab is not None:
            raise ValueError("QuantGridCNN 不接受 tabular 输入")
        h = self._conv_relu("stem", self.stem, x)  # type: ignore[arg-type]
        for si, stage in enumerate(self.stages):
            net = stage.net  # type: ignore[attr-defined]
            for li, layer in enumerate(net):
                if isinstance(layer, FoldedConvReLU):
                    h = self._conv_relu(f"stages.{si}.net.{li}", layer, h)
                else:
                    h = layer(h)
        h = self.pool(h).flatten(1)
        h = self._linear("head.0", self.head[0], h)  # type: ignore[index]
        h = self.head[1](h)
        h = self.head[2](h)
        h = self._linear("head.3", self.head[3], h)  # type: ignore[index]
        return h


def iter_quant_layers(folded: nn.Module) -> list[tuple[str, str, nn.Module]]:
    out: list[tuple[str, str, nn.Module]] = []
    stem = folded.stem  # type: ignore[attr-defined]
    if not isinstance(stem, FoldedConvReLU):
        raise TypeError(f"stem 应为 FoldedConvReLU，实际为 {type(stem)}；请先 fold_batchnorm")
    out.append(("stem", "conv", stem))
    for si, stage in enumerate(folded.stages):  # type: ignore[attr-defined]
        for li, layer in enumerate(stage.net):  # type: ignore[attr-defined]
            if isinstance(layer, FoldedConvReLU):
                out.append((f"stages.{si}.net.{li}", "conv", layer))
    head = folded.head  # type: ignore[attr-defined]
    out.append(("head.0", "linear", head[0]))
    out.append(("head.3", "linear", head[3]))
    return out


def _module_fp_forward(model: QuantGridCNN, x: torch.Tensor) -> torch.Tensor:
    """走子模块 forward（触发 hook），用于校准 / 采层输入。"""
    h = model.stem(x)
    for stage in model.stages:
        h = stage(h)
    h = model.pool(h).flatten(1)
    return model.head(h)


# ---------------------------------------------------------------------------
# Calibration + AdaRound
# ---------------------------------------------------------------------------


@torch.no_grad()
def calibrate_activations(
    model: QuantGridCNN,
    loader: DataLoader,
    device: torch.device,
    pct: float = 0.999,
    max_elems_per_batch: int = 100_000,
) -> dict[str, tuple[float, float]]:
    """收集各量化层输入激活的分位范围（按 batch 子采样，降低峰值内存）。"""
    model.eval()
    model.enable_act_quant = False
    model.enable_weight_quant = False
    ranges: dict[str, list[torch.Tensor]] = {n: [] for n, _, _ in iter_quant_layers(model)}

    hooks = []

    def make_hook(name: str):
        def _hook(_m, inp, _out):
            t = inp[0].detach().float().reshape(-1)
            if t.numel() > max_elems_per_batch:
                idx = torch.randint(0, t.numel(), (max_elems_per_batch,), device=t.device)
                t = t[idx]
            ranges[name].append(t.cpu())

        return _hook

    for name, _kind, mod in iter_quant_layers(model):
        hooks.append(mod.register_forward_hook(make_hook(name)))

    for x, *_rest in loader:
        _ = _module_fp_forward(model, x.to(device))
    for h in hooks:
        h.remove()

    out: dict[str, tuple[float, float]] = {}
    for name, chunks in ranges.items():
        if not chunks:
            out[name] = (-1.0, 1.0)
            continue
        t = torch.cat(chunks, dim=0)
        if t.numel() > 5_000_000:
            idx = torch.randperm(t.numel())[:5_000_000]
            t = t[idx]
        lo = float(torch.quantile(t, 1.0 - pct))
        hi = float(torch.quantile(t, pct))
        out[name] = (min(lo, 0.0), max(hi, 0.0))
    return out


def apply_act_ranges(model: QuantGridCNN, act_ranges: dict[str, tuple[float, float]]) -> None:
    for name, (lo, hi) in act_ranges.items():
        scale, zp = qparams_per_tensor_asymmetric(lo, hi)
        spec = model.layer_specs[name]
        spec.act_in_scale = scale
        spec.act_in_zp = zp
        model.set_spec(spec)


@torch.no_grad()
def collect_layer_input(
    model: QuantGridCNN,
    layer_name: str,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 8,
    *,
    quantize_prev: bool = False,
) -> torch.Tensor:
    """采集目标层输入。

    ``quantize_prev=True`` 时：前序层走 QuantGridCNN 量化前向（硬权重 + 激活量化），
    返回进入目标层之前的激活（目标层输入尚未量化，由 AdaRound 调用方处理）。

    注意：QuantGridCNN 内部使用 ``F.conv2d``，不能依赖子模块 forward hook。
    """
    model.eval()
    names = [n for n, _, _ in iter_quant_layers(model)]
    if layer_name not in names:
        raise KeyError(f"未找到量化层: {layer_name}")

    buf: list[torch.Tensor] = []

    def _forward_until(x: torch.Tensor) -> torch.Tensor:
        # 复现 QuantGridCNN.forward 直至目标层入口
        model.enable_act_quant = bool(quantize_prev)
        model.enable_weight_quant = bool(quantize_prev)
        model.use_hard_round = bool(quantize_prev)

        h = x
        # stem
        if layer_name == "stem":
            return h
        h = model._conv_relu("stem", model.stem, h)  # type: ignore[arg-type]

        for si, stage in enumerate(model.stages):
            net = stage.net  # type: ignore[attr-defined]
            for li, layer in enumerate(net):
                qname = f"stages.{si}.net.{li}"
                if isinstance(layer, FoldedConvReLU):
                    if layer_name == qname:
                        return h
                    h = model._conv_relu(qname, layer, h)
                else:
                    h = layer(h)

        h = model.pool(h).flatten(1)
        if layer_name == "head.0":
            return h
        h = model._linear("head.0", model.head[0], h)  # type: ignore[index]
        h = model.head[1](h)
        h = model.head[2](h)
        if layer_name == "head.3":
            return h
        raise RuntimeError(f"未能走到层入口: {layer_name}")

    for bi, (x, *_rest) in enumerate(loader):
        h = _forward_until(x.to(device))
        buf.append(h.detach().float().cpu())
        if bi + 1 >= max_batches:
            break

    model.enable_act_quant = False
    model.enable_weight_quant = False
    model.use_hard_round = False
    if not buf:
        raise RuntimeError(f"未能采集层输入: {layer_name}")
    return torch.cat(buf, dim=0)


def adaround_layer(
    model: QuantGridCNN,
    layer_name: str,
    kind: str,
    module: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    steps: int = 500,
    batch_take: int = 4,
) -> None:
    """对单层做 AdaRound 重构优化（输入侧施加该层激活 fake-quant，对齐最终 W8A8）。"""
    spec = model.layer_specs[layer_name]
    if spec.keep_fp32:
        return
    weight = module.conv.weight if kind == "conv" else module.weight  # type: ignore[attr-defined]
    scale = torch.tensor(spec.weight_scale, device=device, dtype=weight.dtype)
    ar = AdaRoundWeight(weight.detach().to(device), scale.to(device))
    ar_key = layer_name.replace(".", "__")
    model.adaround[ar_key] = ar

    x_all = collect_layer_input(
        model, layer_name, calib_loader, device, max_batches=batch_take, quantize_prev=True
    )
    x_all = x_all.to(device)
    if x_all.shape[0] > 256:
        x_all = x_all[:256]
    # 对齐最终推理：对当前层输入做激活量化
    x_q = fake_quant_act_per_tensor(x_all, spec.act_in_scale, spec.act_in_zp)

    with torch.no_grad():
        if kind == "conv":
            conv = module.conv  # type: ignore[attr-defined]
            # 与最终 W8A8 对齐：在同一量化输入上，让量化权重逼近 FP32 权重输出
            y_ref = F.conv2d(
                x_q, weight.to(device), conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups
            )
        else:
            y_ref = F.linear(x_q, weight.to(device), module.bias)  # type: ignore[attr-defined]

    opt = torch.optim.Adam([ar.alpha], lr=1e-2)
    ar.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        temp = max(0.1, 1.0 - 0.9 * step / max(steps - 1, 1))
        w_hat = ar.soft_weight(temp=temp)
        if kind == "conv":
            conv = module.conv  # type: ignore[attr-defined]
            y_hat = F.conv2d(x_q, w_hat, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        else:
            y_hat = F.linear(x_q, w_hat, module.bias)  # type: ignore[attr-defined]
        h = torch.sigmoid(ar.alpha / temp)
        reg = (1 - (2 * h - 1).abs().pow(1.0)).mean()
        beta = 0.01 * (step / max(steps - 1, 1))
        loss = F.mse_loss(y_hat, y_ref) + beta * reg
        loss.backward()
        opt.step()
    ar.eval()
    with torch.no_grad():
        hard = ar.hard_weight().detach()
        if kind == "conv":
            module.conv.weight.data.copy_(hard.to(module.conv.weight.device))  # type: ignore[attr-defined]
        else:
            module.weight.data.copy_(hard.to(module.weight.device))  # type: ignore[attr-defined]


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
    m["precision_from_counts"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
    m["recall_from_counts"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return m


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _drop(a: float, b: float) -> dict[str, float]:
    abs_d = float(b - a)
    rel = abs_d / a if abs(a) > 1e-12 else float("nan")
    return {"abs": abs_d, "rel": rel}


def pack_int8_weights(
    qmodel: QuantGridCNN,
    export: dict,
    pack_path: Path,
    *,
    model_cfg: dict,
) -> dict[str, float]:
    """将可量化层权重打包为 int8 + scale（v2，含 model_cfg），返回体积统计（MB）。"""
    from dram2cnn.engine.infer_int8 import bias_key_for_layer, weight_key_for_layer

    specs = export["layer_specs"]
    sd = export["state_dict"]
    packed: dict[str, Any] = {}
    fp32_weight_bytes = 0
    packed_weight_bytes = 0

    for name, spec in specs.items():
        wk = weight_key_for_layer(name)
        if wk not in sd:
            raise KeyError(f"export state_dict 缺少 {wk}")
        w = sd[wk]
        fp32_weight_bytes += int(w.numel()) * 4
        if spec["keep_fp32"]:
            packed[wk] = {"dtype": "fp32", "tensor": w.clone()}
            packed_weight_bytes += int(w.numel()) * 4
        else:
            scale = torch.tensor(spec["weight_scale"], dtype=torch.float32)
            shape = [w.shape[0]] + [1] * (w.ndim - 1)
            s = scale.reshape(shape)
            q = torch.clamp(torch.round(w / s), -128, 127).to(torch.int8)
            packed[wk] = {
                "dtype": "int8",
                "scale": scale,
                "zero_point": 0,
                "qweight": q,
                "shape": list(w.shape),
            }
            packed_weight_bytes += int(q.numel()) * 1 + int(scale.numel()) * 4

    other = 0
    other_state: dict[str, torch.Tensor] = {}
    weight_keys = {weight_key_for_layer(n) for n in specs}
    for k, v in sd.items():
        if not torch.is_tensor(v) or "adaround" in k:
            continue
        if k in weight_keys:
            continue
        other += int(v.numel()) * 4
        other_state[k] = v

    for name, spec in specs.items():
        bk = bias_key_for_layer(name, spec["kind"])
        if bk in sd and bk not in other_state:
            other_state[bk] = sd[bk]

    payload = {
        "format": PACKED_FORMAT,
        "threshold": export["threshold"],
        "model_cfg": dict(model_cfg),
        "protect_stem": bool(export.get("protect_stem", True)),
        "protect_head_out": bool(export.get("protect_head_out", True)),
        "layer_specs": specs,
        "packed_weights": packed,
        "other_state": other_state,
    }
    # 结构化剪枝后拓扑：INT8 加载时需先按 masks 重建通道
    if export.get("channel_masks") is not None:
        payload["channel_masks"] = export["channel_masks"]

    torch.save(payload, pack_path)
    fp32_total = fp32_weight_bytes + other
    packed_total = packed_weight_bytes + other
    return {
        "fp32_weights_and_biases_mb": fp32_total / 1024 / 1024,
        "theoretical_packed_mb": packed_total / 1024 / 1024,
        "packed_int8_file_mb": file_size_mb(pack_path),
        "theoretical_compression_ratio": fp32_total / max(packed_total, 1),
    }


def run_w8a8_ptq(
    *,
    model_cfg: dict,
    ckpt: Path | None = None,
    splits_dir: Path,
    out_dir: Path,
    threshold: float,
    device: torch.device,
    calib_size: int = 1024,
    adaround_steps: int = 400,
    batch_size: int = 128,
    protect_stem: bool = True,
    protect_head_out: bool = True,
    seed: int = 42,
    log: LogFn | None = None,
    fp32_model: GridCNN | None = None,
    channel_masks: dict | None = None,
) -> dict[str, Any]:
    """执行完整 W8A8 PTQ，导出 fake-quant 与 packed INT8，并做 INT8 参考推理校验。

    可从 ``ckpt`` 加载，或直接传入 ``fp32_model``（先剪枝后量化）。
    结构化剪枝请传 ``channel_masks``，以便 packed INT8 可重建拓扑。
    """
    import copy

    def _log(msg: str, *args: Any) -> None:
        if log is not None:
            log(msg, *args)
        else:
            print(msg % args if args else msg)

    if int(model_cfg.get("tabular_dim", 0)) > 0:
        raise ValueError("W8A8 PTQ 当前要求 model.tabular_dim=0")
    if fp32_model is None and ckpt is None:
        raise ValueError("必须提供 ckpt 或 fp32_model")

    out_dir.mkdir(parents=True, exist_ok=True)
    pin = device.type == "cuda"
    val_ds = NpyTensorDataset(str(splits_dir / "X_val.npy"), str(splits_dir / "y_val.npy"))
    test_ds = NpyTensorDataset(str(splits_dir / "X_test.npy"), str(splits_dir / "y_test.npy"))
    rng = np.random.RandomState(seed)
    calib_idx = rng.choice(len(val_ds), size=min(calib_size, len(val_ds)), replace=False).tolist()
    calib_loader = make_loader(val_ds, batch_size, calib_idx, pin_memory=pin)
    test_loader = make_loader(test_ds, batch_size, pin_memory=pin)

    _log(
        "Device: %s | cuda_available=%s | device_count=%s | current=%s",
        device,
        torch.cuda.is_available(),
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
        torch.cuda.current_device() if device.type == "cuda" else "n/a",
    )
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            _log("  GPU%d: %s", i, torch.cuda.get_device_name(i))
        _log(
            "说明：本 PTQ（逐层 AdaRound + 激活校准）为单进程单卡流程；"
            "与全量训练 DDP 不同，默认只用一张可见 GPU，属正常设计。"
        )

    # FP32（ckpt 或内存中的已剪枝模型）
    if fp32_model is not None:
        fp32 = fp32_model.to(device)
        fp32.eval()
        ckpt_label = "in_memory_fp32_model"
    else:
        assert ckpt is not None
        fp32 = build_model(model_cfg).to(device)
        state = _load_checkpoint_state(ckpt, device)
        fp32.load_state_dict(state)
        fp32.eval()
        ckpt_label = str(Path(ckpt).resolve())

    n_params = sum(p.numel() for p in fp32.parameters() if p.requires_grad)
    t0 = time.time()
    y_t, p_t = predict_probs(fp32, test_loader, device)
    fp32_metrics = eval_at_threshold(y_t, p_t, threshold)
    fp32_sec = time.time() - t0
    _log("FP32 deployment metrics: %s", json.dumps(fp32_metrics, ensure_ascii=False))
    fp32_path = out_dir / "fp32_grid_cnn.pt"
    torch.save(fp32.state_dict(), fp32_path)

    # BN fold（深拷贝，避免破坏传入的剪枝模型）
    folded = copy.deepcopy(fp32).cpu()
    fold_batchnorm(folded)
    folded.to(device).eval()
    y_f, p_f = predict_probs(folded, test_loader, device)
    folded_metrics = eval_at_threshold(y_f, p_f, threshold)
    _log("BN-Fold FP32 metrics: %s", json.dumps(folded_metrics, ensure_ascii=False))

    qmodel = QuantGridCNN(folded).to(device)
    for name, kind, mod in iter_quant_layers(qmodel):
        w = mod.conv.weight if kind == "conv" else mod.weight  # type: ignore[attr-defined]
        scale, _zp = qparams_per_channel_symmetric(w)
        keep = (name == "stem" and protect_stem) or (name == "head.3" and protect_head_out)
        qmodel.set_spec(
            LayerQuantSpec(
                name=name,
                kind=kind,
                weight_scale=scale.cpu().tolist(),
                act_in_scale=1.0,
                act_in_zp=0,
                keep_fp32=keep,
            )
        )

    # 初始激活校准（AdaRound 前，提供合理 act qparams）
    _log("Calibrating activations (pre-AdaRound) on val subset n=%d …", len(calib_idx))
    apply_act_ranges(qmodel, calibrate_activations(qmodel, calib_loader, device))

    _log("AdaRound layer-wise reconstruction (steps=%d) …", adaround_steps)
    for name, kind, mod in iter_quant_layers(qmodel):
        if qmodel.layer_specs[name].keep_fp32:
            _log("  skip FP32 protect: %s", name)
            continue
        _log("  adaround: %s", name)
        adaround_layer(
            qmodel,
            name,
            kind,
            mod,
            calib_loader,
            device,
            steps=adaround_steps,
            batch_take=6,
        )

    # AdaRound 后二次校准：权重已硬量化，激活分布更接近最终 W8A8
    _log("Re-calibrating activations (post-AdaRound) …")
    qmodel.use_hard_round = True
    qmodel.enable_weight_quant = True
    qmodel.enable_act_quant = False
    apply_act_ranges(qmodel, calibrate_activations(qmodel, calib_loader, device))

    qmodel.use_hard_round = True
    qmodel.enable_act_quant = True
    qmodel.enable_weight_quant = True
    qmodel.eval()

    t1 = time.time()
    y_q, p_q = predict_probs(qmodel, test_loader, device)
    q_metrics = eval_at_threshold(y_q, p_q, threshold)
    q_sec = time.time() - t1
    _log("W8A8 fake-quant deployment metrics: %s", json.dumps(q_metrics, ensure_ascii=False))

    def _prob_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        return {
            "mean_prob": float(np.mean(p)),
            "mean_prob_pos": float(np.mean(p[y == 1])) if (y == 1).any() else float("nan"),
            "mean_prob_neg": float(np.mean(p[y == 0])) if (y == 0).any() else float("nan"),
            "frac_pred_pos": float(np.mean(p >= threshold)),
        }

    fp32_prob_s = _prob_summary(y_t, p_t)
    w8_prob_s = _prob_summary(y_q, p_q)
    _log("FP32 prob summary: %s", json.dumps(fp32_prob_s))
    _log("W8A8 fake-quant prob summary: %s", json.dumps(w8_prob_s))

    export = {
        "format": "gridcnn_w8a8_ptq_adaround_v1",
        "threshold": threshold,
        "protect_stem": protect_stem,
        "protect_head_out": protect_head_out,
        "model_cfg": dict(model_cfg),
        "channel_masks": channel_masks,
        "layer_specs": {k: asdict(v) for k, v in qmodel.layer_specs.items()},
        "state_dict": {
            k: v.detach().cpu() for k, v in qmodel.state_dict().items() if not k.startswith("adaround")
        },
    }
    q_path = out_dir / "grid_cnn_w8a8_ptq.pt"
    torch.save(export, q_path)
    pack_path = out_dir / "grid_cnn_w8a8_packed.pt"
    size_stats = pack_int8_weights(qmodel, export, pack_path, model_cfg=model_cfg)
    size_stats["fp32_state_dict_mb"] = file_size_mb(fp32_path)
    size_stats["quant_export_file_mb"] = file_size_mb(q_path)

    # INT8 参考推理（真实加载 packed int8 权重）
    from dram2cnn.engine.infer_int8 import load_packed_int8

    int8_model, _int8_meta = load_packed_int8(pack_path, model_cfg=model_cfg, map_location=device)
    t2 = time.time()
    y_i, p_i = predict_probs(int8_model, test_loader, device)
    int8_metrics = eval_at_threshold(y_i, p_i, threshold)
    int8_sec = time.time() - t2
    _log("W8A8 packed-INT8 deployment metrics: %s", json.dumps(int8_metrics, ensure_ascii=False))
    max_prob_diff = float(np.max(np.abs(p_q - p_i))) if len(p_q) == len(p_i) else float("nan")
    _log("max |p_fakequant - p_int8| = %.6g", max_prob_diff)

    pred_path = out_dir / "test_predictions_compare.csv"
    with pred_path.open("w", encoding="utf-8") as f:
        f.write(
            "sample_index,y_true,y_prob_fp32,y_pred_fp32,"
            "y_prob_w8a8_fakequant,y_pred_w8a8_fakequant,"
            "y_prob_w8a8_int8,y_pred_w8a8_int8\n"
        )
        for i in range(len(y_t)):
            f.write(
                f"{i},{int(y_t[i])},{p_t[i]:.8f},{int(p_t[i] >= threshold)},"
                f"{p_q[i]:.8f},{int(p_q[i] >= threshold)},"
                f"{p_i[i]:.8f},{int(p_i[i] >= threshold)}\n"
            )

    report = {
        "method": "PTQ + BN Folding + AdaRound-style + post-AdaRound act recalib + packed INT8 infer",
        "scheme": "W8A8",
        "weight_quant": "per-channel symmetric int8",
        "activation_quant": "per-tensor asymmetric int8",
        "bn_folding": True,
        "qat": False,
        "device": str(device),
        "gpu_note": "PTQ AdaRound 为单卡逐层优化；不使用 DDP 多卡属正常。",
        "calib": {
            "split": "val",
            "n_samples": len(calib_idx),
            "sampling": f"uniform random seed={seed} without replacement",
            "act_observer": "percentile 99.9% asymmetric range including 0",
            "recalib_after_adaround": True,
        },
        "protect_layers": {"stem": protect_stem, "head.3": protect_head_out},
        "model": {
            "arch": "GridCNN-Deep",
            "conv_layers": int(fp32.count_conv_layers()) if hasattr(fp32, "count_conv_layers") else 11,
            "n_params_fp32": n_params,
            "input_shape": [
                int(model_cfg.get("in_channels", 31)),
                int(model_cfg.get("grid_height", 128)),
                int(model_cfg.get("grid_width", 32)),
            ],
            "ckpt": ckpt_label,
            "has_channel_masks": channel_masks is not None,
        },
        "sizes": size_stats,
        "deployment_threshold": threshold,
        "fp32_test_deployment": fp32_metrics,
        "bn_fold_fp32_test_deployment": folded_metrics,
        "w8a8_test_deployment": q_metrics,
        "w8a8_int8_packed_test_deployment": int8_metrics,
        "fakequant_vs_int8": {
            "max_abs_prob_diff": max_prob_diff,
            "f1_abs_diff": abs(float(q_metrics["f1"]) - float(int8_metrics["f1"])),
        },
        "prob_summary": {"fp32": fp32_prob_s, "w8a8_fakequant": w8_prob_s, "w8a8_int8": _prob_summary(y_i, p_i)},
        "drops_vs_fp32": {
            "f1": _drop(fp32_metrics["f1"], q_metrics["f1"]),
            "precision": _drop(fp32_metrics["precision"], q_metrics["precision"]),
            "recall": _drop(fp32_metrics["recall"], q_metrics["recall"]),
            "roc_auc": _drop(fp32_metrics["roc_auc"], q_metrics["roc_auc"]),
            "f1_int8": _drop(fp32_metrics["f1"], int8_metrics["f1"]),
        },
        "latency_sec_test_forward": {
            "fp32": fp32_sec,
            "w8a8_fakequant": q_sec,
            "w8a8_int8_packed": int8_sec,
            "note": "参考实现墙钟时间，不可与 TensorRT/cuDNN INT8 内核时延对比。",
        },
        "paths": {
            "out_dir": str(out_dir.resolve()),
            "fp32_ref": str(fp32_path.resolve()),
            "quant_model": str(q_path.resolve()),
            "packed_model": str(pack_path.resolve()),
            "predictions_compare": str(pred_path.resolve()),
        },
        "notes": [
            "w8a8_test_deployment 为 fake-quant 仿真精度；w8a8_int8_packed_test_deployment 为加载 int8 打包权重的参考推理。",
            "可用 bash scripts/infer_int8.sh 对 packed 模型单独推理。",
            "主评价指标为部署阶段 F1（固定验证集阈值）。",
            "Precision 上升通常因固定阈值下预测正例变少（FP 下降快于 TP），伴随 Recall 下降。",
        ],
    }
    return report
