"""GridCNN-Deep 打包 INT8（W8A8）参考推理。

加载 ``grid_cnn_w8a8_packed.pt``（由 PTQ 导出），按 **INT8 算术语义** 前向：

- 可量化层：激活先量化为 uint8 域，权重以 **int8 缓冲区** 驻留；卷积/线性用
  ``(x_q - zp) * w_q`` 再乘 ``act_scale * weight_scale``（标准 QDQ / 参考 INT8 路径）。
- 保护层（默认 stem / head.3）：保持 FP32 权重，跳过该层输入激活量化。

说明：本实现保证与打包权重一致的数值语义与内存布局（权重以 int8 存储），
便于校验与对接 TensorRT/ORT；**不**调用 cuDNN/TensorRT INT8 内核，故墙钟时延
不可与专用加速引擎对比。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dram2cnn.data.dataset import NpyTensorDataset
from dram2cnn.engine.metrics import classification_metrics
from dram2cnn.engine.quantize_ptq import (
    FoldedConvReLU,
    LayerQuantSpec,
    fold_batchnorm,
    iter_quant_layers,
)
from dram2cnn.models.factory import build_model
from dram2cnn.utils.io import dump_json

logger = logging.getLogger(__name__)

PACKED_FORMAT_V1 = "gridcnn_w8a8_packed_v1"
PACKED_FORMAT_V2 = "gridcnn_w8a8_packed_v2"
SUPPORTED_PACKED_FORMATS = frozenset({PACKED_FORMAT_V1, PACKED_FORMAT_V2})

_WEIGHT_KEY = {
    "stem": "stem.conv.weight",
    "head.0": "head.0.weight",
    "head.3": "head.3.weight",
}


def weight_key_for_layer(name: str) -> str:
    return _WEIGHT_KEY.get(name, f"{name}.conv.weight")


def bias_key_for_layer(name: str, kind: str) -> str:
    if kind == "conv":
        if name == "stem":
            return "stem.conv.bias"
        return f"{name}.conv.bias"
    return f"{name}.bias"


def _load_torch(path: Path, map_location) -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def quantize_activation_uint8(x: torch.Tensor, scale: float, zp: int) -> torch.Tensor:
    """返回量化后的整数域张量（仍为 float32，值为 0..255）。"""
    if scale <= 0:
        raise ValueError(f"act scale must be > 0, got {scale}")
    q = torch.round(x / scale + float(zp)).clamp(0, 255)
    return q


class Int8PackedGridCNN(nn.Module):
    """由 BN-folded 拓扑 + packed INT8 权重构成的参考推理模型。"""

    def __init__(
        self,
        folded: nn.Module,
        layer_specs: dict[str, LayerQuantSpec],
        packed_weights: dict[str, dict[str, Any]],
        other_state: dict[str, torch.Tensor],
    ):
        super().__init__()
        if getattr(folded, "tab_proj", None) is not None:
            raise ValueError(
                "Int8PackedGridCNN 当前不支持 tabular_dim>0；请使用 inventory 拼入网格通道的配置。"
            )
        self.stem = folded.stem
        self.stages = folded.stages
        self.pool = folded.pool
        self.head = folded.head
        self.layer_specs = dict(layer_specs)
        self._w_int8 = nn.ParameterDict()
        self._w_scale = nn.ParameterDict()
        self._w_fp32 = nn.ParameterDict()
        self._bias = nn.ParameterDict()
        self._meta: dict[str, dict[str, Any]] = {}

        for name, kind, mod in iter_quant_layers(folded):
            spec = self.layer_specs.get(name)
            if spec is None:
                raise KeyError(f"packed 缺少 layer_specs[{name}]")
            wk = weight_key_for_layer(name)
            bk = bias_key_for_layer(name, kind)
            buf_key = name.replace(".", "__")
            entry = packed_weights.get(wk)
            if entry is None:
                raise KeyError(f"packed_weights 缺少 {wk}")

            if kind == "conv":
                conv = mod.conv  # type: ignore[attr-defined]
                self._meta[name] = {
                    "kind": "conv",
                    "stride": conv.stride,
                    "padding": conv.padding,
                    "dilation": conv.dilation,
                    "groups": conv.groups,
                }
            else:
                self._meta[name] = {"kind": "linear"}

            if spec.keep_fp32 or entry.get("dtype") == "fp32":
                w = entry["tensor"] if "tensor" in entry else entry["qweight"]
                self._w_fp32[buf_key] = nn.Parameter(w.detach().float().clone(), requires_grad=False)
            else:
                q = entry["qweight"].detach().to(torch.int8).contiguous()
                scale = entry["scale"].detach().float().contiguous()
                self._w_int8[buf_key] = nn.Parameter(q, requires_grad=False)
                self._w_scale[buf_key] = nn.Parameter(scale, requires_grad=False)

            if bk in other_state:
                self._bias[buf_key] = nn.Parameter(
                    other_state[bk].detach().float().clone(), requires_grad=False
                )

        # Dropout 推理期关闭
        self.eval()

    def _bias_of(self, name: str) -> torch.Tensor | None:
        key = name.replace(".", "__")
        return self._bias[key] if key in self._bias else None

    def _conv_relu(self, name: str, x: torch.Tensor) -> torch.Tensor:
        spec = self.layer_specs[name]
        meta = self._meta[name]
        key = name.replace(".", "__")
        bias = self._bias_of(name)
        if spec.keep_fp32 or key in self._w_fp32:
            w = self._w_fp32[key]
            y = F.conv2d(x, w, bias, meta["stride"], meta["padding"], meta["dilation"], meta["groups"])
            return F.relu(y, inplace=True)

        x_q = quantize_activation_uint8(x, spec.act_in_scale, spec.act_in_zp)
        x_c = x_q - float(spec.act_in_zp)
        w_q = self._w_int8[key].float()
        y = F.conv2d(
            x_c, w_q, None, meta["stride"], meta["padding"], meta["dilation"], meta["groups"]
        )
        scale = self._w_scale[key].reshape(1, -1, 1, 1) * float(spec.act_in_scale)
        y = y * scale
        if bias is not None:
            y = y + bias.reshape(1, -1, 1, 1)
        return F.relu(y, inplace=True)

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        spec = self.layer_specs[name]
        key = name.replace(".", "__")
        bias = self._bias_of(name)
        if spec.keep_fp32 or key in self._w_fp32:
            return F.linear(x, self._w_fp32[key], bias)

        x_q = quantize_activation_uint8(x, spec.act_in_scale, spec.act_in_zp)
        x_c = x_q - float(spec.act_in_zp)
        w_q = self._w_int8[key].float()
        y = F.linear(x_c, w_q, None)
        scale = self._w_scale[key].reshape(1, -1) * float(spec.act_in_scale)
        y = y * scale
        if bias is not None:
            y = y + bias
        return y

    def forward(self, x: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        if tab is not None:
            raise ValueError("Int8PackedGridCNN 不接受 tabular 输入")
        h = self._conv_relu("stem", x)
        for si, stage in enumerate(self.stages):
            net = stage.net  # type: ignore[attr-defined]
            for li, layer in enumerate(net):
                if isinstance(layer, FoldedConvReLU):
                    h = self._conv_relu(f"stages.{si}.net.{li}", h)
                else:
                    h = layer(h)
        h = self.pool(h).flatten(1)
        h = self._linear("head.0", h)
        h = self.head[1](h)  # ReLU
        h = self.head[2](h)  # Dropout (eval → identity)
        return self._linear("head.3", h)

    def int8_weight_nbytes(self) -> int:
        return sum(int(p.numel()) for p in self._w_int8.parameters())

    def fp32_protect_nbytes(self) -> int:
        return sum(int(p.numel()) * 4 for p in self._w_fp32.parameters())


def _specs_from_payload(raw: dict[str, Any]) -> dict[str, LayerQuantSpec]:
    out: dict[str, LayerQuantSpec] = {}
    for name, d in raw.items():
        if isinstance(d, LayerQuantSpec):
            out[name] = d
            continue
        out[name] = LayerQuantSpec(
            name=str(d.get("name", name)),
            kind=str(d["kind"]),
            weight_scale=[float(x) for x in d["weight_scale"]],
            act_in_scale=float(d["act_in_scale"]),
            act_in_zp=int(d["act_in_zp"]),
            keep_fp32=bool(d.get("keep_fp32", False)),
        )
    return out


def load_packed_int8(
    pack_path: Path | str,
    model_cfg: dict | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[Int8PackedGridCNN, dict[str, Any]]:
    """加载 packed INT8 模型；返回 ``(model, meta)``，meta 含 threshold / format 等。"""
    path = Path(pack_path)
    if not path.is_file():
        raise FileNotFoundError(f"packed 模型不存在: {path}")
    payload = _load_torch(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"packed 文件应为 dict，得到 {type(payload)}")
    fmt = str(payload.get("format", ""))
    if fmt not in SUPPORTED_PACKED_FORMATS:
        raise ValueError(f"不支持的 packed format={fmt!r}；支持 {sorted(SUPPORTED_PACKED_FORMATS)}")

    cfg = model_cfg or payload.get("model_cfg")
    if not cfg:
        raise ValueError(
            "packed 文件缺少 model_cfg；请传入 model_cfg，或使用 quantize 导出的 v2 打包文件。"
        )

    folded = build_model(cfg)
    # 结构化剪枝：先按 channel_masks 重建通道，再 BN fold
    masks = payload.get("channel_masks")
    if masks and masks.get("layers"):
        from dram2cnn.engine.prune.structured_utils import apply_channel_keeps

        keeps = [torch.tensor(layer["keep_out"], dtype=torch.long) for layer in masks["layers"]]
        apply_channel_keeps(folded, keeps)  # type: ignore[arg-type]
    # 权重来自 packed，无需 load FP32 state；只需折叠后的拓扑（Conv+ReLU / Dropout / Pool）
    fold_batchnorm(folded)
    folded.eval()

    specs = _specs_from_payload(payload["layer_specs"])
    packed_weights = payload["packed_weights"]
    other_state = {k: v for k, v in payload.get("other_state", {}).items() if torch.is_tensor(v)}

    model = Int8PackedGridCNN(folded, specs, packed_weights, other_state)
    device = torch.device(map_location) if not isinstance(map_location, torch.device) else map_location
    model.to(device)
    model.eval()

    meta = {
        "format": fmt,
        "threshold": float(payload.get("threshold", 0.5)),
        "pack_path": str(path.resolve()),
        "model_cfg": dict(cfg),
        "protect_stem": bool(payload.get("protect_stem", specs.get("stem", LayerQuantSpec("stem", "conv", [], 1.0, 0)).keep_fp32)),
        "protect_head_out": bool(
            payload.get(
                "protect_head_out",
                specs.get("head.3", LayerQuantSpec("head.3", "linear", [], 1.0, 0)).keep_fp32,
            )
        ),
    }
    return model, meta


@torch.no_grad()
def predict_int8(
    *,
    model_cfg: dict,
    pack_path: str | Path,
    splits_dir: str | Path,
    output_csv: str | Path,
    device: str = "cpu",
    batch_size: int = 128,
    threshold: float | None = None,
) -> dict[str, Any]:
    """对 ``X_test`` 做 INT8 参考推理，写出与 FP32 infer 兼容的 ``predictions.csv``。"""
    splits = Path(splits_dir)
    x_test = splits / "X_test.npy"
    y_test = splits / "y_test.npy"
    if not x_test.is_file() or not y_test.is_file():
        raise FileNotFoundError(f"缺少切分文件: {x_test} / {y_test}")

    dev = torch.device(device)
    model, meta = load_packed_int8(pack_path, model_cfg=model_cfg, map_location=dev)
    thr = float(meta["threshold"] if threshold is None else threshold)

    ds = NpyTensorDataset(str(x_test), str(y_test), None)
    if int(model_cfg.get("tabular_dim", 0)) > 0:
        raise ValueError("INT8 infer 当前仅支持 tabular_dim=0")
    pin = dev.type == "cuda"
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=pin)

    rows: list[dict[str, Any]] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    cursor = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        logits = model(x.to(dev))
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = (prob >= thr).long()
        bs = int(y.shape[0])
        y_np = y.numpy()
        p_np = prob.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        ys.append(y_np)
        ps.append(p_np)
        for i in range(bs):
            rows.append(
                {
                    "sample_index": cursor + i,
                    "y_true": int(y_np[i]),
                    "y_pred": int(pred_np[i]),
                    "y_prob": float(p_np[i]),
                    "threshold": thr,
                }
            )
        cursor += bs

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= thr).astype(int)
    metrics = classification_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = thr

    meta_path = splits / "meta_test.csv"
    if meta_path.is_file():
        meta_df = pd.read_csv(meta_path)
        if len(meta_df) == len(rows):
            for i, row in enumerate(rows):
                if "sid" in meta_df.columns:
                    row["sid"] = meta_df.iloc[i]["sid"]
                if "predict_time" in meta_df.columns:
                    row["predict_time"] = meta_df.iloc[i]["predict_time"]

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    dump_json(
        out.with_name("predictions_int8_manifest.json"),
        {
            "backend": "int8_packed_reference",
            "pack_path": str(Path(pack_path).resolve()),
            "pack_format": meta["format"],
            "splits_dir": str(splits.resolve()),
            "threshold": thr,
            "n_samples": len(rows),
            "device": str(dev),
            "metrics": metrics,
            "int8_weight_bytes": model.int8_weight_nbytes(),
            "fp32_protect_weight_bytes": model.fp32_protect_nbytes(),
        },
    )
    logger.info("INT8 infer done: %s | F1=%.4f P=%.4f R=%.4f", out, metrics["f1"], metrics["precision"], metrics["recall"])
    return {"predictions_csv": str(out.resolve()), "metrics": metrics, "threshold": thr, "meta": meta}
