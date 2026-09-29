"""官方 ``torch.ao.quantization`` 对称 signed INT8：BN 折叠后的 Eager FakeQuant PTQ + QAT + QDQ ONNX。

设计原则（吸取 v1 教训）
------------------------
1. **先 BN fold，再量化**：``fuse_modules``（eval）把 Conv+BN+ReLU 折成 ConvReLU2d，
   部署拓扑与 FakeQuant 作用对象一致；**禁止**对未折叠的 ``ConvBnReLU2d`` 做
   ``fuse_modules_qat`` 近似路径（否则 FakeQuant 作用在 γ/√var 缩放权重上，
   图中仍残留 BN/Div，打包易错）。
2. **评估 / 导出 / 打包同一套折叠 FakeQuant 模型**：``weight_fake_quant`` 的 scale
   对应模块 ``.weight``（已是折叠权重）；packed int8 = ``round(w/scale)``。
3. **不 ``convert`` 到 FBGEMM 整型算子**（该后端要求激活 quint8，与对称 signed
   INT8 约定冲突）。指标为 **BN 已折叠的对称 FakeQuant 仿真**；ONNX 为无 BN 的
   QDQ 标注浮点图（QuantizeLinear/DequantizeLinear），可供 ORT/TRT 再整型化。
4. stem / 末层 Linear（``head_out``）保持 FP32；中间卷积与 ``head_fc1`` 对称 INT8。
"""
from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dram2cnn.data.dataset import IndexMappedDataset, NpyTensorDataset
from dram2cnn.engine.imbalance import (
    build_positive_oversample_index_map,
    compute_class_weights_tensor,
    focal_gamma_from_cfg,
    positive_oversample_factor_from_cfg,
)
from dram2cnn.engine.prune.common import count_parameters, load_checkpoint_state, make_split_loader
from dram2cnn.engine.prune.reeval_thresholds import dual_threshold_eval
from dram2cnn.models.factory import build_model
from dram2cnn.models.grid_cnn import GridCNN, _ConvBNReLU
from dram2cnn.models.losses import classification_loss
from dram2cnn.utils.io import dump_json

LogFn = Callable[..., None]

PACKED_FORMAT = "gridcnn_official_symint8_bnfold_fakequant_v2"


def _log(log: LogFn | None, msg: str, *args: Any) -> None:
    if log:
        log(msg, *args)


def make_symmetric_qconfig() -> Any:
    """对称 signed INT8（W/A，per-tensor，zp=0）。"""
    from torch.ao.quantization import QConfig
    from torch.ao.quantization.fake_quantize import FakeQuantize
    from torch.ao.quantization.observer import MovingAverageMinMaxObserver

    act = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )
    wt = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )
    return QConfig(activation=act, weight=wt)


def _cbr_to_sequential(block: _ConvBNReLU) -> nn.Sequential:
    return nn.Sequential(block.conv, block.bn, block.relu)


class QuantizableGridCNN(nn.Module):
    """可 BN-fold / FakeQuant 的 GridCNN；stem 与末层 Linear 保持 FP32。"""

    def __init__(self, model_cfg: dict):
        super().__init__()
        from torch.ao.quantization import DeQuantStub, QuantStub

        src = GridCNN(model_cfg)
        self.model_cfg = dict(model_cfg)
        self.stem = _cbr_to_sequential(src.stem)  # type: ignore[arg-type]
        stages: list[nn.Module] = []
        for stage in src.stages:
            layers: list[nn.Module] = []
            for layer in stage.net:  # type: ignore[attr-defined]
                if isinstance(layer, _ConvBNReLU):
                    layers.append(_cbr_to_sequential(layer))
                else:
                    layers.append(layer)
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.pool = src.pool
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.head_fc1 = src.head[0]
        self.head_relu = src.head[1]
        self.head_drop = src.head[2]
        self.head_out = src.head[3]
        if getattr(src, "tab_proj", None) is not None:
            raise ValueError("QuantizableGridCNN 当前不支持 tabular_dim>0")

    def forward(self, x: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        if tab is not None:
            raise ValueError("QuantizableGridCNN 不接受 tabular 输入")
        h = self.stem(x)
        h = self.quant(h)
        h = self.stages(h)
        h = self.pool(h).flatten(1)
        h = self.head_fc1(h)
        h = self.head_relu(h)
        h = self.head_drop(h)
        h = self.dequant(h)
        return self.head_out(h)

    @classmethod
    def from_fp32_checkpoint(cls, model_cfg: dict, ckpt: Path, map_location="cpu") -> QuantizableGridCNN:
        base = build_model(model_cfg)
        state = load_checkpoint_state(ckpt, map_location=map_location)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            cleaned = {}
            for k, v in state.items():
                nk = k[7:] if str(k).startswith("module.") else k
                cleaned[nk] = v
            state = cleaned
        base.load_state_dict(state, strict=True)
        q = cls(model_cfg)
        q.load_from_grid_cnn(base)
        return q

    def load_from_grid_cnn(self, src: GridCNN) -> None:
        self.stem = _cbr_to_sequential(src.stem)  # type: ignore[arg-type]
        stages: list[nn.Module] = []
        for stage in src.stages:
            layers: list[nn.Module] = []
            for layer in stage.net:  # type: ignore[attr-defined]
                if isinstance(layer, _ConvBNReLU):
                    layers.append(_cbr_to_sequential(layer))
                else:
                    layers.append(layer)
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.pool = src.pool
        self.head_fc1 = src.head[0]
        self.head_relu = src.head[1]
        self.head_drop = src.head[2]
        self.head_out = src.head[3]

    def collect_fuse_pairs(self, *, include_stem: bool = True) -> list[list[str]]:
        """Conv-BN-ReLU 路径（含 stem：仅折叠 BN，仍不量化）。"""
        pairs: list[list[str]] = []
        if include_stem and isinstance(self.stem, nn.Sequential) and len(self.stem) == 3:
            if (
                isinstance(self.stem[0], nn.Conv2d)
                and isinstance(self.stem[1], nn.BatchNorm2d)
                and isinstance(self.stem[2], nn.ReLU)
            ):
                pairs.append(["stem.0", "stem.1", "stem.2"])
        for si, stage in enumerate(self.stages):
            for li, layer in enumerate(stage):  # type: ignore[arg-type]
                if (
                    isinstance(layer, nn.Sequential)
                    and len(layer) == 3
                    and isinstance(layer[0], nn.Conv2d)
                    and isinstance(layer[1], nn.BatchNorm2d)
                    and isinstance(layer[2], nn.ReLU)
                ):
                    base = f"stages.{si}.{li}"
                    pairs.append([f"{base}.0", f"{base}.1", f"{base}.2"])
        return pairs

    def apply_qconfig(self, qconfig: Any) -> None:
        self.qconfig = qconfig
        # stem 整体不量化（其子模块在 prepare 时跳过）
        self.stem.qconfig = None  # type: ignore[attr-defined]
        self.head_out.qconfig = None  # type: ignore[attr-defined]


def fold_batchnorm(model: QuantizableGridCNN) -> QuantizableGridCNN:
    """eval 下 fuse_modules：BN 折入 Conv，得到部署拓扑（无 BN）。"""
    from torch.ao.quantization import fuse_modules

    model.eval()
    pairs = model.collect_fuse_pairs(include_stem=True)
    if not pairs:
        raise RuntimeError("未找到可折叠的 Conv-BN-ReLU；请检查模型是否已折叠或结构异常")
    fuse_modules(model, pairs, inplace=True)
    return model


def prepare_fakequant(model: QuantizableGridCNN) -> QuantizableGridCNN:
    """在 **已 BN-fold** 的模型上插入对称 FakeQuant（W/A）。

    使用 ``prepare_qat`` 以便同时挂上 weight FakeQuantize；调用方若只做 PTQ，
    则随后 ``eval`` + 校准、不反传即可。
    """
    from torch.ao.quantization import prepare_qat

    # 确认无残留 BN（fold 后应为 Identity 占位或已消失）
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            raise RuntimeError(f"prepare_fakequant 前仍存在 BatchNorm2d: {name}；必须先 fold_batchnorm")

    model.train()
    model.apply_qconfig(make_symmetric_qconfig())
    prepare_qat(model, inplace=True)
    return model


def freeze_fakequant(model: nn.Module) -> nn.Module:
    model.eval()
    model.apply(torch.ao.quantization.disable_observer)
    model.apply(torch.ao.quantization.enable_fake_quant)
    return model


@torch.no_grad()
def calibrate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> int:
    model.eval()
    model.apply(torch.ao.quantization.enable_observer)
    model.apply(torch.ao.quantization.enable_fake_quant)
    model.to(device)
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= int(max_batches):
            break
        model(batch[0].to(device))
        n += 1
    return n


def _file_mb(path: Path) -> float:
    return float(Path(path).stat().st_size) / (1024.0 * 1024.0)


def export_qdq_onnx(
    model: nn.Module,
    out_path: Path,
    example: torch.Tensor,
    *,
    opset: int = 17,
) -> dict[str, Any]:
    """从 BN 已折叠的 FakeQuant 模型导出 QDQ ONNX（应无 BatchNormalization）。"""
    import onnx

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(model).cpu().eval()
    freeze_fakequant(m)
    torch.onnx.export(
        m,
        example.detach().cpu(),
        str(out_path),
        dynamo=False,
        opset_version=int(opset),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    g = onnx.load(str(out_path))
    ops = sorted({n.op_type for n in g.graph.node})
    n_bn = sum(1 for n in g.graph.node if n.op_type == "BatchNormalization")
    n_q = sum(1 for n in g.graph.node if n.op_type == "QuantizeLinear")
    n_dq = sum(1 for n in g.graph.node if n.op_type == "DequantizeLinear")
    if n_bn > 0:
        raise RuntimeError(f"QDQ ONNX 仍含 BatchNormalization×{n_bn}；BN fold 未生效或导出图异常")
    if n_q == 0 or n_dq == 0:
        raise RuntimeError("QDQ ONNX 缺少 QuantizeLinear/DequantizeLinear")
    return {
        "path": str(out_path.resolve()),
        "file_mb": _file_mb(out_path),
        "opset": int(opset),
        "ops": ops,
        "n_quantize_linear": n_q,
        "n_dequantize_linear": n_dq,
        "n_batchnorm": n_bn,
        "has_quantize_linear": True,
        "has_dequantize_linear": True,
        "bn_folded": True,
    }


def _quantize_weight_symmetric(w: torch.Tensor, scale: float) -> torch.Tensor:
    s = float(scale) if float(scale) > 0 else 1.0
    return torch.round(w.float() / s).clamp(-128, 127).to(torch.int8)


def audit_pack_vs_fakequant(model: nn.Module, packed: dict[str, Any]) -> dict[str, Any]:
    """校验 packed int8 与 FakeQuant 重构一致（折叠权重）。"""
    details = []
    max_err = 0.0
    for name, entry in packed.get("packed_weights", {}).items():
        mod = dict(model.named_modules()).get(name)
        if mod is None:
            raise KeyError(f"packed 层 {name} 不在模型中")
        wfq = getattr(mod, "weight_fake_quant", None)
        w = getattr(mod, "weight", None)
        if wfq is None or w is None:
            raise RuntimeError(f"{name} 缺少 weight / weight_fake_quant")
        scale = float(wfq.scale.detach().cpu().reshape(-1)[0].item())
        w_cpu = w.detach().cpu().float()
        q_ref = _quantize_weight_symmetric(w_cpu, scale)
        q_pack = entry["qweight"].cpu()
        if not torch.equal(q_ref, q_pack):
            raise RuntimeError(f"{name}: packed qweight 与 round(w/scale) 不一致")
        with torch.no_grad():
            w_fq = wfq(w).detach().cpu().float()
        recon = q_pack.float() * scale
        err = float((recon - w_fq).abs().max().item())
        max_err = max(max_err, err)
        details.append({"name": name, "scale": scale, "recon_err_vs_fq": err, "q_absmax": int(q_pack.abs().max())})
    return {"ok": True, "max_recon_err_vs_fq": max_err, "n_layers": len(details), "layers": details}


def pack_fakequant_int8(model: nn.Module, out_path: Path, *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """打包：对每个带 ``weight_fake_quant`` 的模块，量化其 ``.weight``（已是折叠权重）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packed_weights: dict[str, Any] = {}
    fp32_protect: dict[str, Any] = {}
    n_int8 = 0
    n_fp32 = 0

    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if not torch.is_tensor(w):
            continue
        wfq = getattr(mod, "weight_fake_quant", None)
        w_cpu = w.detach().cpu()
        if wfq is not None:
            scale = float(wfq.scale.detach().cpu().reshape(-1)[0].item())
            # 完整性：FQ 应作用在同一份 weight 上（折叠后不应再乘 BN scale）
            with torch.no_grad():
                w_fq = wfq(w).detach().cpu().float()
            q = _quantize_weight_symmetric(w_cpu, scale)
            recon = q.float() * scale
            if float((recon - w_fq).abs().max().item()) > 1e-4 + scale:
                raise RuntimeError(
                    f"{name}: FakeQuant 重构与 round(w/scale) 偏差过大；"
                    "可能仍在使用未折叠 ConvBn 路径"
                )
            packed_weights[name] = {
                "qweight": q.contiguous(),
                "scale": torch.tensor([scale], dtype=torch.float32),
                "zero_point": 0,
                "qscheme": "per_tensor_symmetric",
                "dtype": "int8",
                "weight_source": "module.weight_after_bn_fold",
            }
            n_int8 += int(q.numel())
            continue
        # stem 折叠后权重可能在 stem.0（QAT ConvReLU2d）或 stem.0.0（intrinsic 内层 Conv）
        if name in {"stem.0", "stem.0.0", "head_out"}:
            fp32_protect[name] = w_cpu.float().contiguous()
            n_fp32 += int(w_cpu.numel())

    if len(packed_weights) < 1:
        raise RuntimeError("未打包到任何 INT8 层")
    has_stem = any(k.startswith("stem") for k in fp32_protect)
    if (not has_stem) or ("head_out" not in fp32_protect):
        raise RuntimeError(f"保护层缺失: protect={list(fp32_protect)}（需要 stem.* 与 head_out）")

    other_state: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        b = getattr(mod, "bias", None)
        if torch.is_tensor(b) and (
            name in packed_weights or name in fp32_protect or getattr(mod, "weight_fake_quant", None) is not None
        ):
            other_state[f"{name}.bias"] = b.detach().cpu().float().contiguous()

    payload = {
        "format": PACKED_FORMAT,
        "packed_weights": packed_weights,
        "fp32_protect_weights": fp32_protect,
        "other_state": other_state,
        "meta": {
            **(meta or {}),
            "bn_folded": True,
            "note": "INT8 权重对应 BN 折叠后的 Conv/Linear.weight；与 FakeQuant 仿真一致",
        },
    }
    audit = audit_pack_vs_fakequant(model, payload)
    payload["meta"]["pack_audit"] = {k: audit[k] for k in ("ok", "max_recon_err_vs_fq", "n_layers")}
    torch.save(payload, str(out_path))

    theo_bytes = n_int8 * 1 + n_fp32 * 4 + len(packed_weights) * 4
    return {
        "path": str(out_path.resolve()),
        "n_int8_weights": n_int8,
        "n_fp32_protect_weights": n_fp32,
        "theoretical_packed_bytes": theo_bytes,
        "theoretical_packed_mb": theo_bytes / (1024.0 * 1024.0),
        "measured_packed_mb": _file_mb(out_path),
        "n_quant_layers": len(packed_weights),
        "n_protect_layers": len(fp32_protect),
        "quant_layer_names": list(packed_weights.keys()),
        "protect_layer_names": list(fp32_protect.keys()),
        "pack_audit": audit,
        "bn_folded": True,
    }


def build_prepared_model(model_cfg: dict, ckpt: Path) -> QuantizableGridCNN:
    """FP32 → BN fold → FakeQuant 插入（未校准）。"""
    m = QuantizableGridCNN.from_fp32_checkpoint(model_cfg, ckpt, map_location="cpu")
    fold_batchnorm(m)
    prepare_fakequant(m)
    return m


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


def run_qat_epochs(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    class_weight: torch.Tensor | None,
    focal_gamma: float,
    log: LogFn | None = None,
    rank: int = 0,
    sampler: DistributedSampler | None = None,
) -> dict[str, Any]:
    model.train()
    # FakeQuant 训练期保持 observer 开启，便于尺度随权重更新
    model.apply(torch.ao.quantization.enable_observer)
    model.apply(torch.ao.quantization.enable_fake_quant)
    opt = torch.optim.Adam(_unwrap(model).parameters(), lr=float(lr))
    history: list[dict[str, Any]] = []
    for ep in range(1, int(epochs) + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = classification_loss(model(x), y, class_weight=class_weight, focal_gamma=focal_gamma)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().item())
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        row = {"epoch": ep, "train_loss": avg, "sec": time.time() - t0}
        history.append(row)
        if rank == 0:
            _log(log, "QAT epoch %d/%d loss=%.6f (%.1fs)", ep, epochs, avg, row["sec"])
    return {"history": history, "final_train_loss": history[-1]["train_loss"] if history else None}


def build_train_loader(
    splits_dir: Path,
    batch_size: int,
    train_cfg: dict,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler | None, torch.Tensor | None, float]:
    ds = NpyTensorDataset(str(splits_dir / "X_train.npy"), str(splits_dir / "y_train.npy"))
    y_np = ds.y.numpy()
    imb = train_cfg.get("imbalance", {}) if train_cfg else {}
    factor = positive_oversample_factor_from_cfg(imb)
    idx = build_positive_oversample_index_map(y_np, factor)
    train_ds: Any = IndexMappedDataset(ds, idx) if factor > 1 else ds
    sampler: DistributedSampler | None = None
    if distributed:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True
        )
    else:
        loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
        )
    cw = compute_class_weights_tensor(y_np, imb) if bool(imb.get("use_class_weight", True)) else None
    return loader, sampler, cw, focal_gamma_from_cfg(imb)


def eval_model_dual(
    model: nn.Module,
    *,
    splits_dir: Path,
    device: torch.device,
    deploy_threshold: float,
    batch_size: int,
    t_cfg: dict | None = None,
) -> dict[str, Any]:
    t_cfg = t_cfg or {}
    pin = device.type == "cuda"
    return dual_threshold_eval(
        model,
        val_loader=make_split_loader(splits_dir, "val", batch_size, pin_memory=pin),
        test_loader=make_split_loader(splits_dir, "test", batch_size, pin_memory=pin),
        device=device,
        deploy_threshold=float(deploy_threshold),
        t_min=float(t_cfg.get("min", 0.1)),
        t_max=float(t_cfg.get("max", 0.9)),
        t_step=float(t_cfg.get("step", 0.02)),
        objective=str(t_cfg.get("objective", "f1")),
    )


def size_report_fp32(model: nn.Module, ckpt: Path) -> dict[str, Any]:
    n = count_parameters(model)
    return {
        "n_params": n,
        "checkpoint_mb": _file_mb(ckpt),
        "state_dict_mb_est": n * 4 / (1024.0 * 1024.0),
    }


def _primary_metrics(dual: dict[str, Any]) -> dict[str, Any]:
    m = dual["test_at_val_threshold"]
    vb = dual["val_best_threshold"]
    return {
        "f1": m.get("f1"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "roc_auc": m.get("roc_auc"),
        "avg_precision": m.get("avg_precision"),
        "threshold": float(vb["threshold"]),
        "threshold_source": "val_retuned_f1",
        "confusion": m.get("confusion"),
        "val_at_val_threshold": dual.get("val_at_val_threshold"),
        "test_at_deploy_threshold": dual.get("test_at_deploy_threshold"),
    }


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _render_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# 官方对称 INT8（BN 折叠 + FakeQuant）：FP32 vs PTQ vs QAT",
        "",
        f"- API：`{summary.get('api')}`",
        f"- 拓扑：`{summary.get('topology')}`",
        f"- 保护层：stem / head_out = FP32",
        f"- QAT：{summary.get('qat_epochs')} epoch，lr={summary.get('qat_lr')}，DDP={summary.get('distributed')}×{summary.get('world_size')}",
        f"- ckpt：`{summary.get('ckpt')}`",
        "",
        "> 指标为 **BN 已折叠** 的对称 signed INT8 FakeQuant 仿真；未 convert 到 FBGEMM 整型算子。"
        " ONNX 为无 BN 的 QDQ 图；packed 权重与 FakeQuant 的 ``round(w/scale)`` 一致。",
        "",
        "## 对比表（test @ val 重搜 τ）",
        "",
        "| 模型 | F1 | P | R | τ | params | ckpt MB | packed 实测 MB | packed 理论 MB | QDQ ONNX | BN folded |",
        "|------|----|---|---|---|--------|---------|----------------|----------------|----------|-----------|",
    ]
    for r in summary.get("comparison_rows") or []:
        lines.append(
            "| {model} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {threshold:.2f} | {n_params} | {ckpt} | {pm} | {pt} | {onnx} | {bn} |".format(
                model=r.get("model"),
                f1=float(r.get("f1") or 0),
                precision=float(r.get("precision") or 0),
                recall=float(r.get("recall") or 0),
                threshold=float(r.get("threshold") or 0),
                n_params=r.get("n_params"),
                ckpt=_fmt(r.get("checkpoint_mb")),
                pm=_fmt(r.get("packed_measured_mb")),
                pt=_fmt(r.get("packed_theoretical_mb")),
                onnx=r.get("onnx_qdq"),
                bn=r.get("bn_folded"),
            )
        )
    lines += ["", "## 产物", "", "- `ptq/` / `qat/`：FakeQuant 状态、无 BN 的 QDQ ONNX、审计通过的 packed INT8", ""]
    return "\n".join(lines)


def run_official_quant(
    *,
    model_cfg: dict,
    train_cfg: dict,
    ckpt: Path,
    splits_dir: Path,
    out_dir: Path,
    deploy_threshold: float,
    device: torch.device,
    batch_size: int = 128,
    calib_batches: int = 50,
    qat_epochs: int = 20,
    qat_lr: float = 1e-4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    log: LogFn | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ptq").mkdir(exist_ok=True)
        (out_dir / "qat").mkdir(exist_ok=True)

    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    t_search = (train_cfg or {}).get("threshold_search", {})
    summary: dict[str, Any] = {
        "api": "torch.ao.quantization eager FakeQuantize, symmetric signed INT8 W/A",
        "topology": "BN-fold (fuse_modules) then FakeQuant; stem/head_out FP32",
        "evaluation_semantics": "bn_folded_fakequant_simulation_not_fbgemm_int8_kernels",
        "protect": {"stem": "fp32", "head_out": "fp32"},
        "ckpt": str(Path(ckpt).resolve()),
        "splits_dir": str(Path(splits_dir).resolve()),
        "deploy_threshold_train": float(deploy_threshold),
        "qat_epochs": int(qat_epochs),
        "qat_lr": float(qat_lr),
        "distributed": bool(distributed),
        "world_size": int(world_size),
        "pipeline_version": 2,
    }

    # ----- FP32 -----
    if rank == 0:
        _log(log, "=== FP32 baseline ===")
        fp32 = QuantizableGridCNN.from_fp32_checkpoint(model_cfg, ckpt, map_location="cpu")
        fp32.eval().to(device)
        sizes_fp32 = size_report_fp32(fp32, ckpt)
        dual_fp32 = eval_model_dual(
            fp32, splits_dir=splits_dir, device=device, deploy_threshold=deploy_threshold,
            batch_size=batch_size, t_cfg=t_search,
        )
        summary["fp32"] = {"sizes": sizes_fp32, "metrics": _primary_metrics(dual_fp32), "dual": dual_fp32}
        dump_json(out_dir / "fp32_metrics.json", summary["fp32"])
        _log(
            log,
            "FP32 test@val_τ: F1=%.4f P=%.4f R=%.4f τ=%.4f ckpt=%.3fMB params=%d",
            summary["fp32"]["metrics"]["f1"],
            summary["fp32"]["metrics"]["precision"],
            summary["fp32"]["metrics"]["recall"],
            summary["fp32"]["metrics"]["threshold"],
            sizes_fp32["checkpoint_mb"],
            sizes_fp32["n_params"],
        )
    if distributed:
        torch.distributed.barrier()

    # ----- PTQ -----
    if rank == 0:
        _log(log, "=== PTQ (BN-fold + symmetric FakeQuant calib) ===")
        ptq = build_prepared_model(model_cfg, ckpt)
        calib_loader = make_split_loader(splits_dir, "train", batch_size, pin_memory=device.type == "cuda")
        n_cal = calibrate(ptq, calib_loader, device, max_batches=calib_batches)
        freeze_fakequant(ptq)
        dual_ptq = eval_model_dual(
            ptq, splits_dir=splits_dir, device=device, deploy_threshold=deploy_threshold,
            batch_size=batch_size, t_cfg=t_search,
        )
        ex = next(iter(make_split_loader(splits_dir, "val", 2)))[0][:1]
        onnx_ptq = export_qdq_onnx(ptq, out_dir / "ptq" / "grid_cnn_ptq_qdq.onnx", ex)
        pack_ptq = pack_fakequant_int8(
            ptq.cpu(),
            out_dir / "ptq" / "grid_cnn_ptq_symint8_packed.pt",
            meta={"stage": "ptq", "calib_batches": n_cal},
        )
        torch.save(ptq.state_dict(), out_dir / "ptq" / "fakequant_state.pt")
        summary["ptq"] = {
            "calib_batches": n_cal,
            "metrics": _primary_metrics(dual_ptq),
            "dual": dual_ptq,
            "onnx": onnx_ptq,
            "packed": pack_ptq,
            "n_params": count_parameters(ptq),
            "bn_folded": True,
        }
        dump_json(out_dir / "ptq" / "metrics.json", summary["ptq"])
        _log(
            log,
            "PTQ test@val_τ: F1=%.4f P=%.4f R=%.4f τ=%.4f onnx_QDQ=%s nBN=%d packed=%.3fMB audit_err=%.2e",
            summary["ptq"]["metrics"]["f1"],
            summary["ptq"]["metrics"]["precision"],
            summary["ptq"]["metrics"]["recall"],
            summary["ptq"]["metrics"]["threshold"],
            True,
            onnx_ptq["n_batchnorm"],
            pack_ptq["measured_packed_mb"],
            pack_ptq["pack_audit"]["max_recon_err_vs_fq"],
        )
    if distributed:
        torch.distributed.barrier()

    # ----- QAT（同样先 BN-fold，再 FakeQuant 短训）-----
    _log(log, "=== QAT %d epochs lr=%g (ddp=%s, BN already folded) ===", qat_epochs, qat_lr, distributed)
    qat = build_prepared_model(model_cfg, ckpt)
    qat.to(device)
    train_loader, sampler, cw, gamma = build_train_loader(
        splits_dir, batch_size, train_cfg, distributed=distributed, rank=rank, world_size=world_size,
    )
    if cw is not None:
        cw = cw.to(device)
    train_model: nn.Module = qat
    if distributed:
        train_model = DDP(qat, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    qat_hist = run_qat_epochs(
        train_model,
        train_loader=train_loader,
        device=device,
        epochs=qat_epochs,
        lr=qat_lr,
        class_weight=cw,
        focal_gamma=gamma,
        log=log,
        rank=rank,
        sampler=sampler,
    )
    qat_core = _unwrap(train_model)
    freeze_fakequant(qat_core)

    if rank == 0:
        dual_qat = eval_model_dual(
            qat_core, splits_dir=splits_dir, device=device, deploy_threshold=deploy_threshold,
            batch_size=batch_size, t_cfg=t_search,
        )
        ex = next(iter(make_split_loader(splits_dir, "val", 2)))[0][:1]
        onnx_qat = export_qdq_onnx(qat_core, out_dir / "qat" / "grid_cnn_qat_qdq.onnx", ex)
        pack_qat = pack_fakequant_int8(
            qat_core.cpu(),
            out_dir / "qat" / "grid_cnn_qat_symint8_packed.pt",
            meta={"stage": "qat", "epochs": qat_epochs, "lr": qat_lr},
        )
        torch.save(qat_core.state_dict(), out_dir / "qat" / "fakequant_state.pt")
        dump_json(out_dir / "qat" / "train_history.json", qat_hist)
        summary["qat"] = {
            "epochs": qat_epochs,
            "lr": qat_lr,
            "train": qat_hist,
            "metrics": _primary_metrics(dual_qat),
            "dual": dual_qat,
            "onnx": onnx_qat,
            "packed": pack_qat,
            "n_params": count_parameters(qat_core),
            "bn_folded": True,
        }
        dump_json(out_dir / "qat" / "metrics.json", summary["qat"])
        _log(
            log,
            "QAT test@val_τ: F1=%.4f P=%.4f R=%.4f τ=%.4f onnx_QDQ=%s nBN=%d packed=%.3fMB audit_err=%.2e",
            summary["qat"]["metrics"]["f1"],
            summary["qat"]["metrics"]["precision"],
            summary["qat"]["metrics"]["recall"],
            summary["qat"]["metrics"]["threshold"],
            True,
            onnx_qat["n_batchnorm"],
            pack_qat["measured_packed_mb"],
            pack_qat["pack_audit"]["max_recon_err_vs_fq"],
        )

        rows = []
        for tag in ("fp32", "ptq", "qat"):
            block = summary.get(tag) or {}
            m = block.get("metrics") or {}
            sizes = block.get("sizes") or {}
            packed = block.get("packed") or {}
            rows.append(
                {
                    "model": tag.upper(),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "threshold": m.get("threshold"),
                    "roc_auc": m.get("roc_auc"),
                    "n_params": block.get("n_params") or sizes.get("n_params"),
                    "checkpoint_mb": sizes.get("checkpoint_mb"),
                    "packed_measured_mb": packed.get("measured_packed_mb"),
                    "packed_theoretical_mb": packed.get("theoretical_packed_mb"),
                    "onnx_qdq": (block.get("onnx") or {}).get("has_quantize_linear"),
                    "onnx_n_bn": (block.get("onnx") or {}).get("n_batchnorm"),
                    "bn_folded": block.get("bn_folded", tag == "fp32"),
                    "pack_audit_max_err": (packed.get("pack_audit") or {}).get("max_recon_err_vs_fq"),
                }
            )
        summary["comparison_rows"] = rows
        _write_comparison_csv(out_dir / "comparison.csv", rows)
        dump_json(out_dir / "summary.json", summary)
        (out_dir / "summary.md").write_text(_render_summary_md(summary), encoding="utf-8")
        _log(log, "done. out_dir=%s", out_dir.resolve())

    if distributed:
        torch.distributed.barrier()
    return summary
