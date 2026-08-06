"""加载剪枝产物：重建结构化通道拓扑并载入权重。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from dram2cnn.engine.prune.common import load_checkpoint_state
from dram2cnn.engine.prune.structured_utils import apply_channel_keeps
from dram2cnn.models.factory import build_model
from dram2cnn.models.grid_cnn import GridCNN


def load_channel_masks(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reconstruct_pruned_model(
    model_cfg: dict,
    *,
    channel_masks: dict[str, Any] | None = None,
) -> GridCNN:
    """按 channel_masks 重建结构化剪枝后的 GridCNN；无 masks 则返回原结构。"""
    model: GridCNN = build_model(model_cfg)  # type: ignore[assignment]
    if channel_masks and channel_masks.get("layers"):
        keeps = [torch.tensor(layer["keep_out"], dtype=torch.long) for layer in channel_masks["layers"]]
        apply_channel_keeps(model, keeps)
    return model


def load_pruned_model(
    pruned_pt: Path | str,
    model_cfg: dict,
    *,
    channel_masks_path: Path | str | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[GridCNN, dict[str, Any]]:
    """加载 ``pruned_model.pt``，返回 ``(model, meta)``。"""
    path = Path(pruned_pt)
    payload = load_checkpoint_state(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise TypeError(f"期望含 state_dict 的剪枝产物 dict，得到 {type(payload)}: {path}")

    method = str(payload.get("method", "unknown"))
    cfg = dict(payload.get("model_cfg") or model_cfg)
    masks_path = Path(channel_masks_path) if channel_masks_path else path.parent / "channel_masks.json"
    channel_masks = load_channel_masks(masks_path if masks_path.is_file() else None)

    # 结构化剪枝必须有 masks 才能对齐张量形状
    if method in ("bn_channel", "filter", "filter_l1") and not channel_masks:
        raise FileNotFoundError(
            f"结构化剪枝方法 {method} 需要 channel_masks.json（相邻于 {path} 或显式传入）"
        )

    model = reconstruct_pruned_model(cfg, channel_masks=channel_masks)
    model.load_state_dict(payload["state_dict"])
    device = torch.device(map_location) if not isinstance(map_location, torch.device) else map_location
    model.to(device)
    model.eval()
    meta = {
        "method": method,
        "target_ratio": float(payload.get("target_sparsity_or_prune_ratio", float("nan"))),
        "threshold": float(payload.get("threshold", 0.5)),
        "model_cfg": cfg,
        "channel_masks": channel_masks,
        "pruned_pt": str(path.resolve()),
        "finetune": payload.get("finetune"),
    }
    return model, meta
