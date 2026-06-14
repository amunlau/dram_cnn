from __future__ import annotations

"""测试集推理：加载 GridCNN 权重，对 `X_test` 前向，写出 predictions CSV。

阈值默认读训练目录 **`best_threshold.json`**（若无则用 0.5），用于将正类概率转为 `y_pred`。
多卡时由 `torchrun` 启动，各 rank 分片推理后在 rank0 合并写出 CSV。
"""

import inspect
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from dram2cnn.data.dataset import NpyTensorDataset
from dram2cnn.models.factory import build_model
from dram2cnn.utils.device import resolve_device
from dram2cnn.utils.io import dump_json
from dram2cnn.utils.experiment_guards import state_dict_has_non_finite
from dram2cnn.utils.model_shape import validate_model_input_shape

logger = logging.getLogger(__name__)


def _load_checkpoint_state(path: Path, map_location):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError, OSError, pickle.UnpicklingError) as exc:
        logger.warning("weights_only checkpoint load failed (%s), retrying without weights_only: %s", path, exc)
        return torch.load(str(path), map_location=map_location)


def _resolve_threshold(run_dir: Path, default=0.5):
    tfile = run_dir / "best_threshold.json"
    if not tfile.is_file():
        return float(default)
    try:
        payload = json.loads(tfile.read_text(encoding="utf-8"))
        return float(payload.get("threshold", default))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"无法解析阈值文件 {tfile}: {exc}") from exc


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def predict(model_cfg: dict, infer_cfg: dict, ckpt_path: str, splits_dir: str, output_csv: str, device: str = "auto") -> None:
    """对 `X_test` 推理并写出 predictions CSV。

    Args:
        model_cfg: `cfg["model"]`。
        infer_cfg: `cfg["infer"]`（含 `batch_size`）。
        ckpt_path: `grid_cnn.pt` 路径。
        splits_dir: 含 `X_test.npy`、`y_test.npy`、可选 `meta_test.csv`。
        output_csv: 输出 CSV 路径；同目录写入 `predictions_manifest.json`。
        device: 推理设备。

    输出列：`y_true`, `y_pred`, `y_prob`, `threshold`；若存在 meta 则含 `sid`, `predict_time`。
    阈值来自训练 run 目录的 `best_threshold.json`，缺省 0.5。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_enabled = world_size > 1

    ckpt = Path(ckpt_path)
    x_test = Path(splits_dir) / "X_test.npy"
    y_test = Path(splits_dir) / "y_test.npy"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not x_test.exists() or not y_test.exists():
        raise FileNotFoundError(f"Missing split files: {x_test} / {y_test}")

    resolved_device = resolve_device(device)
    use_cuda = str(resolved_device).startswith("cuda")

    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("device is cuda but torch.cuda.is_available() == False")
        if ddp_enabled:
            torch.cuda.set_device(local_rank)
            dev = torch.device(f"cuda:{local_rank}")
        else:
            dev = torch.device(resolved_device)
    else:
        dev = torch.device("cpu")

    if ddp_enabled and not _is_dist():
        backend = "nccl" if use_cuda else "gloo"
        init_kw: dict = {"backend": backend}
        if use_cuda and "device_id" in inspect.signature(dist.init_process_group).parameters:
            init_kw["device_id"] = dev
        dist.init_process_group(**init_kw)

    tab_path = Path(splits_dir) / "tabular_test.npy"
    ds = NpyTensorDataset(
        str(x_test),
        str(y_test),
        str(tab_path) if tab_path.is_file() else None,
    )
    tab_shape = ds.tab_shape
    validate_model_input_shape(model_cfg, ds.x_shape, tab_shape)

    rank_indices: np.ndarray | None = None
    if ddp_enabled:
        rank_indices = np.arange(rank, len(ds), world_size, dtype=np.int64)
        infer_ds: NpyTensorDataset | Subset = Subset(ds, rank_indices.tolist())
    else:
        infer_ds = ds

    use_pin = use_cuda
    dl = DataLoader(
        infer_ds,
        batch_size=int(infer_cfg.get("batch_size", 512)),
        shuffle=False,
        pin_memory=use_pin,
    )

    model = build_model(model_cfg).to(dev)
    state = _load_checkpoint_state(ckpt, dev)
    if state_dict_has_non_finite(state):
        raise ValueError(
            f"Checkpoint 含 NaN/Inf 权重: {ckpt}。"
            "常见原因：GridCNN-Deep + focal_gamma>0 + 默认 lr 导致训练发散。"
            "请将 train.imbalance.focal_gamma 设为 0 后重训。"
        )
    model.load_state_dict(state)
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)
    model.eval()

    train_artifact_dir = ckpt.resolve().parent.parent
    threshold = _resolve_threshold(train_artifact_dir, default=0.5)

    local_rows: list[dict] = []
    cursor = 0
    with torch.no_grad():
        for batch in dl:
            if len(batch) == 3:
                x, tab, y = batch[0], batch[1], batch[2]
                x, tab = x.to(dev), tab.to(dev)
                logits = model(x, tab)
            else:
                x, y = batch[0], batch[1]
                x = x.to(dev)
                logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob >= threshold).long()
            bs = int(y.shape[0])
            if ddp_enabled and rank_indices is not None:
                batch_indices = rank_indices[cursor : cursor + bs]
            else:
                batch_indices = np.arange(cursor, cursor + bs, dtype=np.int64)
            cursor += bs
            for i in range(bs):
                local_rows.append(
                    {
                        "sample_index": int(batch_indices[i]),
                        "y_true": int(y[i].item()),
                        "y_pred": int(pred[i].item()),
                        "y_prob": float(prob[i].item()),
                    }
                )

    if ddp_enabled:
        gathered: list[list[dict] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_rows)
        if rank != 0:
            if _is_dist():
                dist.destroy_process_group()
            return
        all_rows: list[dict] = []
        for part in gathered:
            if part:
                all_rows.extend(part)
        all_rows.sort(key=lambda r: r["sample_index"])
    else:
        all_rows = local_rows

    n = len(all_rows)
    ys = [r["y_true"] for r in all_rows]
    yp = [r["y_pred"] for r in all_rows]
    yprob = [r["y_prob"] for r in all_rows]
    y_arr = np.asarray(ys, dtype=np.int64)

    out_path = Path(output_csv)
    infer_artifact_dir = out_path.parent
    meta_path = Path(splits_dir) / "meta_test.csv"
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        if len(meta_df) != n:
            raise ValueError(f"meta_test.csv rows ({len(meta_df)}) != predictions ({n})")
        for col in ("sid", "label"):
            if col not in meta_df.columns:
                raise ValueError(f"meta_test.csv missing `{col}`")
        if not np.array_equal(meta_df["label"].to_numpy(dtype=np.int64), y_arr):
            raise ValueError("meta_test label mismatch vs y_test; rerun preprocess.")
        base_cols = {
            "sample_index": np.arange(n, dtype=np.int64),
            "sid": meta_df["sid"].values,
            "meta_label": meta_df["label"].values,
            "y_true": ys,
            "y_pred": yp,
            "y_prob": yprob,
            "threshold": threshold,
        }
        if "predict_time" in meta_df.columns:
            base_cols["predict_time"] = meta_df["predict_time"].values
        out = pd.DataFrame(base_cols)
    else:
        out = pd.DataFrame(
            {
                "sample_index": np.arange(n, dtype=np.int64),
                "y_true": ys,
                "y_pred": yp,
                "y_prob": yprob,
                "threshold": threshold,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    y_prob_np = np.asarray(yprob, dtype=np.float64)
    if n > 0 and not np.isfinite(y_prob_np).all():
        n_bad = int(np.sum(~np.isfinite(y_prob_np)))
        raise ValueError(
            f"推理 y_prob 含 {n_bad}/{n} 个非有限值；checkpoint 可能已数值崩溃。"
            "请检查训练日志并重训（GridCNN-Deep 建议 focal_gamma=0）。"
        )

    dump_json(
        infer_artifact_dir / "predictions_manifest.json",
        {
            "n_rows": int(n),
            "threshold": float(threshold),
            "checkpoint": str(ckpt.resolve()),
            "splits_dir": str(Path(splits_dir).resolve()),
            "device_resolved": str(resolved_device),
            "world_size": world_size,
            "y_prob_summary": {
                "min": float(y_prob_np.min()) if n else None,
                "max": float(y_prob_np.max()) if n else None,
                "mean": float(y_prob_np.mean()) if n else None,
                "nunique": int(len(np.unique(y_prob_np))) if n else 0,
            },
            "columns": list(out.columns),
        },
    )

    if ddp_enabled and _is_dist():
        dist.destroy_process_group()
