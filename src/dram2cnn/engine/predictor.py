from __future__ import annotations

"""测试集推理：加载 GridCNN 权重，对 `X_test` 前向，写出 predictions CSV。

阈值默认读训练目录 **`best_threshold.json`**（若无则用 0.5），用于将正类概率转为 `y_pred`。
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dram2cnn.data.dataset import NpyTensorDataset
from dram2cnn.models.factory import build_model
from dram2cnn.utils.device import resolve_device
from dram2cnn.utils.io import dump_json


def _validate_model_input_shape(
    model_cfg: dict, sample_shape: tuple[int, ...], tabular_shape: tuple[int, ...] | None
) -> None:
    if len(sample_shape) != 4:
        raise ValueError(f"Expected 4D tensor [N,C,H,W], got shape: {sample_shape}")
    _, c, h, w = sample_shape
    mismatch = []
    if int(model_cfg.get("in_channels", c)) != c:
        mismatch.append("in_channels mismatch")
    if int(model_cfg.get("grid_height", h)) != h:
        mismatch.append("grid_height mismatch")
    if int(model_cfg.get("grid_width", w)) != w:
        mismatch.append("grid_width mismatch")
    if mismatch:
        raise ValueError("Model/data shape mismatch: " + "; ".join(mismatch))
    td = int(model_cfg.get("tabular_dim", 0))
    if td > 0:
        if tabular_shape is None or len(tabular_shape) != 2:
            raise ValueError("model.tabular_dim>0 需要 tabular_test.npy")
        if int(tabular_shape[1]) != td:
            raise ValueError(f"tabular F={tabular_shape[1]} != model.tabular_dim={td}")
    elif tabular_shape is not None:
        raise ValueError("存在 tabular_test.npy 但 model.tabular_dim=0")


def _load_checkpoint_state(path: Path, map_location):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError, OSError, pickle.UnpicklingError):
        return torch.load(str(path), map_location=map_location)


def _resolve_threshold(run_dir: Path, default=0.5):
    tfile = run_dir / "best_threshold.json"
    if tfile.exists():
        try:
            return float(json.loads(tfile.read_text(encoding="utf-8")).get("threshold", default))
        except Exception:
            return float(default)
    return float(default)


def predict(model_cfg: dict, infer_cfg: dict, ckpt_path: str, splits_dir: str, output_csv: str, device: str = "auto") -> None:
    ckpt = Path(ckpt_path)
    x_test = Path(splits_dir) / "X_test.npy"
    y_test = Path(splits_dir) / "y_test.npy"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not x_test.exists() or not y_test.exists():
        raise FileNotFoundError(f"Missing split files: {x_test} / {y_test}")

    resolved_device = resolve_device(device)
    tab_path = Path(splits_dir) / "tabular_test.npy"
    ds = NpyTensorDataset(
        str(x_test),
        str(y_test),
        str(tab_path) if tab_path.is_file() else None,
    )
    tab_shape = tuple(ds.tab.shape) if ds.tab is not None else None
    _validate_model_input_shape(model_cfg, tuple(ds.x.shape), tab_shape)
    use_pin = str(resolved_device).startswith("cuda")
    dl = DataLoader(ds, batch_size=int(infer_cfg.get("batch_size", 512)), shuffle=False, pin_memory=use_pin)

    model = build_model(model_cfg).to(resolved_device)
    model.load_state_dict(_load_checkpoint_state(ckpt, resolved_device))
    model.eval()

    out_path = Path(output_csv)
    infer_artifact_dir = out_path.parent
    train_artifact_dir = ckpt.resolve().parent.parent
    threshold = _resolve_threshold(train_artifact_dir, default=0.5)

    ys, yp, yprob = [], [], []
    with torch.no_grad():
        for batch in dl:
            if len(batch) == 3:
                x, tab, y = batch[0], batch[1], batch[2]
                x, tab = x.to(resolved_device), tab.to(resolved_device)
                logits = model(x, tab)
            else:
                x, y = batch[0], batch[1]
                x = x.to(resolved_device)
                logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob >= threshold).long()
            ys.extend(y.tolist())
            yp.extend(pred.cpu().tolist())
            yprob.extend(prob.cpu().tolist())

    n = len(ys)
    y_arr = np.asarray(ys, dtype=np.int64)

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
    dump_json(
        infer_artifact_dir / "predictions_manifest.json",
        {
            "n_rows": int(n),
            "threshold": float(threshold),
            "checkpoint": str(ckpt.resolve()),
            "splits_dir": str(Path(splits_dir).resolve()),
            "device_resolved": str(resolved_device),
            "y_prob_summary": {
                "min": float(y_prob_np.min()) if n else None,
                "max": float(y_prob_np.max()) if n else None,
                "mean": float(y_prob_np.mean()) if n else None,
                "nunique": int(len(np.unique(y_prob_np))) if n else 0,
            },
            "columns": list(out.columns),
        },
    )
