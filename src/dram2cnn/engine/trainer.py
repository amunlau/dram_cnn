from __future__ import annotations

"""Grid CNN 训练循环（单头分类、可选 Focal 与过采样）。

早停指标（`train.early_stop_metric`）在验证集 AUC/PR 为 nan 或全负类时会退化为可比较的标量；
`best_score` 自 `-inf` 起算，避免首轮无法落盘。若整轮仍无「更好」checkpoint，rank0 在结束时
保存最后一轮权重，保证 `grid_cnn.pt` 存在供 infer 使用。详见 `docs/troubleshooting.md`。
"""

from pathlib import Path
import inspect
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_

from dram2cnn.data.dataset import IndexMappedDataset, NpyTensorDataset
from dram2cnn.engine.imbalance import (
    build_positive_oversample_index_map,
    compute_class_weights_tensor,
    compute_per_sample_weights,
)
from dram2cnn.engine.metrics import classification_metrics, search_best_threshold
from dram2cnn.models.factory import build_model
from dram2cnn.models.losses import classification_loss
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
        mismatch.append(f"in_channels={model_cfg.get('in_channels')} != tensor C={c}")
    if int(model_cfg.get("grid_height", h)) != h:
        mismatch.append(f"grid_height={model_cfg.get('grid_height')} != tensor H={h}")
    if int(model_cfg.get("grid_width", w)) != w:
        mismatch.append(f"grid_width={model_cfg.get('grid_width')} != tensor W={w}")
    if mismatch:
        raise ValueError("Model/data shape mismatch: " + "; ".join(mismatch))
    td = int(model_cfg.get("tabular_dim", 0))
    if td > 0:
        if tabular_shape is None or len(tabular_shape) != 2:
            raise ValueError("model.tabular_dim>0 需要 splits 中的 tabular_train.npy [N,F]")
        if int(tabular_shape[1]) != td:
            raise ValueError(f"tabular F={tabular_shape[1]} != model.tabular_dim={td}")
    elif tabular_shape is not None:
        raise ValueError("存在 tabular 数据但 model.tabular_dim=0，请在 model.yaml 中设置 tabular_dim 与之一致")


def _collect_eval(model, loader, device):
    model.eval()
    ys, yp, yprob = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, tab, y = batch[0], batch[1], batch[2]
                x, tab, y = x.to(device), tab.to(device), y.to(device)
                logits = model(x, tab)
            else:
                x, y = batch[0], batch[1]
                x, y = x.to(device), y.to(device)
                logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
            ys.extend(y.cpu().tolist())
            yp.extend(pred.cpu().tolist())
            yprob.extend(prob.cpu().tolist())
    return ys, yp, yprob


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_dist()) or dist.get_rank() == 0


def _metrics_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (np.asarray(y_prob) >= float(threshold)).astype(int)
    return classification_metrics(y_true, y_pred, y_prob)


def _early_stop_score(
    early_stop_metric: str,
    m_at_05: dict,
    m_best: dict,
    roc_auc_val,
    ap_val,
) -> float:
    """标量越大越好；无效指标（nan / 仅一类等）时退回 -1.0，配合 best_score=-inf 保证至少首轮可落盘。"""
    if early_stop_metric == "best_f1":
        x = float(m_best["f1"])
    elif early_stop_metric == "roc_auc":
        x = float(roc_auc_val) if roc_auc_val is not None else -1.0
    elif early_stop_metric == "avg_precision":
        x = float(ap_val) if ap_val is not None else -1.0
    elif early_stop_metric == "f1_at_0p5":
        x = float(m_at_05["f1"])
    else:
        raise ValueError(f"unexpected early_stop_metric for scoring: {early_stop_metric!r}")
    return x if math.isfinite(x) else -1.0


def _plot_curve(x, y, title: str, ylabel: str, out_png: Path) -> None:
    if not x or not y:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", linewidth=1.8, markersize=3.5)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _save_training_figures(run_dir: Path, epoch_rows: list[dict]) -> None:
    if not epoch_rows:
        return
    fig_dir = run_dir / "figures"
    epochs = [int(r["epoch"]) for r in epoch_rows]
    _plot_curve(epochs, [float(r["train_loss"]) for r in epoch_rows], "Train Loss by Epoch", "Loss", fig_dir / "train_loss_curve.png")
    _plot_curve(epochs, [float(r["best_f1"]) for r in epoch_rows], "Best-F1 by Epoch", "F1", fig_dir / "val_best_f1_curve.png")
    _plot_curve(epochs, [float(r["roc_auc"]) for r in epoch_rows], "ROC-AUC by Epoch (Val @0.5)", "ROC-AUC", fig_dir / "val_roc_auc_curve.png")
    _plot_curve(
        epochs,
        [float(r["avg_precision"]) for r in epoch_rows],
        "Average Precision (PR-AUC) by Epoch (Val)",
        "AP",
        fig_dir / "val_avg_precision_curve.png",
    )
    _plot_curve(epochs, [float(r["f1_at_0p5"]) for r in epoch_rows], "F1@0.5 vs Best-F1 by Epoch", "F1", fig_dir / "val_f1_compare_curve.png")
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [float(r["f1_at_0p5"]) for r in epoch_rows], marker="o", linewidth=1.8, markersize=3.5, label="F1@0.5")
    plt.plot(epochs, [float(r["best_f1"]) for r in epoch_rows], marker="o", linewidth=1.8, markersize=3.5, label="Best-F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Validation F1 Curves by Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "val_f1_dual_curve.png", dpi=150)
    plt.close()


CKPT_NAME = "grid_cnn.pt"


def train_model(
    model_cfg: dict,
    train_cfg: dict,
    splits_dir: str,
    run_dir: Path,
    device: str = "auto",
    logger=None,
    rng_seed: int = 42,
) -> Path:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_enabled = world_size > 1

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
        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
        torch.backends.cudnn.deterministic = not bool(train_cfg.get("cudnn_benchmark", True))
    else:
        dev = torch.device("cpu")

    if ddp_enabled and not _is_dist():
        backend = "nccl" if use_cuda else "gloo"
        init_kw: dict = {"backend": backend}
        if use_cuda and "device_id" in inspect.signature(dist.init_process_group).parameters:
            init_kw["device_id"] = dev
        dist.init_process_group(**init_kw)

    split_p = Path(splits_dir)
    train_tab = split_p / "tabular_train.npy"
    val_tab = split_p / "tabular_val.npy"
    train_base = NpyTensorDataset(
        str(split_p / "X_train.npy"),
        str(split_p / "y_train.npy"),
        str(train_tab) if train_tab.is_file() else None,
    )
    val_ds = NpyTensorDataset(
        str(split_p / "X_val.npy"),
        str(split_p / "y_val.npy"),
        str(val_tab) if val_tab.is_file() else None,
    )
    if len(train_base) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train/val split is empty. Please rerun preprocess with adjusted split settings.")
    tab_shape = tuple(train_base.tab.shape) if train_base.tab is not None else None
    _validate_model_input_shape(model_cfg, tuple(train_base.x.shape), tab_shape)

    imb_cfg = train_cfg.get("imbalance", {})
    y_train_np = np.load(f"{splits_dir}/y_train.npy")
    pos_os_raw = imb_cfg.get("positive_oversample_factor", 1)
    pos_os = int(pos_os_raw) if pos_os_raw is not None else 1
    idx_map = build_positive_oversample_index_map(y_train_np, pos_os)
    if idx_map.shape[0] > len(y_train_np):
        train_ds = IndexMappedDataset(train_base.x, train_base.y, idx_map, train_base.tab)
    else:
        train_ds = train_base

    train_sampler = None
    if ddp_enabled:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    elif bool(imb_cfg.get("use_weighted_sampler", False)):
        sw = compute_per_sample_weights(y_train_np, imb_cfg)
        g_ws = torch.Generator()
        g_ws.manual_seed(int(rng_seed))
        train_sampler = WeightedRandomSampler(
            torch.from_numpy(sw).double(),
            num_samples=len(train_ds),
            replacement=True,
            generator=g_ws,
        )
        if logger is not None:
            logger.info("Using WeightedRandomSampler for training (single-process). See train.imbalance.")
    if ddp_enabled and bool(imb_cfg.get("use_weighted_sampler", False)) and rank == 0:
        msg = "train.imbalance.use_weighted_sampler is ignored under DDP; using DistributedSampler and class weights only."
        if logger is not None:
            logger.info(msg)

    nw = int(train_cfg.get("num_workers", 0))
    pin_mem = bool(use_cuda)
    g_train = torch.Generator()
    g_train.manual_seed(int(rng_seed))
    use_shuffle = train_sampler is None
    train_dl = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 256)),
        shuffle=use_shuffle,
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=pin_mem,
        persistent_workers=bool(nw > 0),
        generator=g_train if use_shuffle else None,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_mem,
        persistent_workers=bool(nw > 0),
    )

    model = build_model(model_cfg).to(dev)
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)
    optim = torch.optim.Adam(
        model.parameters(), lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-4))
    )

    use_weight = bool(imb_cfg.get("use_class_weight", True))
    class_w = None
    if use_weight:
        class_w = compute_class_weights_tensor(y_train_np, imb_cfg).to(dev)

    _fg = imb_cfg.get("focal_gamma")
    focal_gamma = float(_fg) if _fg is not None else 0.0

    if _is_main_process() and logger is not None:
        logger.info(
            "Imbalance: use_class_weight=%s scheme=%s cap=%s focal_gamma=%s "
            "positive_oversample_factor=%s train_virtual_len=%s weighted_sampler=%s ddp=%s",
            use_weight,
            imb_cfg.get("weight_scheme"),
            imb_cfg.get("positive_weight_cap"),
            focal_gamma,
            pos_os,
            len(train_ds),
            bool(imb_cfg.get("use_weighted_sampler", False)),
            ddp_enabled,
        )

    epochs = int(train_cfg.get("epochs", 12))
    patience = int(train_cfg.get("patience", 3))
    max_grad_norm = train_cfg.get("max_grad_norm")
    early_stop_metric = str(train_cfg.get("early_stop_metric", "best_f1")).lower()
    if early_stop_metric not in {"best_f1", "roc_auc", "f1_at_0p5", "avg_precision"}:
        raise ValueError("train.early_stop_metric must be one of: best_f1, roc_auc, f1_at_0p5, avg_precision")
    if _is_main_process() and logger is not None:
        logger.info(
            "Early stopping: metric=%s patience=%d（该标量越大越好；连续 patience 轮无提升则停止）。"
            " 日志中 best_f1 为「验证集上阈值搜索后的 F1」，f1@0.5 为固定 0.5 阈值，二者不可混为一谈。",
            early_stop_metric,
            patience,
        )
        n_va_ds = len(val_ds)
        if n_va_ds < 32:
            logger.warning(
                "验证集 DataLoader 样本数 N_val=%d 较小：roc_auc/pr_ap 易呈阶梯状或长期不变。",
                n_va_ds,
            )
    best_score, bad_epochs = float("-inf"), 0
    epoch_rows = []

    ckpt = run_dir / "checkpoints" / CKPT_NAME
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch in train_dl:
            if len(batch) == 3:
                x, tab, y = batch[0], batch[1], batch[2]
                x, tab, y = x.to(dev), tab.to(dev), y.to(dev)
                logits = model(x, tab)
            else:
                x, y = batch[0], batch[1]
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
            loss = classification_loss(logits, y, class_weight=class_w, focal_gamma=focal_gamma)
            optim.zero_grad()
            loss.backward()
            if max_grad_norm is not None and float(max_grad_norm) > 0:
                clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optim.step()
            running_loss += float(loss.detach().item())
            batch_count += 1

        stop_now = False
        # 各 rank 先同步到此：避免 rank0 在验证时其它 rank 已进入下一轮 train 导致 NCCL 顺序错乱。
        if ddp_enabled and _is_dist():
            dist.barrier()

        if _is_main_process():
            avg_train_loss = float(running_loss / max(1, batch_count))
            epoch_seconds = float(time.time() - t0)
            eval_model = model.module if isinstance(model, DDP) else model
            ys, yp, yprob = _collect_eval(eval_model, val_dl, dev)
            m_at_05 = classification_metrics(ys, yp, yprob)
            best_t = search_best_threshold(
                np.asarray(ys),
                np.asarray(yprob),
                float(train_cfg.get("threshold_search", {}).get("min", 0.1)),
                float(train_cfg.get("threshold_search", {}).get("max", 0.9)),
                float(train_cfg.get("threshold_search", {}).get("step", 0.02)),
            )
            m_best = _metrics_at_threshold(ys, yprob, best_t["threshold"])

            roc_auc_val = m_at_05.get("roc_auc")
            ap_val = m_at_05.get("avg_precision")
            stop_score = _early_stop_score(early_stop_metric, m_at_05, m_best, roc_auc_val, ap_val)
            row = {
                "epoch": ep,
                "train_loss": avg_train_loss,
                "lr": float(optim.param_groups[0]["lr"]),
                "epoch_seconds": epoch_seconds,
                "batches": int(batch_count),
                "f1_at_0p5": m_at_05["f1"],
                "precision_at_0p5": m_at_05["precision"],
                "recall_at_0p5": m_at_05["recall"],
                "accuracy_at_0p5": m_at_05["accuracy"],
                "roc_auc": float("nan") if roc_auc_val is None else float(roc_auc_val),
                "avg_precision": float("nan") if ap_val is None else float(ap_val),
                "best_threshold": best_t["threshold"],
                "best_f1": m_best["f1"],
                "best_precision": m_best["precision"],
                "best_recall": m_best["recall"],
                "best_accuracy": m_best["accuracy"],
                "early_stop_score": stop_score,
            }
            epoch_rows.append(row)
            pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
            _save_training_figures(run_dir, epoch_rows)

            if logger is not None:
                auc_log = float("nan") if roc_auc_val is None else float(roc_auc_val)
                ap_log = float("nan") if ap_val is None else float(ap_val)
                logger.info(
                    "Epoch %d/%d | loss=%.6f | roc_auc=%.6f | pr_ap=%.6f | f1@0.5=%.6f | best_f1=%.6f | best_t=%.4f | "
                    "%s=%.6f | sec=%.2f",
                    ep,
                    epochs,
                    avg_train_loss,
                    auc_log,
                    ap_log,
                    float(m_at_05["f1"]),
                    float(m_best["f1"]),
                    float(best_t["threshold"]),
                    early_stop_metric,
                    float(stop_score),
                    epoch_seconds,
                )

            if stop_score > best_score:
                best_score = stop_score
                bad_epochs = 0
                state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save(state_dict, ckpt)
                dump_json(
                    run_dir / "best_val_metrics.json",
                    {
                        "epoch": ep,
                        "early_stop_metric": early_stop_metric,
                        "early_stop_score": stop_score,
                        "metrics_at_0p5": m_at_05,
                        "metrics_at_best_threshold": m_best,
                        "best_threshold": best_t,
                    },
                )
                pd.DataFrame({"y_true": ys, "y_pred": yp, "y_prob": yprob}).to_csv(run_dir / "val_predictions.csv", index=False)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if logger is not None:
                        logger.info(
                            "Early stop at epoch %d: metric=%s 已连续 %d 轮无提升（patience=%d）；"
                            "当前 best_%s=%.6f。",
                            ep,
                            early_stop_metric,
                            bad_epochs,
                            patience,
                            early_stop_metric,
                            float(best_score),
                        )
                    stop_now = True

        if ddp_enabled:
            # rank0 根据验证指标决定是否早停；广播标量使所有 rank 同时 break，避免僵尸进程。
            stop_t = torch.tensor([1 if stop_now else 0], device=dev)
            dist.broadcast(stop_t, src=0)
            if int(stop_t.item()) == 1:
                break
        elif stop_now:
            break

    ts_cfg = train_cfg.get("threshold_search", {})
    if _is_main_process() and bool(ts_cfg.get("enabled", True)) and (run_dir / "val_predictions.csv").exists():
        vdf = pd.read_csv(run_dir / "val_predictions.csv")
        best_t = search_best_threshold(
            vdf["y_true"].to_numpy(),
            vdf["y_prob"].to_numpy(),
            float(ts_cfg.get("min", 0.1)),
            float(ts_cfg.get("max", 0.9)),
            float(ts_cfg.get("step", 0.02)),
        )
        dump_json(run_dir / "best_threshold.json", best_t)

    if _is_main_process() and not ckpt.is_file():
        eval_model = model.module if isinstance(model, DDP) else model
        torch.save(eval_model.state_dict(), ckpt)
        if logger is not None:
            logger.warning(
                "验证早停指标未产生可比较的提升（常见于验证集无正例或 PR-AUC 为 nan）；"
                "已写入最后一轮权重到 %s，供推理使用。",
                ckpt,
            )

    if ddp_enabled and _is_dist():
        # 等所有 rank 结束训练循环后再销毁进程组，防止尾部 I/O 与 teardown 竞态。
        dist.barrier()
        dist.destroy_process_group()

    return ckpt
