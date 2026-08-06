"""对剪枝 / 先剪枝后量化产物按验证集重搜阈值，并在全量时间分层测试集上重评。

背景：若一律沿用基线 FP32 的部署阈值 τ（如 0.84），剪枝+量化后概率分布通常
更「保守」，在高阈值下易出现 FP=0 → Precision=1.0，同时 Recall 被压低。
正确对比口径与训练一致：在 **val** 上 ``search_threshold(objective=f1)``，
再用该 τ 评估 **test**。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dram2cnn.engine.infer_int8 import load_packed_int8
from dram2cnn.engine.metrics import search_threshold
from dram2cnn.engine.prune.common import (
    eval_at_threshold,
    load_checkpoint_state,
    make_split_loader,
    predict_probs,
    sparsity_dirname,
)
from dram2cnn.engine.prune.load_pruned import load_pruned_model
from dram2cnn.engine.prune.prune_then_quantize import (
    _file_mb,
    _render_big_table,
    _write_csv,
    discover_pruned_runs,
)
from dram2cnn.models.factory import build_model
from dram2cnn.utils.io import dump_json

LogFn = Callable[..., None]


def load_split_manifest(splits_dir: Path) -> dict[str, Any]:
    path = Path(splits_dir) / "split_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_splits(splits_dir: Path) -> dict[str, Any]:
    """核对评估数据是否为时间分层全量切分。"""
    spl = Path(splits_dir)
    man = load_split_manifest(spl)
    counts: dict[str, int] = {}
    for tag in ("train", "val", "test"):
        y = np.load(spl / f"y_{tag}.npy", mmap_mode="r")
        counts[f"n_{tag}"] = int(y.shape[0])
        counts[f"n_pos_{tag}"] = int((np.asarray(y) == 1).sum())
    return {
        "splits_dir": str(spl.resolve()),
        "split_strategy": man.get("split_strategy"),
        "manifest": man,
        "counts": counts,
        "ok_time_stratified": man.get("split_strategy") == "time_stratified",
        "ok_full_sized": (
            counts.get("n_train", 0) >= 10000
            and counts.get("n_val", 0) >= 1000
            and counts.get("n_test", 0) >= 1000
        ),
    }


def dual_threshold_eval(
    model: nn.Module,
    *,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    deploy_threshold: float,
    t_min: float = 0.1,
    t_max: float = 0.9,
    t_step: float = 0.02,
    objective: str = "f1",
) -> dict[str, Any]:
    """val 搜阈值 + test 双口径（部署 τ / val 重搜 τ）。"""
    y_v, p_v = predict_probs(model, val_loader, device)
    y_t, p_t = predict_probs(model, test_loader, device)
    val_best = search_threshold(
        y_v,
        p_v,
        t_min=t_min,
        t_max=t_max,
        step=t_step,
        objective=objective,
    )
    thr_val = float(val_best["threshold"])
    m_deploy = eval_at_threshold(y_t, p_t, float(deploy_threshold))
    m_val = eval_at_threshold(y_t, p_t, thr_val)
    m_val_on_val = eval_at_threshold(y_v, p_v, thr_val)
    return {
        "n_val": int(len(y_v)),
        "n_test": int(len(y_t)),
        "n_pos_test": int((np.asarray(y_t) == 1).sum()),
        "deploy_threshold": float(deploy_threshold),
        "val_best_threshold": val_best,
        "test_at_deploy_threshold": m_deploy,
        "test_at_val_threshold": m_val,
        "val_at_val_threshold": m_val_on_val,
        "threshold_protocol": (
            f"search on val ({objective}, [{t_min},{t_max}] step={t_step}); "
            "report primary metrics on time-stratified full test"
        ),
    }


def _apply_primary_from_dual(row: dict[str, Any], dual: dict[str, Any]) -> None:
    """主表指标改为 val 重搜阈值在 test 上的结果；保留固定部署 τ 对照列。"""
    m = dual["test_at_val_threshold"]
    md = dual["test_at_deploy_threshold"]
    vb = dual["val_best_threshold"]
    row["f1"] = m.get("f1")
    row["precision"] = m.get("precision")
    row["recall"] = m.get("recall")
    row["roc_auc"] = m.get("roc_auc")
    row["threshold"] = float(vb["threshold"])
    row["threshold_source"] = "val_retuned_f1"
    row["confusion"] = m.get("confusion")
    row["f1_at_deploy_tau"] = md.get("f1")
    row["precision_at_deploy_tau"] = md.get("precision")
    row["recall_at_deploy_tau"] = md.get("recall")
    row["confusion_at_deploy_tau"] = md.get("confusion")
    row["deploy_threshold"] = dual["deploy_threshold"]
    row["n_test"] = dual["n_test"]
    row["n_pos_test"] = dual["n_pos_test"]


def reeval_prune_then_quant_session(
    *,
    model_cfg: dict,
    splits_dir: Path,
    prune_dir: Path,
    quant_dir: Path,
    deploy_threshold: float,
    device: torch.device,
    batch_size: int = 128,
    t_min: float = 0.1,
    t_max: float = 0.9,
    t_step: float = 0.02,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """重评既有 prune_compare + prune_then_quant 会话，写回 summary / 大表。"""

    def _log(msg: str, *args: Any) -> None:
        if log is not None:
            log(msg, *args)
        else:
            print(msg % args if args else msg)

    audit = audit_splits(splits_dir)
    _log(
        "splits audit: strategy=%s n_train/val/test=%s/%s/%s time_stratified=%s",
        audit.get("split_strategy"),
        audit["counts"].get("n_train"),
        audit["counts"].get("n_val"),
        audit["counts"].get("n_test"),
        audit.get("ok_time_stratified"),
    )
    if not audit.get("ok_time_stratified"):
        raise RuntimeError(
            f"期望 split_strategy=time_stratified，实际={audit.get('split_strategy')!r} @ {splits_dir}"
        )

    pin = device.type == "cuda"
    val_loader = make_split_loader(splits_dir, "val", batch_size, pin_memory=pin)
    test_loader = make_split_loader(splits_dir, "test", batch_size, pin_memory=pin)

    summary_path = quant_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"缺少 {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = list(summary.get("rows") or [])

    # baseline FP32
    prune_summary = {}
    if (prune_dir / "summary.json").is_file():
        prune_summary = json.loads((prune_dir / "summary.json").read_text(encoding="utf-8"))
    ckpt = Path(prune_summary.get("ckpt") or summary.get("baseline_ckpt") or "")
    if not ckpt.is_file():
        raise FileNotFoundError(f"找不到 baseline ckpt: {ckpt}")

    _log("=== reeval baseline_fp32 ===")
    base = build_model(model_cfg)
    base.load_state_dict(load_checkpoint_state(ckpt, device))
    base.to(device).eval()
    dual_b = dual_threshold_eval(
        base,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        deploy_threshold=deploy_threshold,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
    )
    dump_json(prune_dir / "baseline" / "metrics_val_retuned.json", dual_b)

    for row in rows:
        if row.get("pipeline") == "baseline_fp32":
            _apply_primary_from_dual(row, dual_b)
            row["ok"] = True
            _log(
                "  baseline valτ=%.2f → test F1=%.4f P=%.4f R=%.4f | deployτ=%.2f F1=%.4f P=%.4f",
                row["threshold"],
                row["f1"],
                row["precision"],
                row["recall"],
                deploy_threshold,
                row["f1_at_deploy_tau"],
                row["precision_at_deploy_tau"],
            )

    # prune-only + prune→quant
    items = discover_pruned_runs(prune_dir)
    by_key = {(it["method"], round(float(it["ratio"]), 4)): it for it in items}

    for row in rows:
        method = str(row.get("prune_method") or "")
        ratio = round(float(row.get("prune_ratio") or 0.0), 4)
        pipe = row.get("pipeline")
        if pipe == "baseline_fp32":
            continue
        key = (method, ratio)
        item = by_key.get(key)
        if item is None:
            _log("  skip missing prune item %s", key)
            continue

        if pipe == "prune_only_fp32":
            _log("=== reeval prune_only %s/%.2f ===", method, ratio)
            model, _meta = load_pruned_model(
                item["pruned_pt"],
                model_cfg,
                channel_masks_path=item["channel_masks"] if item["channel_masks"].is_file() else None,
                map_location=device,
            )
            dual = dual_threshold_eval(
                model,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                deploy_threshold=deploy_threshold,
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
            )
            dump_json(item["run_dir"] / "metrics_val_retuned.json", dual)
            _apply_primary_from_dual(row, dual)
            row["model_size_mb"] = _file_mb(item["pruned_pt"]) or row.get("model_size_mb")
            _log(
                "  valτ=%.2f → F1=%.4f P=%.4f R=%.4f | @deployτ P=%.4f R=%.4f conf=%s",
                row["threshold"],
                row["f1"],
                row["precision"],
                row["recall"],
                row["precision_at_deploy_tau"],
                row["recall_at_deploy_tau"],
                row.get("confusion"),
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        elif pipe == "prune_then_w8a8":
            tag = f"{method}/{sparsity_dirname(ratio)}"
            q_out = quant_dir / method / sparsity_dirname(ratio)
            packed = q_out / "grid_cnn_w8a8_packed.pt"
            if not packed.is_file():
                _log("  missing packed %s", packed)
                row["ok"] = False
                row["error"] = f"missing {packed.name}"
                continue
            _log("=== reeval prune→W8A8 %s ===", tag)
            int8_model, _meta = load_packed_int8(packed, model_cfg=model_cfg, map_location=device)
            int8_model.to(device).eval()
            dual = dual_threshold_eval(
                int8_model,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                deploy_threshold=deploy_threshold,
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
            )
            dump_json(q_out / "metrics_val_retuned.json", dual)
            _apply_primary_from_dual(row, dual)
            row["model_size_mb"] = _file_mb(packed) or row.get("model_size_mb")
            _log(
                "  valτ=%.2f → F1=%.4f P=%.4f R=%.4f | @deployτ=%.2f P=%.4f (conf=%s)",
                row["threshold"],
                row["f1"],
                row["precision"],
                row["recall"],
                deploy_threshold,
                row["precision_at_deploy_tau"],
                dual["test_at_deploy_threshold"].get("confusion"),
            )
            del int8_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary["rows"] = rows
    summary["deploy_threshold_baseline"] = float(deploy_threshold)
    summary["threshold_protocol"] = (
        "primary metrics: val-retuned F1 threshold → full time-stratified test; "
        f"also retain *_at_deploy_tau columns for baseline τ={deploy_threshold}"
    )
    summary["data_audit"] = audit
    summary["notes"] = [
        "仅「先剪枝后量化」；不考虑先量化后剪枝。",
        "主指标：各模型在时间分层 **验证集** 上重搜 F1 最优阈值，再于 **全量测试集** 评估。",
        f"对照列 *_at_deploy_tau：固定使用基线部署阈值 τ={deploy_threshold}（易在量化后出现 P=1.0）。",
        f"数据：{audit.get('split_strategy')} @ {audit.get('splits_dir')} "
        f"(n_train/val/test="
        f"{audit['counts'].get('n_train')}/{audit['counts'].get('n_val')}/{audit['counts'].get('n_test')})",
        "prune_then_w8a8 主指标为 packed INT8 参考推理。",
    ]
    # keep threshold field as protocol note (not a single τ)
    summary["threshold"] = float(deploy_threshold)
    dump_json(quant_dir / "summary.json", summary)
    md = _render_big_table(summary)
    (quant_dir / "summary.md").write_text(md, encoding="utf-8")
    _write_csv(quant_dir / "comparison_table.csv", rows)
    return summary
