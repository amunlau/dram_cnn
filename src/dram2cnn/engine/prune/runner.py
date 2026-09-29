"""剪枝对比实验编排：baseline + 多方法 × 多稀疏率 → 归档。

主指标默认：验证集 F1 阈值搜索 → 全量 time-stratified 测试集评估。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from dram2cnn.engine.prune.bn_channel import prune_bn_channels
from dram2cnn.engine.prune.common import (
    clone_model,
    load_checkpoint_state,
    make_split_loader,
    make_test_loader,
    model_stats,
    sparsity_dirname,
)
from dram2cnn.engine.prune.filter_prune import prune_filters_l1
from dram2cnn.engine.prune.finetune import short_finetune
from dram2cnn.engine.prune.reeval_thresholds import dual_threshold_eval
from dram2cnn.engine.prune.unstructured import (
    finalize_unstructured_pruning,
    prune_unstructured_l1,
    prune_unstructured_l2,
)
from dram2cnn.models.factory import build_model
from dram2cnn.models.grid_cnn import GridCNN
from dram2cnn.utils.io import dump_json, load_json

LogFn = Callable[..., None]

METHOD_ALIASES = {
    "unstructured": "unstructured_l1",
    "unstructured_l1": "unstructured_l1",
    "unstructured_l2": "unstructured_l2",
    "bn": "bn_channel",
    "bn_channel": "bn_channel",
    "slimming": "bn_channel",
    "filter": "filter",
    "filter_l1": "filter",
}

_UNSTRUCTURED_METHODS = frozenset({"unstructured_l1", "unstructured_l2"})


def run_prune_compare(
    *,
    model_cfg: dict,
    ckpt: Path,
    splits_dir: Path,
    out_dir: Path,
    threshold: float,
    device: torch.device,
    methods: list[str],
    sparsities: list[float],
    batch_size: int = 128,
    min_keep_channels: int = 8,
    global_ranking: bool = False,
    protect_stem: bool = True,
    protect_last_conv: bool = True,
    finetune_epochs: int = 0,
    seed: int = 42,
    skip_baseline: bool = False,
    shard_tag: str | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """执行剪枝对比；结果写入 ``out_dir``，返回 summary 字典。

    ``skip_baseline=True`` 时复用已有 ``out_dir/baseline/``（便于多卡并行分片）。
    ``shard_tag`` 非空时写入 ``summary_shard_<tag>.json``，不覆盖总 ``summary.json``。
    """

    def _log(msg: str, *args: Any) -> None:
        if log is not None:
            log(msg, *args)
        else:
            print(msg % args if args else msg)

    if finetune_epochs and finetune_epochs > 0:
        _log("剪枝后将微调 finetune_epochs=%d（结构化方法收益更大；非结构化也会微调）。", finetune_epochs)

    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pin = device.type == "cuda"
    test_loader = make_test_loader(splits_dir, batch_size, pin_memory=pin)
    val_loader = make_split_loader(splits_dir, "val", batch_size, pin_memory=pin)

    input_size = (
        int(model_cfg.get("in_channels", 31)),
        int(model_cfg.get("grid_height", 128)),
        int(model_cfg.get("grid_width", 32)),
    )

    base_model: GridCNN = build_model(model_cfg).to(device)  # type: ignore[assignment]
    state = load_checkpoint_state(ckpt, device)
    base_model.load_state_dict(state)
    base_model.eval()

    baseline_dir = out_dir / "baseline"
    if skip_baseline:
        metrics_path = baseline_dir / "metrics.json"
        stats_path = baseline_dir / "stats.json"
        if not metrics_path.is_file() or not stats_path.is_file():
            raise FileNotFoundError(
                f"skip_baseline 需要已有 {metrics_path} 与 {stats_path}；请先跑 baseline"
            )
        baseline_metrics = load_json(metrics_path)
        baseline_stats = load_json(stats_path)
        _log(
            "复用已有 baseline: F1=%.4f P=%.4f R=%.4f",
            baseline_metrics["f1"],
            baseline_metrics["precision"],
            baseline_metrics["recall"],
        )
    else:
        _log("Evaluating FP32 baseline（val 重搜 τ → 全量 time-stratified test）…")
        dual_b = dual_threshold_eval(
            base_model,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            deploy_threshold=threshold,
        )
        baseline_metrics = dict(dual_b["test_at_val_threshold"])
        baseline_metrics["threshold_source"] = "val_retuned_f1"
        baseline_metrics["deploy_threshold"] = float(threshold)
        baseline_metrics["metrics_at_deploy_tau"] = dual_b["test_at_deploy_threshold"]
        baseline_metrics["val_best_threshold"] = dual_b["val_best_threshold"]
        baseline_stats = model_stats(base_model, input_size=input_size, sparsity_for_macs=0.0)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        dump_json(baseline_dir / "metrics.json", baseline_metrics)
        dump_json(baseline_dir / "metrics_val_retuned.json", dual_b)
        dump_json(baseline_dir / "stats.json", baseline_stats)
        _log(
            "Baseline: valτ=%.2f F1=%.4f P=%.4f R=%.4f | @deployτ=%.2f P=%.4f | params=%d | GMACs=%.4f",
            float(dual_b["val_best_threshold"]["threshold"]),
            baseline_metrics["f1"],
            baseline_metrics["precision"],
            baseline_metrics["recall"],
            threshold,
            dual_b["test_at_deploy_threshold"]["precision"],
            baseline_stats["n_params"],
            baseline_stats["theoretical_gmacs"],
        )

    norm_methods: list[str] = []
    for m in methods:
        key = METHOD_ALIASES.get(m.strip().lower())
        if key is None:
            raise ValueError(
                f"未知剪枝方法: {m}；可选 unstructured_l1,unstructured_l2,bn_channel,filter"
            )
        if key not in norm_methods:
            norm_methods.append(key)

    rows: list[dict[str, Any]] = []
    for method in norm_methods:
        for sp in sparsities:
            tag = f"{method}/{sparsity_dirname(sp)}"
            run_dir = out_dir / method / sparsity_dirname(sp)
            run_dir.mkdir(parents=True, exist_ok=True)
            _log("=== %s target=%.2f ===", method, sp)
            try:
                model = clone_model(base_model)
                # 非结构化：微调前必须保留 prune mask（finalize=False），否则 Adam 会把零权重训回稠密。
                if method == "unstructured_l1":
                    stats = prune_unstructured_l1(
                        model, sp, input_size=input_size, finalize=False
                    )
                elif method == "unstructured_l2":
                    stats = prune_unstructured_l2(
                        model, sp, input_size=input_size, finalize=False
                    )
                elif method == "bn_channel":
                    stats = prune_bn_channels(
                        model,
                        sp,
                        min_keep_channels=min_keep_channels,
                        global_ranking=global_ranking,
                        protect_stem=protect_stem,
                        protect_last_conv=protect_last_conv,
                        input_size=input_size,
                    )
                elif method == "filter":
                    stats = prune_filters_l1(
                        model,
                        sp,
                        min_keep_channels=min_keep_channels,
                        global_ranking=global_ranking,
                        protect_stem=protect_stem,
                        protect_last_conv=protect_last_conv,
                        input_size=input_size,
                    )
                else:
                    raise ValueError(f"未实现的剪枝方法: {method}")

                ft_info: dict[str, Any] = {"epochs": 0, "skipped": True}
                if finetune_epochs > 0:
                    _log("  short finetune %d epochs …", finetune_epochs)
                    ft_info = short_finetune(
                        model,
                        splits_dir,
                        device,
                        epochs=int(finetune_epochs),
                        batch_size=batch_size,
                        lr=1e-4,
                        log=_log,
                    )

                if method in _UNSTRUCTURED_METHODS:
                    n_rm = finalize_unstructured_pruning(model)
                    # 固化后重测稀疏率 / 理论 MACs，与评估、存盘模型一致
                    after = model_stats(model, input_size=input_size, sparsity_for_macs=None)
                    before = stats["before"]
                    stats["after"] = after
                    stats["achieved_weight_sparsity"] = after["weight_sparsity"]
                    stats["param_reduction_ratio"] = 1.0 - (
                        after["n_params_nonzero"] / max(before["n_params"], 1)
                    )
                    stats["theoretical_macs_reduction_ratio"] = 1.0 - (
                        after["theoretical_macs"] / max(before["dense_macs"], 1.0)
                    )
                    stats["finalize"] = True
                    stats["n_params_removed_hooks"] = int(n_rm)
                    _log(
                        "  unstructured finalize: remove=%d sparsity=%.4f",
                        n_rm,
                        float(after["weight_sparsity"]),
                    )

                dual = dual_threshold_eval(
                    model,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    deploy_threshold=threshold,
                )
                metrics = dict(dual["test_at_val_threshold"])
                metrics["threshold_source"] = "val_retuned_f1"
                metrics["deploy_threshold"] = float(threshold)
                metrics["metrics_at_deploy_tau"] = dual["test_at_deploy_threshold"]
                metrics["val_best_threshold"] = dual["val_best_threshold"]
                torch.save(
                    {
                        "method": method,
                        "target_sparsity_or_prune_ratio": float(sp),
                        "threshold": float(dual["val_best_threshold"]["threshold"]),
                        "deploy_threshold": float(threshold),
                        "model_cfg": dict(model_cfg),
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "prune_stats": {k: v for k, v in stats.items() if k != "channel_masks"},
                        "finetune": ft_info,
                    },
                    run_dir / "pruned_model.pt",
                )
                dump_json(run_dir / "metrics.json", metrics)
                dump_json(run_dir / "metrics_val_retuned.json", dual)
                dump_json(
                    run_dir / "stats.json",
                    {
                        **{k: v for k, v in stats.items() if k != "channel_masks"},
                        "finetune": ft_info,
                    },
                )
                if "channel_masks" in stats:
                    dump_json(run_dir / "channel_masks.json", stats["channel_masks"])

                row = {
                    "method": method,
                    "target_ratio": float(sp),
                    "threshold": float(dual["val_best_threshold"]["threshold"]),
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "roc_auc": metrics["roc_auc"],
                    "precision_at_deploy_tau": dual["test_at_deploy_threshold"]["precision"],
                    "recall_at_deploy_tau": dual["test_at_deploy_threshold"]["recall"],
                    "f1_drop_vs_baseline": float(metrics["f1"] - baseline_metrics["f1"]),
                    "n_params_before": stats["before"]["n_params"],
                    "n_params_after": stats["after"]["n_params"],
                    "n_params_nonzero_after": stats["after"]["n_params_nonzero"],
                    "param_reduction_ratio": stats["param_reduction_ratio"],
                    "weight_sparsity": stats["after"]["weight_sparsity"],
                    "dense_gmacs_before": stats["before"]["dense_macs"] / 1e9,
                    "theoretical_gmacs_after": stats["after"]["theoretical_gmacs"],
                    "theoretical_macs_reduction_ratio": stats["theoretical_macs_reduction_ratio"],
                    "finetune_epochs": int(ft_info.get("epochs", 0)),
                    "path": str(run_dir.resolve()),
                    "ok": True,
                    "error": None,
                }
                rows.append(row)
                params_show = (
                    row["n_params_nonzero_after"]
                    if method in _UNSTRUCTURED_METHODS
                    else row["n_params_after"]
                )
                _log(
                    "  valτ=%.2f F1=%.4f P=%.4f R=%.4f (Δ%+.4f) | @deployτ P=%.4f | params %d→%d (red=%.2f%%) | theoGMACs %.4f→%.4f",
                    row["threshold"],
                    metrics["f1"],
                    metrics["precision"],
                    metrics["recall"],
                    row["f1_drop_vs_baseline"],
                    row["precision_at_deploy_tau"],
                    row["n_params_before"],
                    params_show,
                    100 * row["param_reduction_ratio"],
                    row["dense_gmacs_before"],
                    row["theoretical_gmacs_after"],
                )
            except Exception as exc:
                _log("  FAILED %s: %s", tag, exc)
                err_row = {
                    "method": method,
                    "target_ratio": float(sp),
                    "ok": False,
                    "error": str(exc),
                    "path": str(run_dir.resolve()),
                }
                rows.append(err_row)
                dump_json(run_dir / "error.json", err_row)

    summary = {
        "ckpt": str(ckpt.resolve()),
        "threshold": float(threshold),
        "device": str(device),
        "methods": norm_methods,
        "sparsities": [float(s) for s in sparsities],
        "min_keep_channels": min_keep_channels,
        "global_ranking": global_ranking,
        "protect_stem": protect_stem,
        "protect_last_conv": protect_last_conv,
        "finetune_epochs": int(finetune_epochs),
        "baseline": {"metrics": baseline_metrics, "stats": baseline_stats},
        "results": rows,
        "method_notes": {
            "unstructured_l1": "全局 L1 非结构化（|w|）；微调保留 mask，结束后固化；理论 MACs≈dense*(1-sparsity)。",
            "unstructured_l2": "全局 L2 非结构化（w²）；对标量权重与 |w| 排序等价；微调同样保留 mask。",
            "bn_channel": "按现成 BN|γ| 逐层淘汰通道并切片衔接；未做 γ-L1 再训练；默认可短时微调。",
            "filter": "按 filter 权重 L1-norm 淘汰输出通道；衔接同 bn_channel。",
        },
    }
    if shard_tag:
        dump_json(out_dir / f"summary_shard_{shard_tag}.json", summary)
        _log("已写入分片 summary_shard_%s.json（ok=%d/%d）", shard_tag, sum(1 for r in rows if r.get("ok")), len(rows))
    else:
        dump_json(out_dir / "summary.json", summary)
        (out_dir / "summary.md").write_text(_render_summary_md(summary), encoding="utf-8")
    return summary


def merge_prune_shards(out_dir: Path, *, log: LogFn | None = None) -> dict[str, Any]:
    """合并 ``summary_shard_*.json`` 为总 ``summary.json`` / ``summary.md``。"""

    def _log(msg: str, *args: Any) -> None:
        if log is not None:
            log(msg, *args)
        else:
            print(msg % args if args else msg)

    shards = sorted(out_dir.glob("summary_shard_*.json"))
    if not shards:
        raise FileNotFoundError(f"未找到分片: {out_dir}/summary_shard_*.json")

    baseline = None
    all_rows: list[dict[str, Any]] = []
    methods: list[str] = []
    meta: dict[str, Any] | None = None
    for sp in shards:
        part = load_json(sp)
        if baseline is None:
            baseline = part["baseline"]
            meta = {k: part[k] for k in part if k not in ("results", "methods", "baseline")}
        for m in part.get("methods", []):
            if m not in methods:
                methods.append(m)
        all_rows.extend(part.get("results", []))

    # 稳定排序：方法顺序 + ratio
    method_order = {m: i for i, m in enumerate(methods)}
    all_rows.sort(key=lambda r: (method_order.get(r.get("method", ""), 99), float(r.get("target_ratio", 0))))

    assert meta is not None and baseline is not None
    summary = {
        **meta,
        "methods": methods,
        "baseline": baseline,
        "results": all_rows,
        "shards": [p.name for p in shards],
    }
    # 合并 method_notes
    notes: dict[str, str] = {}
    for sp in shards:
        notes.update(load_json(sp).get("method_notes") or {})
    summary["method_notes"] = notes

    dump_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(_render_summary_md(summary), encoding="utf-8")
    n_ok = sum(1 for r in all_rows if r.get("ok"))
    _log("合并完成: results=%d ok=%d → summary.json / summary.md", len(all_rows), n_ok)
    return summary


def _render_summary_md(summary: dict[str, Any]) -> str:
    b = summary["baseline"]["metrics"]
    lines = [
        "# GridCNN 剪枝对比实验报告",
        "",
        f"- checkpoint: `{summary['ckpt']}`",
        f"- primary threshold: **val-retuned F1**（各模型独立；与训练 threshold_search 一致）",
        f"- baseline deploy τ（对照）: **{summary['threshold']}**",
        f"- device: `{summary.get('device', '')}`",
        f"- sparsities / prune ratios: {summary['sparsities']}",
        "",
        "## Baseline (FP32)",
        "",
        f"| F1 | Precision | Recall | ROC-AUC | params |",
        f"|----|-----------|--------|---------|--------|",
        f"| {b['f1']:.4f} | {b['precision']:.4f} | {b['recall']:.4f} | {b['roc_auc']:.4f} | "
        f"{summary['baseline']['stats']['n_params']} |",
        "",
        "## Results",
        "",
        "| method | ratio | valτ | F1 | ΔF1 | P | R | params_after | param_red | theo_GMACs | ok |",
        "|--------|-------|------|----|-----|---|---|--------------|-----------|------------|----|",
    ]
    for r in summary["results"]:
        if not r.get("ok"):
            lines.append(
                f"| {r['method']} | {r['target_ratio']:.2f} | - | - | - | - | - | - | - | - | FAIL: {r.get('error')} |"
            )
            continue
        params_show = (
            r["n_params_nonzero_after"]
            if r["method"] in _UNSTRUCTURED_METHODS
            else r["n_params_after"]
        )
        thr = r.get("threshold")
        thr_s = f"{float(thr):.2f}" if thr is not None else "—"
        lines.append(
            f"| {r['method']} | {r['target_ratio']:.2f} | {thr_s} | {r['f1']:.4f} | {r['f1_drop_vs_baseline']:+.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {params_show} | "
            f"{100 * r['param_reduction_ratio']:.1f}% | {r['theoretical_gmacs_after']:.4f} | ok |"
        )
    lines.extend(
        [
            "",
            "## 方法说明",
            "",
            "### 1. unstructured_l1（非结构化 L1）",
            "",
            "- 对 Conv/Linear 权重做全局 L1 magnitude 剪枝（`torch.nn.utils.prune`）。",
            "- 不改网络结构；稀疏加速需专用内核，报告中 MACs 为理论值。",
            "",
            "### 2. unstructured_l2（非结构化 L2）",
            "",
            "- 全局按权重平方（w²）排序剪枝；对单个标量权重与 `|w|` 排序等价。",
            "- 目录与 L1 分开归档，便于对照；理想情况下指标应与 L1 一致。",
            "",
            "### 3. bn_channel（BN-γ 通道剪枝）",
            "",
            "- Network Slimming **剪枝阶段**：按 `|γ|` **逐层**淘汰通道（可用 `--global-ranking`），切片 Conv/BN 并同步后续 `in_channels` 与 `head.0`。",
            "- **注意**：全量训练未对 γ 施加 L1；此处用现成 γ 作重要性；默认短时微调。",
            "",
            "### 4. filter（Filter L1 结构化）",
            "",
            "- 按每个输出卷积核权重 L1-norm 剪通道；通道衔接与 bn_channel 相同。",
            "",
            "## 注意事项",
            "",
            "- **主指标**：各模型在 val 上重搜 F1 阈值后再评 time-stratified 全量 test。",
            "- 默认**短时微调**（`--finetune-epochs`）；设为 0 可对比「纯剪枝」掉点。",
            "- 结构化剪枝每层至少保留 `min_keep_channels` 个通道。",
            "- 分类末层 `head.3` 输出维不剪；stem 输入通道（31）不剪。",
            "- 全量训练未对 BN γ 做 L1；bn_channel 仅用现成 γ 作重要性。",
            "",
        ]
    )
    if int(summary.get("finetune_epochs") or 0) > 0:
        lines.insert(
            6,
            f"- finetune_epochs: **{summary['finetune_epochs']}**",
        )
    return "\n".join(lines) + "\n"
