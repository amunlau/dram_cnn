"""先剪枝后量化（prune → W8A8 PTQ）编排与总表。

仅支持 **先剪枝、后量化**；从已有 ``prune_compare_*`` 产物加载剪枝模型，
再跑 PTQ，合并 baseline / 仅剪枝 / 剪枝+量化 的完整对比表。
结束后调用 ``reeval_thresholds``：主指标 = 各模型 val 重搜 F1 τ → 全量 test；
另保留固定基线部署 τ 对照（量化后可能出现 P=1.0）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from dram2cnn.engine.prune.common import sparsity_dirname
from dram2cnn.engine.prune.load_pruned import load_pruned_model
from dram2cnn.engine.quantize_ptq import run_w8a8_ptq
from dram2cnn.utils.io import dump_json

LogFn = Callable[..., None]


def _file_mb(path: Path | None) -> float | None:
    if path is None or not Path(path).is_file():
        return None
    return Path(path).stat().st_size / (1024 * 1024)


def discover_pruned_runs(prune_dir: Path) -> list[dict[str, Any]]:
    """扫描 prune_compare 目录下各 method/sparsity_XX。"""
    found: list[dict[str, Any]] = []
    for method_dir in sorted(prune_dir.iterdir()):
        if not method_dir.is_dir() or method_dir.name in ("baseline",):
            continue
        method = method_dir.name
        if method not in ("unstructured", "bn_channel", "filter"):
            continue
        for sp_dir in sorted(method_dir.glob("sparsity_*")):
            pt = sp_dir / "pruned_model.pt"
            if not pt.is_file():
                continue
            metrics = {}
            stats = {}
            if (sp_dir / "metrics.json").is_file():
                metrics = json.loads((sp_dir / "metrics.json").read_text(encoding="utf-8"))
            if (sp_dir / "stats.json").is_file():
                stats = json.loads((sp_dir / "stats.json").read_text(encoding="utf-8"))
            # sparsity_30 → 0.30
            name = sp_dir.name
            ratio = float(name.split("_", 1)[1]) / 100.0
            found.append(
                {
                    "method": method,
                    "ratio": ratio,
                    "run_dir": sp_dir,
                    "pruned_pt": pt,
                    "channel_masks": sp_dir / "channel_masks.json",
                    "prune_metrics": metrics,
                    "prune_stats": stats,
                }
            )
    return found


def run_prune_then_quantize(
    *,
    model_cfg: dict,
    splits_dir: Path,
    prune_dir: Path,
    out_dir: Path,
    threshold: float,
    device: torch.device,
    methods: list[str] | None = None,
    sparsities: list[float] | None = None,
    batch_size: int = 128,
    calib_size: int = 1024,
    adaround_steps: int = 200,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """对 prune_compare 中的剪枝模型逐个做 W8A8 PTQ，并写总对比表。"""

    def _log(msg: str, *args: Any) -> None:
        if log is not None:
            log(msg, *args)
        else:
            print(msg % args if args else msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    method_filter = set(methods) if methods else None
    spars_filter = {round(s, 4) for s in sparsities} if sparsities else None

    # baseline from prune summary if present
    prune_summary_path = prune_dir / "summary.json"
    prune_summary = {}
    if prune_summary_path.is_file():
        prune_summary = json.loads(prune_summary_path.read_text(encoding="utf-8"))

    baseline_m = (prune_summary.get("baseline") or {}).get("metrics") or {}
    baseline_s = (prune_summary.get("baseline") or {}).get("stats") or {}
    if not baseline_m and (prune_dir / "baseline" / "metrics.json").is_file():
        baseline_m = json.loads((prune_dir / "baseline" / "metrics.json").read_text(encoding="utf-8"))
        baseline_s = json.loads((prune_dir / "baseline" / "stats.json").read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []

    # Row: FP32 baseline (no prune, no quant)
    fp32_ckpt = Path(prune_summary.get("ckpt") or "")
    baseline_size = _file_mb(fp32_ckpt) if fp32_ckpt.is_file() else baseline_s.get("n_params", 0) * 4 / 1024 / 1024
    rows.append(
        {
            "pipeline": "baseline_fp32",
            "prune_method": "none",
            "prune_ratio": 0.0,
            "quant_scheme": "none_fp32",
            "f1": baseline_m.get("f1"),
            "precision": baseline_m.get("precision"),
            "recall": baseline_m.get("recall"),
            "roc_auc": baseline_m.get("roc_auc"),
            "n_params": baseline_s.get("n_params"),
            "n_params_nonzero": baseline_s.get("n_params_nonzero", baseline_s.get("n_params")),
            "weight_sparsity": baseline_s.get("weight_sparsity", 0.0),
            "param_reduction_vs_baseline": 0.0,
            "theoretical_gmacs": baseline_s.get("theoretical_gmacs")
            or (baseline_s.get("dense_macs", 0) / 1e9 if baseline_s.get("dense_macs") else None),
            "model_size_mb": baseline_size,
            "model_size_note": "fp32_checkpoint",
            "ok": True,
            "error": None,
            "source": str(prune_dir / "baseline"),
        }
    )

    runs = discover_pruned_runs(prune_dir)
    for item in runs:
        if method_filter and item["method"] not in method_filter:
            continue
        if spars_filter and round(item["ratio"], 4) not in spars_filter:
            continue

        method = item["method"]
        ratio = item["ratio"]
        pm = item["prune_metrics"]
        ps = item["prune_stats"]
        after = ps.get("after") or {}
        before = ps.get("before") or {}
        pruned_size = _file_mb(item["pruned_pt"])

        # Row: prune-only FP32
        rows.append(
            {
                "pipeline": "prune_only_fp32",
                "prune_method": method,
                "prune_ratio": ratio,
                "quant_scheme": "none_fp32",
                "f1": pm.get("f1"),
                "precision": pm.get("precision"),
                "recall": pm.get("recall"),
                "roc_auc": pm.get("roc_auc"),
                "n_params": after.get("n_params"),
                "n_params_nonzero": after.get("n_params_nonzero"),
                "weight_sparsity": after.get("weight_sparsity"),
                "param_reduction_vs_baseline": ps.get("param_reduction_ratio"),
                "theoretical_gmacs": after.get("theoretical_gmacs"),
                "model_size_mb": pruned_size,
                "model_size_note": "pruned_fp32_pt",
                "ok": True,
                "error": None,
                "source": str(item["run_dir"]),
            }
        )

        # Quantize
        tag = f"{method}/{sparsity_dirname(ratio)}"
        q_out = out_dir / method / sparsity_dirname(ratio)
        q_out.mkdir(parents=True, exist_ok=True)
        _log("=== prune→quant %s ===", tag)
        try:
            model, meta = load_pruned_model(
                item["pruned_pt"],
                model_cfg,
                channel_masks_path=item["channel_masks"] if item["channel_masks"].is_file() else None,
                map_location=device,
            )
            report = run_w8a8_ptq(
                model_cfg=model_cfg,
                ckpt=None,
                fp32_model=model,
                channel_masks=meta.get("channel_masks"),
                splits_dir=splits_dir,
                out_dir=q_out,
                threshold=threshold,
                device=device,
                calib_size=calib_size,
                adaround_steps=adaround_steps,
                batch_size=batch_size,
                log=_log,
            )
            dump_json(q_out / "quantization_report.json", report)
            qm = report["w8a8_int8_packed_test_deployment"]
            sizes = report["sizes"]
            rows.append(
                {
                    "pipeline": "prune_then_w8a8",
                    "prune_method": method,
                    "prune_ratio": ratio,
                    "quant_scheme": "W8A8_packed_int8",
                    "f1": qm.get("f1"),
                    "precision": qm.get("precision"),
                    "recall": qm.get("recall"),
                    "roc_auc": qm.get("roc_auc"),
                    "f1_fakequant": report["w8a8_test_deployment"].get("f1"),
                    "n_params": report["model"]["n_params_fp32"],
                    "n_params_nonzero": after.get("n_params_nonzero"),
                    "weight_sparsity": after.get("weight_sparsity"),
                    "param_reduction_vs_baseline": ps.get("param_reduction_ratio"),
                    "theoretical_gmacs": after.get("theoretical_gmacs"),
                    "model_size_mb": sizes.get("packed_int8_file_mb"),
                    "model_size_theoretical_mb": sizes.get("theoretical_packed_mb"),
                    "fp32_size_mb": sizes.get("fp32_state_dict_mb"),
                    "compression_ratio": sizes.get("theoretical_compression_ratio"),
                    "model_size_note": "packed_int8",
                    "ok": True,
                    "error": None,
                    "source": str(q_out.resolve()),
                    "max_abs_prob_diff_fq_vs_int8": report.get("fakequant_vs_int8", {}).get(
                        "max_abs_prob_diff"
                    ),
                }
            )
            _log(
                "  prune→W8A8 INT8: F1=%.4f P=%.4f R=%.4f | packed=%.3f MB",
                qm["f1"],
                qm["precision"],
                qm["recall"],
                float(sizes.get("packed_int8_file_mb") or 0),
            )
        except Exception as exc:
            _log("  FAILED %s: %s", tag, exc)
            err = {
                "pipeline": "prune_then_w8a8",
                "prune_method": method,
                "prune_ratio": ratio,
                "quant_scheme": "W8A8_packed_int8",
                "ok": False,
                "error": str(exc),
                "source": str(q_out.resolve()),
            }
            rows.append(err)
            dump_json(q_out / "error.json", err)

    summary = {
        "pipeline": "prune_then_quantize",
        "prune_dir": str(prune_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "threshold": float(threshold),
        "device": str(device),
        "adaround_steps": adaround_steps,
        "calib_size": calib_size,
        "baseline_ckpt": prune_summary.get("ckpt"),
        "rows": rows,
        "notes": [
            "仅「先剪枝后量化」；不考虑先量化后剪枝。",
            "主指标：各模型在时间分层验证集上重搜 F1 最优阈值，再于全量测试集评估（与训练 threshold_search 一致）。",
            f"对照：固定基线部署阈值 τ={threshold} 的指标见 CSV 列 *_at_deploy_tau（量化后可能出现 P=1.0）。",
            "prune_only_fp32 指标来自既有 prune_compare 产物（含微调后结果）。",
            "prune_then_w8a8 主指标为 packed INT8 参考推理；另附 f1_fakequant。",
            "model_size_mb：baseline/prune_only 为文件体积；prune_then_w8a8 为 packed INT8 文件体积。",
        ],
    }
    dump_json(out_dir / "summary.json", summary)
    # 统一用 val 重搜阈值覆盖主表（避免固定基线 τ 导致量化后假性 P=1.0）
    from dram2cnn.engine.prune.reeval_thresholds import reeval_prune_then_quant_session

    summary = reeval_prune_then_quant_session(
        model_cfg=model_cfg,
        splits_dir=splits_dir,
        prune_dir=prune_dir,
        quant_dir=out_dir,
        deploy_threshold=float(threshold),
        device=device,
        batch_size=batch_size,
        log=_log,
    )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "pipeline",
        "prune_method",
        "prune_ratio",
        "quant_scheme",
        "threshold",
        "threshold_source",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "f1_at_deploy_tau",
        "precision_at_deploy_tau",
        "recall_at_deploy_tau",
        "deploy_threshold",
        "f1_fakequant",
        "n_params",
        "n_params_nonzero",
        "weight_sparsity",
        "param_reduction_vs_baseline",
        "theoretical_gmacs",
        "model_size_mb",
        "model_size_theoretical_mb",
        "fp32_size_mb",
        "compression_ratio",
        "model_size_note",
        "n_test",
        "n_pos_test",
        "ok",
        "error",
        "source",
    ]
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if v is None:
                v = ""
            s = str(v).replace(",", ";")
            vals.append(s)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _render_big_table(summary: dict[str, Any]) -> str:
    rows = summary["rows"]
    deploy_tau = summary.get("deploy_threshold_baseline", summary.get("threshold"))
    audit = summary.get("data_audit") or {}
    counts = audit.get("counts") or {}
    lines = [
        "# GridCNN 先剪枝后量化 · 完整对比表",
        "",
        f"- prune 来源：`{summary['prune_dir']}`",
        f"- 本实验输出：`{summary['out_dir']}`",
        f"- 数据：`{audit.get('split_strategy', 'time_stratified')}` @ `{audit.get('splits_dir', '')}` "
        f"(train/val/test="
        f"{counts.get('n_train', '—')}/{counts.get('n_val', '—')}/{counts.get('n_test', '—')})",
        f"- 主指标阈值：各模型 **val 重搜 F1**（与训练 threshold_search 一致）",
        f"- 对照固定阈值：基线部署 τ=**{deploy_tau}**（见「固定τ下P/R」列）",
        f"- device=`{summary['device']}` | AdaRound steps={summary['adaround_steps']}",
        "",
        "## 总览大表",
        "",
        "| 流水线 | 剪枝方法 | 剪枝比例 | 量化方案 | valτ | F1 | Precision | Recall | ROC-AUC | 固定τ下P | 固定τ下R | 参数量 | 非零参数 | 权重稀疏率 | 参数降幅 | 理论GMACs | 模型大小MB | 理论打包MB | 压缩比 | 状态 |",
        "|--------|----------|----------|----------|------|----|-----------|--------|---------|----------|----------|--------|----------|------------|----------|-----------|------------|------------|--------|------|",
    ]
    for r in rows:
        if not r.get("ok", True) and r.get("pipeline") == "prune_then_w8a8":
            lines.append(
                f"| {r.get('pipeline')} | {r.get('prune_method')} | {_fmt(r.get('prune_ratio'), 2)} | "
                f"{r.get('quant_scheme')} | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | FAIL |"
            )
            continue
        red = r.get("param_reduction_vs_baseline")
        red_s = f"{100 * float(red):.1f}%" if red is not None else "—"
        sp = r.get("weight_sparsity")
        sp_s = f"{100 * float(sp):.1f}%" if sp is not None else "—"
        lines.append(
            "| {pipe} | {pm} | {pr} | {qs} | {thr} | {f1} | {p} | {rc} | {auc} | {pd} | {rd} | {np_} | {nnz} | {sp} | {red} | {g} | {sz} | {tsz} | {cr} | {ok} |".format(
                pipe=r.get("pipeline", ""),
                pm=r.get("prune_method", ""),
                pr=_fmt(r.get("prune_ratio"), 2),
                qs=r.get("quant_scheme", ""),
                thr=_fmt(r.get("threshold"), 2),
                f1=_fmt(r.get("f1")),
                p=_fmt(r.get("precision")),
                rc=_fmt(r.get("recall")),
                auc=_fmt(r.get("roc_auc")),
                pd=_fmt(r.get("precision_at_deploy_tau")),
                rd=_fmt(r.get("recall_at_deploy_tau")),
                np_=r.get("n_params") if r.get("n_params") is not None else "—",
                nnz=r.get("n_params_nonzero") if r.get("n_params_nonzero") is not None else "—",
                sp=sp_s,
                red=red_s,
                g=_fmt(r.get("theoretical_gmacs")),
                sz=_fmt(r.get("model_size_mb"), 3),
                tsz=_fmt(r.get("model_size_theoretical_mb"), 3),
                cr=_fmt(r.get("compression_ratio"), 2),
                ok="ok" if r.get("ok", True) else "FAIL",
            )
        )

    lines.extend(
        [
            "",
            "### 列说明",
            "",
            "| 列 | 含义 |",
            "|----|------|",
            "| 流水线 | `baseline_fp32` / `prune_only_fp32` / `prune_then_w8a8` |",
            "| 剪枝方法 | none / unstructured / bn_channel / filter |",
            "| 剪枝比例 | 非结构化=权重稀疏率；结构化=通道剪除比例 |",
            "| 量化方案 | `none_fp32` 或 `W8A8_packed_int8` |",
            "| valτ | 该模型在时间分层验证集上 F1 最优阈值 |",
            "| F1/P/R | 用 valτ 在全量时间分层测试集上的主指标 |",
            f"| 固定τ下P/R | 强制使用基线部署 τ={deploy_tau}（量化后易出现 P=1.0） |",
            "| 模型大小MB | baseline/prune_only=文件体积；prune_then_w8a8=packed INT8 文件 |",
            "| 理论打包MB / 压缩比 | 仅量化行有意义 |",
            "",
            "## 注意",
            "",
            "- 本实验为 **先剪枝 → 再 W8A8 PTQ**；不包含先量化后剪枝。",
            "- 评估数据为实验 YAML 指定的 **time_stratified 全量** splits（非 smoke）。",
            "- `prune_then_w8a8` 主指标为 **packed INT8** 参考推理；fake-quant F1 见 CSV 的 `f1_fakequant`。",
            f"- 若固定 τ={deploy_tau} 出现 Precision=1.0：表示该阈值下 FP=0（模型更保守），不是算错或泄漏。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
