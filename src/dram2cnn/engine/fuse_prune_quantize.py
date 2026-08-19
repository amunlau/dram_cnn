"""融合(Conv+BN+ReLU) → 强制无 bias → 非结构化剪枝 → W8A8 量化。

面向硬件无偏置交付的专用编排；日志写入会话目录。
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from dram2cnn.engine.prune.common import (
    load_checkpoint_state,
    make_split_loader,
    make_test_loader,
    model_stats,
)
from dram2cnn.engine.prune.finetune import short_finetune
from dram2cnn.engine.prune.reeval_thresholds import dual_threshold_eval
from dram2cnn.engine.prune.unstructured import prune_unstructured_l1
from dram2cnn.engine.quantize_ptq import (
    assert_no_biases,
    fold_batchnorm,
    run_w8a8_ptq,
)
from dram2cnn.models.factory import build_model
from dram2cnn.models.grid_cnn import GridCNN
from dram2cnn.utils.io import dump_json

LogFn = Callable[..., None]


def run_fuse_prune_quantize(
    *,
    model_cfg: dict,
    ckpt: Path,
    splits_dir: Path,
    out_dir: Path,
    threshold: float,
    device: torch.device,
    sparsity: float = 0.3,
    finetune_epochs: int = 5,
    batch_size: int = 128,
    calib_size: int = 1024,
    adaround_steps: int = 200,
    seed: int = 42,
    protect_stem: bool = False,
    protect_head_out: bool = False,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """执行：BN fold + 无 bias → unstructured prune → W8A8 PTQ。

    默认 ``protect_stem/protect_head_out=False``：主干首层与分类末层一并 INT8（整网全量化）。
    """

    def _log(msg: str, *args: Any) -> None:
        if log is not None:
            log(msg, *args)
        else:
            print(msg % args if args else msg)

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

    stages: dict[str, Any] = {"threshold_deploy": float(threshold), "sparsity": float(sparsity)}
    t_all = time.time()

    # --- 0) 加载 FP32 ---
    model: GridCNN = build_model(model_cfg).to(device)  # type: ignore[assignment]
    state = load_checkpoint_state(ckpt, device)
    model.load_state_dict(state)
    model.eval()
    _log("Loaded ckpt=%s | params=%d | head_bias=%s", ckpt, model.count_parameters(), model_cfg.get("head_bias"))

    dual0 = dual_threshold_eval(
        model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        deploy_threshold=threshold,
    )
    stages["fp32_train"] = {
        "test_at_deploy_tau": dual0["test_at_deploy_threshold"],
        "test_at_val_tau": dual0["test_at_val_threshold"],
        "val_threshold": dual0["val_best_threshold"],
        "stats": model_stats(model, input_size=input_size),
    }
    _log(
        "Stage0 FP32(train): deploy F1=%.4f | valτ F1=%.4f (τ=%.2f)",
        float(dual0["test_at_deploy_threshold"]["f1"]),
        float(dual0["test_at_val_threshold"]["f1"]),
        float(dual0["val_best_threshold"]["threshold"]),
    )

    # --- 1) Fuse Conv+BN+ReLU，强制无 bias ---
    fused = copy.deepcopy(model).cpu()
    fold_batchnorm(fused, force_zero_bias=True)
    bad = assert_no_biases(fused)
    if bad:
        raise RuntimeError(f"融合后仍有 bias: {bad}")
    fused.to(device).eval()
    torch.save(
        {"state_dict": fused.state_dict(), "model_cfg": model_cfg, "force_zero_bias": True, "folded": True},
        out_dir / "fused_nobias_fp32.pt",
    )
    dual1 = dual_threshold_eval(
        fused,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        deploy_threshold=threshold,
    )
    stages["fused_nobias"] = {
        "test_at_deploy_tau": dual1["test_at_deploy_threshold"],
        "test_at_val_tau": dual1["test_at_val_threshold"],
        "val_threshold": dual1["val_best_threshold"],
        "stats": model_stats(fused, input_size=input_size),
        "bias_params": assert_no_biases(fused),
    }
    _log(
        "Stage1 Fuse+no-bias: deploy F1=%.4f | valτ F1=%.4f (τ=%.2f)",
        float(dual1["test_at_deploy_threshold"]["f1"]),
        float(dual1["test_at_val_threshold"]["f1"]),
        float(dual1["val_best_threshold"]["threshold"]),
    )

    # --- 2) 非结构化剪枝 ---
    pruned = copy.deepcopy(fused)
    prune_info = prune_unstructured_l1(pruned, sparsity, input_size=input_size)
    _log(
        "Stage2 prune unstructured L1 sparsity=%.2f → achieved=%.4f",
        sparsity,
        float(prune_info["achieved_weight_sparsity"]),
    )
    ft_info: dict[str, Any] = {"epochs": 0, "skipped": True}
    if finetune_epochs > 0:
        _log("Stage2 short finetune epochs=%d …", finetune_epochs)
        ft_info = short_finetune(
            pruned,  # type: ignore[arg-type]
            splits_dir,
            device,
            epochs=int(finetune_epochs),
            batch_size=batch_size,
            log=_log,
        )
    pruned.eval()
    torch.save(
        {
            "state_dict": pruned.state_dict(),
            "model_cfg": model_cfg,
            "prune": prune_info,
            "finetune": ft_info,
            "force_zero_bias": True,
            "folded": True,
        },
        out_dir / "fused_nobias_pruned_fp32.pt",
    )
    dual2 = dual_threshold_eval(
        pruned,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        deploy_threshold=threshold,
    )
    stages["fused_pruned"] = {
        "test_at_deploy_tau": dual2["test_at_deploy_threshold"],
        "test_at_val_tau": dual2["test_at_val_threshold"],
        "val_threshold": dual2["val_best_threshold"],
        "prune": prune_info,
        "finetune": ft_info,
        "bias_params": assert_no_biases(pruned),
    }
    _log(
        "Stage2 after prune(+ft): deploy F1=%.4f | valτ F1=%.4f (τ=%.2f)",
        float(dual2["test_at_deploy_threshold"]["f1"]),
        float(dual2["test_at_val_threshold"]["f1"]),
        float(dual2["val_best_threshold"]["threshold"]),
    )

    # --- 3) W8A8（已 fold，skip_fold；force_zero_bias 再确认）---
    q_out = out_dir / "w8a8"
    q_out.mkdir(parents=True, exist_ok=True)
    # 使用剪枝后在 val 上重搜的阈值作为部署 τ（更贴近交付）
    thr_q = float(dual2["val_best_threshold"]["threshold"])
    _log("Stage3 W8A8 PTQ (skip_fold, force_zero_bias) threshold=%.4f …", thr_q)
    _log(
        "Stage3 protect_stem=%s protect_head_out=%s（False=该层也走 INT8）",
        protect_stem,
        protect_head_out,
    )
    report = run_w8a8_ptq(
        model_cfg=model_cfg,
        ckpt=None,
        fp32_model=pruned,  # type: ignore[arg-type]
        splits_dir=splits_dir,
        out_dir=q_out,
        threshold=thr_q,
        device=device,
        calib_size=calib_size,
        adaround_steps=adaround_steps,
        batch_size=batch_size,
        protect_stem=protect_stem,
        protect_head_out=protect_head_out,
        force_zero_bias=True,
        skip_fold=True,
        log=_log,
    )
    dump_json(q_out / "quantization_report.json", report)
    stages["w8a8"] = {
        "threshold": thr_q,
        "protect_stem": bool(protect_stem),
        "protect_head_out": bool(protect_head_out),
        "fp32_input_metrics": report.get("fp32_test_deployment"),
        "folded_metrics": report.get("bn_fold_fp32_test_deployment"),
        "fakequant_metrics": report.get("w8a8_test_deployment"),
        "int8_packed_metrics": report.get("w8a8_int8_packed_test_deployment"),
        "sizes": report.get("sizes"),
        "pack_path": str((q_out / "grid_cnn_w8a8_packed.pt").resolve()),
    }
    qm = report.get("w8a8_int8_packed_test_deployment") or {}
    _log(
        "Stage3 INT8 packed: F1=%.4f P=%.4f R=%.4f | packed=%.3f MB",
        float(qm.get("f1") or 0),
        float(qm.get("precision") or 0),
        float(qm.get("recall") or 0),
        float((report.get("sizes") or {}).get("packed_int8_file_mb") or 0),
    )

    all_int8 = (not protect_stem) and (not protect_head_out)
    summary = {
        "pipeline": "fuse_nobias_prune_w8a8",
        "ckpt": str(Path(ckpt).resolve()),
        "out_dir": str(out_dir.resolve()),
        "force_zero_bias": True,
        "protect_stem": bool(protect_stem),
        "protect_head_out": bool(protect_head_out),
        "all_layers_int8": bool(all_int8),
        "sparsity": float(sparsity),
        "finetune_epochs": int(finetune_epochs),
        "elapsed_sec": time.time() - t_all,
        "stages": stages,
        "notes": [
            "顺序：训练权重 → Conv+BN+ReLU 融合并丢弃 bias → 非结构化 L1 剪枝 → W8A8 PTQ。",
            "force_zero_bias 会丢弃 BN 折合偏置，数值上不等价于标准 BN fold；属硬件约束。",
            (
                "整网全 INT8：主干首层与分类末层均量化（protect_stem/protect_head_out=False）。"
                if all_int8
                else f"保护层：stem={protect_stem} head.3={protect_head_out}（True 表示该层保持单精度）。"
            ),
            "量化方案：逐层（per-tensor）对称 signed INT8，权重与激活 zero-point=0；非逐通道、非 uint8。",
            "主交付模型：w8a8/grid_cnn_w8a8_packed.pt",
        ],
    }
    dump_json(out_dir / "summary.json", summary)
    _write_summary_md(out_dir / "summary.md", summary)
    return summary


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    st = summary.get("stages") or {}
    lines = [
        "# 融合无偏置 → 剪枝 → W8A8 摘要",
        "",
        f"- 流程：`{summary.get('pipeline')}`",
        f"- 目标稀疏度：**{summary.get('sparsity')}**",
        f"- 剪枝后微调轮数：**{summary.get('finetune_epochs')}**",
        f"- 强制无偏置：**{summary.get('force_zero_bias')}**",
        f"- 保护主干首层：**{summary.get('protect_stem')}**（False=该层 INT8）",
        f"- 保护分类末层：**{summary.get('protect_head_out')}**（False=该层 INT8）",
        f"- 整网全 INT8：**{summary.get('all_layers_int8')}**",
        "- 量化：逐层对称 signed INT8（zp=0；非逐通道、非 uint8）",
        f"- 耗时（秒）：{summary.get('elapsed_sec'):.1f}" if summary.get("elapsed_sec") else "",
        "",
        "## 各阶段测试指标",
        "",
        "| 阶段 | F1@deployτ | F1@valτ | P@valτ | R@valτ |",
        "|------|------------|---------|--------|--------|",
    ]
    for key, label in (
        ("fp32_train", "FP32 训练权重"),
        ("fused_nobias", "融合+无 bias"),
        ("fused_pruned", "剪枝(+微调)"),
    ):
        block = st.get(key) or {}
        d = block.get("test_at_deploy_tau") or {}
        v = block.get("test_at_val_tau") or {}
        lines.append(
            f"| {label} | {_fmt(d.get('f1'))} | {_fmt(v.get('f1'))} | {_fmt(v.get('precision'))} | {_fmt(v.get('recall'))} |"
        )
    w = st.get("w8a8") or {}
    qm = w.get("int8_packed_metrics") or {}
    lines.append(
        f"| INT8 packed | {_fmt(qm.get('f1'))} | — | {_fmt(qm.get('precision'))} | {_fmt(qm.get('recall'))} |"
    )
    lines.extend(
        [
            "",
            f"- INT8 部署阈值 τ={w.get('threshold')}",
            f"- packed: `{w.get('pack_path')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "—"
