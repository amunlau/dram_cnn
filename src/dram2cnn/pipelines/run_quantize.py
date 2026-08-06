"""W8A8 PTQ 命令行入口（GridCNN-Deep）。

布局对齐训练/推理：产物与日志写入训练会话下 ``quant_w8a8_<timestamp>/``，
日志文件为 ``quantize.log``（``get_logger`` 同时打控制台）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from dram2cnn.engine.quantize_ptq import run_w8a8_ptq
from dram2cnn.utils.config import load_experiment_config, project_root
from dram2cnn.utils.logger import get_logger
from dram2cnn.utils.run_layout import new_timestamp_id, read_pointer_dir, write_pointer


def _resolve_train_dir(root: Path, base: dict, train_dir: str | None) -> Path:
    if train_dir:
        p = Path(train_dir)
        return p if p.is_absolute() else (root / p)
    run_name = str(base.get("run_name", "exp_tianchi_full_f1_ddp"))
    run_root = root / "runs" / run_name
    latest = read_pointer_dir(run_root, ".latest_train_session", "latest_train")
    if latest is None:
        raise FileNotFoundError(
            f"未找到最新训练会话指针：{run_root}/.latest_train_session；请传 --train-dir"
        )
    return latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GridCNN-Deep W8A8 PTQ（BN Folding + AdaRound 风格，无 QAT）",
        epilog="示例: python -m dram2cnn.pipelines.run_quantize configs/experiments/exp_full_f1_ddp.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_full_f1_ddp.yaml",
        help="实验 YAML",
    )
    parser.add_argument("--train-dir", default=None, help="训练会话目录（默认 latest_train）")
    parser.add_argument("--ckpt", default=None, help="权重路径（默认 <train-dir>/checkpoints/grid_cnn.pt）")
    parser.add_argument("--calib-size", type=int, default=1024)
    parser.add_argument("--adaround-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-protect", action="store_true", help="首末层也做 W8A8")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PTQ 使用设备；默认单卡 cuda:0（AdaRound 逐层，非 DDP）",
    )
    args = parser.parse_args(argv)

    root = project_root()
    cfg = load_experiment_config(args.exp_config)
    train_dir = _resolve_train_dir(root, cfg["base"], args.train_dir)
    ckpt = Path(args.ckpt) if args.ckpt else train_dir / "checkpoints" / "grid_cnn.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    thr_path = train_dir / "best_threshold.json"
    if not thr_path.is_file():
        raise FileNotFoundError(f"缺少 best_threshold.json: {thr_path}")
    threshold = float(json.loads(thr_path.read_text(encoding="utf-8"))["threshold"])

    out_name = f"quant_w8a8_{new_timestamp_id()}"
    out_dir = train_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pointer(train_dir, ".latest_quant_session", out_name)

    logger = get_logger(out_dir / "quantize.log")
    logger.info("exp_config=%s", args.exp_config)
    logger.info("train_dir=%s", train_dir.resolve())
    logger.info("ckpt=%s", ckpt.resolve())
    logger.info("out_dir=%s", out_dir.resolve())
    logger.info("threshold=%.4f (from best_threshold.json)", threshold)
    logger.info("device=%s (PTQ 单卡；多卡训练权重可在此单卡上量化)", args.device)

    splits = Path(cfg["data"].get("splits_dir", "data/splits"))
    if not splits.is_absolute():
        splits = root / splits

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    report = run_w8a8_ptq(
        model_cfg=cfg["model"],
        ckpt=ckpt if ckpt.is_absolute() else root / ckpt,
        splits_dir=splits,
        out_dir=out_dir,
        threshold=threshold,
        device=device,
        calib_size=int(args.calib_size),
        adaround_steps=int(args.adaround_steps),
        batch_size=int(args.batch_size),
        protect_stem=not args.no_protect,
        protect_head_out=not args.no_protect,
        log=logger.info,
    )

    report_path = out_dir / "quantization_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fp32_m = report["fp32_test_deployment"]
    q_m = report["w8a8_test_deployment"]
    i_m = report.get("w8a8_int8_packed_test_deployment", q_m)
    drops = report["drops_vs_fp32"]
    sizes = report["sizes"]
    fq_vs = report.get("fakequant_vs_int8", {})
    md = f"""# GridCNN-Deep W8A8 PTQ 量化报告

**训练会话**：`{train_dir}`  
**本次量化输出**：`{out_dir}`  
**日志**：`{out_dir / "quantize.log"}`

## GPU 说明

{report.get("gpu_note", "")}  
本次 device=`{report.get("device")}`。

## 1. 量化前后模型对比

| 项 | FP32 | W8A8 PTQ |
|----|------|----------|
| 参数量（个数） | {report["model"]["n_params_fp32"]} | 同左（位宽变化） |
| 模型文件大小（MB） | {sizes.get("fp32_state_dict_mb", 0):.3f} | 打包 {sizes.get("packed_int8_file_mb", 0):.3f}；理论 {sizes.get("theoretical_packed_mb", 0):.3f} |
| 理论压缩比 | 1.0× | **{sizes.get("theoretical_compression_ratio", 0):.2f}×** |

## 2. 精度对比（测试集部署阶段，τ={report["deployment_threshold"]}）

| 指标 | FP32 | W8A8 fake-quant | W8A8 packed-INT8 | ΔF1(vs FP32, fake) |
|------|------|-----------------|------------------|---------------------|
| F1 | {fp32_m["f1"]:.6f} | {q_m["f1"]:.6f} | {i_m["f1"]:.6f} | {drops["f1"]["abs"]:+.6f} |
| Precision | {fp32_m["precision"]:.6f} | {q_m["precision"]:.6f} | {i_m["precision"]:.6f} | {drops["precision"]["abs"]:+.6f} |
| Recall | {fp32_m["recall"]:.6f} | {q_m["recall"]:.6f} | {i_m["recall"]:.6f} | {drops["recall"]["abs"]:+.6f} |
| ROC-AUC | {fp32_m["roc_auc"]:.6f} | {q_m["roc_auc"]:.6f} | {i_m["roc_auc"]:.6f} | {drops["roc_auc"]["abs"]:+.6f} |

fake-quant 与 packed-INT8 最大概率差：`{fq_vs.get("max_abs_prob_diff", float("nan"))}`。

### 混淆矩阵（解释 Precision 变化）

| | FP32 | W8A8 fake-quant |
|--|------|-----------------|
| TP / FP / FN / TN | {fp32_m["confusion"]["tp"]} / {fp32_m["confusion"]["fp"]} / {fp32_m["confusion"]["fn"]} / {fp32_m["confusion"]["tn"]} | {q_m["confusion"]["tp"]} / {q_m["confusion"]["fp"]} / {q_m["confusion"]["fn"]} / {q_m["confusion"]["tn"]} |
| Precision(counts) | {fp32_m["precision_from_counts"]:.6f} | {q_m["precision_from_counts"]:.6f} |
| Recall(counts) | {fp32_m["recall_from_counts"]:.6f} | {q_m["recall_from_counts"]:.6f} |

## 3. 量化配置

- PTQ + BN Folding + AdaRound 风格 + **AdaRound 后二次激活校准**；**无 QAT**
- 权重 Per-channel 对称 INT8；激活 Per-tensor 非对称 INT8
- 校准：验证集 {report["calib"]["n_samples"]} 条
- 保护层：stem={report["protect_layers"]["stem"]}，head.3={report["protect_layers"]["head.3"]}
- 推理：`bash scripts/infer_int8.sh` 加载 `grid_cnn_w8a8_packed.pt`（INT8 参考路径）

## 4. 路径

- 日志：`{out_dir / "quantize.log"}`
- 报告 JSON：`{report_path}`
- 对比预测：`{report["paths"]["predictions_compare"]}`
- 量化模型：`{report["paths"]["quant_model"]}`
- INT8 打包：`{report["paths"]["packed_model"]}`
"""
    (out_dir / "quantization_report.md").write_text(md, encoding="utf-8")
    logger.info("Wrote %s", report_path)
    logger.info(
        "部署阶段: FP32 F1=%.4f | fake-quant F1=%.4f | packed-INT8 F1=%.4f | ΔF1(fake)=%+.4f",
        fp32_m["f1"],
        q_m["f1"],
        i_m["f1"],
        drops["f1"]["abs"],
    )
    print(f"Quantize done: {out_dir}")
    print(
        f"  部署阶段: FP32 F1={fp32_m['f1']:.4f} | "
        f"fake-quant F1={q_m['f1']:.4f} | packed-INT8 F1={i_m['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
