"""先剪枝后量化实验入口。

默认读取训练会话下 ``.latest_prune_session``，对其中各剪枝模型做 W8A8 PTQ，
写出 ``prune_then_quant_<ts>/``（含总对比表 summary.md / comparison_table.csv）。
主表指标在结束后统一按 val 重搜阈值覆盖；另保留固定基线部署 τ 对照列。
仅支持先剪枝后量化。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from dram2cnn.engine.prune.prune_then_quantize import run_prune_then_quantize
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


def _parse_float_list(s: str | None) -> list[float] | None:
    if not s:
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_methods(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GridCNN 先剪枝后 W8A8 量化对比实验",
        epilog="示例: python -m dram2cnn.pipelines.run_prune_quantize configs/experiments/exp_full_f1_ddp.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_full_f1_ddp.yaml",
        help="实验 YAML",
    )
    parser.add_argument("--train-dir", default=None, help="训练会话目录")
    parser.add_argument(
        "--prune-dir",
        default=None,
        help="prune_compare_* 目录（默认 <train-dir>/.latest_prune_session）",
    )
    parser.add_argument("--methods", default=None, help="可选过滤：unstructured,bn_channel,filter")
    parser.add_argument("--sparsities", default=None, help="可选过滤，如 0.3,0.5,0.7")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calib-size", type=int, default=1024)
    parser.add_argument("--adaround-steps", type=int, default=200)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)

    root = project_root()
    cfg = load_experiment_config(args.exp_config)
    train_dir = _resolve_train_dir(root, cfg["base"], args.train_dir)

    if args.prune_dir:
        p = Path(args.prune_dir)
        prune_dir = p if p.is_absolute() else (root / p if p.exists() else train_dir / args.prune_dir)
    else:
        prune_dir = read_pointer_dir(train_dir, ".latest_prune_session", "latest_prune")
        if prune_dir is None:
            raise FileNotFoundError(
                f"未找到 {train_dir}/.latest_prune_session；请先 bash scripts/prune.sh 或传 --prune-dir"
            )

    thr_path = train_dir / "best_threshold.json"
    if not thr_path.is_file():
        raise FileNotFoundError(f"缺少 best_threshold.json: {thr_path}")
    threshold = float(json.loads(thr_path.read_text(encoding="utf-8"))["threshold"])

    out_name = f"prune_then_quant_{new_timestamp_id()}"
    out_dir = train_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pointer(train_dir, ".latest_prune_quant_session", out_name)

    logger = get_logger(out_dir / "prune_quantize.log")
    logger.info("exp_config=%s", args.exp_config)
    logger.info("train_dir=%s", train_dir.resolve())
    logger.info("prune_dir=%s", prune_dir.resolve())
    logger.info("out_dir=%s", out_dir.resolve())
    logger.info("threshold=%.4f device=%s", threshold, args.device)

    splits = Path(cfg["data"].get("splits_dir", "data/splits"))
    if not splits.is_absolute():
        splits = root / splits

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    summary = run_prune_then_quantize(
        model_cfg=cfg["model"],
        splits_dir=splits,
        prune_dir=prune_dir,
        out_dir=out_dir,
        threshold=threshold,
        device=device,
        methods=_parse_methods(args.methods),
        sparsities=_parse_float_list(args.sparsities),
        batch_size=int(args.batch_size),
        calib_size=int(args.calib_size),
        adaround_steps=int(args.adaround_steps),
        log=logger.info,
    )

    n_ok = sum(1 for r in summary["rows"] if r.get("ok", True) or r.get("pipeline") != "prune_then_w8a8")
    n_q = sum(1 for r in summary["rows"] if r.get("pipeline") == "prune_then_w8a8")
    n_q_ok = sum(1 for r in summary["rows"] if r.get("pipeline") == "prune_then_w8a8" and r.get("ok"))
    print(f"Prune→Quant done: {out_dir}")
    print(f"  quant runs ok={n_q_ok}/{n_q} | table: summary.md / comparison_table.csv")
    return 0 if n_q_ok == n_q else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
