"""GridCNN 三种粒度剪枝对比实验入口。

产物写入训练会话下 ``prune_compare_<timestamp>/``（含 prune.log、summary、各方法子目录）。
主评价指标：各模型在验证集上重搜 F1 最优阈值后，再于全量 time-stratified 测试集评估。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from dram2cnn.engine.prune.runner import run_prune_compare
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


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_methods(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GridCNN 剪枝对比：unstructured / bn_channel / filter",
        epilog="示例: python -m dram2cnn.pipelines.run_prune configs/experiments/exp_full_f1_ddp.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_full_f1_ddp.yaml",
        help="实验 YAML",
    )
    parser.add_argument("--train-dir", default=None, help="训练会话目录（默认 latest_train）")
    parser.add_argument("--ckpt", default=None, help="权重路径（默认 <train-dir>/checkpoints/grid_cnn.pt）")
    parser.add_argument(
        "--methods",
        default="unstructured,bn_channel,filter",
        help="逗号分隔：unstructured,bn_channel,filter",
    )
    parser.add_argument(
        "--sparsities",
        default="0.3,0.5,0.7",
        help="目标稀疏率/通道剪除比例，逗号分隔",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-keep-channels", type=int, default=8)
    parser.add_argument(
        "--global-ranking",
        action="store_true",
        help="结构化剪枝时跨层全局排序淘汰（默认逐层同比例）",
    )
    parser.add_argument(
        "--no-protect-structured",
        action="store_true",
        help="结构化剪枝时不保护 stem/末层卷积",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=5,
        help="剪枝后短时微调轮数（0=关闭；默认 5，利于结构化剪枝恢复精度）",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="评估设备（单卡）",
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

    out_name = f"prune_compare_{new_timestamp_id()}"
    out_dir = train_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pointer(train_dir, ".latest_prune_session", out_name)

    logger = get_logger(out_dir / "prune.log")
    logger.info("exp_config=%s", args.exp_config)
    logger.info("train_dir=%s", train_dir.resolve())
    logger.info("ckpt=%s", ckpt.resolve())
    logger.info("out_dir=%s", out_dir.resolve())
    logger.info("threshold=%.4f", threshold)
    logger.info("methods=%s sparsities=%s device=%s", args.methods, args.sparsities, args.device)

    splits = Path(cfg["data"].get("splits_dir", "data/splits"))
    if not splits.is_absolute():
        splits = root / splits

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    summary = run_prune_compare(
        model_cfg=cfg["model"],
        ckpt=ckpt if ckpt.is_absolute() else root / ckpt,
        splits_dir=splits,
        out_dir=out_dir,
        threshold=threshold,
        device=device,
        methods=_parse_methods(args.methods),
        sparsities=_parse_float_list(args.sparsities),
        batch_size=int(args.batch_size),
        min_keep_channels=int(args.min_keep_channels),
        global_ranking=bool(args.global_ranking),
        protect_stem=not bool(args.no_protect_structured),
        protect_last_conv=not bool(args.no_protect_structured),
        finetune_epochs=int(args.finetune_epochs),
        log=logger.info,
    )

    n_ok = sum(1 for r in summary["results"] if r.get("ok"))
    n_all = len(summary["results"])
    print(f"Prune compare done: {out_dir}")
    print(f"  baseline F1={summary['baseline']['metrics']['f1']:.4f}")
    print(f"  runs ok={n_ok}/{n_all} → summary.md / summary.json / prune.log")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
