"""融合无 bias →（可选 BN-γ）→ 非结构化剪枝 → W8A8 量化入口。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from dram2cnn.engine.fuse_prune_quantize import run_fuse_prune_quantize
from dram2cnn.utils.config import load_experiment_config, project_root
from dram2cnn.utils.logger import get_logger
from dram2cnn.utils.run_layout import new_timestamp_id, read_pointer_dir, write_pointer


def _resolve_train_dir(root: Path, base: dict, train_dir: str | None) -> Path:
    if train_dir:
        p = Path(train_dir)
        return p if p.is_absolute() else (root / p)
    run_name = str(base.get("run_name", "exp_tianchi_full_f1_nobias_ddp"))
    run_root = root / "runs" / run_name
    latest = read_pointer_dir(run_root, ".latest_train_session", "latest_train")
    if latest is None:
        raise FileNotFoundError(
            f"未找到最新训练会话指针：{run_root}/.latest_train_session；请传 --train-dir"
        )
    return latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GridCNN：可选 BN-γ 通道剪枝 → Conv+BN+ReLU 融合(无 bias) → 非结构化剪枝 → W8A8",
        epilog="示例: python -m dram2cnn.pipelines.run_fuse_prune_quantize "
        "configs/experiments/exp_full_f1_nobias_ddp.yaml --sparsity 0.3 --bn-channel-ratio 0.2",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_full_f1_nobias_ddp.yaml",
    )
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--ckpt", default=None, help="默认 <train-dir>/checkpoints/grid_cnn.pt")
    parser.add_argument("--sparsity", type=float, default=0.3, help="非结构化 L1 稀疏度")
    parser.add_argument(
        "--bn-channel-ratio",
        type=float,
        default=0.0,
        help="融合前 BN-γ 通道剪枝比例（0=跳过结构化剪枝）",
    )
    parser.add_argument("--finetune-epochs", type=int, default=5, help="非结构化剪枝后微调轮数")
    parser.add_argument(
        "--bn-finetune-epochs",
        type=int,
        default=None,
        help="BN-γ 剪枝后微调轮数（默认与 --finetune-epochs 相同）",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calib-size", type=int, default=1024)
    parser.add_argument("--adaround-steps", type=int, default=200)
    parser.add_argument(
        "--protect-ends",
        action="store_true",
        help="保护主干首层与分类末层为单精度（默认关闭：整网全 INT8）",
    )
    parser.add_argument(
        "--no-bn-protect-ends",
        action="store_true",
        help="BN-γ 剪枝时不保护 stem/末层卷积（默认保护）",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="输出子目录名（默认 fuse_prune_quant_<timestamp>）",
    )
    args = parser.parse_args(argv)

    root = project_root()
    cfg = load_experiment_config(args.exp_config)
    train_dir = _resolve_train_dir(root, cfg["base"], args.train_dir)
    ckpt = Path(args.ckpt) if args.ckpt else train_dir / "checkpoints" / "grid_cnn.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"缺少权重: {ckpt}")

    thr_path = train_dir / "best_threshold.json"
    if not thr_path.is_file():
        raise FileNotFoundError(f"缺少 best_threshold.json: {thr_path}")
    threshold = float(json.loads(thr_path.read_text(encoding="utf-8"))["threshold"])

    out_name = args.out_name or f"fuse_prune_quant_{new_timestamp_id()}"
    out_dir = train_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pointer(train_dir, ".latest_fuse_prune_quant_session", out_name)

    logger = get_logger(out_dir / "fuse_prune_quantize.log")
    logger.info("exp_config=%s", args.exp_config)
    logger.info("train_dir=%s", train_dir.resolve())
    logger.info("ckpt=%s", ckpt.resolve())
    logger.info("out_dir=%s", out_dir.resolve())
    logger.info(
        "threshold=%.4f sparsity=%.3f bn_channel_ratio=%.3f finetune_epochs=%d "
        "bn_finetune_epochs=%s protect_ends=%s device=%s",
        threshold,
        args.sparsity,
        args.bn_channel_ratio,
        args.finetune_epochs,
        args.bn_finetune_epochs,
        bool(args.protect_ends),
        args.device,
    )

    splits = Path(cfg["data"].get("splits_dir", "data/splits"))
    if not splits.is_absolute():
        splits = root / splits

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    bn_protect = not bool(args.no_bn_protect_ends)
    summary = run_fuse_prune_quantize(
        model_cfg=cfg["model"],
        ckpt=ckpt,
        splits_dir=splits,
        out_dir=out_dir,
        threshold=threshold,
        device=device,
        sparsity=float(args.sparsity),
        bn_channel_ratio=float(args.bn_channel_ratio),
        finetune_epochs=int(args.finetune_epochs),
        bn_finetune_epochs=args.bn_finetune_epochs,
        batch_size=int(args.batch_size),
        calib_size=int(args.calib_size),
        adaround_steps=int(args.adaround_steps),
        protect_stem=bool(args.protect_ends),
        protect_head_out=bool(args.protect_ends),
        bn_protect_stem=bn_protect,
        bn_protect_last_conv=bn_protect,
        log=logger.info,
    )
    qm = ((summary.get("stages") or {}).get("w8a8") or {}).get("int8_packed_metrics") or {}
    print(f"Fuse→Prune→Quant done: {out_dir}")
    print(
        f"  bn_channel_ratio={args.bn_channel_ratio} | "
        f"INT8 F1={qm.get('f1')} P={qm.get('precision')} R={qm.get('recall')} | summary: summary.md"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
