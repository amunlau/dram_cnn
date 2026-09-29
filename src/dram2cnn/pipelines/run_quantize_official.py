"""官方对称 INT8（torch.ao FakeQuant）PTQ + QAT + QDQ ONNX 入口。

产物目录：``<train_dir>/quant_official_symint8_ptq_qat_<tag>/``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from dram2cnn.engine.quantize_official import run_official_quant
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


def _init_distributed() -> tuple[bool, int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="官方 torch.ao 对称 signed INT8：PTQ + QAT + QDQ ONNX",
        epilog=(
            "示例: torchrun --nproc_per_node=3 -m dram2cnn.pipelines.run_quantize_official "
            "configs/experiments/exp_full_f1_ddp.yaml "
            "--train-dir runs/exp_tianchi_full_f1_ddp/20260917_153434_808743"
        ),
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_full_f1_ddp.yaml",
        help="实验 YAML",
    )
    parser.add_argument("--train-dir", default=None, help="训练会话目录")
    parser.add_argument("--ckpt", default=None, help="权重路径")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calib-batches", type=int, default=50)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument(
        "--tag",
        default=None,
        help="输出目录后缀；默认 auto 时间戳描述",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="单卡设备（非 torchrun 时）",
    )
    args = parser.parse_args(argv)

    distributed, rank, world, local = _init_distributed()
    root = project_root()
    cfg = load_experiment_config(args.exp_config)
    train_dir = _resolve_train_dir(root, cfg["base"], args.train_dir)
    ckpt = Path(args.ckpt) if args.ckpt else train_dir / "checkpoints" / "grid_cnn.pt"
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")

    thr_path = train_dir / "best_threshold.json"
    if not thr_path.is_file():
        raise FileNotFoundError(f"缺少 best_threshold.json: {thr_path}")
    threshold = float(json.loads(thr_path.read_text(encoding="utf-8"))["threshold"])

    tag = args.tag or f"e{args.qat_epochs}_lr{args.qat_lr:g}_{new_timestamp_id()}"
    out_name = f"quant_official_symint8_ptq_qat_{tag}"
    out_dir = train_dir / out_name

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_pointer(train_dir, ".latest_quant_official_session", out_name)
        logger = get_logger(out_dir / "quantize_official.log")
        logger.info("exp_config=%s", args.exp_config)
        logger.info("train_dir=%s", train_dir.resolve())
        logger.info("ckpt=%s", ckpt.resolve())
        logger.info("out_dir=%s", out_dir.resolve())
        logger.info("threshold=%.4f world_size=%d", threshold, world)
    else:
        logger = get_logger(out_dir / f"quantize_official_rank{rank}.log")

    splits = Path(cfg["data"].get("splits_dir", "data/splits"))
    if not splits.is_absolute():
        splits = root / splits

    device = torch.device(args.device if not distributed else f"cuda:{local}")
    if device.type == "cuda" and not distributed:
        torch.cuda.set_device(device)

    def _log(msg: str, *a: object) -> None:
        logger.info(msg, *a)

    run_official_quant(
        model_cfg=cfg["model"],
        train_cfg=cfg.get("train") or {},
        ckpt=ckpt,
        splits_dir=splits,
        out_dir=out_dir,
        deploy_threshold=threshold,
        device=device,
        batch_size=int(args.batch_size),
        calib_batches=int(args.calib_batches),
        qat_epochs=int(args.qat_epochs),
        qat_lr=float(args.qat_lr),
        distributed=distributed,
        rank=rank,
        world_size=world,
        local_rank=local,
        log=_log,
    )

    if distributed:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
