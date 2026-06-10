from __future__ import annotations

"""模型训练命令行入口。

流程概要（rank0）：
1. `validate_experiment`：确认 `data/splits/` 与 `model.yaml` 形状一致。
2. `train_model`：读切分张量训练 GridCNN，权重写入本次会话 `checkpoints/grid_cnn.pt`。
3. 发布 `.latest_train_session` 等指针，供 `infer.sh` 解析最近一次训练目录。

**分布式**：由 `torchrun` 设置 `WORLD_SIZE` / `RANK` / `LOCAL_RANK`；请使用 `bash scripts/train.sh` 启动。
"""

import argparse
import os
import sys

import torch

from dram2cnn.engine.trainer import train_model
from dram2cnn.engine.validator import validate_experiment
from dram2cnn.utils.config import load_pipeline_config, project_root
from dram2cnn.utils.device import resolve_device
from dram2cnn.utils.gpu_launch import merged_scope_section, resolve_ddp_enabled
from dram2cnn.utils.io import dump_json
from dram2cnn.utils.logger import get_logger
from dram2cnn.utils.run_layout import (
    cleanup_train_session_alloc,
    publish_train_session_pointer,
    resolve_train_run_dir,
    train_log_path,
)
from dram2cnn.utils.experiment_guards import assert_full_best_index_not_stub
from dram2cnn.utils.seed import set_seed
from dram2cnn.utils.training_data_audit import log_dataset_health_for_train


def main(exp_config: str = "configs/experiments/exp_baseline.yaml"):
    """加载实验 YAML，校验切分，训练并落盘日志与 checkpoint。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    cfg = load_pipeline_config(exp_config)
    configured_device = str(cfg["base"].get("device", "auto"))
    resolved_device = resolve_device(configured_device)
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    env_bs = os.environ.get("DRAM2CNN_BATCH_SIZE")
    if env_bs:
        cfg.setdefault("train", {})["batch_size"] = int(env_bs)

    root = project_root()
    run_dir = resolve_train_run_dir(root, cfg["base"], world_size=world_size)
    log_path = train_log_path(root, cfg["base"], run_dir)
    logger = get_logger(log_path) if rank == 0 else None

    if rank == 0:
        assert_full_best_index_not_stub(cfg, context="run_train")
        validate_experiment(cfg, require_splits=True)
        dump_json(run_dir / "config_snapshot.json", cfg)
        log_dataset_health_for_train(cfg, logger, root)
        ddp_raw = merged_scope_section(cfg, "ddp", "train").get("enabled", "auto")
        ddp_resolved = resolve_ddp_enabled(ddp_raw, resolved_device, cuda_count)
        if ddp_resolved and world_size == 1 and resolved_device.startswith("cuda") and cuda_count > 1:
            logger.warning(
                "train ddp 已启用（enabled=%s）但当前为单进程 (WORLD_SIZE=1)。"
                "多卡训练请使用 bash scripts/train.sh 或 bash scripts/train_ddp.sh，"
                "勿直接 python -m dram2cnn.pipelines.run_train。",
                ddp_raw,
            )
        if env_bs:
            frac = float(merged_scope_section(cfg, "gpu", "train").get("vram_budget_frac", 0.8))
            logger.info(
                "Using batch_size=%s from VRAM auto-probe (≤%.0f%% GPU memory)",
                env_bs,
                frac * 100,
            )

    set_seed(int(cfg["base"].get("seed", 42)) + rank)
    if logger is not None:
        ddp_raw = merged_scope_section(cfg, "ddp", "train").get("enabled", "auto")
        ddp_resolved = resolve_ddp_enabled(ddp_raw, resolved_device, cuda_count)
        using_all_visible_gpus = bool(
            resolved_device.startswith("cuda")
            and ddp_resolved
            and world_size > 1
            and world_size == cuda_count
        )
        logger.info(
            "Train launch context: world_size=%s, rank=%s, local_rank=%s, configured_device=%s, resolved_device=%s, cuda_available=%s, cuda_count=%s, batch_size=%s, using_all_visible_gpus=%s",
            world_size,
            rank,
            local_rank,
            configured_device,
            resolved_device,
            torch.cuda.is_available(),
            cuda_count,
            cfg.get("train", {}).get("batch_size"),
            using_all_visible_gpus,
        )
    rng_seed = int(cfg["base"].get("seed", 42)) + rank
    ckpt = train_model(
        cfg["model"],
        cfg["train"],
        cfg["data"]["splits_dir"],
        run_dir,
        configured_device,
        logger=logger,
        rng_seed=rng_seed,
    )
    if logger is not None:
        logger.info("Training done. Checkpoint: %s", ckpt)
        publish_train_session_pointer(root, cfg["base"], run_dir)
        cleanup_train_session_alloc(root, cfg["base"])
        logger.info("Run artifacts directory: %s", run_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="训练 GridCNN（需已执行 preprocess 生成 data/splits）。",
        epilog="示例: bash scripts/train.sh configs/experiments/exp_baseline.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_baseline.yaml",
        help="实验 YAML（默认 exp_baseline），内含 data/model/train/base/infer 合并结果",
    )
    args = parser.parse_args()
    try:
        main(args.exp_config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
