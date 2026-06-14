from __future__ import annotations

"""测试集推理命令行入口。

从 `data/splits/` 读取 `X_test.npy`（及可选 `tabular_test.npy`），加载训练产出的
`grid_cnn.pt`，写出 **`predictions.csv`**（含 `y_true`/`y_pred`/`y_prob` 等）。

checkpoint 目录由 **`configs/base.yaml`**（`versioned_runs`、`checkpoint_run_id`、`.latest_train_session`）
解析，通常无需手写路径。

**分布式**：由 `torchrun` 设置 `WORLD_SIZE` / `RANK` / `LOCAL_RANK`；请使用 `bash scripts/infer.sh` 启动。
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from dram2cnn.engine.predictor import predict
from dram2cnn.engine.validator import validate_experiment
from dram2cnn.utils.config import load_pipeline_config, project_root
from dram2cnn.utils.device import resolve_device
from dram2cnn.utils.experiment_guards import (
    assert_checkpoint_matches_experiment,
    warn_unstable_deep_focal_combo,
)
from dram2cnn.utils.gpu_launch import merged_scope_section, resolve_ddp_enabled
from dram2cnn.utils.run_layout import (
    publish_infer_session_pointer,
    resolve_checkpoint_dir,
    resolve_infer_output_dir,
)


def main(exp_config: str = "configs/experiments/exp_baseline.yaml"):
    """加载配置，解析 checkpoint 与输出目录，对测试集批量推理并更新 `.latest_infer_session`。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    cfg = load_pipeline_config(exp_config)
    configured_device = str(cfg["base"].get("device", "auto"))
    resolved_device = resolve_device(configured_device)
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    env_bs = os.environ.get("DRAM2CNN_BATCH_SIZE")
    if env_bs:
        cfg.setdefault("infer", {})["batch_size"] = int(env_bs)

    if rank == 0:
        validate_experiment(cfg, require_splits=True)
        warn_unstable_deep_focal_combo(cfg)

    root = project_root()
    ckpt_dir = resolve_checkpoint_dir(root, cfg["base"])
    if rank == 0:
        assert_checkpoint_matches_experiment(cfg, ckpt_dir, root)
    infer_dir = resolve_infer_output_dir(root, cfg["base"])
    ckpt = ckpt_dir / "checkpoints" / "grid_cnn.pt"
    pred_csv = infer_dir / "predictions.csv"

    if rank == 0:
        ddp_raw = merged_scope_section(cfg, "ddp", "infer").get("enabled", "auto")
        ddp_resolved = resolve_ddp_enabled(ddp_raw, resolved_device, cuda_count)
        if ddp_resolved and world_size == 1 and resolved_device.startswith("cuda") and cuda_count > 1:
            print(
                f"WARNING: infer ddp 已启用（enabled={ddp_raw}）但当前为单进程 (WORLD_SIZE=1)。"
                "多卡推理请使用 bash scripts/infer.sh（内部 torchrun）。",
                file=sys.stderr,
            )
        if env_bs:
            frac = float(merged_scope_section(cfg, "gpu", "infer").get("vram_budget_frac", 0.8))
            print(
                f"Using infer batch_size={env_bs} from VRAM auto-probe (≤{frac * 100:.0f}% GPU memory)",
                flush=True,
            )

    predict(
        cfg["model"],
        cfg["infer"],
        str(ckpt),
        cfg["data"]["splits_dir"],
        str(pred_csv),
        cfg["base"].get("device", "auto"),
    )

    if rank == 0:
        publish_infer_session_pointer(root, cfg["base"], infer_dir)
        print(f"Inference done: checkpoint_dir={ckpt_dir}")
        print(f"  predictions: {pred_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用训练得到的权重对 X_test 推理，写出 predictions.csv。",
        epilog="示例: bash scripts/infer.sh configs/experiments/exp_baseline.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_baseline.yaml",
        help="实验 YAML（需已完成 train 且存在 checkpoint）",
    )
    args = parser.parse_args()
    try:
        main(args.exp_config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
