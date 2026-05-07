from __future__ import annotations

"""测试集推理命令行入口。

从 `data/splits/` 读取 `X_test.npy`（及可选 `tabular_test.npy`），加载训练产出的
`grid_cnn.pt`，写出 **`predictions.csv`**（含 `y_true`/`y_pred`/`y_prob` 等）。

checkpoint 目录由 **`configs/base.yaml`**（`versioned_runs`、`checkpoint_run_id`、`.latest_train_session`）
解析，通常无需手写路径。
"""

import argparse
from pathlib import Path

from dram2cnn.engine.predictor import predict
from dram2cnn.utils.config import load_experiment_config
from dram2cnn.utils.run_layout import (
    publish_infer_session_pointer,
    resolve_checkpoint_dir,
    resolve_infer_output_dir,
)


def _project_root() -> Path:
    """返回仓库根目录路径。"""
    return Path(__file__).resolve().parents[3]


def main(exp_config: str = "configs/experiments/exp_baseline.yaml"):
    """加载配置，解析 checkpoint 与输出目录，对测试集批量推理并更新 `.latest_infer_session`。"""
    cfg = load_experiment_config(exp_config)
    root = _project_root()
    ckpt_dir = resolve_checkpoint_dir(root, cfg["base"])
    infer_dir = resolve_infer_output_dir(root, cfg["base"])
    ckpt = ckpt_dir / "checkpoints" / "grid_cnn.pt"
    pred_csv = infer_dir / "predictions.csv"
    predict(cfg["model"], cfg["infer"], str(ckpt), cfg["data"]["splits_dir"], str(pred_csv), cfg["base"].get("device", "auto"))
    publish_infer_session_pointer(root, cfg["base"], infer_dir)
    print(f"Inference done: checkpoint_dir={ckpt_dir}")
    print(f"  predictions: {pred_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用训练得到的权重对 X_test 推理，写出 predictions.csv。",
        epilog="示例: python -m dram2cnn.pipelines.run_infer configs/experiments/exp_baseline.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_baseline.yaml",
        help="实验 YAML（需已完成 train 且存在 checkpoint）",
    )
    args = parser.parse_args()
    main(args.exp_config)
