from __future__ import annotations

"""预处理入口：`materialize_dataset` 写出 `processed/`（`X.npy`、`y.npy`、`meta.csv`、`preprocess_manifest.json`；`tabular.npy` 视 `grid_representation` / `inventory_as_grid_channels` 而定）；再按配置做时间或 sid 切分。

约定见 `docs/tianchi_pipeline.md`。 **`spatial_heatmap`** 下 H×W 由分箱与 `model.grid_*` 决定；**`legacy_rc_et`** 下由 `grid_max_*` 与 `model.grid_*` 取 min。"""

import argparse
from pathlib import Path

from dram2cnn.data.splitter import split_by_sid, split_by_time
from dram2cnn.data.tianchi_build import materialize_dataset
from dram2cnn.utils.config import load_experiment_config


def main(exp_config: str = "configs/experiments/exp_baseline.yaml"):
    """读入实验配置，物化 `processed/`，再按 `split_strategy` 写入 `splits/` 并做标准化。"""
    cfg = load_experiment_config(exp_config)
    manifest = materialize_dataset(
        cfg["data"],
        cfg["model"],
        Path(cfg["data"]["processed_dir"]),
    )
    split_strategy = cfg["data"].get("split_strategy", "time")
    if split_strategy == "sid":
        split_by_sid(
            cfg["data"]["processed_dir"],
            cfg["data"]["splits_dir"],
            random_state=int(cfg["base"].get("seed", 42)),
        )
    else:
        ts = cfg["data"].get("time_split", {})
        split_by_time(
            cfg["data"]["processed_dir"],
            cfg["data"]["splits_dir"],
            float(ts.get("train_ratio", 0.7)),
            float(ts.get("val_ratio", 0.15)),
            float(ts.get("test_ratio", 0.15)),
        )
    print("Preprocess + split completed.", manifest.get("grid_shape"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 raw 表构建 X/y/meta 与 manifest，并切分、标准化为 data/splits。",
        epilog="示例: python -m dram2cnn.pipelines.run_preprocess configs/experiments/exp_baseline.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_baseline.yaml",
        help="实验 YAML（需能访问 data 中 mcelog/索引等路径）",
    )
    args = parser.parse_args()
    main(args.exp_config)
