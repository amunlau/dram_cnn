from __future__ import annotations

"""预处理入口：`materialize_dataset` 写出 `processed/`（`X.npy`、`y.npy`、`meta.csv`、`preprocess_manifest.json`；`tabular.npy` 视 `grid_representation` / `inventory_as_grid_channels` 而定）；再按配置做时间或 sid 切分。

约定见 `docs/tianchi_pipeline.md`。 **`spatial_heatmap`** 下 H×W 由分箱与 `model.grid_*` 决定；**`legacy_rc_et`** 下由 `grid_max_*` 与 `model.grid_*` 取 min。"""

import argparse
import sys
from pathlib import Path

from dram2cnn.data.splitter import split_by_sid, split_by_time
from dram2cnn.data.tianchi_build import materialize_dataset
from dram2cnn.engine.validator import validate_experiment
from dram2cnn.utils.config import load_experiment_config
from dram2cnn.utils.experiment_guards import assert_full_best_index_not_stub, assert_processed_matches_experiment
from dram2cnn.utils.preprocess_preflight import check_preprocess_raw_inputs


def main(exp_config: str = "configs/experiments/exp_baseline.yaml"):
    """两阶段：① 只写 processed/；② 从 processed 读入再写 splits/ 并按训练集标准化。

    exp_config 一般为 configs/experiments/*.yaml；load_experiment_config 会合并片段并校验路径。
    """
    check_preprocess_raw_inputs(exp_config)
    cfg = load_experiment_config(exp_config)
    assert_full_best_index_not_stub(cfg, context="run_preprocess")
    try:
        assert_processed_matches_experiment(cfg)
    except ValueError as exc:
        raise ValueError(
            f"{exc}\n"
            f"请先清空 {cfg['data']['processed_dir']} 与 {cfg['data']['splits_dir']} 后重跑 preprocess。"
        ) from exc
    # 阶段一：全量索引样本 → X/y/meta/manifest（不写 train/val/test 拆分）
    manifest = materialize_dataset(
        cfg["data"],
        cfg["model"],
        Path(cfg["data"]["processed_dir"]),
        seed=int(cfg["base"].get("seed", 42)),
    )
    # 阶段二：splitter 读 processed，写出 X_train 等并 normalize_split_features_by_train
    split_strategy = cfg["data"].get("split_strategy", "time")
    if split_strategy == "sid":
        split_by_sid(
            cfg["data"]["processed_dir"],
            cfg["data"]["splits_dir"],
            random_state=int(cfg["base"].get("seed", 42)),
        )
    else:
        # 默认 time：按 predict_ts 排序后按比例切段，减轻时间泄漏
        ts = cfg["data"].get("time_split", {})
        split_by_time(
            cfg["data"]["processed_dir"],
            cfg["data"]["splits_dir"],
            float(ts.get("train_ratio", 0.7)),
            float(ts.get("val_ratio", 0.15)),
            float(ts.get("test_ratio", 0.15)),
        )
    print("Preprocess + split completed.", manifest.get("grid_shape"), flush=True)

    validate_experiment(cfg, require_splits=True)
    print("Validate after preprocess: ok", flush=True)


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
    try:
        main(args.exp_config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
