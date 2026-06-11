#!/usr/bin/env python3
"""从 `data/raw` 生成真实索引表（sid, predict_time）。

``predict_time`` 以 1970-01-01 + 相对秒 的 ISO 字符串存储；``predict_ts`` 在 preprocess 中为 ≥0 的相对秒。

示例：
  # 冒烟（默认 5 分钟预测窗 + 5 分钟步长，写入 features_5m_index_demo.csv）
  bash scripts/build_index.sh smoke
  # 或：PYTHONPATH=src python3 scripts/build_index.py --mode smoke

  # 全量（扫描完整 mcelog，耗时较长）
  PYTHONPATH=src python3 scripts/build_index.py --mode full \\
    --out data/raw/features_index_h300s_step300s_full.csv

  # 设计空间：30 分钟预测窗、10 分钟采样（须同时指定 window 与 horizon 一致）
  PYTHONPATH=src python3 scripts/build_index.py --mode full \\
    --label-horizon-seconds 1800 --sample-interval-seconds 600 --window-seconds 1800 \\
    --out data/raw/features_index_h1800s_step600s_full.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dram2cnn.data.index_builder import IndexBuildConfig, build_index_to_csv
from dram2cnn.utils.config import label_horizon_seconds_from_dict, load_yaml, project_root


def _default_out_path(root: Path, horizon: float, interval: float, mode: str) -> Path:
    """按 horizon、步长与模式生成默认输出路径。

    默认 5 分钟冒烟（300s/300s）写入 ``features_5m_index_demo.csv``，与 ``data.yaml`` 的
    ``index_features_path`` 一致；其余组合使用 ``features_index_h{H}s_step{S}s_{mode}.csv``。
    """
    h = int(horizon)
    s = int(interval)
    if mode == "smoke" and h == 300 and s == 300:
        return root / "data/index_stubs" / "features_5m_index_demo.csv"
    name = f"features_index_h{h}s_step{s}s_{mode}.csv"
    if mode == "smoke":
        return root / "data/index_stubs" / name
    return root / "data/raw" / name


def main() -> None:
    """解析 CLI，合并 data.yaml / preset，调用 `build_index_to_csv` 写出索引表。"""
    ap = argparse.ArgumentParser(description="从 mcelog + 工单生成索引表")
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--out", type=Path, default=None, help="输出 CSV 路径")
    ap.add_argument("--data-yaml", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--sample-interval-seconds", type=float, default=None)
    ap.add_argument("--label-horizon-seconds", type=float, default=None)
    ap.add_argument("--window-seconds", type=float, default=None)
    ap.add_argument("--positive-prefault-seconds", type=float, default=None)
    ap.add_argument("--positive-sample-interval-seconds", type=float, default=None)
    ap.add_argument("--target-positive-ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    root = project_root()
    data_cfg = load_yaml(root / args.data_yaml if not args.data_yaml.is_absolute() else args.data_yaml)
    ib_raw = dict(data_cfg.get("index_build") or {})
    ib_raw.setdefault("mcelog_path", data_cfg.get("mcelog_path", "data/raw/mcelog.csv"))
    ib_raw.setdefault("tickets_path", data_cfg.get("tickets_path", "data/raw/trouble_tickets.csv"))
    ib_raw.setdefault("failure_types", data_cfg.get("label_failure_types", [1, 2, 3]))
    ib_raw.setdefault("mcelog_chunk_rows", int(data_cfg.get("mcelog_chunk_rows", 1_000_000)))
    ib_raw.setdefault("grid_representation", data_cfg.get("grid_representation", "spatial_heatmap"))
    if data_cfg.get("mcelog_error_type_filter") is not None:
        ib_raw.setdefault("mcelog_error_type_filter", data_cfg.get("mcelog_error_type_filter"))

    # horizon / window 优先 CLI → 顶层 data.yaml（index_build 段不再重复 horizon，避免与顶层不一致）
    horizon = (
        float(args.label_horizon_seconds)
        if args.label_horizon_seconds is not None
        else label_horizon_seconds_from_dict(data_cfg)
    )
    if args.sample_interval_seconds is not None:
        interval = float(args.sample_interval_seconds)
    elif args.label_horizon_seconds is not None:
        # CLI 指定 horizon 时默认步长与 horizon 一致（与 window 逻辑对齐）
        interval = horizon
    else:
        interval = float(ib_raw.get("sample_interval_seconds", horizon))
    # 索引锚点有效区间依赖特征窗 W
    if args.window_seconds is not None:
        window = float(args.window_seconds)
    elif args.label_horizon_seconds is not None:
        window = horizon
    elif data_cfg.get("window_seconds") is not None:
        window = float(data_cfg["window_seconds"])
    elif ib_raw.get("window_seconds") is not None:
        window = float(ib_raw["window_seconds"])
    else:
        window = horizon

    ib_raw["label_horizon_seconds"] = horizon
    ib_raw["sample_interval_seconds"] = interval
    ib_raw["window_seconds"] = window
    ib_raw["seed"] = args.seed

    if args.positive_prefault_seconds is not None:
        ib_raw["positive_prefault_seconds"] = args.positive_prefault_seconds
    if args.positive_sample_interval_seconds is not None:
        ib_raw["positive_sample_interval_seconds"] = args.positive_sample_interval_seconds
    if args.target_positive_ratio is not None:
        ib_raw["target_positive_ratio"] = args.target_positive_ratio

    presets = dict(data_cfg.get("index_build_presets") or {})
    preset = presets.get(args.mode, {})
    ib_raw.update({k: v for k, v in preset.items() if v is not None})

    if args.mode == "full":
        ib_raw.setdefault("mcelog_scan_rows", None)
        ib_raw.setdefault("smoke_max_sids", None)
        ib_raw.setdefault("max_rows", args.max_rows)

    if args.max_rows is not None:
        ib_raw["max_rows"] = args.max_rows

    cfg = IndexBuildConfig.from_dict(ib_raw, root=root)
    out = args.out
    if out is None:
        out = _default_out_path(root, horizon, interval, args.mode)
    elif not out.is_absolute():
        out = root / out

    print(
        f"Building index mode={args.mode} horizon={horizon}s interval={interval}s window={window}s "
        f"pos_interval={cfg.positive_sample_interval_seconds}s → {out}",
        flush=True,
    )
    summary = build_index_to_csv(cfg, out)
    print(
        f"Done: rows={summary['n_rows']} pos={summary['n_pos']} neg={summary['n_neg']} "
        f"pos_rate={summary['pos_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
