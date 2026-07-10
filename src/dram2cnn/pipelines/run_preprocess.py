from __future__ import annotations

"""预处理入口：`materialize_dataset` 写出实验 `processed_dir`；再按配置切分至 `splits_dir`。

``--split-only``：跳过特征生成，在已有 ``processed_dir`` 上仅重跑切分（用于 sid vs time_stratified 对比，避免重复扫 mcelog）。
说明见 `docs/tianchi_pipeline.md`。"""

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from dram2cnn.data.splitter import split_by_sid, split_by_time, split_by_time_stratified
from dram2cnn.data.tianchi_build import materialize_dataset
from dram2cnn.engine.validator import validate_experiment
from dram2cnn.utils.config import load_experiment_config, project_root
from dram2cnn.utils.experiment_guards import assert_full_scale_index_not_stub, assert_processed_matches_experiment
from dram2cnn.utils.io import dump_json
from dram2cnn.utils.preprocess_preflight import check_preprocess_raw_inputs


def _resolve_data_dir(raw: str | Path, root: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def _clear_dir_contents(path: Path) -> None:
    """删除目录内文件/子目录，保留目录本身（与 clean.sh 对 processed/splits 行为一致）。"""
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _run_split_phase(cfg: dict) -> str:
    split_strategy = str(cfg["data"].get("split_strategy", "time_stratified"))
    norm_method = cfg["data"].get("normalization_method", "robust")
    norm_clip = cfg["data"].get("normalization_clip", 20.0)
    if norm_clip in (None, "null", False):
        norm_clip_val = None
    else:
        norm_clip_val = float(norm_clip)
    split_kw = dict(normalization_method=norm_method, normalization_clip=norm_clip_val)
    if split_strategy == "sid":
        ts = cfg["data"].get("time_split", {})
        split_by_sid(
            cfg["data"]["processed_dir"],
            cfg["data"]["splits_dir"],
            float(ts.get("train_ratio", 0.7)),
            float(ts.get("val_ratio", 0.15)),
            float(ts.get("test_ratio", 0.15)),
            random_state=int(cfg["base"].get("seed", 42)),
            **split_kw,
        )
    elif split_strategy == "time":
        ts = cfg["data"].get("time_split", {})
        split_by_time(
            cfg["data"]["processed_dir"],
            cfg["data"]["splits_dir"],
            float(ts.get("train_ratio", 0.7)),
            float(ts.get("val_ratio", 0.15)),
            float(ts.get("test_ratio", 0.15)),
            **split_kw,
        )
    else:
        ts = cfg["data"].get("time_split", {})
        split_by_time_stratified(
            cfg["data"]["processed_dir"],
            cfg["data"]["splits_dir"],
            float(ts.get("train_ratio", 0.7)),
            float(ts.get("val_ratio", 0.15)),
            float(ts.get("test_ratio", 0.15)),
            min_pos_per_split=int(ts.get("min_pos_per_split", 1)),
            **split_kw,
        )
    return split_strategy


def _write_split_manifest(cfg: dict, split_strategy: str) -> None:
    spl = Path(cfg["data"]["splits_dir"])
    info: dict = {
        "split_strategy": split_strategy,
        "splits_dir": str(spl.resolve()),
        "processed_dir": str(Path(cfg["data"]["processed_dir"]).resolve()),
        "seed": int(cfg["base"].get("seed", 42)),
    }
    for tag in ("train", "val", "test"):
        mp = spl / f"meta_{tag}.csv"
        if not mp.is_file():
            continue
        m = pd.read_csv(mp)
        info[f"n_{tag}"] = int(len(m))
        if "label" in m.columns:
            info[f"n_pos_{tag}"] = int(m["label"].sum())
        if "sid" in m.columns:
            info[f"n_sid_{tag}"] = int(m["sid"].nunique())
        if "empty_mcelog_window" in m.columns:
            info[f"n_empty_mcelog_{tag}"] = int(m["empty_mcelog_window"].sum())
    dump_json(spl / "split_manifest.json", info)


def main(exp_config: str = "configs/experiments/exp_baseline.yaml", *, split_only: bool = False):
    """两阶段：① 生成 processed/；② 切分 splits/ 并按训练集标准化。``split_only=True`` 时跳过①。"""
    if not split_only:
        check_preprocess_raw_inputs(exp_config)
    cfg = load_experiment_config(exp_config)
    assert_full_scale_index_not_stub(cfg, context="run_preprocess")
    try:
        assert_processed_matches_experiment(cfg)
    except ValueError as exc:
        if split_only:
            raise ValueError(
                f"{exc}\n"
                "split-only 需要与当前实验配置一致的 processed_dir；"
                "请先对同一 index/horizon 跑完整 preprocess，或去掉 --split-only。"
            ) from exc
        # 完整 preprocess：旧输出与当前实验不一致时自动清空后重新生成
        # （典型场景：04 F1 drop_empty_positive=false → 05 工程实验 true）
        root = project_root()
        proc_dir = _resolve_data_dir(cfg["data"]["processed_dir"], root)
        split_dir = _resolve_data_dir(cfg["data"]["splits_dir"], root)
        print(
            f"提示：已有 processed 与当前实验不一致（{exc}）。\n"
            f"将自动清空 {proc_dir} 与 {split_dir} 后重新 preprocess。",
            flush=True,
        )
        _clear_dir_contents(proc_dir)
        _clear_dir_contents(split_dir)
        # sid 等旁路切分目录若仍指向同一 processed，也清掉，避免后续误用旧 N
        for extra in ("data/splits_sid", "data/splits_sid_balanced"):
            extra_p = _resolve_data_dir(extra, root)
            if extra_p.is_dir() and extra_p.resolve() != split_dir.resolve():
                _clear_dir_contents(extra_p)

    manifest: dict
    if split_only:
        man_p = Path(cfg["data"]["processed_dir"]) / "preprocess_manifest.json"
        if not man_p.is_file():
            raise FileNotFoundError(f"split-only 需要 {man_p}；请先运行完整 preprocess。")
        manifest = json.loads(man_p.read_text(encoding="utf-8"))
        print(
            f"Split-only: 复用 processed_dir（n_samples={manifest.get('n_samples')}），"
            f"策略={cfg['data'].get('split_strategy')}",
            flush=True,
        )
    else:
        manifest = materialize_dataset(
            cfg["data"],
            cfg["model"],
            Path(cfg["data"]["processed_dir"]),
            seed=int(cfg["base"].get("seed", 42)),
        )
        gc.collect()

    split_strategy = _run_split_phase(cfg)
    _write_split_manifest(cfg, split_strategy)
    print(
        f"Preprocess + split completed: N={manifest.get('n_samples')}, "
        f"pos_rate={float(manifest.get('pos_rate', 0.0)):.6f}, "
        f"grid_shape={manifest.get('grid_shape')}",
        flush=True,
    )

    validate_experiment(cfg, require_splits=True)
    print("Validate after preprocess: ok", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 raw 表构建 X/y/meta 与 manifest，并切分、标准化到实验 YAML 指定的 splits_dir。",
        epilog="示例: python -m dram2cnn.pipelines.run_preprocess configs/experiments/exp_baseline.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_baseline.yaml",
        help="实验 YAML",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="跳过 materialize，仅在已有 processed/ 上重跑切分（写入 cfg 指定的 splits_dir）",
    )
    args = parser.parse_args()
    try:
        main(args.exp_config, split_only=bool(args.split_only))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
