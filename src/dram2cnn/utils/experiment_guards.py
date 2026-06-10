from __future__ import annotations

"""实验语义守卫：全量训练误用 stub 索引、evaluate 与 infer 产物不一致等。"""

import json
from pathlib import Path
from typing import Any

from dram2cnn.utils.config import project_root as repo_root


_STUB_MARKERS = ("index_stubs", "features_5m_index_demo")


def is_stub_index_path(path: str | Path) -> bool:
    s = str(path).replace("\\", "/").lower()
    return any(m in s for m in _STUB_MARKERS)


def assert_full_best_index_not_stub(cfg: dict[str, Any], *, context: str = "train") -> None:
    """`exp_full_best_ddp` 等全量实验不得仍指向 ~500 行演示索引。"""
    run_name = str(cfg.get("base", {}).get("run_name", ""))
    idx = str(cfg.get("data", {}).get("index_features_path", ""))
    if "full_best" not in run_name and not run_name.endswith("full_best_ddp"):
        return
    if not is_stub_index_path(idx):
        return
    raise RuntimeError(
        f"[{context}] 全量实验仍使用演示索引: {idx}\n"
        "请先 bash scripts/build_index.sh full，并在实验 YAML 设置\n"
        "  index_features_path: data/raw/features_index_h300s_step300s_full.csv\n"
        "然后清空 data/processed、data/splits 并重跑 preprocess。"
    )


def assert_predictions_match_experiment(
    cfg: dict[str, Any],
    artifact_dir: Path,
    root: Path,
) -> None:
    """evaluate 前比对 infer 写入的 manifest 与当前实验 splits 配置。"""
    man_p = artifact_dir / "predictions_manifest.json"
    pred_p = artifact_dir / "predictions.csv"
    if not man_p.is_file():
        if pred_p.is_file():
            raise ValueError(
                f"{pred_p} 存在但缺少 predictions_manifest.json；"
                "请对当前实验重跑 infer，避免误用旧 predictions 做 evaluate。"
            )
        return
    man = json.loads(man_p.read_text(encoding="utf-8"))
    man_splits = man.get("splits_dir")
    if not man_splits:
        return
    cfg_splits = Path(str(cfg["data"]["splits_dir"]))
    if not cfg_splits.is_absolute():
        cfg_splits = (root / cfg_splits).resolve()
    if Path(str(man_splits)).resolve() != cfg_splits.resolve():
        raise ValueError(
            f"predictions 来自 splits_dir={man_splits}，与当前实验 {cfg_splits} 不一致；"
            "请对当前实验重跑 infer，或切换为生成该 predictions 的实验 YAML。"
        )


def assert_processed_matches_experiment(
    cfg: dict[str, Any],
    root: Path | None = None,
) -> None:
    """磁盘 preprocess manifest 须与当前实验 data 段一致（horizon/索引/网格）。"""
    from dram2cnn.engine.validator import check_preprocess_manifest

    base = root or repo_root()
    processed_raw = Path(cfg["data"]["processed_dir"])
    processed_dir = processed_raw if processed_raw.is_absolute() else base / processed_raw
    man_p = processed_dir / "preprocess_manifest.json"
    if not man_p.is_file():
        return

    model = cfg.get("model", {})
    expected = (
        int(model.get("in_channels", 31)),
        int(model.get("grid_height", 128)),
        int(model.get("grid_width", 32)),
    )
    check_preprocess_manifest(
        processed_dir,
        model,
        expected,
        data_cfg=cfg.get("data"),
        project_root_path=base,
    )


def splits_match_processed(cfg: dict[str, Any], root: Path | None = None) -> bool:
    """splits 文件齐全、manifest 与 cfg 对齐，且切分行数等于 processed 样本数。"""
    import numpy as np

    base = root or repo_root()
    splits_raw = Path(cfg["data"]["splits_dir"])
    splits_dir = splits_raw if splits_raw.is_absolute() else base / splits_raw
    for tag in ("train", "val", "test"):
        if not (splits_dir / f"X_{tag}.npy").is_file() or not (splits_dir / f"y_{tag}.npy").is_file():
            return False
    if not (splits_dir / "normalization_stats.json").is_file():
        return False
    try:
        assert_processed_matches_experiment(cfg, base)
    except ValueError:
        return False

    processed_raw = Path(cfg["data"]["processed_dir"])
    processed_dir = processed_raw if processed_raw.is_absolute() else base / processed_raw
    man_p = processed_dir / "preprocess_manifest.json"
    if not man_p.is_file():
        return False
    man = json.loads(man_p.read_text(encoding="utf-8"))
    n_proc = int(man.get("n_samples") or 0)
    if n_proc <= 0:
        return False
    split_total = sum(int(len(np.load(splits_dir / f"y_{tag}.npy", mmap_mode="r"))) for tag in ("train", "val", "test"))
    return split_total == n_proc
