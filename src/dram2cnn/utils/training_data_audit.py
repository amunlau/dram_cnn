"""训练前数据规模与「全量实验」语义提示（rank0 日志）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dram2cnn.utils.experiment_guards import is_stub_index_path


def log_dataset_health_for_train(cfg: dict[str, Any], logger: Any, project_root: Path) -> None:
    """在 validate 通过后打印：索引路径、manifest 样本数、train/val 体量与正例数；stub 索引与小 val 集告警。"""
    if logger is None:
        return
    data = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    patience = int(train_cfg.get("patience", 50))

    idx_path = Path(str(data.get("index_features_path", "")))
    proc_dir = Path(str(data.get("processed_dir", "data/processed")))
    spl_dir = Path(str(data.get("splits_dir", "data/splits")))
    if not idx_path.is_absolute():
        idx_path = project_root / idx_path
    if not proc_dir.is_absolute():
        proc_dir = project_root / proc_dir
    if not spl_dir.is_absolute():
        spl_dir = project_root / spl_dir

    n_manifest: int | None = None
    pos_rate_m: float | None = None
    man_p = proc_dir / "preprocess_manifest.json"
    if man_p.is_file():
        try:
            man = json.loads(man_p.read_text(encoding="utf-8"))
            n_manifest = int(man.get("n_samples", 0)) or None
            pos_rate_m = float(man.get("pos_rate", 0.0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if logger is not None:
                logger.debug("无法解析 preprocess_manifest.json: %s", exc)

    yt = spl_dir / "y_train.npy"
    yv = spl_dir / "y_val.npy"
    if not yt.is_file() or not yv.is_file():
        logger.warning("未找到 y_train/y_val，跳过数据集体量摘要。")
        return
    y_train = np.load(yt, mmap_mode="r")
    y_val = np.load(yv, mmap_mode="r")
    n_tr, n_va = int(len(y_train)), int(len(y_val))
    pos_tr = int(np.sum(y_train == 1))
    pos_va = int(np.sum(y_val == 1))

    logger.info(
        "数据摘要: index_features_path=%s | preprocess_manifest n_samples=%s pos_rate=%s | "
        "train N=%d pos=%d | val N=%d pos=%d",
        idx_path,
        n_manifest if n_manifest is not None else "?",
        f"{pos_rate_m:.4f}" if pos_rate_m is not None else "?",
        n_tr,
        pos_tr,
        n_va,
        pos_va,
    )

    if is_stub_index_path(idx_path):
        logger.warning(
            "当前索引仍指向仓库内 **演示/stub CSV**（约 500 行）。"
            " `exp_full_best_ddp` 中的「全量」仅表示 **取消 max_index_rows / mcelog_max_rows 截断**，"
            " **不会**自动替换为天池全量索引表。全量请先 `bash scripts/build_index.sh full`，再在 "
            "`configs/data.yaml` 或实验 `overrides.data.index_features_path` 中改为 "
            "`data/raw/features_index_*.csv` 后 **重跑 preprocess**。"
        )

    if n_va < 32:
        logger.warning(
            "验证集极小 (N_val=%d)：验证集 ROC/PR/F1 方差大，`train.early_stop_metric`（如 avg_precision）"
            " 易在首轮即达平台，随后连续 patience=%d 轮无提升会触发早停；这不等于数据泄漏。",
            n_va,
            patience,
        )

    if n_tr < 200:
        logger.warning(
            "训练集原始样本数较少 (N_train=%d)；配合 positive_oversample_factor 后 train_virtual_len 会变大，"
            " 但泛化仍受限于索引行数与特征信息量。",
            n_tr,
        )

    if man_p.is_file():
        try:
            man2 = json.loads(man_p.read_text(encoding="utf-8"))
            n_s = int(man2.get("n_samples") or 0)
            n_em = int(man2.get("n_empty_grids") or 0)
            n_drop = int(man2.get("n_empty_grids_dropped") or 0)
            denom = max(n_s + n_drop, 1)
            if n_em / denom >= 0.5:
                logger.warning(
                    "preprocess_manifest: n_empty_grids=%d，已丢弃负样本空窗 %d（保留正样本空窗 %d）；"
                    "冒烟 mcelog 截断下常见；全量请取消 mcelog_max_rows。",
                    n_em,
                    n_drop,
                    int(man2.get("n_empty_grids_kept_positive") or 0),
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
