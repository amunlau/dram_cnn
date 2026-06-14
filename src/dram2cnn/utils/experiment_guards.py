from __future__ import annotations

"""实验语义守卫：索引 stub、checkpoint 与配置一致性、evaluate/infer 产物对齐等。"""

import json
import warnings
from pathlib import Path
from typing import Any

import torch

from dram2cnn.utils.config import project_root as repo_root

# 影响网络结构与输入形状的 model 键；infer 前与 config_snapshot 比对。
_CHECKPOINT_MODEL_KEYS = (
    "arch",
    "in_channels",
    "grid_height",
    "grid_width",
    "num_classes",
    "tabular_dim",
    "stem_channels",
    "backbone_channels",
    "blocks_per_stage",
    "use_residual",
    "head_hidden",
    "dropout",
)


_STUB_MARKERS = ("index_stubs", "features_5m_index_demo")


def is_stub_index_path(path: str | Path) -> bool:
    s = str(path).replace("\\", "/").lower()
    return any(m in s for m in _STUB_MARKERS)


def is_deep_grid_cnn(model_cfg: dict[str, Any]) -> bool:
    """Heuristic：与 configs/model.yaml（GridCNN-Deep）对齐的深度 backbone。"""
    return (
        int(model_cfg.get("blocks_per_stage", 1)) >= 2
        and int(model_cfg.get("stem_channels", 32)) >= 64
        and len(model_cfg.get("backbone_channels") or []) >= 4
    )


def warn_unstable_deep_focal_combo(cfg: dict[str, Any], logger=None) -> None:
    """GridCNN-Deep + focal_gamma>0 在默认 lr 下已知易第 2 步梯度 NaN；rank0 打 WARNING。"""
    from dram2cnn.engine.imbalance import focal_gamma_from_cfg

    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("train") or {}
    imb = train_cfg.get("imbalance") or {}
    gamma = focal_gamma_from_cfg(imb)
    if not is_deep_grid_cnn(model_cfg) or gamma <= 0.0:
        return
    msg = (
        "GridCNN-Deep（model.yaml）与 focal_gamma=%s 在默认 lr 下已知易训练发散（loss/grad NaN）。"
        " 推荐实验（exp_baseline / exp_full_best_ddp 等）已默认 focal_gamma=0；"
        "若仍启用 Focal，请同时降低 train.lr 并监控首轮 loss。"
    ) % gamma
    if logger is not None:
        logger.warning(msg)
    else:
        import warnings

        warnings.warn(msg, stacklevel=2)


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


def state_dict_has_non_finite(state_dict: dict) -> bool:
    """checkpoint ``state_dict`` 是否含 NaN/Inf 浮点权重。"""
    for v in state_dict.values():
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
            return True
    return False


def assert_checkpoint_matches_experiment(
    cfg: dict[str, Any],
    ckpt_dir: Path,
    root: Path | None = None,
) -> None:
    """infer 前比对训练目录 ``config_snapshot.json`` 与当前实验 ``model`` 段是否一致。

    缺 snapshot 时仅 WARNING（兼容旧 run）；存在 snapshot 且 model 键不一致则 fail-fast。
    """
    base = root or repo_root()
    ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else (base / ckpt_dir).resolve()
    snap_p = ckpt_dir / "config_snapshot.json"
    if not snap_p.is_file():
        warnings.warn(
            f"checkpoint 目录缺少 config_snapshot.json（{ckpt_dir}）；"
            "无法校验 model 配置是否与当前实验 YAML 一致。建议对当前实验重训。",
            stacklevel=2,
        )
        return

    snap = json.loads(snap_p.read_text(encoding="utf-8"))
    saved_model = snap.get("model") or {}
    current_model = cfg.get("model") or {}
    mismatches: list[str] = []
    for key in _CHECKPOINT_MODEL_KEYS:
        saved_val = saved_model.get(key)
        current_val = current_model.get(key)
        if saved_val != current_val:
            mismatches.append(f"  {key}: snapshot={saved_val!r} != current={current_val!r}")
    if mismatches:
        run_saved = (snap.get("base") or {}).get("run_name")
        run_current = (cfg.get("base") or {}).get("run_name")
        raise RuntimeError(
            "checkpoint 与当前实验 model 配置不一致，拒绝 infer。\n"
            + "\n".join(mismatches)
            + f"\n  run_name: snapshot={run_saved!r} current={run_current!r}\n"
            "请使用生成该 checkpoint 的实验 YAML，或设置 base.checkpoint_run_id 指向匹配的训练会话后重训。"
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
