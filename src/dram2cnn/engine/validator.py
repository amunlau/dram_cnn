from __future__ import annotations

"""训练/推理前预检：配置、`processed/`、`data/splits/` 张量形状与 manifest 一致性。

`require_splits=False` 时跳过切分文件检查，便于「尚未 preprocess」环境下只做路径校验。
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dram2cnn.utils.config import label_horizon_seconds_from_dict, project_root
from dram2cnn.utils.device import resolve_device


def _check_run_dir_writable(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = run_dir / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _check_xy_shape_pair(
    x_path: Path, y_path: Path, expected: dict[str, int] | None = None
) -> tuple[int, int, int, int]:
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"缺少数据文件（不存在其一或两者）:\n"
            f"  X: {x_path}\n"
            f"  y: {y_path}\n"
            "请先运行 `bash scripts/preprocess.sh <实验yaml>` 生成 processed/ 与 splits/。\n"
            "若仅需校验配置与原始路径是否存在，请执行:\n"
            "  python -m dram2cnn.pipelines.run_validate <实验yaml> --no-require-splits"
        )
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    if x.ndim != 4:
        raise ValueError(f"{x_path} expected 4D [N,C,H,W], got shape={x.shape}")
    if y.ndim != 1:
        raise ValueError(f"{y_path} expected 1D [N], got shape={y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"N mismatch: X_N={x.shape[0]}, y_N={y.shape[0]}")
    if expected:
        if int(expected["in_channels"]) != int(x.shape[1]):
            raise ValueError(f"{x_path.name} C mismatch: cfg={expected['in_channels']}, data={x.shape[1]}")
        if int(expected["grid_height"]) != int(x.shape[2]):
            raise ValueError(f"{x_path.name} H mismatch: cfg={expected['grid_height']}, data={x.shape[2]}")
        if int(expected["grid_width"]) != int(x.shape[3]):
            raise ValueError(f"{x_path.name} W mismatch: cfg={expected['grid_width']}, data={x.shape[3]}")
    return int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])


def _norm_cfg_path(root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def check_preprocess_manifest(
    processed_dir: Path,
    model: dict[str, Any],
    expected_chw: tuple[int, int, int],
    data_cfg: dict[str, Any] | None = None,
    project_root_path: Path | None = None,
) -> None:
    """若存在 `preprocess_manifest.json`，则与 model/data 配置及磁盘上 `X.npy` 对齐，避免漏跑 preprocess。"""
    man_p = processed_dir / "preprocess_manifest.json"
    if not man_p.is_file():
        return
    man = json.loads(man_p.read_text(encoding="utf-8"))
    gs = man.get("grid_shape")
    if gs is not None and len(gs) == 3:
        g_c, g_h, g_w = int(gs[0]), int(gs[1]), int(gs[2])
        m_c, m_h, m_w = expected_chw
        if g_c != m_c or g_h != m_h or g_w != m_w:
            raise ValueError(
                f"{man_p.name} 中 grid_shape={gs} 与 model.yaml "
                f"(in_channels,grid_height,grid_width)=({m_c},{m_h},{m_w}) 不一致；请重跑 preprocess 或修正配置。"
            )
        x_path = processed_dir / "X.npy"
        if x_path.is_file():
            try:
                x = np.load(x_path, mmap_mode="r")
            except ValueError as exc:
                raise ValueError(
                    f"{x_path.name} 文件损坏或不完整（可能在 preprocess 中断时写入），请重跑 preprocess。"
                ) from exc
            chw = tuple(int(x.shape[i]) for i in range(1, 4)) if x.ndim == 4 else ()
            if x.ndim == 4 and chw != (g_c, g_h, g_w):
                raise ValueError(
                    f"{x_path.name} 形状 CHW={chw} 与 manifest grid_shape={gs} 不一致；请重跑 preprocess。"
                )
    td_m = man.get("tabular_dim")
    if td_m is not None:
        m_td = int(model.get("tabular_dim", 0))
        if int(td_m) != m_td:
            raise ValueError(
                f"{man_p.name} 中 tabular_dim={td_m} 与 model.tabular_dim={m_td} 不一致；请重跑 preprocess 或修正配置。"
            )

    if data_cfg is not None:
        root = project_root_path or project_root()

        cfg_win = float(data_cfg["window_seconds"]) if "window_seconds" in data_cfg else None
        cfg_hor = label_horizon_seconds_from_dict(data_cfg)
        man_win = man.get("window_seconds")
        man_hor = man.get("label_horizon_seconds")
        if man_win is not None and cfg_win is not None and float(man_win) != cfg_win:
            raise ValueError(
                f"{man_p.name} 中 window_seconds={man_win} 与当前配置 {cfg_win} 不一致；"
                "切换 horizon/窗口后请清空 processed/splits 并重跑 preprocess。"
            )
        if man_hor is not None and cfg_hor is not None and float(man_hor) != cfg_hor:
            raise ValueError(
                f"{man_p.name} 中 label_horizon_seconds={man_hor} 与当前配置 {cfg_hor} 不一致；"
                "切换 horizon 实验后请清空 processed/splits 并重跑 preprocess。"
            )

        man_idx = man.get("index_features_path")
        cfg_idx_raw = data_cfg.get("index_features_path")
        if man_idx and cfg_idx_raw:
            man_pth = _norm_cfg_path(root, str(man_idx))
            cfg_pth = _norm_cfg_path(root, str(cfg_idx_raw))
            if man_pth and cfg_pth and man_pth != cfg_pth:
                raise ValueError(
                    f"{man_p.name} 中 index_features_path={man_idx} 与当前配置 {cfg_idx_raw} 不一致；"
                    "改索引路径后请重跑 preprocess。"
                )

        def _norm_mcelog_cap(raw: Any) -> int | None:
            if raw is None or raw == "null":
                return None
            return int(raw)

        if "mcelog_max_rows" in man:
            man_cap = _norm_mcelog_cap(man.get("mcelog_max_rows"))
            cfg_cap = _norm_mcelog_cap(data_cfg.get("mcelog_max_rows"))
            if man_cap != cfg_cap:
                raise ValueError(
                    f"{man_p.name} 中 mcelog_max_rows={man.get('mcelog_max_rows')} "
                    f"与当前配置 {data_cfg.get('mcelog_max_rows')} 不一致；"
                    "切换冒烟/全量截断后请清空 processed/splits 并重跑 preprocess。"
                )


def validate_experiment(cfg: dict[str, Any], require_splits: bool = True) -> dict[str, Any]:
    """校验实验配置与磁盘数据是否一致。

    Args:
        cfg: 合并后的实验配置 dict。
        require_splits: True 时检查 `data/splits/` 下 train/val/test 张量形状与 tabular 语义。

    Returns:
        含 `expected_shape`、`checks.processed`、`checks.splits` 等字段的报告 dict。

    Raises:
        FileNotFoundError / ValueError: 路径缺失或 C/H/W/tabular 与 model 段不一致。
    """
    model = cfg.get("model", {})
    expected = {
        "in_channels": int(model.get("in_channels", 31)),
        "grid_height": int(model.get("grid_height", 128)),
        "grid_width": int(model.get("grid_width", 32)),
    }

    run_name = cfg.get("base", {}).get("run_name", "exp_tianchi_baseline")
    run_dir = project_root() / "runs" / run_name
    _check_run_dir_writable(run_dir)
    configured_device = str(cfg.get("base", {}).get("device", "auto"))
    resolved_device = resolve_device(configured_device)

    processed_dir = Path(cfg["data"]["processed_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])

    exp_c = int(expected["in_channels"])
    exp_h = int(expected["grid_height"])
    exp_w = int(expected["grid_width"])

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "device": resolved_device,
        "expected_shape": expected,
        "checks": {},
        "warnings": [],
    }

    manifest_data = cfg.get("data") if require_splits else None
    check_preprocess_manifest(
        processed_dir,
        model,
        (exp_c, exp_h, exp_w),
        data_cfg=manifest_data,
        project_root_path=project_root(),
    )

    man_p = processed_dir / "preprocess_manifest.json"
    if require_splits and man_p.is_file():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        if cfg.get("data", {}).get("index_features_path") and not man.get("index_features_path"):
            report["warnings"].append(
                "preprocess_manifest 缺少 index_features_path（旧产物）；建议重跑 preprocess 以便校验索引路径。"
            )
        n_samples = int(man.get("n_samples") or 0)
        n_empty = int(man.get("n_empty_grids") or 0)
        n_dropped = int(man.get("n_empty_grids_dropped") or 0)
        n_before = int(man.get("n_samples_before_drop") or (n_samples + n_dropped))
        if n_before > 0 and n_empty / n_before >= 0.5:
            report["warnings"].append(
                f"n_empty_grids={n_empty}（已丢弃负样本空窗 {n_dropped}，保留正样本空窗 "
                f"{int(man.get('n_empty_grids_kept_positive') or 0)}）；"
                "冒烟 mcelog 截断下常见，全量请取消 mcelog_max_rows。"
            )

    x_proc = processed_dir / "X.npy"
    y_proc = processed_dir / "y.npy"
    if x_proc.exists() and y_proc.exists():
        n, c, h, w = _check_xy_shape_pair(x_proc, y_proc, expected=expected)
        report["checks"]["processed"] = {"exists": True, "shape": [n, c, h, w]}
    else:
        report["checks"]["processed"] = {"exists": False}

    tab_dim_cfg = int(model.get("tabular_dim", 0))
    if require_splits:
        split_shapes = {}
        meta_rows: dict[str, Any] = {}
        for name in ("train", "val", "test"):
            n, c, h, w = _check_xy_shape_pair(
                splits_dir / f"X_{name}.npy",
                splits_dir / f"y_{name}.npy",
                expected=expected,
            )
            split_shapes[name] = [n, c, h, w]
            tab_path = splits_dir / f"tabular_{name}.npy"
            if tab_dim_cfg > 0:
                if not tab_path.is_file():
                    raise FileNotFoundError(f"model.tabular_dim={tab_dim_cfg} 但缺少 {tab_path}")
                t = np.load(tab_path, mmap_mode="r")
                if t.ndim != 2 or int(t.shape[0]) != n or int(t.shape[1]) != tab_dim_cfg:
                    raise ValueError(f"{tab_path.name} 形状 {t.shape} 与期望 [N={n},F={tab_dim_cfg}] 不符")
            elif tab_path.is_file():
                raise ValueError(
                    f"发现 {tab_path} 但 model.tabular_dim=0，请设置 tabular_dim 或删除 tabular 切分文件后重跑 preprocess"
                )
            meta_path = splits_dir / f"meta_{name}.csv"
            if meta_path.exists():
                mdf = pd.read_csv(meta_path)
                if len(mdf) != n:
                    raise ValueError(f"meta_{name}.csv rows={len(mdf)} vs X N={n}")
                meta_rows[name] = len(mdf)
            else:
                meta_rows[name] = None
        split_total = sum(int(split_shapes[name][0]) for name in ("train", "val", "test"))
        proc_n = None
        proc_check = report["checks"].get("processed") or {}
        if proc_check.get("exists"):
            proc_n = int(proc_check["shape"][0])
        elif man_p.is_file():
            proc_n = int(json.loads(man_p.read_text(encoding="utf-8")).get("n_samples") or 0)
        if proc_n is not None and proc_n > 0 and split_total != proc_n:
            raise ValueError(
                f"splits 总行数 {split_total} 与 processed 样本数 {proc_n} 不一致；"
                "可能 split 未完成或使用了陈旧 data/splits，请重跑 preprocess。"
            )
        norm_stats = splits_dir / "normalization_stats.json"
        report["checks"]["splits"] = {
            "exists": True,
            "shapes": split_shapes,
            "meta_rows": meta_rows,
            "total_rows": split_total,
            "processed_rows": proc_n,
            "normalization_stats_exists": norm_stats.exists(),
        }
    else:
        report["checks"]["splits"] = {"exists": (splits_dir / "X_train.npy").exists()}

    return report
