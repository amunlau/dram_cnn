from __future__ import annotations

"""训练/推理前预检：配置、`processed/`、`data/splits/` 张量形状与 manifest 一致性。

`require_splits=False` 时跳过切分文件检查，便于「尚未 preprocess」环境下只做路径校验。
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dram2cnn.utils.device import resolve_device


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _check_run_dir_writable(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = run_dir / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _check_device_available(device: str) -> None:
    resolve_device(device)


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


def _check_preprocess_manifest(
    processed_dir: Path,
    model: dict[str, Any],
    expected_chw: tuple[int, int, int],
) -> None:
    """若存在 `preprocess_manifest.json`，则与 `model.yaml` 及磁盘上 `X.npy` 的 CHW 对齐，避免漏跑 preprocess。"""
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
            x = np.load(x_path, mmap_mode="r")
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


def validate_experiment(cfg: dict[str, Any], require_splits: bool = True) -> dict[str, Any]:
    model = cfg.get("model", {})
    expected = {
        "in_channels": int(model.get("in_channels", 3)),
        "grid_height": int(model.get("grid_height", 64)),
        "grid_width": int(model.get("grid_width", 64)),
    }

    run_name = cfg.get("base", {}).get("run_name", "exp_tianchi_baseline")
    run_dir = _project_root() / "runs" / run_name
    _check_run_dir_writable(run_dir)
    configured_device = str(cfg.get("base", {}).get("device", "auto"))
    _check_device_available(configured_device)
    resolved_device = resolve_device(configured_device)

    processed_dir = Path(cfg["data"]["processed_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])

    exp_c = int(expected["in_channels"])
    exp_h = int(expected["grid_height"])
    exp_w = int(expected["grid_width"])
    _check_preprocess_manifest(processed_dir, model, (exp_c, exp_h, exp_w))

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "device": resolved_device,
        "expected_shape": expected,
        "checks": {},
    }

    x_proc = processed_dir / "X.npy"
    y_proc = processed_dir / "y.npy"
    if x_proc.exists() and y_proc.exists():
        n, c, h, w = _check_xy_shape_pair(x_proc, y_proc, expected=None)
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
                raise ValueError(f"发现 {tab_path} 但 model.tabular_dim=0，请设置 tabular_dim 或删除旧切分")
            meta_path = splits_dir / f"meta_{name}.csv"
            if meta_path.exists():
                mdf = pd.read_csv(meta_path)
                if len(mdf) != n:
                    raise ValueError(f"meta_{name}.csv rows={len(mdf)} vs X N={n}")
                meta_rows[name] = len(mdf)
            else:
                meta_rows[name] = None
        norm_stats = splits_dir / "normalization_stats.json"
        report["checks"]["splits"] = {
            "exists": True,
            "shapes": split_shapes,
            "meta_rows": meta_rows,
            "normalization_stats_exists": norm_stats.exists(),
        }
    else:
        report["checks"]["splits"] = {"exists": (splits_dir / "X_train.npy").exists()}

    return report
