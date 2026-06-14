"""Notebook / 批量流水线辅助：marker 断点续跑、preprocess 前预检。"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from dram2cnn.utils.config import load_experiment_config, project_root


def pipeline_log_dir(root: Path | None = None) -> Path:
    root = root or project_root()
    d = root / "runs" / "pipeline-logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pipeline_marker_dir(root: Path | None = None) -> Path:
    d = pipeline_log_dir(root) / "markers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_step(
    exp_name: str,
    step_name: str,
    command: str,
    *,
    project_root_path: Path | None = None,
) -> None:
    """执行一步 shell 命令；成功则写 marker，便于断点续跑。"""
    root = project_root_path or project_root()
    marker_dir = pipeline_marker_dir(root)
    log_dir = pipeline_log_dir(root)
    marker = marker_dir / f"{exp_name}__{step_name}.done"
    if marker.is_file():
        print(f"[SKIP] {exp_name}/{step_name} (marker: {marker.name})")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{exp_name}__{step_name}__{ts}.log"
    src = root / "src"
    cmd = (
        f'bash -lc "set -o pipefail; export PYTHONUNBUFFERED=1; '
        f"export PYTHONPATH={src}:${{PYTHONPATH:-}}; "
        f'{command} 2>&1 | tee \\"{log_file}\\""'
    )
    print(f"[RUN ] {exp_name}/{step_name}")
    print("      command:", command)
    print("      log:", log_file)
    subprocess.run(cmd, shell=True, check=True, cwd=str(root))
    marker.write_text(ts, encoding="utf-8")
    print(f"[DONE] {exp_name}/{step_name}\n")


def clear_exp_markers(exp_name: str, *, project_root_path: Path | None = None) -> int:
    """删除某一实验 marker 前缀下的全部 `.done` 文件。"""
    marker_dir = pipeline_marker_dir(project_root_path)
    removed = 0
    for p in sorted(marker_dir.glob(f"{exp_name}__*.done")):
        p.unlink(missing_ok=True)
        removed += 1
    print(f"已删除 {exp_name} 的 marker 数量: {removed}")
    return removed


def splits_dir_for_exp(exp_yaml: str | Path, root: Path | None = None) -> Path:
    root = root or project_root()
    exp_path = Path(exp_yaml)
    if not exp_path.is_absolute():
        exp_path = root / exp_path
    cfg = load_experiment_config(exp_path)
    p = Path(cfg["data"]["splits_dir"])
    return p if p.is_absolute() else root / p


def preflight_splits_vs_model(exp_yaml: str | Path, root: Path | None = None) -> None:
    """比对 X_train 与当前实验 model 配置的 (C,H,W)。"""
    root = root or project_root()
    spl = splits_dir_for_exp(exp_yaml, root)
    x_train = spl / "X_train.npy"
    if not x_train.is_file():
        print(f"[preflight] 尚无 {x_train}，请先 preprocess / resplit")
        return
    exp_path = Path(exp_yaml)
    if not exp_path.is_absolute():
        exp_path = root / exp_path
    cfg = load_experiment_config(exp_path)
    model_cfg = cfg["model"]
    exp_c = int(model_cfg["in_channels"])
    exp_h = int(model_cfg["grid_height"])
    exp_w = int(model_cfg["grid_width"])
    x = np.load(x_train, mmap_mode="r")
    _n, c, h, w = (int(x.shape[i]) for i in range(4))
    if (c, h, w) == (exp_c, exp_h, exp_w):
        print(f"[preflight] 通过 ({exp_yaml}): X_train (C,H,W)=({c},{h},{w}) splits={spl}")
    else:
        print(
            f"[preflight] 不匹配 ({exp_yaml}): X_train=({c},{h},{w}) "
            f"vs model=({exp_c},{exp_h},{exp_w})；删对应 preprocess/resplit marker 后重跑"
        )


def preflight_split_labels(exp_yaml: str | Path, root: Path | None = None) -> None:
    """打印各 split 正负样本数；无正样本时警告。"""
    root = root or project_root()
    spl = splits_dir_for_exp(exp_yaml, root)
    man = spl / "split_manifest.json"
    if man.is_file():
        print(
            f"[preflight] split_manifest ({man.name}):",
            json.loads(man.read_text(encoding="utf-8"))["split_strategy"],
        )
    for tag in ("train", "val", "test"):
        p = spl / f"y_{tag}.npy"
        if not p.is_file():
            continue
        y = np.load(p)
        pos, n = int(y.sum()), int(len(y))
        if pos == 0:
            print(f"[preflight] 警告 {exp_yaml} {tag}: 无正样本 → AUC/PR 可能为 nan")
        else:
            print(f"[preflight] {exp_yaml} {tag}: N={n} pos={pos} rate={pos / max(n, 1):.4f}")
