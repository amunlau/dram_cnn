"""Notebook / 批量流水线辅助：断点标记中断后续跑、preprocess 前预先检查。"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from dram2cnn.utils.config import load_experiment_config, project_root
from dram2cnn.utils.notebook_bootstrap import clear_dram2cnn_modules


def reload_dram2cnn_modules() -> int:
    """丢弃已 import 的 dram2cnn 子模块，便于 Notebook 长会话加载最新源码。

    典型场景：内核启动后仓库更新了 ``experiment_guards`` 等模块，直接 ``from … import`` 新符号会 ImportError。
    返回清除的模块数量。
    """
    return clear_dram2cnn_modules()


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
    """执行一步 shell 命令；成功则写 marker，便于中断后续跑。"""
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



def _data_dir_from_cfg(cfg: dict, key: str, root: Path) -> Path:
    """``processed_dir`` / ``splits_dir``：相对传入 ``root`` 解析（兼容 config 已绝对化）。"""
    p = Path(str(cfg["data"][key]))
    repo = project_root()
    if p.is_absolute():
        try:
            return root / p.relative_to(repo)
        except ValueError:
            return p
    return root / p


def processed_dir_for_exp(exp_yaml: str | Path, root: Path | None = None) -> Path:
    """实验 YAML 中 ``data.processed_dir`` 的绝对路径。"""
    root = root or project_root()
    exp_path = Path(exp_yaml)
    if not exp_path.is_absolute():
        exp_path = root / exp_path
    cfg = load_experiment_config(exp_path)
    return _data_dir_from_cfg(cfg, "processed_dir", root)


def splits_dir_for_exp(exp_yaml: str | Path, root: Path | None = None) -> Path:
    root = root or project_root()
    exp_path = Path(exp_yaml)
    if not exp_path.is_absolute():
        exp_path = root / exp_path
    cfg = load_experiment_config(exp_path)
    return _data_dir_from_cfg(cfg, "splits_dir", root)


def have_processed_for_exp(exp_yaml: str | Path, root: Path | None = None) -> bool:
    proc = processed_dir_for_exp(exp_yaml, root)
    return (proc / "X.npy").is_file() and (proc / "y.npy").is_file()


def have_splits_train_for_exp(exp_yaml: str | Path, root: Path | None = None) -> bool:
    spl = splits_dir_for_exp(exp_yaml, root)
    return (spl / "X_train.npy").is_file() and (spl / "y_train.npy").is_file()


def have_feature_data_for_exp(exp_yaml: str | Path, root: Path | None = None) -> bool:
    """``processed/X.npy`` 或 ``splits/X_train.npy`` 至少一组齐全。"""
    return have_processed_for_exp(exp_yaml, root) or have_splits_train_for_exp(exp_yaml, root)


def resolve_feature_xy_paths(
    exp_yaml: str | Path,
    *,
    root: Path | None = None,
    prefer_processed: bool = True,
) -> tuple[Path, Path, str]:
    """按实验配置解析特征 ``(X_path, y_path, source_label)``。

    优先 ``processed_dir`` 下的全量池 ``X.npy``；否则改用 ``splits_dir/X_train.npy``。
    两者均不存在时抛出 ``FileNotFoundError``。
    """
    root = root or project_root()
    proc = processed_dir_for_exp(exp_yaml, root)
    spl = splits_dir_for_exp(exp_yaml, root)
    have_proc = have_processed_for_exp(exp_yaml, root)
    have_spl = have_splits_train_for_exp(exp_yaml, root)

    try:
        proc_rel = proc.relative_to(root)
    except ValueError:
        proc_rel = proc
    try:
        spl_rel = spl.relative_to(root)
    except ValueError:
        spl_rel = spl

    if prefer_processed and have_proc:
        return (
            proc / "X.npy",
            proc / "y.npy",
            f"{proc_rel}/（全量池，未按 train 重标准化）",
        )
    if have_spl:
        return (
            spl / "X_train.npy",
            spl / "y_train.npy",
            f"{spl_rel}/ 训练集（未找到 processed/X.npy；splitter 按 train 统计量标准化）",
        )
    if have_proc:
        return proc / "X.npy", proc / "y.npy", f"{proc_rel}/（全量池）"

    exp_path = Path(exp_yaml)
    if not exp_path.is_absolute():
        exp_path = root / exp_path
    raise FileNotFoundError(
        f"实验 {exp_path} 下未找到特征文件：\n"
        f"  processed: {proc / 'X.npy'}\n"
        f"  splits:    {spl / 'X_train.npy'}\n"
        "请检查 data/raw/、实验 YAML 的 processed_dir/splits_dir，或重新执行预处理。"
    )


def require_preprocess_manifest(
    exp_yaml: str | Path,
    *,
    root: Path | None = None,
    hint_cmd: str | None = None,
) -> Path:
    """resplit 等步骤前置：确认 ``processed_dir/preprocess_manifest.json`` 存在。"""
    root = root or project_root()
    proc = processed_dir_for_exp(exp_yaml, root)
    man = proc / "preprocess_manifest.json"
    if man.is_file():
        return man
    exp_path = Path(exp_yaml)
    if not exp_path.is_absolute():
        exp_path = root / exp_path
    hint = hint_cmd or f"bash scripts/preprocess.sh {exp_path.relative_to(root)}"
    raise RuntimeError(
        f"缺少 {man}。\n"
        f"请先对同一 processed_dir 完成 preprocess，例如：\n  {hint}"
    )


def preflight_full_scale_index(
    exp_yaml: str | Path,
    *,
    root: Path | None = None,
    context: str = "preflight",
) -> dict:
    """全量实验预先检查：刷新模块缓存并确认 ``index_features_path`` 非演示 stub。"""
    reload_dram2cnn_modules()
    from dram2cnn.utils.experiment_guards import assert_full_scale_index_not_stub

    root = root or project_root()
    exp_path = Path(exp_yaml)
    if not exp_path.is_absolute():
        exp_path = root / exp_path
    cfg = load_experiment_config(exp_path)
    assert_full_scale_index_not_stub(cfg, context=context)
    return cfg


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
            f"vs model=({exp_c},{exp_h},{exp_w})；删对应 preprocess/resplit marker 后重新执行"
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
