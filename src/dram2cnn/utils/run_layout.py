from __future__ import annotations

"""训练/推理运行目录布局：时间戳子目录、避免覆盖、latest 指针解析。

当 base.versioned_runs 为真：每次训练写入 runs/<run_name>/<时间戳>/，推理写入
runs/<run_name>/infer_<时间戳>/，并通过 .latest_train_session / .latest_infer_session
单行文本记录相对子目录名，供后续 infer/evaluate 默认解析最近一次会话，避免覆盖其它运行写入的子目录。
"""

import os
import time
from datetime import datetime
from pathlib import Path

from dram2cnn.utils.io import ensure_dir


def new_timestamp_id() -> str:
    """用于目录名的时间戳（含微秒，降低同秒冲突概率）。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _run_root(project_root: Path, run_name: str) -> Path:
    return project_root / "runs" / run_name


def _try_symlink(run_root: Path, link_name: str, target_dirname: str) -> None:
    """可选：创建相对 symlink，失败则忽略（仍依赖指针文件）。"""
    link = run_root / link_name
    try:
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            return
        link.symlink_to(target_dirname, target_is_directory=True)
    except OSError:
        pass


def write_pointer(run_root: Path, pointer_filename: str, session_dirname: str) -> None:
    """写入单行相对目录名，供跨平台解析「最新一次」产物目录。"""
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / pointer_filename).write_text(session_dirname.strip() + "\n", encoding="utf-8")


def read_pointer_dir(run_root: Path, pointer_filename: str, symlink_name: str | None = None) -> Path | None:
    """读取指针得到 `runs/<run_name>/<session>/` 绝对路径；不存在或非法则返回 None。"""
    if symlink_name:
        lp = run_root / symlink_name
        if lp.is_symlink():
            raw = lp.readlink()
            cand = raw.resolve() if raw.is_absolute() else (run_root / raw).resolve()
            if cand.is_dir():
                return cand
    pf = run_root / pointer_filename
    if not pf.is_file():
        return None
    name = pf.read_text(encoding="utf-8").strip().splitlines()[0]
    if not name:
        return None
    cand = (run_root / name).resolve()
    if cand.is_dir():
        return cand
    return None


def resolve_train_run_dir(project_root: Path, base: dict, *, world_size: int = 1) -> Path:
    """解析本次训练写入目录。versioned_runs 为真时新建时间戳子目录（DDP 下由 rank0 分配并同步）。"""
    run_name = str(base.get("run_name", "exp_tianchi_baseline"))
    run_root = _run_root(project_root, run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    versioned = bool(base.get("versioned_runs", True))
    explicit = base.get("run_id")
    if not versioned:
        # 平铺模式：始终写入 runs/<run_name>/，忽略 run_id，避免与「单目录可覆盖」语义冲突。
        return ensure_dir(run_root)
    if explicit:
        return ensure_dir(run_root / str(explicit))

    env_sess = (os.environ.get("DRAM2CNN_TRAIN_SESSION") or "").strip()
    if env_sess:
        return ensure_dir(run_root / env_sess)

    rank = int(os.environ.get("RANK", "0"))
    alloc = run_root / ".train_session_alloc"
    if world_size <= 1:
        sub = new_timestamp_id()
        return ensure_dir(run_root / sub)

    if rank == 0:
        sub = new_timestamp_id()
        out = ensure_dir(run_root / sub)
        alloc.write_text(sub + "\n", encoding="utf-8")
        return out

    for _ in range(600):
        if alloc.exists():
            sub = alloc.read_text(encoding="utf-8").strip().splitlines()[0]
            if sub:
                rd = run_root / sub
                if rd.is_dir():
                    return rd
        time.sleep(0.05)
    raise RuntimeError(
        f"DDP: 等待 rank0 在 {run_root} 下分配训练子目录超时；"
        "也可在启动前设置环境变量 DRAM2CNN_TRAIN_SESSION 为统一目录名。"
    )


def publish_train_session_pointer(project_root: Path, base: dict, train_run_dir: Path) -> None:
    """训练成功后由 rank0 调用：更新「最新训练会话」指针，便于推理自动找 checkpoint。"""
    if not bool(base.get("versioned_runs", True)):
        return
    run_name = str(base.get("run_name", "exp_tianchi_baseline"))
    run_root = _run_root(project_root, run_name)
    try:
        rel = train_run_dir.resolve().relative_to(run_root.resolve())
    except ValueError:
        return
    if rel.parts == ():
        return
    name = str(rel)
    if name == ".":
        return
    write_pointer(run_root, ".latest_train_session", name)
    _try_symlink(run_root, "latest_train", name)


def _find_newest_checkpoint_subdir(run_root: Path) -> Path | None:
    if not run_root.is_dir():
        return None
    best: Path | None = None
    best_m = -1.0
    for p in run_root.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        ck = p / "checkpoints" / "grid_cnn.pt"
        if ck.is_file():
            m = ck.stat().st_mtime
            if m > best_m:
                best_m = m
                best = p
    return best


def resolve_checkpoint_dir(project_root: Path, base: dict) -> Path:
    """推理加载权重所在目录（含 `checkpoints/grid_cnn.pt`）。"""
    run_name = str(base.get("run_name", "exp_tianchi_baseline"))
    run_root = _run_root(project_root, run_name)
    ck_rid = base.get("checkpoint_run_id")
    if ck_rid:
        d = (run_root / str(ck_rid)).resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"checkpoint_run_id 目录不存在: {d}")
        return d

    ptr = read_pointer_dir(run_root, ".latest_train_session", "latest_train")
    if ptr is not None and (ptr / "checkpoints" / "grid_cnn.pt").is_file():
        return ptr

    flat_ck = run_root / "checkpoints" / "grid_cnn.pt"
    if flat_ck.is_file():
        return run_root

    guess = _find_newest_checkpoint_subdir(run_root)
    if guess is not None:
        return guess

    raise FileNotFoundError(
        f"未找到 checkpoint：在 {run_root} 下训练一次，"
        "或在配置 base.checkpoint_run_id 中指定含 checkpoints/grid_cnn.pt 的子目录名。"
    )


def resolve_infer_output_dir(project_root: Path, base: dict) -> Path:
    """推理预测 CSV 等产物目录；versioned_runs 为真时默认新建 infer_<时间戳>/。"""
    run_name = str(base.get("run_name", "exp_tianchi_baseline"))
    run_root = _run_root(project_root, run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    versioned = bool(base.get("versioned_runs", True))
    explicit = base.get("infer_run_id")
    if not versioned:
        return ensure_dir(run_root)
    if explicit:
        return ensure_dir(run_root / str(explicit))
    sub = f"infer_{new_timestamp_id()}"
    return ensure_dir(run_root / sub)


def publish_infer_session_pointer(project_root: Path, base: dict, infer_run_dir: Path) -> None:
    """推理结束后更新「最新推理会话」指针，便于 evaluate 默认读取。"""
    if not bool(base.get("versioned_runs", True)):
        return
    run_name = str(base.get("run_name", "exp_tianchi_baseline"))
    run_root = _run_root(project_root, run_name)
    try:
        rel = infer_run_dir.resolve().relative_to(run_root.resolve())
    except ValueError:
        return
    name = str(rel)
    if not name or name == ".":
        return
    write_pointer(run_root, ".latest_infer_session", name)
    _try_symlink(run_root, "latest_infer", name)


def resolve_evaluate_artifact_dir(project_root: Path, base: dict) -> Path:
    """评估阶段：含 `predictions.csv` 的目录（metrics 等写入同目录）。"""
    run_name = str(base.get("run_name", "exp_tianchi_baseline"))
    run_root = _run_root(project_root, run_name)
    ev_rid = base.get("evaluate_run_id")
    if ev_rid:
        d = (run_root / str(ev_rid)).resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"evaluate_run_id 目录不存在: {d}")
        return d

    ptr = read_pointer_dir(run_root, ".latest_infer_session", "latest_infer")
    if ptr is not None and (ptr / "predictions.csv").is_file():
        return ptr

    if (run_root / "predictions.csv").is_file():
        return run_root

    raise FileNotFoundError(
        f"未找到 {run_root / 'predictions.csv'}。"
        "请先执行推理，或在 base.evaluate_run_id 指定含 predictions.csv 的子目录（如某次 infer_<时间戳>）。"
    )


def train_log_path(project_root: Path, base: dict, train_run_dir: Path) -> Path:
    """平铺 runs/<run_name>/ 时用 `train.log`；时间戳或自定义子目录下用 `train_<子目录名>.log`。"""
    rr = _run_root(project_root, str(base.get("run_name", "exp_tianchi_baseline")))
    if train_run_dir.resolve() == rr.resolve():
        return train_run_dir / "train.log"
    return train_run_dir / f"train_{train_run_dir.name}.log"


def cleanup_train_session_alloc(project_root: Path, base: dict) -> None:
    """训练正常结束后由 rank0 删除分配占位文件（可选清理）。"""
    run_name = str(base.get("run_name", "exp_tianchi_baseline"))
    alloc = _run_root(project_root, run_name) / ".train_session_alloc"
    try:
        alloc.unlink(missing_ok=True)
    except OSError:
        pass
