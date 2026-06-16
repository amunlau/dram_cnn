"""Jupyter Notebook 内核初始化（仅 stdlib，可在 import dram2cnn 前/后安全调用）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def clear_dram2cnn_modules() -> int:
    """丢弃 ``sys.modules`` 中已缓存的 dram2cnn 子模块，便于加载最新源码。"""
    removed = 0
    for name in list(sys.modules):
        if name == "dram2cnn" or name.startswith("dram2cnn."):
            del sys.modules[name]
            removed += 1
    return removed


def infer_project_root(*, cwd: Path | None = None) -> Path:
    """从 Notebook 工作目录推断仓库根（``notebooks/`` 的上一级）。"""
    cwd = (cwd or Path.cwd()).resolve()
    return cwd.parent if cwd.name == "notebooks" else cwd


def prepare_notebook_kernel(*, cwd: Path | None = None) -> Path:
    """设置 ``PYTHONPATH``、清除 dram2cnn 缓存，返回仓库根目录（与 ``project_root()`` 一致）。"""
    root = infer_project_root(cwd=cwd)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    clear_dram2cnn_modules()
    os.environ["PYTHONPATH"] = f"{src}:{os.environ.get('PYTHONPATH', '')}"
    return root


def notebook_setup(*, cwd: Path | None = None) -> Path:
    """Notebook 首格推荐入口：等价于 ``prepare_notebook_kernel``，返回 canonical ``project_root()``。"""
    prepare_notebook_kernel(cwd=cwd)
    from dram2cnn.utils.config import project_root

    root = project_root()
    os.environ["PYTHONPATH"] = f"{root / 'src'}:{os.environ.get('PYTHONPATH', '')}"
    return root
