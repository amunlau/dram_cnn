from __future__ import annotations

"""日志工具：同时输出到文件与控制台。"""

import logging
from pathlib import Path


def get_logger(log_file: Path) -> logging.Logger:
    """创建独立 logger，写入 `log_file` 并同步输出到 stdout。"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger
