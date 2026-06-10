from __future__ import annotations

"""目录创建与 JSON 序列化工具。"""

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """若目录不存在则创建，并返回对应 `Path`。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj: Any) -> None:
    """将对象写入 UTF-8 JSON 文件（带缩进，便于阅读）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
