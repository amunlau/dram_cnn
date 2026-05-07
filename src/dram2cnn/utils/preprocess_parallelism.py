"""预处理阶段 CPU 并行度（与 `configs/data.yaml` 对齐）。

Linux 上 `tianchi_build` 对 **mcelog 按 sid 时间排序** 与 **按索引行填网格 X** 使用 ``fork`` + ``Pool``；
进程预算由本模块解析，避免在多处硬编码比例。

规则：

- **`preprocess_workers` 为非 null 正整数**：固定使用该进程数（基线 / 全量实验 yaml 均可覆盖）。
- **`preprocess_workers` 为 null**：``max(1, floor(os.cpu_count() * preprocess_parallel_frac))``，
  其中 ``os.cpu_count()`` 为 Python 可见的**逻辑 CPU 数**（通常含超线程）。

默认 **`preprocess_parallel_frac = 0.6`**，即约使用 **60%** 逻辑核作为并行预算，为系统与其它进程留出余量。
可在 `configs/data.yaml` 或任意实验 ``overrides.data`` 中调整；比例会被钳制到 ``[0.05, 1.0]``。
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_PREPROCESS_PARALLEL_FRAC = 0.6


def resolve_preprocess_worker_count(data_cfg: dict[str, Any]) -> int:
    """返回预处理 fork Pool 的进程预算（至少为 1）。"""
    explicit = data_cfg.get("preprocess_workers")
    if explicit is not None:
        return max(1, int(explicit))
    frac = float(data_cfg.get("preprocess_parallel_frac", DEFAULT_PREPROCESS_PARALLEL_FRAC))
    frac = max(0.05, min(1.0, frac))
    cpu = os.cpu_count() or 1
    return max(1, int(cpu * frac))


def preprocess_workers_log_line(data_cfg: dict[str, Any], nw: int) -> str:
    """供 ``materialize_dataset`` 打印一行人类可读说明。"""
    explicit = data_cfg.get("preprocess_workers")
    if explicit is not None:
        return (
            f"预处理并行度: worker 预算={nw}（来自显式 data.preprocess_workers={int(explicit)}；"
            "基线/全量可在实验 yaml 的 overrides.data 中覆盖）"
        )
    frac = float(data_cfg.get("preprocess_parallel_frac", DEFAULT_PREPROCESS_PARALLEL_FRAC))
    cpu = os.cpu_count() or 1
    return (
        f"预处理并行度: worker 预算={nw}（data.preprocess_workers=null → "
        f"max(1, floor(os.cpu_count()×preprocess_parallel_frac)) = max(1, floor({cpu}×{frac:g}))；"
        "顺序读 mcelog CSV 仍为单线程；Linux 上满足阈值时对「按 sid 排序」「填网格」多进程）"
    )


def effective_preprocess_parallel_frac_recorded(data_cfg: dict[str, Any]) -> float:
    """写入 manifest：若未配置键则用默认比例。"""
    return float(data_cfg.get("preprocess_parallel_frac", DEFAULT_PREPROCESS_PARALLEL_FRAC))
