"""预处理阶段 CPU 并行度（与 `configs/data.yaml` 一致）。

Linux 上 `tianchi_build` 对 **mcelog 按 sid 时间排序** 与 **按索引行填网格 X** 使用 ``fork`` + ``Pool``；
进程 budget 由本模块解析，避免在多处写死比例常数。

规则：

- **`preprocess_workers` 为非 null 正整数**：CPU 上限为该进程数（仍会被 **RAM 预算** 压低，除非 ``preprocess_ignore_ram_budget: true``）。
- **`preprocess_workers` 为 null**：``max(1, floor(os.cpu_count() * preprocess_parallel_frac))``。
- **`ram_budget_frac` 非 null**：按 ``ram_total_gib × ram_budget_frac`` 估算 preprocess 峰值，取 **min(CPU 预算, RAM 预算 worker)**。

默认 **`preprocess_parallel_frac = 0.6`**；**`ram_budget_frac`** 在 ``configs/data.yaml`` 默认为 **0.5**（系统内存峰值 ≤50% MemTotal，非显存）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_PREPROCESS_PARALLEL_FRAC = 0.6
DEFAULT_RAM_BUDGET_FRAC = 0.5
# 排序 + 填网格 fork 时，每增加 1 个 worker 的保守 COW 放大系数（相对单进程 X+events 体量）
_FORK_RAM_FACTOR_PER_EXTRA_WORKER = 0.14
# OS / Jupyter / Python 运行时预留（GiB）
_RAM_OVERHEAD_GIB = 3.0
# mcelog 解析为 events 结构后的体量 ≈ 扫描 CSV 字节 × 该系数
_MCELOG_EVENTS_SIZE_FACTOR = 1.0
# mcelog 截断扫描时按行估算字节（保守；全文件扫描时用实际文件大小）
_MCELOG_BYTES_PER_ROW_EST = 120.0


def read_mem_total_gib() -> float:
    """读取 ``/proc/meminfo`` 的 MemTotal（GiB）；失败时返回 64.0。"""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return 64.0


def effective_ram_total_gib(data_cfg: dict[str, Any]) -> float:
    """解析 ``ram_total_gib``；``null`` 时读 ``/proc/meminfo``。"""
    total_raw = data_cfg.get("ram_total_gib")
    return float(total_raw) if total_raw is not None else read_mem_total_gib()


def effective_ram_budget_frac(data_cfg: dict[str, Any]) -> float | None:
    """解析 ``ram_budget_frac``；键不存在或值为 ``null`` 时不启用 RAM 预算。"""
    if "ram_budget_frac" not in data_cfg:
        return None
    raw = data_cfg.get("ram_budget_frac")
    if raw is None:
        return None
    frac = float(raw)
    if not (0.0 < frac <= 1.0):
        return None
    return frac


def _estimate_mcelog_events_gib(
    mcelog_path: Path | str | None,
    mcelog_max_rows: int | None = None,
) -> float:
    """估算 mcelog 解析后 events 结构的 RSS（GiB）。"""
    if mcelog_path is None:
        return 0.0
    p = Path(mcelog_path)
    if not p.is_file():
        return 0.0
    file_bytes = p.stat().st_size
    if file_bytes <= 0:
        return 0.0
    if mcelog_max_rows is None:
        scanned_bytes = file_bytes
    else:
        cap_bytes = int(mcelog_max_rows) * _MCELOG_BYTES_PER_ROW_EST
        scanned_bytes = min(file_bytes, cap_bytes)
    return scanned_bytes / (1024**3) * _MCELOG_EVENTS_SIZE_FACTOR


def estimate_split_normalize_peak_gib(
    n_samples: int,
    grid_shape: tuple[int, int, int],
    *,
    train_ratio: float = 0.7,
    split_chunk_rows: int = 256,
) -> float:
    """保守估算 split + 按通道归一化阶段峰值 RSS（GiB）。

    假设 processed ``X.npy`` 以 memmap 读取（不重复整表 RSS），峰值来自：
    - 分块写出最大 split 时的单块 materialize；
    - 按通道 robust/zscore 拟合时的 float64 通道板 + percentile 临时数组。
    """
    c, h, w = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
    n = max(0, int(n_samples))
    n_train = max(1, int(round(n * float(train_ratio))))
    chunk_gib = (
        max(1, int(split_chunk_rows)) * c * h * w * 4 / (1024**3)
    )
    channel_gib = n_train * h * w * 8 / (1024**3)
    normalize_gib = channel_gib * 3.0
    return max(chunk_gib, normalize_gib) + _RAM_OVERHEAD_GIB


def estimate_preprocess_peak_gib(
    n_samples: int,
    grid_shape: tuple[int, int, int],
    *,
    mcelog_path: Path | str | None = None,
    mcelog_max_rows: int | None = None,
    n_workers: int = 1,
) -> float:
    """保守估算 preprocess 峰值 RSS（GiB）：X 数组 + events + fork 放大 + 固定开销。"""
    c, h, w = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
    n = max(0, int(n_samples))
    x_gib = n * c * h * w * 4 / (1024**3)
    events_gib = _estimate_mcelog_events_gib(mcelog_path, mcelog_max_rows)
    fork_factor = 1.0 + _FORK_RAM_FACTOR_PER_EXTRA_WORKER * max(0, int(n_workers) - 1)
    return (x_gib + events_gib) * fork_factor + _RAM_OVERHEAD_GIB


def _workers_from_cpu(data_cfg: dict[str, Any]) -> int:
    explicit = data_cfg.get("preprocess_workers")
    if explicit is not None:
        return max(1, int(explicit))
    frac = float(data_cfg.get("preprocess_parallel_frac", DEFAULT_PREPROCESS_PARALLEL_FRAC))
    frac = max(0.05, min(1.0, frac))
    cpu = os.cpu_count() or 1
    return max(1, int(cpu * frac))


def _workers_from_ram_budget(
    data_cfg: dict[str, Any],
    *,
    n_samples: int,
    grid_shape: tuple[int, int, int],
    mcelog_path: Path | str | None,
    mcelog_max_rows: int | None,
    cpu_cap: int,
) -> int | None:
    frac = effective_ram_budget_frac(data_cfg)
    if frac is None:
        return None
    total_gib = effective_ram_total_gib(data_cfg)
    budget_gib = total_gib * frac
    best = 0
    for w in range(1, max(1, cpu_cap) + 1):
        peak = estimate_preprocess_peak_gib(
            n_samples,
            grid_shape,
            mcelog_path=mcelog_path,
            mcelog_max_rows=mcelog_max_rows,
            n_workers=w,
        )
        if peak <= budget_gib:
            best = w
        else:
            break
    return max(1, best)


def ram_budget_snapshot(
    data_cfg: dict[str, Any],
    nw: int,
    *,
    n_samples: int,
    grid_shape: tuple[int, int, int],
    mcelog_path: Path | str | None = None,
    mcelog_max_rows: int | None = None,
) -> dict[str, Any] | None:
    """返回 RAM 预算快照（供日志/manifest）；未启用 RAM 预算时返回 ``None``。"""
    if data_cfg.get("preprocess_ignore_ram_budget") is True:
        return None
    frac = effective_ram_budget_frac(data_cfg)
    if frac is None:
        return None
    total_gib = effective_ram_total_gib(data_cfg)
    budget_gib = total_gib * frac
    peak_gib = estimate_preprocess_peak_gib(
        n_samples,
        grid_shape,
        mcelog_path=mcelog_path,
        mcelog_max_rows=mcelog_max_rows,
        n_workers=nw,
    )
    split_peak_gib = estimate_split_normalize_peak_gib(n_samples, grid_shape)
    return {
        "ram_total_gib": total_gib,
        "ram_budget_frac": frac,
        "ram_budget_gib": budget_gib,
        "estimated_peak_gib": peak_gib,
        "split_normalize_estimated_peak_gib": split_peak_gib,
        "exceeds_budget": peak_gib > budget_gib,
        "split_normalize_exceeds_budget": split_peak_gib > budget_gib,
    }


def resolve_preprocess_worker_count(
    data_cfg: dict[str, Any],
    *,
    n_samples: int | None = None,
    grid_shape: tuple[int, int, int] | None = None,
    mcelog_path: Path | str | None = None,
    mcelog_max_rows: int | None = None,
) -> int:
    """返回预处理 fork Pool 的进程 budget（至少为 1）。"""
    if mcelog_max_rows is None:
        cap_raw = data_cfg.get("mcelog_max_rows")
        mcelog_max_rows = int(cap_raw) if cap_raw is not None else None

    cpu_w = _workers_from_cpu(data_cfg)
    if data_cfg.get("preprocess_ignore_ram_budget") is True:
        return cpu_w
    if n_samples is None or grid_shape is None:
        return cpu_w
    ram_w = _workers_from_ram_budget(
        data_cfg,
        n_samples=n_samples,
        grid_shape=grid_shape,
        mcelog_path=mcelog_path,
        mcelog_max_rows=mcelog_max_rows,
        cpu_cap=cpu_w,
    )
    if ram_w is None:
        return cpu_w
    return max(1, min(cpu_w, ram_w))


def preprocess_workers_log_line(
    data_cfg: dict[str, Any],
    nw: int,
    *,
    n_samples: int | None = None,
    grid_shape: tuple[int, int, int] | None = None,
    mcelog_path: Path | str | None = None,
    mcelog_max_rows: int | None = None,
) -> str:
    """供 ``materialize_dataset`` 打印一行便于阅读的说明。"""
    if mcelog_max_rows is None:
        cap_raw = data_cfg.get("mcelog_max_rows")
        mcelog_max_rows = int(cap_raw) if cap_raw is not None else None

    cpu_w = _workers_from_cpu(data_cfg)
    explicit = data_cfg.get("preprocess_workers")
    if explicit is not None:
        base = (
            f"预处理并行度: worker 预算={nw}（显式 preprocess_workers={int(explicit)}"
        )
    else:
        frac = float(data_cfg.get("preprocess_parallel_frac", DEFAULT_PREPROCESS_PARALLEL_FRAC))
        cpu = os.cpu_count() or 1
        base = (
            f"预处理并行度: worker 预算={nw}（CPU 上限={cpu_w}，"
            f"floor({cpu}×{frac:g})"
        )
    ram_note = ""
    if n_samples is not None and grid_shape is not None:
        snap = ram_budget_snapshot(
            data_cfg,
            nw,
            n_samples=n_samples,
            grid_shape=grid_shape,
            mcelog_path=mcelog_path,
            mcelog_max_rows=mcelog_max_rows,
        )
        if snap is not None:
            pct = int(round(snap["ram_budget_frac"] * 100))
            ram_note = (
                f"，RAM 预算={snap['ram_budget_gib']:.1f}GiB"
                f"（{snap['ram_total_gib']:.0f}GiB×{snap['ram_budget_frac']:g}，目标≤{pct}%），"
                f"估算峰值≈{snap['estimated_peak_gib']:.1f}GiB"
            )
            if nw < cpu_w:
                ram_note += f"，已由 {cpu_w} 压至 {nw}"
            if snap["exceeds_budget"]:
                ram_note += "，警告：即使最小 worker 估算峰值仍超预算（全量 X 体量过大，请增内存或减索引）"
            split_peak = snap.get("split_normalize_estimated_peak_gib")
            if split_peak is not None:
                ram_note += f"；split/归一化估算≈{split_peak:.1f}GiB"
                if snap.get("split_normalize_exceeds_budget"):
                    ram_note += "（超预算，已启用 memmap+分通道归一化降峰值）"
    return base + ram_note + "；顺序读 mcelog 仍为单线程；Linux 上对排序/填网格 fork 并行）"


def effective_preprocess_parallel_frac_recorded(data_cfg: dict[str, Any]) -> float:
    """写入 manifest：若未配置键则用默认比例。"""
    return float(data_cfg.get("preprocess_parallel_frac", DEFAULT_PREPROCESS_PARALLEL_FRAC))
