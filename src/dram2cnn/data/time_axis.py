from __future__ import annotations

"""时间轴：将 raw CSV 中的绝对 Unix 秒统一平移为「相对秒」（全局最早时刻 = 0）。

全链路（索引构建、preprocess、mcelog 事件、工单打标）须使用同一 ``time_epoch_offset``（raw 时间全局最小值）。
索引 CSV 的 ``predict_time`` 以 **1970-01-01 + 相对秒** 的 ISO 字符串存储，便于阅读且与 ``predict_ts`` 数值一致。
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 索引/ meta 中 predict_time 的可读展示纪元（与 Unix 1970 对齐，解析后数值即相对秒）
DISPLAY_EPOCH = pd.Timestamp("1970-01-01 00:00:00")

TIME_AXIS_RELATIVE_MARKER = "relative_seconds_since_global_min"

# 相对展示格式下，距 t₀ 的秒数通常远小于真实公历 Unix（2000-01-01 ≈ 9.5e8）
_LEGACY_CALENDAR_MIN_UNIX = 1e9


def series_to_absolute_seconds(s: pd.Series) -> np.ndarray:
    """pandas 时间列 → 相对 Unix 纪元（1970-01-01）的秒 float64；无法解析为 NaN。"""
    t = pd.to_datetime(s, errors="coerce", format="mixed")
    sec = (t - DISPLAY_EPOCH) / pd.Timedelta(seconds=1)
    return sec.to_numpy(dtype=np.float64)


def index_time_axis_from_sidecar(path: Path | str) -> bool | None:
    """读取索引 sidecar（``*.csv.stats.json``）中的 ``time_axis`` 标记。

    返回 ``True``/``False`` 表示是否为相对秒展示；无 sidecar 或无法解析时返回 ``None``。
    """
    p = Path(path)
    sidecar = p.parent / (p.name + ".stats.json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    axis = data.get("time_axis")
    if axis == TIME_AXIS_RELATIVE_MARKER:
        return True
    if axis is not None:
        return False
    return None


def predict_time_series_looks_relative(predict_time: pd.Series, abs_ts: np.ndarray) -> bool:
    """判断 ``predict_time`` 是否为 ``build_index`` 相对展示格式（1970 纪元 + 相对秒）。

    与 ``load_index_table`` 使用同一规则，避免把公历 2000+ 或 AD 0001 误判为相对格式。
    """
    finite = abs_ts[np.isfinite(abs_ts)]
    if finite.size == 0:
        return False
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo < 0:
        return False
    if hi >= _LEGACY_CALENDAR_MIN_UNIX:
        return False

    dt = pd.to_datetime(predict_time, errors="coerce", format="mixed")
    years = dt.dt.year
    if years.isna().any():
        return False
    if (years < 1000).any():
        return False
    if (years >= 2000).any():
        return False

    # build_index 产物：自 1970-01-01 起算，lo 为距全局 t₀ 的秒数（通常 < 1e8）
    if int(years.min()) == 1970 and lo < 1e8:
        return True
    return False


def index_timestamps_look_relative(
    abs_ts: np.ndarray,
    *,
    predict_time_strs: pd.Series | None = None,
) -> bool:
    """判断索引 ``predict_time`` 是否已为相对展示格式（优先 sidecar/字符串规则）。"""
    if predict_time_strs is not None:
        return predict_time_series_looks_relative(predict_time_strs, abs_ts)
    finite = abs_ts[np.isfinite(abs_ts)]
    if finite.size == 0:
        return False
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo < 0 or hi >= _LEGACY_CALENDAR_MIN_UNIX:
        return False
    return lo < 1e8 and hi < 5e8


def index_file_uses_relative_display(path: Path | str, *, chunk_rows: int = 1000) -> bool:
    """抽样索引 CSV，判断是否为本仓库 ``build_index`` 相对展示产物。"""
    p = Path(path)
    sidecar = index_time_axis_from_sidecar(p)
    if sidecar is not None:
        return sidecar
    if not p.is_file():
        return False
    for chunk in pd.read_csv(p, usecols=["predict_time"], chunksize=int(chunk_rows)):
        abs_ts = series_to_absolute_seconds(chunk["predict_time"])
        return predict_time_series_looks_relative(chunk["predict_time"], abs_ts)
    return False


def resolve_index_predict_ts(
    predict_time: pd.Series,
    *,
    time_epoch_offset: float | None,
    index_path: Path | str | None = None,
) -> np.ndarray:
    """将索引 ``predict_time`` 列解析为 **相对秒** ``predict_ts``。"""
    abs_ts = series_to_absolute_seconds(predict_time)
    sidecar = index_time_axis_from_sidecar(index_path) if index_path is not None else None
    if sidecar is True or predict_time_series_looks_relative(predict_time, abs_ts):
        return abs_ts
    if time_epoch_offset is None:
        raise ValueError(
            "索引 predict_time 为遗留低年份/公历格式，请先运行 build_index 重建索引，"
            "或在 load_index_table 传入 time_epoch_offset（与 preprocess 一致）。"
        )
    return abs_ts - float(time_epoch_offset)


def relative_seconds_to_predict_time_str(rel_sec: float) -> str:
    """相对秒 → ISO 可读字符串（1970-01-01 + rel），读回后 ``series_to_absolute_seconds`` 即得 rel。"""
    return (DISPLAY_EPOCH + pd.Timedelta(seconds=float(rel_sec))).isoformat(sep=" ", timespec="seconds")


def compute_global_time_epoch_offset(
    mcelog_path: Path | str,
    tickets_path: Path | str,
    *,
    index_path: Path | str | None = None,
    mcelog_chunk_rows: int = 1_000_000,
    mcelog_max_rows: int | None = None,
) -> float:
    """扫描 mcelog / tickets /（可选）遗留 index，返回全局最早时间的绝对 Unix 秒 ``t₀``。"""
    if mcelog_max_rows is not None:
        warnings.warn(
            f"mcelog_max_rows={mcelog_max_rows}：仅扫描 mcelog 前若干行计算 time_epoch_offset，"
            "可能不是全文件全局最小值；冒烟/截断模式不应用于生产预处理。",
            UserWarning,
            stacklevel=2,
        )

    mins: list[float] = []

    mcelog_p = Path(mcelog_path)
    t_min = _scan_csv_time_column_min(
        mcelog_p,
        "error_time",
        usecols=["sid", "error_time"],
        chunk_rows=mcelog_chunk_rows,
        max_rows=mcelog_max_rows,
    )
    if t_min is not None:
        mins.append(t_min)

    tickets_p = Path(tickets_path)
    f_min = _scan_csv_time_column_min(
        tickets_p,
        "failed_time",
        usecols=["sid", "failure_type", "failed_time"],
        chunk_rows=mcelog_chunk_rows,
        max_rows=None,
    )
    if f_min is not None:
        mins.append(f_min)

    if index_path is not None:
        idx_p = Path(index_path)
        if idx_p.is_file() and not index_file_uses_relative_display(idx_p, chunk_rows=mcelog_chunk_rows):
            i_min = _scan_csv_time_column_min(
                idx_p,
                "predict_time",
                usecols=["sid", "predict_time"],
                chunk_rows=mcelog_chunk_rows,
                max_rows=None,
            )
            if i_min is not None:
                mins.append(i_min)

    if not mins:
        raise ValueError(
            f"无法从 {mcelog_p} / {tickets_p} 解析任何有效时间戳以计算 time_epoch_offset"
        )
    return float(min(mins))


def _scan_csv_time_column_min(
    path: Path,
    time_col: str,
    *,
    usecols: list[str],
    chunk_rows: int,
    max_rows: int | None,
) -> float | None:
    if not path.is_file():
        return None
    gmin = np.inf
    rows_read = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunk_rows), low_memory=False):
        if max_rows is not None and rows_read >= max_rows:
            break
        if max_rows is not None:
            remain = max_rows - rows_read
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]
        rows_read += len(chunk)
        if time_col not in chunk.columns:
            continue
        ts = series_to_absolute_seconds(chunk[time_col])
        valid = ts[np.isfinite(ts)]
        if valid.size:
            gmin = min(gmin, float(valid.min()))
    return None if not np.isfinite(gmin) else float(gmin)


def to_relative_seconds_array(abs_ts: np.ndarray, offset: float) -> np.ndarray:
    out = abs_ts.astype(np.float64, copy=True)
    out = out - float(offset)
    return out
