from __future__ import annotations

"""天池 DRAM 网格样本构建。

支持两种网格表示（`data.grid_representation`）：

- **`spatial_heatmap`**（天池 132973 / 2D CNN 报告默认）：按 row/col **分箱**到固定 H×W（如 128×32），
  **bankid**（0–15）作为通道维累积 CE 计数，`log1p` 后可与 inventory one-hot **按通道拼接到同一网格**
  （可选 `inventory_as_grid_channels`）；CE 类型默认仅保留 Read/Scrub（error_type 1、2）。
- **`legacy_rc_et`**：原始 row/col 直接作为网格索引 + error_type 分通道，用于小网格调试与单元测试。

数据流：
1. **索引表**：每行 `(sid, predict_time)` → `predict_ts`（相对秒，全局最早 raw 时刻 = 0）。
2. **标签** `merge_ticket_labels`：`predict_ts <= failed_ts <= predict_ts + label_horizon_seconds`（字段别名 `label_backtrace_seconds`）。
3. **窗口** `window_slice`：半开区间 `[predict_ts - window_seconds, predict_ts)`。
4. **inventory**：见 `inventory_encode`；若拼入网格则通常不写单独 `tabular.npy`（`model.tabular_dim=0`）。

产物：`X.npy`、`y.npy`、可选 `tabular.npy`、`meta.csv`、`preprocess_manifest.json`。"""

import gc
import json
import multiprocessing
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dram2cnn.data.inventory_encode import (
    TABULAR_DIM,
    TABULAR_FEATURE_NAMES,
    load_inventory_by_sid,
    stack_tabular_for_index,
)
from dram2cnn.data.time_axis import (
    compute_global_time_epoch_offset,
    relative_seconds_to_predict_time_str,
    resolve_index_predict_ts,
    series_to_absolute_seconds,
    to_relative_seconds_array,
)
from dram2cnn.utils.config import label_horizon_seconds_from_dict
from dram2cnn.utils.preprocess_parallelism import (
    effective_preprocess_parallel_frac_recorded,
    preprocess_workers_log_line,
    ram_budget_snapshot,
    resolve_preprocess_worker_count,
)


def normalize_sid(s: object) -> str:
    """统一 sid 字符串（去首尾空白），避免索引/mcelog/tickets 对齐失败。"""
    return str(s).strip()


# ---------- 多进程预处理（仅 Linux + fork）----------
# 子进程通过写时复制读父进程的 events / buckets，避免 spawn 下整表 pickle 到子进程。
# _WORKER_CTX：填网格时 pool.map(_fill_indices_batch) 各子进程读的全局上下文。
# _BUCKETS_RAW：mcelog 聚合后按 sid 排序阶段，各子进程读的原始 list bucket（排序前）。
_WORKER_CTX: dict[str, Any] | None = None
_BUCKETS_RAW: dict[str, dict[str, list]] | None = None


def _preprocess_fork_parallel_available() -> bool:
    # Windows/macOS 默认不用 fork Pool 填网格，避免复制大对象与语义差异。
    return sys.platform.startswith("linux")


def _fill_indices_batch(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """fork Pool 子任务：处理一批全局样本行号 indices，返回 (indices, X 块, 空窗数, 空窗行号)。

    父进程在 pool 前设置 _WORKER_CTX；每个子进程只读 events/tabular，写本地 out 再拼回主数组。
    """
    ctx = _WORKER_CTX
    if ctx is None:
        raise RuntimeError(
            "preprocess parallel worker: context not initialized")
    mode = str(ctx.get("mode", "legacy"))
    events = ctx["events"]
    win = float(ctx["win"])
    gh = int(ctx["grid_h"])
    gw = int(ctx["grid_w"])
    inch = int(ctx["in_ch"])
    sids = ctx["sids"]
    pts = ctx["pts"]
    out = np.zeros((len(indices), inch, gh, gw), dtype=np.float32)
    n_skip = 0
    skipped: list[int] = []
    for j, ii in enumerate(indices):
        i = int(ii)
        sid = normalize_sid(sids[i])
        pt = float(pts[i])
        sl = window_slice(events, sid, pt, win)
        if sl is None:
            n_skip += 1
            skipped.append(i)
            continue
        if mode == "spatial":
            r, c, _e, _t, bk = sl
            row_bs = int(ctx["row_bin_size"])
            col_bs = int(ctx["col_bin_size"])
            nb = int(ctx["num_bank_channels"])
            base = fill_spatial_bank_grid(r, c, bk, gh, gw, nb, row_bs, col_bs)
            if bool(ctx["inventory_as_grid_channels"]):
                tab_rows = ctx["tabular_rows"]
                vec = tab_rows[i]
                out[j] = concat_spatial_and_inventory_channels(base, vec)
            else:
                out[j] = base
        else:
            r, c, e, _t, _bk = sl
            leg_in = int(ctx.get("legacy_in_channels", 3))
            out[j] = fill_log1p_grid(r, c, e, gh, gw, leg_in)
    return indices.astype(np.int64), out, n_skip, np.asarray(skipped, dtype=np.int64)


def _finalize_sid_chunk(sids: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    """对一批 sid 的 list bucket 做时间排序并转为 numpy 数组，供 window_slice 二分查找。

    输入为 sid 元组（一批任务）；输出 dict 可 merge 进总 events。
    """
    global _BUCKETS_RAW
    b = _BUCKETS_RAW
    if b is None:
        raise RuntimeError("preprocess finalize worker: buckets not set")
    out: dict[str, dict[str, np.ndarray]] = {}
    for sid in sids:
        d = b.get(sid)
        if not d:
            continue
        t = np.asarray(d["t"], dtype=np.float64)
        if t.size == 0:
            continue
        order = np.argsort(t, kind="mergesort")
        out[sid] = {
            "t": t[order],
            "row": np.asarray(d["r"], dtype=np.int32)[order],
            "col": np.asarray(d["c"], dtype=np.int32)[order],
            "etype": np.asarray(d["e"], dtype=np.int8)[order],
            "bank": np.asarray(d["b"], dtype=np.int8)[order],
        }
    return out


def failure_in_horizon(
    predict_ts: float,
    fail_times: np.ndarray | None,
    horizon: float,
) -> bool:
    """锚点 ``predict_ts`` 的闭区间 ``[predict_ts, predict_ts + horizon]`` 内是否存在工单失败时刻。"""
    if fail_times is None or fail_times.size == 0:
        return False
    lo, hi = float(predict_ts), float(predict_ts) + float(horizon)
    j = int(np.searchsorted(fail_times, lo, side="left"))
    return j < len(fail_times) and float(fail_times[j]) <= hi


def load_ticket_failures(
    tickets_path: Path,
    failure_types: list[int],
    time_epoch_offset: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """读取工单表，按 sid 聚合升序 ``failed_ts``（**相对秒**）。"""
    if not tickets_path.is_file():
        raise FileNotFoundError(f"tickets 文件不存在: {tickets_path}")
    try:
        tix = pd.read_csv(tickets_path, low_memory=False)
    except pd.errors.ParserError as exc:
        raise RuntimeError(f"无法解析 tickets CSV {tickets_path}: {exc}") from exc
    for col in ("sid", "failure_type", "failed_time"):
        if col not in tix.columns:
            raise ValueError(f"tickets 需含 `{col}`，当前: {list(tix.columns)}")
    ft_set = set(int(x) for x in failure_types)
    tix = tix[tix["failure_type"].isin(ft_set)].copy()
    tix["sid"] = tix["sid"].astype(str)
    tix["failed_ts"] = series_to_absolute_seconds(
        tix["failed_time"]) - float(time_epoch_offset)
    tix = tix.dropna(subset=["failed_ts"])
    by_sid: dict[str, np.ndarray] = {}
    for sid, g in tix.groupby("sid"):
        by_sid[str(sid)] = np.sort(g["failed_ts"].to_numpy(dtype=np.float64))
    return tix, by_sid


def scan_mcelog_rowcol_max(
    mcelog_path: Path,
    chunk_rows: int,
    max_rows: int | None,
    *,
    spatial_heatmap: bool = False,
    mcelog_error_type_filter: list[int] | None = None,
) -> tuple[int, int]:
    """轻量首遍扫描：只读 row/col/error_type，得到全局 max row/col。

    legacy 用于与 model 网格取 min；spatial 用于提示物理地址是否超出分箱 clip 范围。
    """
    max_r, max_c = 0, 0
    usecols = ["row", "col", "error_type"]
    seen = 0
    cap = int(max_rows) if max_rows is not None and int(max_rows) > 0 else None
    for chunk in pd.read_csv(mcelog_path, chunksize=int(chunk_rows), usecols=usecols, low_memory=False):
        if spatial_heatmap:
            ft = mcelog_error_type_filter if mcelog_error_type_filter is not None else [
                1, 2]
            chunk = chunk[chunk["error_type"].isin(ft)].copy()
        else:
            chunk = chunk[chunk["error_type"] < 3]
        if chunk.empty:
            continue
        max_r = max(max_r, int(chunk["row"].max()))
        max_c = max(max_c, int(chunk["col"].max()))
        seen += len(chunk)
        if cap is not None and seen >= cap:
            break
    return max_r, max_c


def _mcelog_read_columns(mcelog_path: Path) -> tuple[list[str], bool]:
    """只读表头决定列集合：有 bankid 则 spatial 路径需要分箱+银行通道。"""
    hdr = pd.read_csv(mcelog_path, nrows=0).columns.tolist()
    has_bank = "bankid" in hdr
    if has_bank:
        return ["sid", "row", "col", "bankid", "error_type", "error_time"], True
    return ["sid", "row", "col", "error_type", "error_time"], False


def build_events_by_sid(
    mcelog_path: Path,
    chunk_rows: int,
    max_rows: int | None = None,
    *,
    finalize_workers: int = 1,
    spatial_heatmap: bool = False,
    mcelog_error_type_filter: list[int] | None = None,
    time_epoch_offset: float = 0.0,
) -> tuple[dict[str, dict[str, np.ndarray]], bool]:
    """分块读取 mcelog，按 sid 聚合后按时间排序。

    spatial_heatmap=True 时默认仅保留 error_type∈{1,2}（Read/Scrub CE），并读取 bankid（缺失则全 0）。
    """
    usecols, has_bank = _mcelog_read_columns(mcelog_path)

    def _new_bkt() -> dict[str, list]:
        return {"t": [], "r": [], "c": [], "e": [], "b": []}

    # 阶段 A：单线程顺序 read_csv + Python 循环 append（无法用 fork 加速，否则需分片合并 bucket）
    buckets: dict[str, dict[str, list]] = defaultdict(_new_bkt)

    seen = 0
    cap = int(max_rows) if max_rows is not None and int(max_rows) > 0 else None
    for chunk in pd.read_csv(mcelog_path, chunksize=int(chunk_rows), usecols=usecols, low_memory=False):
        if cap is not None and seen >= cap:
            break
        if spatial_heatmap:
            ft = mcelog_error_type_filter if mcelog_error_type_filter is not None else [
                1, 2]
            chunk = chunk[chunk["error_type"].isin(list(ft))].copy()
        else:
            chunk = chunk[chunk["error_type"] < 3].copy()
        if chunk.empty:
            continue
        if cap is not None:
            room = cap - seen
            if room <= 0:
                break
            if len(chunk) > room:
                chunk = chunk.iloc[:room]
        tsec = to_relative_seconds_array(series_to_absolute_seconds(
            chunk["error_time"]), time_epoch_offset)
        rows = chunk["row"].to_numpy(dtype=np.int32, copy=False)
        cols = chunk["col"].to_numpy(dtype=np.int32, copy=False)
        et = chunk["error_type"].to_numpy(dtype=np.int8, copy=False)
        sids = chunk["sid"].map(normalize_sid).to_numpy(copy=False)
        if has_bank:
            banks = np.clip(chunk["bankid"].to_numpy(
                dtype=np.int32, copy=False), 0, 15)
        else:
            banks = np.zeros(len(chunk), dtype=np.int32)
        for i in range(len(chunk)):
            sid = normalize_sid(sids[i])
            if np.isnan(tsec[i]):
                continue
            buckets[sid]["t"].append(float(tsec[i]))
            buckets[sid]["r"].append(int(rows[i]))
            buckets[sid]["c"].append(int(cols[i]))
            buckets[sid]["e"].append(int(et[i]))
            buckets[sid]["b"].append(int(banks[i]))
        seen += len(chunk)

    # 阶段 B：每个 sid 内按时间排序；sid 足够多时在 Linux 上用 fork Pool 并行 argsort
    out: dict[str, dict[str, np.ndarray]] = {}
    events_finalize_parallel = False
    n_sid = len(buckets)
    min_sid_mp = max(32, finalize_workers)
    use_finalize_mp = (
        _preprocess_fork_parallel_available()
        and finalize_workers > 1
        and n_sid >= min_sid_mp
    )
    if use_finalize_mp:
        try:
            multiprocessing.get_context("fork")
        except ValueError:
            use_finalize_mp = False

    if use_finalize_mp:
        keys = list(buckets.keys())
        k = min(finalize_workers, max(1, len(keys)))
        chunk_sz = max(1, (len(keys) + k - 1) // k)
        batches: list[tuple[str, ...]] = []
        for i in range(0, len(keys), chunk_sz):
            batches.append(tuple(keys[i: i + chunk_sz]))
        print(
            f"Sorting mcelog buckets by time (fork Pool, {len(batches)} tasks, ~{finalize_workers} workers budget) …",
            flush=True,
        )
        global _BUCKETS_RAW
        _BUCKETS_RAW = buckets
        try:
            ctx_mp = multiprocessing.get_context("fork")
            with ctx_mp.Pool(processes=len(batches)) as pool:
                for part in pool.map(_finalize_sid_chunk, batches):
                    out.update(part)
            events_finalize_parallel = True
        finally:
            _BUCKETS_RAW = None
    else:
        if (
            _preprocess_fork_parallel_available()
            and finalize_workers > 1
            and n_sid > 0
            and n_sid < min_sid_mp
        ):
            print(
                f"提示：distinct sid 数={n_sid}（<{min_sid_mp}），mcelog 按 sid 排序保持单进程。",
                flush=True,
            )
        for sid, d in buckets.items():
            t = np.asarray(d["t"], dtype=np.float64)
            if t.size == 0:
                continue
            order = np.argsort(t, kind="mergesort")
            out[sid] = {
                "t": t[order],
                "row": np.asarray(d["r"], dtype=np.int32)[order],
                "col": np.asarray(d["c"], dtype=np.int32)[order],
                "etype": np.asarray(d["e"], dtype=np.int8)[order],
                "bank": np.asarray(d["b"], dtype=np.int8)[order],
            }
    return out, events_finalize_parallel


def load_index_table(
    path: Path,
    max_rows: int | None,
    time_epoch_offset: float | None = None,
) -> pd.DataFrame:
    """读索引 CSV，解析 ``predict_time`` 并写入 ``predict_ts``（**相对秒**，全局最早时刻 = 0）。

    本仓库 ``build_index`` 产物以 1970-01-01 起算的 ISO 字符串存储，解析后即为相对秒。
    遗留「公元 0001」等低年份索引须传入 ``time_epoch_offset``（与 preprocess 扫描的 ``t₀`` 一致）。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少索引特征表 {path}。请放置天池流水线导出的 CSV（至少含 sid, predict_time 列）。"
        )
    df = pd.read_csv(path, low_memory=False)
    for col in ("sid", "predict_time"):
        if col not in df.columns:
            raise ValueError(f"索引表必须包含列 `{col}`，当前: {list(df.columns)}")
    df["sid"] = df["sid"].map(normalize_sid)
    pts = pd.to_datetime(df["predict_time"], errors="coerce", format="mixed")
    if pts.isna().any():
        bad = int(pts.isna().sum())
        raise ValueError(f"predict_time 存在 {bad} 行无法解析为时间")
    rel_ts = resolve_index_predict_ts(
        df["predict_time"],
        time_epoch_offset=time_epoch_offset,
        index_path=path,
    )
    df["predict_ts"] = rel_ts
    if max_rows is not None and int(max_rows) > 0 and len(df) > int(max_rows):
        n_keep = int(max_rows)
        df_sorted = df.sort_values(
            "predict_ts", kind="mergesort").reset_index(drop=True)
        if len(df_sorted) > n_keep:
            warnings.warn(
                f"索引表 {path.name} 共 {len(df_sorted)} 行，max_index_rows={n_keep} "
                "将按时间轴均匀子采样（非仅保留最早段）。",
                UserWarning,
                stacklevel=2,
            )
            pick = np.linspace(0, len(df_sorted) - 1, n_keep, dtype=int)
            df = df_sorted.iloc[pick].reset_index(drop=True)
        else:
            df = df_sorted
    df["predict_time"] = [relative_seconds_to_predict_time_str(
        float(t)) for t in df["predict_ts"]]
    return df


def merge_ticket_labels(
    index_df: pd.DataFrame,
    tickets_path: Path,
    failure_types: list[int],
    label_horizon_seconds: float,
    time_epoch_offset: float = 0.0,
    *,
    by_sid_fail: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """按工单规则打标：存在工单满足 predict_ts <= failed_ts <= predict_ts + label_horizon_seconds。

    ``index_df.predict_ts`` 与工单 ``failed_ts`` 均为**相对秒**（须与 mcelog 事件时间同一 ``time_epoch_offset``）。
    传入 ``by_sid_fail`` 时可跳过重复读 tickets CSV（索引构建阶段复用）。
    """
    if by_sid_fail is None:
        _, by_sid_fail = load_ticket_failures(
            tickets_path, failure_types, time_epoch_offset)

    y = np.zeros(len(index_df), dtype=np.int64)
    bt = float(label_horizon_seconds)
    sids = index_df["sid"].astype(str).to_numpy()
    pts_arr = index_df["predict_ts"].to_numpy(dtype=np.float64)
    for sid in np.unique(sids):
        fail_times = by_sid_fail.get(str(sid))
        if fail_times is None or fail_times.size == 0:
            continue
        m = sids == sid
        pts = pts_arr[m]
        pos = np.flatnonzero(m)
        for ft in fail_times:
            hit = (pts <= float(ft)) & (pts + bt >= float(ft))
            y[pos[hit]] = 1
    return y


def window_slice(
    events: dict[str, np.ndarray] | None,
    sid: str,
    t_end: float,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """半开窗口 [t_end - window_sec, t_end) 内的事件切片；t 已按时间递增故用 searchsorted 定起止下标。"""
    if events is None or sid not in events:
        return None
    ev = events[sid]
    t = ev["t"]
    start = float(t_end) - float(window_sec)
    end = float(t_end)
    i0 = int(np.searchsorted(t, start, side="left"))
    i1 = int(np.searchsorted(t, end, side="left"))
    if i0 >= i1:
        return None
    bank = ev["bank"][i0:i1]
    return ev["row"][i0:i1], ev["col"][i0:i1], ev["etype"][i0:i1], t[i0:i1], bank


def fill_log1p_grid(
    rows: np.ndarray,
    cols: np.ndarray,
    etypes: np.ndarray,
    grid_h: int,
    grid_w: int,
    in_channels: int = 3,
) -> np.ndarray:
    """legacy：在固定 H×W 上按 (row,col) 累加计数，再 log1p 堆成 [C,H,W]。

    read(1)→ch0，scrub(2)→ch1；in_channels>=3 时 error_type==0→ch2。
    """
    gh, gw = int(grid_h), int(grid_w)
    acc0 = np.zeros((gh, gw), dtype=np.float32)
    acc1 = np.zeros((gh, gw), dtype=np.float32)
    acc2 = np.zeros((gh, gw), dtype=np.float32)
    if rows.shape[0] > 0:
        r = rows.astype(np.int64, copy=False)
        c = cols.astype(np.int64, copy=False)
        et = etypes.astype(np.int64, copy=False)
        valid = (r >= 0) & (c >= 0) & (r < gh) & (c < gw)
        r, c, et = r[valid], c[valid], et[valid]
        if r.size:
            m1 = et == 1
            if m1.any():
                np.add.at(acc0, (r[m1], c[m1]), 1.0)
            m2 = et == 2
            if m2.any():
                np.add.at(acc1, (r[m2], c[m2]), 1.0)
            if int(in_channels) >= 3:
                m0 = et == 0
                if m0.any():
                    np.add.at(acc2, (r[m0], c[m0]), 1.0)
    if int(in_channels) == 2:
        return np.stack([np.log1p(acc0), np.log1p(acc1)], axis=0)
    if int(in_channels) == 3:
        return np.stack([np.log1p(acc0), np.log1p(acc1), np.log1p(acc2)], axis=0)
    raise ValueError(f"in_channels={in_channels} 仅支持 2 或 3")


def fill_spatial_bank_grid(
    rows: np.ndarray,
    cols: np.ndarray,
    banks: np.ndarray,
    grid_h: int,
    grid_w: int,
    n_banks: int,
    row_bin_size: int,
    col_bin_size: int,
) -> np.ndarray:
    """spatial：物理 row/col 映射到 bin (rb,cb)，再按 bank 维用 np.add.at 累加，最后 log1p。"""
    gh, gw = int(grid_h), int(grid_w)
    nb = max(1, int(n_banks))
    acc = np.zeros((nb, gh, gw), dtype=np.float64)
    if rows.shape[0] == 0:
        return np.zeros((nb, gh, gw), dtype=np.float32)
    r = rows.astype(np.int64, copy=False)
    c = cols.astype(np.int64, copy=False)
    bk = banks.astype(np.int64, copy=False)
    rb = (r // int(row_bin_size)).clip(0, gh - 1)
    cb = (c // int(col_bin_size)).clip(0, gw - 1)
    bi = bk.clip(0, nb - 1)
    np.add.at(acc, (bi, rb, cb), 1.0)
    return np.log1p(acc.astype(np.float32))


def concat_spatial_and_inventory_channels(spatial_banks: np.ndarray, inv_vec: np.ndarray) -> np.ndarray:
    """`spatial_banks` [B,H,W] + inventory one-hot 逐通道平铺 [F,H,W] → [B+F,H,W]。"""
    _, h, w = spatial_banks.shape
    f = int(inv_vec.shape[0])
    tiled = np.broadcast_to(inv_vec.reshape(
        f, 1, 1).astype(np.float32, copy=False), (f, h, w))
    return np.concatenate([spatial_banks.astype(np.float32, copy=False), tiled], axis=0)


def _validate_index_sidecar_against_config(
    idx_stats: dict[str, Any],
    data_cfg: dict[str, Any],
    *,
    mcelog_cap: int | None,
    time_epoch_offset: float,
) -> None:
    """校验索引 sidecar 与当前 preprocess 配置一致（时间轴、截断行数、W/H）。"""
    idx_off = idx_stats.get("time_epoch_offset_absolute")
    if idx_off is not None and abs(float(idx_off) - time_epoch_offset) > 1.0:
        raise ValueError(
            f"索引 sidecar time_epoch_offset_absolute={float(idx_off):.3f} "
            f"与 preprocess 扫描值 {time_epoch_offset:.3f} 不一致；"
            "请重建索引（bash scripts/build_index.sh smoke）并清空 data/processed、data/splits。"
        )

    idx_scan = idx_stats.get("mcelog_scan_rows")
    if idx_scan is None:
        idx_scan = (idx_stats.get("config") or {}).get("mcelog_scan_rows")
    if mcelog_cap is not None and idx_scan is not None and int(idx_scan) != int(mcelog_cap):
        raise ValueError(
            f"索引 sidecar mcelog_scan_rows={idx_scan} 与当前 preprocess mcelog_max_rows={mcelog_cap} "
            "不一致；请用相同截断行数重建索引（bash scripts/build_index.sh smoke）并清空 processed/splits。"
        )

    cfg_hor = label_horizon_seconds_from_dict(data_cfg)
    idx_hor = idx_stats.get("label_horizon_seconds")
    if idx_hor is None:
        idx_hor = (idx_stats.get("config") or {}).get("label_horizon_seconds")
    if idx_hor is not None and abs(float(idx_hor) - cfg_hor) > 1e-6:
        raise ValueError(
            f"索引 sidecar label_horizon_seconds={idx_hor} 与当前配置 {cfg_hor} 不一致；"
            "切换 horizon 实验须重建索引（build_horizon_indexes.sh）并清空 processed/splits。"
        )

    cfg_win = float(data_cfg.get("window_seconds", cfg_hor))
    idx_win = idx_stats.get("window_seconds")
    if idx_win is None:
        idx_win = (idx_stats.get("config") or {}).get("window_seconds")
    if idx_win is not None and abs(float(idx_win) - cfg_win) > 1e-6:
        raise ValueError(
            f"索引 sidecar window_seconds={idx_win} 与当前配置 {cfg_win} 不一致；"
            "请重建与 window 匹配的索引并清空 processed/splits。"
        )


def materialize_dataset(
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    processed_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """从 raw + 索引表物化全量池：写 processed/，不切 train/val/test（由 run_preprocess 调 splitter）。

    主干：扫 mcelog → events → 对每条索引行 window_slice → 填 X；y 由工单规则 merge_ticket_labels。
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    mcelog_path = Path(data_cfg["mcelog_path"])
    chunk = int(data_cfg.get("mcelog_chunk_rows", 1_000_000))
    win = float(data_cfg.get("window_seconds", 300))
    max_rows = data_cfg.get("max_index_rows")
    mcelog_cap = data_cfg.get("mcelog_max_rows")

    rep = str(data_cfg.get("grid_representation",
              "spatial_heatmap")).lower().strip()
    spatial = rep in {"spatial_heatmap", "heatmap", "spatial", "tianchi132973"}

    ft_filter_cfg = data_cfg.get("mcelog_error_type_filter")
    ft_filter: list[int] | None = None
    if ft_filter_cfg is not None:
        ft_filter = [int(x) for x in ft_filter_cfg]

    # spatial 下常把 inventory one-hot 拼进 X 通道，tabular.npy 可不写（tabular_dim=0）
    inv_as_grid = bool(data_cfg.get("inventory_as_grid_channels", True))

    print("Scanning mcelog for row/col max …", flush=True)
    max_r, max_c = scan_mcelog_rowcol_max(
        mcelog_path,
        chunk,
        mcelog_cap,
        spatial_heatmap=spatial,
        mcelog_error_type_filter=ft_filter,
    )

    if spatial:
        # 逻辑 row/col → 分箱索引；网格尺寸来自 model，通道数 = bank 数 ± inventory 拼入
        row_bs = int(data_cfg.get("row_bin_size", 1024))
        col_bs = int(data_cfg.get("col_bin_size", 32))
        grid_h = int(model_cfg.get("grid_height", 128))
        grid_w = int(model_cfg.get("grid_width", 32))
        n_banks = int(data_cfg.get("num_bank_channels", 16))
        extra_ch = TABULAR_DIM if inv_as_grid else 0
        expect_in_ch = n_banks + extra_ch
        cfg_in = int(model_cfg.get("in_channels", expect_in_ch))
        if cfg_in != expect_in_ch:
            raise ValueError(
                f"spatial_heatmap: 期望 in_channels={expect_in_ch}（={n_banks} banks + {extra_ch} inventory），"
                f"但 model.in_channels={cfg_in}"
            )
        in_ch = expect_in_ch
        cap_phys_h = row_bs * grid_h
        cap_phys_w = col_bs * grid_w
        if max_r >= cap_phys_h or max_c >= cap_phys_w:
            print(
                f"提示：mcelog 最大 row/col=({max_r},{max_c})，超出分箱覆盖的物理范围 "
                f"[0,{cap_phys_h})×[0,{cap_phys_w}) 的事件将被 clip 到网格边界。",
                flush=True,
            )
    else:
        # legacy：原始 row/col 直接当网格下标，网格尺寸为 data 与 model 上限的 min
        cap_r = int(data_cfg.get("grid_max_row", max_r)) + 1
        cap_c = int(data_cfg.get("grid_max_col", max_c)) + 1
        model_h = int(model_cfg.get("grid_height", cap_r))
        model_w = int(model_cfg.get("grid_width", cap_c))
        grid_h = min(cap_r, model_h)
        grid_w = min(cap_c, model_w)
        in_ch = int(model_cfg.get("in_channels", 3))
        row_bs, col_bs, n_banks = 1, 1, 0

    print(
        f"Grid representation={rep!r}, (H,W)=({grid_h},{grid_w}), channels={in_ch}", flush=True)

    index_path = Path(data_cfg["index_features_path"])
    print("Computing global time epoch offset (earliest raw timestamp → relative t=0) …", flush=True)
    time_epoch_offset = compute_global_time_epoch_offset(
        mcelog_path,
        Path(data_cfg["tickets_path"]),
        index_path=index_path,
        mcelog_chunk_rows=chunk,
        mcelog_max_rows=mcelog_cap,
    )
    print(f"  time_epoch_offset_absolute={time_epoch_offset:.3f}", flush=True)

    stats_path = index_path.parent / (index_path.name + ".stats.json")
    if stats_path.is_file():
        try:
            idx_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            _validate_index_sidecar_against_config(
                idx_stats,
                data_cfg,
                mcelog_cap=int(mcelog_cap) if mcelog_cap is not None else None,
                time_epoch_offset=time_epoch_offset,
            )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            warnings.warn(
                f"无法解析索引 sidecar {stats_path}: {exc}；跳过 sidecar 校验。"
                "建议重建索引（bash scripts/build_index.sh smoke）。",
                UserWarning,
                stacklevel=2,
            )
        except ValueError:
            raise

    print("Loading index CSV …", flush=True)
    idx_df = load_index_table(index_path, max_rows,
                              time_epoch_offset=time_epoch_offset)
    bt_used = label_horizon_seconds_from_dict(data_cfg)
    y = merge_ticket_labels(
        idx_df,
        Path(data_cfg["tickets_path"]),
        list(data_cfg.get("label_failure_types", [1, 2, 3])),
        bt_used,
        time_epoch_offset=time_epoch_offset,
    )

    inv_path = data_cfg.get("inventory_path")
    if inv_path:
        print("Loading inventory one-hot …", flush=True)
        inv_by_sid = load_inventory_by_sid(Path(inv_path))
        tab = stack_tabular_for_index(idx_df["sid"].to_numpy(), inv_by_sid)
    else:
        tab = np.zeros((len(idx_df), TABULAR_DIM), dtype=np.float32)

    n = len(idx_df)
    nw_cfg = resolve_preprocess_worker_count(
        data_cfg,
        n_samples=n,
        grid_shape=(in_ch, grid_h, grid_w),
        mcelog_path=mcelog_path,
        mcelog_max_rows=int(mcelog_cap) if mcelog_cap is not None else None,
    )
    print(
        preprocess_workers_log_line(
            data_cfg,
            nw_cfg,
            n_samples=n,
            grid_shape=(in_ch, grid_h, grid_w),
            mcelog_path=mcelog_path,
            mcelog_max_rows=int(mcelog_cap) if mcelog_cap is not None else None,
        ),
        flush=True,
    )

    print(
        "Building mcelog time index（顺序流式读 CSV + 按 sid 聚合；排序/填网格阶段见上并行度）…",
        flush=True,
    )
    events, events_finalize_parallel = build_events_by_sid(
        mcelog_path,
        chunk,
        mcelog_cap,
        finalize_workers=nw_cfg,
        spatial_heatmap=spatial,
        mcelog_error_type_filter=ft_filter,
        time_epoch_offset=time_epoch_offset,
    )

    # 按索引行填 X：样本数 n = 索引表行数（与 mcelog 行数无关）
    X = np.zeros((n, in_ch, grid_h, grid_w), dtype=np.float32)
    sids_arr = idx_df["sid"].astype(str).to_numpy()
    pts_arr = idx_df["predict_ts"].to_numpy(dtype=np.float64)
    # 填网格并行：样本太少时进程开销不划算；非 Linux 关闭
    min_samples_mp = max(16, nw_cfg * 2)
    use_mp = _preprocess_fork_parallel_available() and nw_cfg > 1 and n >= min_samples_mp
    if use_mp:
        try:
            multiprocessing.get_context("fork")
        except ValueError:
            use_mp = False

    grid_fill_parallel_done = False
    n_empty_grids = 0
    empty_row_indices: list[int] = []
    global _WORKER_CTX
    leg_in = int(model_cfg.get("in_channels", 3))

    if use_mp:
        k = min(nw_cfg, max(1, n))
        batches = [b for b in np.array_split(
            np.arange(n, dtype=np.int64), k) if len(b) > 0]
        print(
            f"Filling grids with fork Pool ({len(batches)} tasks, ~{nw_cfg} workers budget, Linux COW) …",
            flush=True,
        )
        ctx: dict[str, Any] = {
            "events": events,
            "win": win,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "in_ch": in_ch,
            "sids": sids_arr,
            "pts": pts_arr,
        }
        if spatial:
            ctx["mode"] = "spatial"
            ctx["row_bin_size"] = row_bs
            ctx["col_bin_size"] = col_bs
            ctx["num_bank_channels"] = n_banks
            ctx["inventory_as_grid_channels"] = inv_as_grid
            ctx["tabular_rows"] = tab
        else:
            ctx["mode"] = "legacy"
            ctx["legacy_in_channels"] = leg_in
        _WORKER_CTX = ctx
        try:
            ctx_mp = multiprocessing.get_context("fork")
            with ctx_mp.Pool(processes=len(batches)) as pool:
                for inds, block, skip, skipped in pool.map(_fill_indices_batch, batches):
                    X[inds] = block
                    n_empty_grids += int(skip)
                    if skipped.size > 0:
                        empty_row_indices.extend(int(x) for x in skipped)
        finally:
            _WORKER_CTX = None
        grid_fill_parallel_done = True
    else:
        if nw_cfg > 1:
            if not _preprocess_fork_parallel_available():
                print(
                    "提示：当前 OS 非 Linux，预处理网格填充保持单进程（避免 spawn 复制巨型 events）。",
                    flush=True,
                )
            elif n < min_samples_mp:
                print(
                    f"提示：样本数 n={n} 较小（<{min_samples_mp}），预处理网格填充保持单进程以降低进程开销。",
                    flush=True,
                )
        # 单进程：按 sid 分组再按 predict_ts 排序遍历，顺序与多进程分批填充语义一致，利于复现
        sid_to_rows: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            sid_to_rows[normalize_sid(sids_arr[i])].append(i)
        for _sid, inds in sid_to_rows.items():
            for i in sorted(inds, key=lambda ii: float(pts_arr[ii])):
                sid = normalize_sid(sids_arr[i])
                pts = float(pts_arr[i])
                sl = window_slice(events, sid, pts, win)
                if sl is None:
                    n_empty_grids += 1
                    empty_row_indices.append(i)
                    continue
                if spatial:
                    r, c, _e, _t, bk = sl
                    base = fill_spatial_bank_grid(
                        r, c, bk, grid_h, grid_w, n_banks, row_bs, col_bs
                    )
                    if inv_as_grid:
                        X[i] = concat_spatial_and_inventory_channels(
                            base, tab[i])
                    else:
                        X[i] = base
                else:
                    r, c, e, _t, _bk = sl
                    X[i] = fill_log1p_grid(r, c, e, grid_h, grid_w, leg_in)

    if n_empty_grids > 0:
        print(
            f"警告：{n_empty_grids}/{n} 条样本在 mcelog 窗口内无事件（sid 缺失或窗口为空），对应 X 行为全零。",
            flush=True,
        )

    # 填网格完成后 events 不再使用；尽早释放以降低 drop/save 阶段峰值。
    del events
    gc.collect()

    drop_empty = bool(data_cfg.get("drop_empty_mcelog_windows", True))
    drop_empty_pos = bool(data_cfg.get("drop_empty_positive_windows", False))
    n_before_drop = int(n)
    n_dropped_empty = 0
    n_empty_kept_positive = 0
    n_dropped_empty_positive = 0
    # 空窗 = mcelog 特征窗 [predict_ts−W, predict_ts) 内无事件。
    # 默认仅丢负样本空窗；工程模式 drop_empty_positive_windows=true 时正样本空窗也丢弃（纯 mcelog 可学样本）。
    if drop_empty and empty_row_indices:
        empty_set = set(int(i) for i in empty_row_indices)
        keep = np.ones(n, dtype=bool)
        for i in empty_set:
            if int(y[i]) == 0:
                keep[i] = False
            elif drop_empty_pos:
                keep[i] = False
                n_dropped_empty_positive += 1
            else:
                n_empty_kept_positive += 1
        n_dropped_empty = int((~keep).sum())
        if n_dropped_empty > 0 or n_empty_kept_positive > 0 or n_dropped_empty_positive > 0:
            X = X[keep]
            y = y[keep]
            idx_df = idx_df.loc[keep].reset_index(drop=True)
            tab = tab[keep]
            n = int(keep.sum())
            msg = f"已丢弃 {n_dropped_empty}/{n_before_drop} 条空 mcelog 窗口样本"
            if drop_empty_pos:
                msg += f"（含正样本空窗 {n_dropped_empty_positive} 条；drop_empty_positive_windows=true）"
            else:
                msg += f"（负样本空窗；正样本空窗保留 {n_empty_kept_positive} 条）"
            print(msg + "。", flush=True)
    elif not drop_empty and n_empty_grids > 0:
        print(
            "提示：drop_empty_mcelog_windows=false，空窗样本仍保留（X 对应 mcelog 通道为全零）。",
            flush=True,
        )

    if n == 0:
        raise ValueError(
            "预处理后无剩余样本（可能 drop_empty_mcelog_windows / drop_empty_positive_windows "
            "删除了全部行，或索引与 mcelog 无交集）。请检查索引路径、mcelog 覆盖与空窗策略。"
        )

    # 逐样本标记：特征窗 [predict_ts−W, predict_ts) 内是否无 mcelog 事件（供误差分析 / FN 归因）
    empty_mcelog_flags = np.zeros(n_before_drop, dtype=np.int8)
    if empty_row_indices:
        empty_mcelog_flags[np.asarray(empty_row_indices, dtype=np.int64)] = 1
    if drop_empty and empty_row_indices:
        empty_mcelog_flags = empty_mcelog_flags[keep]

    meta = idx_df[["sid", "predict_time"]].copy()
    meta["label"] = y.astype(np.int64)
    meta["predict_ts"] = idx_df["predict_ts"].astype(np.float64)
    meta["empty_mcelog_window"] = empty_mcelog_flags.astype(np.int64)

    np.save(processed_dir / "X.npy", X)
    np.save(processed_dir / "y.npy", y)
    # spatial 且 inventory 已进 X 通道时不写 tabular.npy，并删除 splits 中多余的 tabular 文件以免 validate 误判
    save_tab = not spatial or not inv_as_grid
    tab_dim_manifest = int(tab.shape[1]) if tab.ndim == 2 else 0
    if save_tab:
        np.save(processed_dir / "tabular.npy", tab)
    else:
        tab_dim_manifest = 0
        tp = processed_dir / "tabular.npy"
        if tp.is_file():
            tp.unlink()

    meta.to_csv(processed_dir / "meta.csv", index=False)

    pos_rate = float(y.mean()) if n else 0.0
    del X, tab
    gc.collect()

    ram_snap = ram_budget_snapshot(
        data_cfg,
        nw_cfg,
        n_samples=n,
        grid_shape=(in_ch, grid_h, grid_w),
        mcelog_path=mcelog_path,
        mcelog_max_rows=int(mcelog_cap) if mcelog_cap is not None else None,
    )

    # manifest：供 validate / train 校验 grid_shape、并行阶段是否启用等
    manifest: dict[str, Any] = {
        "schema": "tianchi_spatial_heatmap" if spatial else "legacy_rc_et",
        "grid_representation": rep,
        "n_samples": int(n),
        "grid_shape": [int(in_ch), int(grid_h), int(grid_w)],
        "tabular_dim": tab_dim_manifest,
        "tabular_feature_names": list(TABULAR_FEATURE_NAMES),
        "window_seconds": win,
        "label_horizon_seconds": bt_used,
        "label_backtrace_seconds": bt_used,
        "label_failure_types": list(data_cfg.get("label_failure_types", [1, 2, 3])),
        "pos_rate": pos_rate,
        "inventory_path": str(inv_path) if inv_path else None,
        "inventory_as_grid_channels": bool(spatial and inv_as_grid),
        "materialize_seed": int(seed),
        "preprocess_workers_requested": int(nw_cfg),
        "preprocess_parallel_frac": effective_preprocess_parallel_frac_recorded(data_cfg),
        "ram_total_gib": ram_snap["ram_total_gib"] if ram_snap else None,
        "ram_budget_frac": ram_snap["ram_budget_frac"] if ram_snap else None,
        "ram_budget_gib": ram_snap["ram_budget_gib"] if ram_snap else None,
        "preprocess_estimated_peak_gib": ram_snap["estimated_peak_gib"] if ram_snap else None,
        "split_normalize_estimated_peak_gib": (
            ram_snap["split_normalize_estimated_peak_gib"] if ram_snap else None
        ),
        "preprocess_ram_budget_exceeded": ram_snap["exceeds_budget"] if ram_snap else None,
        "split_normalize_ram_budget_exceeded": (
            ram_snap["split_normalize_exceeds_budget"] if ram_snap else None
        ),
        "preprocess_events_finalize_parallel": bool(events_finalize_parallel),
        "preprocess_grid_fill_parallel": grid_fill_parallel_done,
        "n_empty_grids": int(n_empty_grids),
        "n_empty_grids_dropped": int(n_dropped_empty),
        "n_empty_grids_kept_positive": int(n_empty_kept_positive),
        "n_empty_positive_dropped": int(n_dropped_empty_positive),
        "n_samples_before_drop": int(n_before_drop),
        "drop_empty_mcelog_windows": bool(drop_empty),
        "drop_empty_positive_windows": bool(drop_empty_pos),
        "index_features_path": str(Path(data_cfg["index_features_path"]).resolve()),
        "time_axis": "relative_seconds_since_global_min",
        "time_epoch_offset_absolute": float(time_epoch_offset),
        "mcelog_max_rows": int(mcelog_cap) if mcelog_cap is not None else None,
    }
    if spatial:
        manifest["row_bin_size"] = int(row_bs)
        manifest["col_bin_size"] = int(col_bs)
        manifest["num_bank_channels"] = int(n_banks)
        manifest["mcelog_error_type_filter"] = ft_filter if ft_filter is not None else [
            1, 2]

    (processed_dir / "preprocess_manifest.json").write_text(json.dumps(manifest,
                                                                       indent=2), encoding="utf-8")
    print(
        f"Preprocess done: N={n}, pos_rate={manifest['pos_rate']:.6f}", flush=True)
    return manifest
