from __future__ import annotations

"""天池 DRAM 网格样本构建。

支持两种网格表示（`data.grid_representation`）：

- **`spatial_heatmap`**（天池 132973 / 2D CNN 报告默认）：按 row/col **分箱**到固定 H×W（如 128×32），
  **bankid**（0–15）作为通道维累积 CE 计数，`log1p` 后可与 inventory one-hot **按通道拼接到同一网格**
  （可选 `inventory_as_grid_channels`）；CE 类型默认仅保留 Read/Scrub（error_type 1、2）。
- **`legacy_rc_et`**：旧版「原始 row/col 直接作为网格索引 + error_type 分通道」，用于兼容小样本调试配置。

数据流：
1. **索引表**：每行 `(sid, predict_time)` → `predict_ts`。
2. **标签** `merge_ticket_labels`：`predict_ts <= failed_ts <= predict_ts + label_backtrace_seconds`。
3. **窗口** `window_slice`：半开区间 `[predict_ts - window_seconds, predict_ts)`。
4. **inventory**：见 `inventory_encode`；若拼入网格则通常不写单独 `tabular.npy`（`model.tabular_dim=0`）。

产物：`X.npy`、`y.npy`、可选 `tabular.npy`、`meta.csv`、`preprocess_manifest.json`。"""

import json
import multiprocessing
import os
import sys
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

# Linux fork + Pool：子进程写时复制共享父进程的 events（只读），避免整表 pickle。
_WORKER_CTX: dict[str, Any] | None = None
# `build_events_by_sid` 末尾按 sid 排序时，共享原始 buckets（仅 Linux fork）。
_BUCKETS_RAW: dict[str, dict[str, list]] | None = None


def _resolve_preprocess_workers(data_cfg: dict[str, Any]) -> int:
    explicit = data_cfg.get("preprocess_workers")
    if explicit is not None:
        return max(1, int(explicit))
    frac = float(data_cfg.get("preprocess_parallel_frac", 0.75))
    cpu = os.cpu_count() or 1
    return max(1, int(cpu * frac))


def _preprocess_fork_parallel_available() -> bool:
    return sys.platform.startswith("linux")


def _fill_indices_batch(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`fork` Pool 子进程：按全局行号填充网格块（依赖模块级 `_WORKER_CTX`）。"""
    ctx = _WORKER_CTX
    if ctx is None:
        raise RuntimeError("preprocess parallel worker: context not initialized")
    mode = str(ctx.get("mode", "legacy"))
    events = ctx["events"]
    win = float(ctx["win"])
    gh = int(ctx["grid_h"])
    gw = int(ctx["grid_w"])
    inch = int(ctx["in_ch"])
    sids = ctx["sids"]
    pts = ctx["pts"]
    out = np.zeros((len(indices), inch, gh, gw), dtype=np.float32)
    for j, ii in enumerate(indices):
        i = int(ii)
        sid = str(sids[i])
        pt = float(pts[i])
        sl = window_slice(events, sid, pt, win)
        if sl is None:
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
    return indices.astype(np.int64), out


def _finalize_sid_chunk(sids: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    """fork Pool：对每个 sid 的 bucket 做 argsort（依赖 `_BUCKETS_RAW`）。"""
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


def _series_to_unix_seconds(s: pd.Series) -> np.ndarray:
    """pandas 时间列（任意 datetime64 精度）→ Unix 秒 float64；无法解析为 NaN。"""
    t = pd.to_datetime(s, errors="coerce", format="mixed")
    sec = (t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    return sec.to_numpy(dtype=np.float64)


def scan_mcelog_rowcol_max(
    mcelog_path: Path,
    chunk_rows: int,
    max_rows: int | None,
    *,
    spatial_heatmap: bool = False,
    mcelog_error_type_filter: list[int] | None = None,
) -> tuple[int, int]:
    """分块扫描 mcelog，得到 row/col 最大值（legacy 裁剪网格；spatial 模式用于校验提示）。"""
    max_r, max_c = 0, 0
    usecols = ["row", "col", "error_type"]
    seen = 0
    cap = int(max_rows) if max_rows is not None and int(max_rows) > 0 else None
    for chunk in pd.read_csv(mcelog_path, chunksize=int(chunk_rows), usecols=usecols, low_memory=False):
        if spatial_heatmap:
            ft = mcelog_error_type_filter if mcelog_error_type_filter is not None else [1, 2]
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
) -> tuple[dict[str, dict[str, np.ndarray]], bool]:
    """分块读取 mcelog，按 sid 聚合后按时间排序。

    spatial_heatmap=True 时默认仅保留 error_type∈{1,2}（Read/Scrub CE），并读取 bankid（缺失则全 0）。
    """
    usecols, has_bank = _mcelog_read_columns(mcelog_path)

    def _new_bkt() -> dict[str, list]:
        return {"t": [], "r": [], "c": [], "e": [], "b": []}

    buckets: dict[str, dict[str, list]] = defaultdict(_new_bkt)

    seen = 0
    cap = int(max_rows) if max_rows is not None and int(max_rows) > 0 else None
    for chunk in pd.read_csv(mcelog_path, chunksize=int(chunk_rows), usecols=usecols, low_memory=False):
        if cap is not None and seen >= cap:
            break
        if spatial_heatmap:
            ft = mcelog_error_type_filter if mcelog_error_type_filter is not None else [1, 2]
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
        tsec = _series_to_unix_seconds(chunk["error_time"])
        rows = chunk["row"].to_numpy(dtype=np.int32, copy=False)
        cols = chunk["col"].to_numpy(dtype=np.int32, copy=False)
        et = chunk["error_type"].to_numpy(dtype=np.int8, copy=False)
        sids = chunk["sid"].astype(str).to_numpy(copy=False)
        if has_bank:
            banks = np.clip(chunk["bankid"].to_numpy(dtype=np.int32, copy=False), 0, 15)
        else:
            banks = np.zeros(len(chunk), dtype=np.int32)
        for i in range(len(chunk)):
            sid = str(sids[i])
            if np.isnan(tsec[i]):
                continue
            buckets[sid]["t"].append(float(tsec[i]))
            buckets[sid]["r"].append(int(rows[i]))
            buckets[sid]["c"].append(int(cols[i]))
            buckets[sid]["e"].append(int(et[i]))
            buckets[sid]["b"].append(int(banks[i]))
        seen += len(chunk)

    out: dict[str, dict[str, np.ndarray]] = {}
    events_finalize_parallel = False
    n_sid = len(buckets)
    min_sid_mp = max(64, finalize_workers * 4)
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
            batches.append(tuple(keys[i : i + chunk_sz]))
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


def load_index_table(path: Path, max_rows: int | None) -> pd.DataFrame:
    """读索引 CSV，解析 `predict_time` 并写入 `predict_ts`（Unix 秒）。

    低年份时间写入文件时建议 ISO 8601（四位年），避免非 ISO 字符串被误解析；见 `docs/troubleshooting.md`。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少索引特征表 {path}。请放置天池流水线导出的 CSV（至少含 sid, predict_time 列）。"
        )
    df = pd.read_csv(path, low_memory=False)
    for col in ("sid", "predict_time"):
        if col not in df.columns:
            raise ValueError(f"索引表必须包含列 `{col}`，当前: {list(df.columns)}")
    df["sid"] = df["sid"].astype(str)
    if max_rows is not None and int(max_rows) > 0 and len(df) > int(max_rows):
        df = df.iloc[: int(max_rows)].copy()
    pts = pd.to_datetime(df["predict_time"], errors="coerce", format="mixed")
    if pts.isna().any():
        bad = int(pts.isna().sum())
        raise ValueError(f"predict_time 存在 {bad} 行无法解析为时间")
    df["predict_ts"] = _series_to_unix_seconds(df["predict_time"])
    return df


def merge_ticket_labels(
    index_df: pd.DataFrame,
    tickets_path: Path,
    failure_types: list[int],
    backtrace_sec: float,
) -> np.ndarray:
    """按天池规则打标：存在工单满足 predict_ts <= failed_ts <= predict_ts + backtrace。

    `j = searchsorted(fail_times, lo, side="left")` 给出第一个 `>= lo` 的工单下标；若该时间仍 `<= hi` 则区间内至少有一条命中。
    """
    tix = pd.read_csv(tickets_path, low_memory=False)
    for col in ("sid", "failure_type", "failed_time"):
        if col not in tix.columns:
            raise ValueError(f"tickets 需含 `{col}`，当前: {list(tix.columns)}")
    tix["sid"] = tix["sid"].astype(str)
    ft_set = set(int(x) for x in failure_types)
    tix = tix[tix["failure_type"].isin(ft_set)].copy()
    tix["failed_ts"] = _series_to_unix_seconds(tix["failed_time"])
    tix = tix.dropna(subset=["failed_ts"])

    by_sid: dict[str, np.ndarray] = {}
    for sid, g in tix.groupby("sid"):
        by_sid[str(sid)] = np.sort(g["failed_ts"].to_numpy(dtype=np.float64))

    y = np.zeros(len(index_df), dtype=np.int64)
    bt = float(backtrace_sec)
    sids = index_df["sid"].astype(str).to_numpy()
    pts_arr = index_df["predict_ts"].to_numpy(dtype=np.float64)
    for i in range(len(index_df)):
        sid = str(sids[i])
        pts = float(pts_arr[i])
        fail_times = by_sid.get(sid)
        if fail_times is None or fail_times.size == 0:
            continue
        lo, hi = pts, pts + bt
        j = int(np.searchsorted(fail_times, lo, side="left"))
        if j < len(fail_times) and fail_times[j] <= hi:
            y[i] = 1
    return y


def window_slice(
    events: dict[str, np.ndarray] | None,
    sid: str,
    t_end: float,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """返回窗口内 row/col/etype/时间与 bank（bank 缺省时 events 内仍为与 etype 对齐的数组）。"""
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
    """read(1)→ch0，scrub(2)→ch1；`in_channels>=3` 时 error_type==0 → ch2（与手工特征里计入 counter 的非 read/scrub 一致）。"""
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
    """按 row/col 分箱 × bank 通道累积计数，再对各通道 `log1p`。形状 `[n_banks, grid_h, grid_w]`。"""
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
    tiled = np.broadcast_to(inv_vec.reshape(f, 1, 1).astype(np.float32, copy=False), (f, h, w))
    return np.concatenate([spatial_banks.astype(np.float32, copy=False), tiled], axis=0)


def materialize_dataset(
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    processed_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """构建 processed/X.npy, y.npy、可选 tabular.npy、meta.csv 与 manifest。"""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    mcelog_path = Path(data_cfg["mcelog_path"])
    chunk = int(data_cfg.get("mcelog_chunk_rows", 500_000))
    win = float(data_cfg.get("window_seconds", 300))
    max_rows = data_cfg.get("max_index_rows")
    mcelog_cap = data_cfg.get("mcelog_max_rows")

    rep = str(data_cfg.get("grid_representation", "spatial_heatmap")).lower().strip()
    spatial = rep in {"spatial_heatmap", "heatmap", "spatial", "tianchi132973"}

    ft_filter_cfg = data_cfg.get("mcelog_error_type_filter")
    ft_filter: list[int] | None = None
    if ft_filter_cfg is not None:
        ft_filter = [int(x) for x in ft_filter_cfg]

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
        cap_r = int(data_cfg.get("grid_max_row", max_r)) + 1
        cap_c = int(data_cfg.get("grid_max_col", max_c)) + 1
        model_h = int(model_cfg.get("grid_height", cap_r))
        model_w = int(model_cfg.get("grid_width", cap_c))
        grid_h = min(cap_r, model_h)
        grid_w = min(cap_c, model_w)
        in_ch = int(model_cfg.get("in_channels", 3))
        row_bs, col_bs, n_banks = 1, 1, 0

    print(f"Grid representation={rep!r}, (H,W)=({grid_h},{grid_w}), channels={in_ch}", flush=True)

    print("Loading index CSV …", flush=True)
    idx_df = load_index_table(Path(data_cfg["index_features_path"]), max_rows)
    bt_used = float(data_cfg.get("label_backtrace_seconds", 86400))
    y = merge_ticket_labels(
        idx_df,
        Path(data_cfg["tickets_path"]),
        list(data_cfg.get("label_failure_types", [1, 2, 3])),
        bt_used,
    )

    inv_path = data_cfg.get("inventory_path")
    if inv_path:
        print("Loading inventory one-hot …", flush=True)
        inv_by_sid = load_inventory_by_sid(Path(inv_path))
        tab = stack_tabular_for_index(idx_df["sid"].to_numpy(), inv_by_sid)
    else:
        tab = np.zeros((len(idx_df), TABULAR_DIM), dtype=np.float32)

    nw_cfg = _resolve_preprocess_workers(data_cfg)

    print("Building mcelog time index (may take long / large RAM) …", flush=True)
    events, events_finalize_parallel = build_events_by_sid(
        mcelog_path,
        chunk,
        mcelog_cap,
        finalize_workers=nw_cfg,
        spatial_heatmap=spatial,
        mcelog_error_type_filter=ft_filter,
    )

    n = len(idx_df)
    X = np.zeros((n, in_ch, grid_h, grid_w), dtype=np.float32)
    sids_arr = idx_df["sid"].astype(str).to_numpy()
    pts_arr = idx_df["predict_ts"].to_numpy(dtype=np.float64)
    min_samples_mp = max(32, nw_cfg * 4)
    use_mp = _preprocess_fork_parallel_available() and nw_cfg > 1 and n >= min_samples_mp
    if use_mp:
        try:
            multiprocessing.get_context("fork")
        except ValueError:
            use_mp = False

    grid_fill_parallel_done = False
    global _WORKER_CTX
    leg_in = int(model_cfg.get("in_channels", 3))

    if use_mp:
        k = min(nw_cfg, max(1, n))
        batches = [b for b in np.array_split(np.arange(n, dtype=np.int64), k) if len(b) > 0]
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
                for inds, block in pool.map(_fill_indices_batch, batches):
                    X[inds] = block
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
        sid_to_rows: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            sid_to_rows[str(sids_arr[i])].append(i)
        for _sid, inds in sid_to_rows.items():
            for i in sorted(inds, key=lambda ii: float(pts_arr[ii])):
                sid = str(sids_arr[i])
                pts = float(pts_arr[i])
                sl = window_slice(events, sid, pts, win)
                if sl is None:
                    continue
                if spatial:
                    r, c, _e, _t, bk = sl
                    base = fill_spatial_bank_grid(
                        r, c, bk, grid_h, grid_w, n_banks, row_bs, col_bs
                    )
                    if inv_as_grid:
                        X[i] = concat_spatial_and_inventory_channels(base, tab[i])
                    else:
                        X[i] = base
                else:
                    r, c, e, _t, _bk = sl
                    X[i] = fill_log1p_grid(r, c, e, grid_h, grid_w, leg_in)

    meta = idx_df[["sid", "predict_time"]].copy()
    meta["label"] = y.astype(np.int64)
    meta["predict_ts"] = idx_df["predict_ts"].astype(np.float64)

    np.save(processed_dir / "X.npy", X)
    np.save(processed_dir / "y.npy", y)
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

    manifest: dict[str, Any] = {
        "schema": "tianchi_spatial_heatmap" if spatial else "tianchi_5m_grid",
        "grid_representation": rep,
        "n_samples": int(n),
        "grid_shape": [int(in_ch), int(grid_h), int(grid_w)],
        "tabular_dim": tab_dim_manifest,
        "tabular_feature_names": list(TABULAR_FEATURE_NAMES),
        "window_seconds": win,
        "label_backtrace_seconds": bt_used,
        "label_failure_types": list(data_cfg.get("label_failure_types", [])),
        "pos_rate": float(y.mean()) if n else 0.0,
        "inventory_path": str(inv_path) if inv_path else None,
        "inventory_as_grid_channels": bool(spatial and inv_as_grid),
        "materialize_seed": int(seed),
        "preprocess_workers_requested": int(nw_cfg),
        "preprocess_parallel_frac": float(data_cfg.get("preprocess_parallel_frac", 0.75)),
        "preprocess_events_finalize_parallel": bool(events_finalize_parallel),
        "preprocess_grid_fill_parallel": grid_fill_parallel_done,
    }
    if spatial:
        manifest["row_bin_size"] = int(row_bs)
        manifest["col_bin_size"] = int(col_bs)
        manifest["num_bank_channels"] = int(n_banks)
        manifest["mcelog_error_type_filter"] = ft_filter if ft_filter is not None else [1, 2]

    (processed_dir / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Preprocess done: N={n}, pos_rate={manifest['pos_rate']:.6f}", flush=True)
    return manifest
