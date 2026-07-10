from __future__ import annotations

"""从真实 mcelog / trouble_tickets 生成索引表 `(sid, predict_time)`。

## 目标

为 preprocess 提供 **真实** 的样本锚点列表（仅 `sid`, `predict_time` 两列）；标签 **y**
在 preprocess 阶段由 `merge_ticket_labels` 与工单表关联后生成，与训练目标一致：

**`predict_ts <= failed_ts <= predict_ts + label_horizon_seconds`**（默认 horizon = 5 分钟）。

## 采样策略（与 `docs/data_processing_guide.md` §3.5 一致）

1. **时间步长**：`sample_interval_seconds` 固定间隔生成锚点（默认 300s）。
2. **正样本区**：每个工单失败时刻前 **label 窗口** `[ft − H, ft]` 内按
   `positive_sample_interval_seconds`（默认 60s）密集采样；标注后 **保留 y=1**。
   （旧实现曾在整个 prefault 窗用统一步长采样，H=300s、步长 300s 时每个故障仅 ~1 个正锚点。）
3. **近故障区**：`[ft − positive_prefault_seconds, ft − H)` 内按 `sample_interval_seconds`
   采样，标注为 **y=0**；在 uniform/random 负采样**不足目标数量时**补入负类池（困难负样本）。
4. **负样本平衡**：按 `target_positive_ratio` 控制正负比例；`uniform` 在网格上均匀取点，
   `random` 随机拒绝采样（冒烟 preset 默认）。
5. **sid 覆盖**：负采样 **优先** 全部工单 sid，再覆盖 mcelog 中其余 sid。
6. **有效锚点区间**：负样本 / 近故障区锚点 clamp 在 **`t_min + W ≤ predict_ts ≤ t_max − H`**。
   **正样本区**在 **`[ft−H, ft]`** 采样（**ft 可 > t_max**）；右删截 **不**限制工单驱动正锚点。

索引粒度为 **sid**（与 trouble_tickets 一致）；mcelog 的 memoryid 等列仅用于后续特征构造。
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

NegStrategy = Literal["uniform", "random"]

import numpy as np
import pandas as pd

from dram2cnn.data.time_axis import (
    compute_global_time_epoch_offset,
    relative_seconds_to_predict_time_str,
    resolve_index_predict_ts,
    series_to_absolute_seconds,
)
from dram2cnn.data.tianchi_build import (
    failure_in_horizon,
    load_index_table,
    load_ticket_failures,
    merge_ticket_labels,
)
from dram2cnn.utils.config import label_horizon_seconds_from_dict, project_root


def _repo_relative(path: Path) -> str:
    """manifest / stats 中优先写入相对仓库根的路径。"""
    root = project_root()
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _time_range(start: float, end: float, step: float) -> np.ndarray:
    """在闭区间 `[start, end]` 上按 `step` 生成等间隔**相对秒**时间戳数组。"""
    if step <= 0:
        raise ValueError("sample_interval_seconds must be > 0")
    if end < start:
        return np.array([], dtype=np.float64)
    n = int(np.floor((end - start) / step)) + 1
    pts = start + np.arange(n, dtype=np.float64) * step
    return pts[pts <= end + 1e-6]


def _attach_predict_ts(df: pd.DataFrame) -> pd.DataFrame:
    """为索引 DataFrame 解析 `predict_time` 并附加 `predict_ts`（相对秒）。

    仅用于 ``build_index`` 内部：此时 ``predict_time`` 已由 ``relative_seconds_to_predict_time_str`` 写出，
    解析后数值即为相对秒（与 ``load_index_table`` 对 索引构建输出的行为一致）。
    """
    out = df.copy()
    out["sid"] = out["sid"].astype(str)
    pts = pd.to_datetime(out["predict_time"], errors="coerce", format="mixed")
    if pts.isna().any():
        bad = int(pts.isna().sum())
        raise ValueError(f"predict_time 存在 {bad} 行无法解析为时间")
    out["predict_ts"] = resolve_index_predict_ts(
        out["predict_time"],
        time_epoch_offset=None,
        index_path=None,
    )
    return out


def _label_index_rows(
    df: pd.DataFrame,
    cfg: IndexBuildConfig,
    time_epoch_offset: float,
    by_sid_fail: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """对索引行调用 `merge_ticket_labels`，无需经临时 CSV 写入磁盘。"""
    if df.empty:
        return np.array([], dtype=np.int64)
    idx = _attach_predict_ts(df)
    return merge_ticket_labels(
        idx,
        cfg.tickets_path,
        cfg.failure_types,
        cfg.label_horizon_seconds,
        time_epoch_offset=time_epoch_offset,
        by_sid_fail=by_sid_fail,
    )


def _is_positive_at(
    sid: str,
    predict_ts: float,
    by_sid_fail: dict[str, np.ndarray],
    horizon: float,
) -> bool:
    """判断锚点 `predict_ts` 在 horizon 内是否命中该 sid 的任一工单失败时刻。

    语义与 `merge_ticket_labels` 一致：闭区间 `[predict_ts, predict_ts + horizon]`。
    """
    return failure_in_horizon(predict_ts, by_sid_fail.get(sid), horizon)


def _merge_chunk_bounds(
    bounds: dict[str, list[float]],
    sids: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    """将当前 mcelog 分块内各 sid 的 min/max 时间合并进全局 `bounds`。"""
    if len(sids) == 0:
        return
    frame = pd.DataFrame({"sid": sids.astype(str), "t": timestamps.astype(np.float64, copy=False)})
    agg = frame.groupby("sid", sort=False)["t"].agg(["min", "max"])
    for sid, row in agg.iterrows():
        lo, hi = float(row["min"]), float(row["max"])
        slot = bounds.get(sid)
        if slot is None:
            bounds[sid] = [lo, hi]
        else:
            if lo < slot[0]:
                slot[0] = lo
            if hi > slot[1]:
                slot[1] = hi


@dataclass
class IndexBuildConfig:
    """索引表生成参数。

    字段来源：`configs/data.yaml` 的 `index_build` 段 + 顶层路径/horizon 字段。
    CLI ``dram2cnn.pipelines.run_build_index``（或 ``scripts/build_index.py``）会合并 preset（smoke/full）后传入 ``from_dict``。

    Attributes:
        mcelog_path / tickets_path: 原始 CSV 路径。
        sample_interval_seconds: 锚点采样步长（秒）。
        label_horizon_seconds: 标注 horizon（秒），须与 preprocess 配置一致。
        window_seconds: 特征窗长度（秒）；锚点下界为 `t_min + window_seconds`。
        positive_prefault_seconds: 近故障困难负样本区起点 = `ft − prefetch`（秒）。
        positive_sample_interval_seconds: 正样本 label 窗口 `[ft−H, ft]` 内采样步长（秒）。
        failure_types: 计入工单的 `failure_type` 列表（同 `label_failure_types`）。
        target_positive_ratio: 输出索引目标正例占比（约 0.05–0.95）。
        neg_stride_multiplier: 负采样网格步长 = interval × 该系数。
        neg_sample_strategy: `uniform`（全量）或 `random`（冒烟）。
        max_neg_grid_points_per_sid: 单 sid 负采样点数上限。
        min_neg_per_ticket_sid: uniform 模式下每个工单 sid 至少采样的负点数。
        seed: 随机采样与 shuffle 种子。
        max_rows: 输出最大行数；截断时 **优先保留全部正样本**。
        smoke_max_sids: 冒烟时参与负采样的 sid 上限（工单 sid 始终保留）。
        mcelog_scan_rows: mcelog 扫描行数上限（None = 全文件）。
        mcelog_chunk_rows: 流式读 mcelog 的分块大小。
        min_positives / min_total_rows: 写入前校验阈值。
        max_positive_candidates: 正样本行数上限（可选；smoke 由 max_rows 控规模，一般不单独截断正样本）。
    """

    mcelog_path: Path
    tickets_path: Path
    sample_interval_seconds: float = 300.0
    label_horizon_seconds: float = 300.0
    window_seconds: float = 300.0
    positive_prefault_seconds: float = 3600.0
    positive_sample_interval_seconds: float = 60.0
    failure_types: list[int] = field(default_factory=lambda: [1, 2, 3])
    target_positive_ratio: float = 0.25
    neg_stride_multiplier: int = 1
    neg_sample_strategy: NegStrategy = "uniform"
    max_neg_grid_points_per_sid: int = 128
    min_neg_per_ticket_sid: int = 1
    seed: int = 42
    max_rows: int | None = None
    smoke_max_sids: int | None = None
    mcelog_scan_rows: int | None = None
    mcelog_chunk_rows: int = 1_000_000
    min_positives: int = 5
    min_total_rows: int = 50
    max_positive_candidates: int | None = None
    # 与 preprocess spatial 模式一致：bounds 扫描仅统计 CE 事件（默认 [1,2]）；legacy 模式用 error_type < 3
    mcelog_error_type_filter: list[int] | None = field(default_factory=lambda: [1, 2])
    grid_representation: str = "spatial_heatmap"

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path | None = None) -> IndexBuildConfig:
        """从 dict / `data.yaml` 片段构造；相对路径相对仓库根目录解析。"""
        root = root or Path.cwd()

        def _p(key: str, default: str | None = None) -> Path:
            v = raw.get(key, default)
            if v is None:
                raise ValueError(f"IndexBuildConfig 缺少路径字段 `{key}`")
            p = Path(v)
            return p if p.is_absolute() else (root / p).resolve()

        horizon = label_horizon_seconds_from_dict(raw)
        sample_interval = float(raw.get("sample_interval_seconds", 300.0))
        pos_interval_raw = raw.get("positive_sample_interval_seconds")
        if pos_interval_raw in (None, "null"):
            pos_interval = min(60.0, sample_interval, horizon)
        else:
            pos_interval = float(pos_interval_raw)
        if pos_interval <= 0:
            raise ValueError("positive_sample_interval_seconds must be > 0")
        if pos_interval > horizon:
            raise ValueError(
                "positive_sample_interval_seconds 不能大于 label_horizon_seconds "
                f"({pos_interval} > {horizon})"
            )
        strategy = str(raw.get("neg_sample_strategy", "uniform")).lower().strip()
        if strategy not in {"uniform", "random"}:
            raise ValueError("neg_sample_strategy must be 'uniform' or 'random'")
        grid_rep = str(raw.get("grid_representation", "spatial_heatmap"))
        et_raw = raw.get("mcelog_error_type_filter")
        if et_raw not in (None, "null"):
            et_filter: list[int] | None = [int(x) for x in et_raw]
        elif grid_rep == "legacy_rc_et":
            et_filter = None
        else:
            et_filter = [1, 2]
        return cls(
            mcelog_path=_p("mcelog_path", "data/raw/mcelog.csv"),
            tickets_path=_p("tickets_path", "data/raw/trouble_tickets.csv"),
            sample_interval_seconds=sample_interval,
            label_horizon_seconds=horizon,
            window_seconds=float(raw.get("window_seconds", horizon)),
            positive_prefault_seconds=float(raw.get("positive_prefault_seconds", 3600.0)),
            positive_sample_interval_seconds=pos_interval,
            failure_types=[
                int(x) for x in raw.get("failure_types", raw.get("label_failure_types", [1, 2, 3]))
            ],
            target_positive_ratio=float(raw.get("target_positive_ratio", 0.25)),
            neg_stride_multiplier=max(1, int(raw.get("neg_stride_multiplier", 1))),
            neg_sample_strategy=strategy,  # type: ignore[arg-type]
            max_neg_grid_points_per_sid=max(1, int(raw.get("max_neg_grid_points_per_sid", 128))),
            min_neg_per_ticket_sid=max(0, int(raw.get("min_neg_per_ticket_sid", 1))),
            seed=int(raw.get("seed", 42)),
            max_rows=(None if raw.get("max_rows") in (None, "null") else int(raw["max_rows"])),
            smoke_max_sids=(
                None
                if raw.get("smoke_max_sids") in (None, "null")
                else int(raw["smoke_max_sids"])
            ),
            mcelog_scan_rows=(
                None
                if raw.get("mcelog_scan_rows") in (None, "null")
                else int(raw["mcelog_scan_rows"])
            ),
            mcelog_chunk_rows=int(raw.get("mcelog_chunk_rows", 1_000_000)),
            min_positives=int(raw.get("min_positives", 5)),
            min_total_rows=int(raw.get("min_total_rows", 50)),
            max_positive_candidates=(
                None
                if raw.get("max_positive_candidates") in (None, "null")
                else int(raw["max_positive_candidates"])
            ),
            mcelog_error_type_filter=et_filter,
            grid_representation=grid_rep,
        )


def _require_readable_csv(path: Path, label: str, required_cols: tuple[str, ...]) -> None:
    """预先检查 CSV 路径可读且含必需列，避免全表扫描后才因 ParserError/缺列失败。"""
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    try:
        head = pd.read_csv(path, nrows=0)
    except pd.errors.ParserError as exc:
        raise RuntimeError(f"无法解析 {label} CSV {path}: {exc}") from exc
    missing = [c for c in required_cols if c not in head.columns]
    if missing:
        raise ValueError(f"{label} 缺少列 {missing}，当前: {list(head.columns)}")


def scan_mcelog_sid_bounds(
    mcelog_path: Path,
    chunk_rows: int = 1_000_000,
    max_rows: int | None = None,
    time_epoch_offset: float = 0.0,
    *,
    mcelog_error_type_filter: list[int] | None = None,
    legacy_error_type_lt: int | None = 3,
) -> dict[str, tuple[float, float]]:
    """流式扫描 mcelog，返回每个 sid 的 `(min_error_ts, max_error_ts)`（**相对秒**）。

    与 preprocess ``build_events_by_sid`` 使用相同的 error_type 过滤，避免 bounds 含 CE 外事件
    而特征窗为空。spatial 默认仅 ``error_type ∈ {1,2}``；legacy 默认 ``error_type < 3``。
    """
    _require_readable_csv(mcelog_path, "mcelog", ("sid", "error_time", "error_type"))
    bounds: dict[str, list[float]] = {}
    rows_read = 0
    usecols = ["sid", "error_time", "error_type"]
    try:
        reader = pd.read_csv(
            mcelog_path,
            usecols=usecols,
            chunksize=chunk_rows,
            low_memory=False,
        )
    except pd.errors.ParserError as exc:
        raise RuntimeError(f"无法解析 mcelog CSV {mcelog_path}: {exc}") from exc
    for chunk in reader:
        if max_rows is not None and rows_read >= max_rows:
            break
        if max_rows is not None:
            remain = max_rows - rows_read
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]
        rows_read += len(chunk)
        if mcelog_error_type_filter is not None:
            chunk = chunk[chunk["error_type"].isin(list(mcelog_error_type_filter))].copy()
        elif legacy_error_type_lt is not None:
            chunk = chunk[chunk["error_type"] < int(legacy_error_type_lt)].copy()
        if chunk.empty:
            continue
        ts = pd.Series(
            series_to_absolute_seconds(chunk["error_time"]) - float(time_epoch_offset),
            index=chunk.index,
        )
        valid = ts.notna()
        if not valid.any():
            continue
        _merge_chunk_bounds(
            bounds,
            chunk.loc[valid, "sid"].to_numpy(),
            ts.loc[valid].to_numpy(dtype=np.float64),
        )
    return {k: (v[0], v[1]) for k, v in bounds.items()}


def _valid_predict_bounds(
    sid: str,
    sid_bounds: dict[str, tuple[float, float]],
    win: float,
    horizon: float,
) -> tuple[float, float] | None:
    """计算 sid 上合法锚点闭区间 `[t_min + win, t_max - horizon]`。

    - 左界：特征半开窗 ``[pts - win, pts)`` 需要 mcelog 自 ``t_min`` 起有覆盖。
    - 右界：右删截，避免 ``pts + horizon`` 超出以 mcelog 时间轴为 proxy 的数据集末尾。
    - 若 ``t_max - t_min < win + horizon``，该 sid 无法放置任何锚点，返回 ``None``。
    """
    b = sid_bounds.get(sid)
    if b is None:
        return None
    t_min, t_max = b
    start = t_min + win
    end = t_max - horizon
    if end < start:
        return None
    return start, end


def _positive_candidates(
    by_sid_fail: dict[str, np.ndarray],
    cfg: IndexBuildConfig,
    sid_bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """在每条工单失败时刻前生成候选锚点（正样本区 + 近故障区两段采样）。

    - **正样本区** ``[ft − horizon, ft]``：步长 ``positive_sample_interval_seconds``；
      上界为工单失败时刻 **ft**（可超出 mcelog ``t_max``；特征窗或为空，标签仍来自 tickets）。
    - **近故障区** ``[ft − prefetch, ft − horizon)``：步长 ``sample_interval_seconds``；
      标注后为 y=0，供困难负样本补齐。

    候选尚未标注；后续 `_label_split_candidates` 会区分 y=1 / y=0。
    """
    pos_step = float(cfg.positive_sample_interval_seconds)
    near_step = float(cfg.sample_interval_seconds)
    prefetch = float(cfg.positive_prefault_seconds)
    win = float(cfg.window_seconds)
    horizon = float(cfg.label_horizon_seconds)
    rows: list[dict[str, str]] = []

    for sid, fail_arr in by_sid_fail.items():
        vb = _valid_predict_bounds(sid, sid_bounds, win, horizon)
        if vb is None:
            continue
        for ft in fail_arr:
            ft = float(ft)
            # 正样本区上界取工单失败时刻 ft（勿将上界截断（clamp） 到 vb[1]）：故障可晚于 mcelog 观测末尾，
            # 此时特征窗可能为空但标签仍来自 tickets（preprocess 保留 y=1 空窗）。
            pos_start = max(ft - horizon, vb[0])
            pos_end = ft
            if pos_end >= pos_start:
                for pts in _time_range(pos_start, pos_end, pos_step):
                    rows.append(
                        {
                            "sid": sid,
                            "predict_time": relative_seconds_to_predict_time_str(float(pts)),
                        }
                    )

            if prefetch > horizon:
                near_start = max(ft - prefetch, vb[0])
                # 半开上界 ft−H：锚点 ft−H 在正样本闭区间 [ft−H, ft] 内打 y=1，不应落入 near_neg
                near_end = min(ft - horizon - 1e-6, vb[1])
                if near_end >= near_start:
                    for pts in _time_range(near_start, near_end, near_step):
                        rows.append(
                            {
                                "sid": sid,
                                "predict_time": relative_seconds_to_predict_time_str(float(pts)),
                            }
                        )

    if not rows:
        return pd.DataFrame(columns=["sid", "predict_time"])
    return pd.DataFrame(rows).drop_duplicates(subset=["sid", "predict_time"]).reset_index(drop=True)


def _negatives_uniform_for_sid(
    sid: str,
    sid_bounds: dict[str, tuple[float, float]],
    by_sid_fail: dict[str, np.ndarray],
    cfg: IndexBuildConfig,
    exclude: set[tuple[str, str]],
    max_pts: int,
) -> list[dict[str, str]]:
    """对单个 sid 在合法时间轴上均匀（或子采样）取负类锚点，跳过 horizon 内会打 y=1 的点。"""
    interval = float(cfg.sample_interval_seconds) * int(cfg.neg_stride_multiplier)
    horizon = float(cfg.label_horizon_seconds)
    win = float(cfg.window_seconds)
    vb = _valid_predict_bounds(sid, sid_bounds, win, horizon)
    if vb is None or max_pts <= 0:
        return []
    start, end = vb
    grid = _time_range(start, end, interval)
    if grid.size == 0:
        return []
    if grid.size > cfg.max_neg_grid_points_per_sid:
        idx = np.linspace(0, grid.size - 1, cfg.max_neg_grid_points_per_sid, dtype=int)
        grid = grid[idx]
    out: list[dict[str, str]] = []
    for pts in grid:
        if _is_positive_at(sid, float(pts), by_sid_fail, horizon):
            continue
        pt_str = relative_seconds_to_predict_time_str(float(pts))
        if (sid, pt_str) in exclude:
            continue
        out.append({"sid": sid, "predict_time": pt_str})
        if len(out) >= max_pts:
            break
    return out


def _negative_candidates_random(
    sid_bounds: dict[str, tuple[float, float]],
    by_sid_fail: dict[str, np.ndarray],
    cfg: IndexBuildConfig,
    exclude: set[tuple[str, str]],
    n_neg_target: int,
    sids: list[str],
) -> pd.DataFrame:
    """随机选择 (sid, predict_ts) 负样本，映射到采样网格并拒绝正类区间内的点。"""
    interval = float(cfg.sample_interval_seconds) * int(cfg.neg_stride_multiplier)
    horizon = float(cfg.label_horizon_seconds)
    win = float(cfg.window_seconds)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set(exclude)
    if not sids or n_neg_target <= 0:
        return pd.DataFrame(columns=["sid", "predict_time"])

    max_attempts = max(n_neg_target * 40, 500)
    sid_arr = np.array(sids, dtype=object)
    attempts = 0
    while len(rows) < n_neg_target and attempts < max_attempts:
        attempts += 1
        sid = str(rng.choice(sid_arr))
        vb = _valid_predict_bounds(sid, sid_bounds, win, horizon)
        if vb is None:
            continue
        start, end = vb
        pts = float(rng.uniform(start, end))
        if interval > 0:
            pts = start + math.floor((pts - start) / interval) * interval
        if pts > end:
            continue
        if _is_positive_at(sid, pts, by_sid_fail, horizon):
            continue
        pt_str = relative_seconds_to_predict_time_str(float(pts))
        key = (sid, pt_str)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"sid": sid, "predict_time": pt_str})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["sid", "predict_time"])


def _negative_candidates(
    sid_bounds: dict[str, tuple[float, float]],
    by_sid_fail: dict[str, np.ndarray],
    cfg: IndexBuildConfig,
    exclude: set[tuple[str, str]],
    n_neg_target: int,
) -> pd.DataFrame:
    """生成目标数量的负样本行：uniform 优先工单 sid，不足时用 random 补齐。"""
    ticket_sids = sorted(s for s in by_sid_fail if s in sid_bounds)
    other_sids = sorted(s for s in sid_bounds if s not in by_sid_fail)
    all_sids = ticket_sids + other_sids

    if cfg.smoke_max_sids is not None and len(all_sids) > cfg.smoke_max_sids:
        rng = np.random.default_rng(cfg.seed)
        rest = list(other_sids)
        rng.shuffle(rest)
        cap = max(0, cfg.smoke_max_sids - len(ticket_sids))
        all_sids = ticket_sids + rest[:cap]
        other_sids = [s for s in all_sids if s not in ticket_sids]

    if cfg.neg_sample_strategy == "random":
        return _negative_candidates_random(
            sid_bounds, by_sid_fail, cfg, exclude, n_neg_target, all_sids
        )

    rows: list[dict[str, str]] = []
    seen = set(exclude)

    if ticket_sids and cfg.min_neg_per_ticket_sid > 0:
        per_ticket = max(
            cfg.min_neg_per_ticket_sid,
            n_neg_target // max(1, len(ticket_sids) * 2),
        )
        per_ticket = min(per_ticket, cfg.max_neg_grid_points_per_sid)
        for sid in ticket_sids:
            for item in _negatives_uniform_for_sid(
                sid, sid_bounds, by_sid_fail, cfg, seen, per_ticket
            ):
                key = (item["sid"], item["predict_time"])
                if key not in seen:
                    seen.add(key)
                    rows.append(item)

    remaining = max(0, n_neg_target - len(rows))
    if remaining > 0 and other_sids:
        per_other = max(1, remaining // len(other_sids))
        per_other = min(per_other, cfg.max_neg_grid_points_per_sid)
        for sid in other_sids:
            if len(rows) >= n_neg_target:
                break
            for item in _negatives_uniform_for_sid(
                sid, sid_bounds, by_sid_fail, cfg, seen, per_other
            ):
                key = (item["sid"], item["predict_time"])
                if key not in seen:
                    seen.add(key)
                    rows.append(item)
                if len(rows) >= n_neg_target:
                    break

    if len(rows) < n_neg_target:
        extra = _negative_candidates_random(
            sid_bounds,
            by_sid_fail,
            cfg,
            seen,
            n_neg_target - len(rows),
            all_sids,
        )
        if len(extra) > 0:
            rows.extend(extra.to_dict("records"))

    df = pd.DataFrame(rows)
    if len(df) > n_neg_target:
        df = df.sample(n=n_neg_target, random_state=cfg.seed).reset_index(drop=True)
    return df if len(df) else pd.DataFrame(columns=["sid", "predict_time"])


def _label_split_candidates(
    pos_df: pd.DataFrame,
    cfg: IndexBuildConfig,
    time_epoch_offset: float,
    by_sid_fail: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对 prefault 候选标注，拆成真阳性、近故障阴性、全量候选三份 DataFrame。"""
    y_all = _label_index_rows(pos_df, cfg, time_epoch_offset, by_sid_fail=by_sid_fail)
    ticket_df = pos_df.reset_index(drop=True)
    pos_out = ticket_df.loc[y_all == 1, ["sid", "predict_time"]].reset_index(drop=True)
    near_neg = ticket_df.loc[y_all == 0, ["sid", "predict_time"]].reset_index(drop=True)
    return pos_out, near_neg, ticket_df


def _truncate_to_max_rows(
    out: pd.DataFrame,
    cfg: IndexBuildConfig,
    time_epoch_offset: float,
    by_sid_fail: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """在 `max_rows` 限制下截断：正样本过多时先随机下采样至配额，再保留负样本至 `max_rows`。"""
    if cfg.max_rows is None or len(out) <= cfg.max_rows:
        return out
    y = _label_index_rows(out, cfg, time_epoch_offset, by_sid_fail=by_sid_fail)
    pos_mask = y == 1
    pos_part = out.loc[pos_mask].copy()
    neg_part = out.loc[~pos_mask]
    if len(pos_part) > int(cfg.max_rows):
        pos_part = pos_part.sample(n=int(cfg.max_rows), random_state=cfg.seed)
    n_neg_keep = max(0, int(cfg.max_rows) - len(pos_part))
    if len(neg_part) > n_neg_keep:
        neg_part = neg_part.sample(n=n_neg_keep, random_state=cfg.seed)
    trimmed = pd.concat([pos_part, neg_part], ignore_index=True)
    return trimmed.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)


def build_index_dataframe(cfg: IndexBuildConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构建索引 DataFrame 与统计 dict（不写磁盘）。

    流程：扫描 mcelog 时间界 → 加载工单 → prefault 正候选 → 标注 → 负采样 → 合并 shuffle。
    """
    _require_readable_csv(cfg.mcelog_path, "mcelog", ("sid", "error_time"))
    _require_readable_csv(cfg.tickets_path, "tickets", ("sid", "failure_type", "failed_time"))
    time_epoch_offset = compute_global_time_epoch_offset(
        cfg.mcelog_path,
        cfg.tickets_path,
        mcelog_chunk_rows=cfg.mcelog_chunk_rows,
        mcelog_max_rows=cfg.mcelog_scan_rows,
    )
    sid_bounds = scan_mcelog_sid_bounds(
        cfg.mcelog_path,
        chunk_rows=cfg.mcelog_chunk_rows,
        max_rows=cfg.mcelog_scan_rows,
        time_epoch_offset=time_epoch_offset,
        mcelog_error_type_filter=cfg.mcelog_error_type_filter,
        legacy_error_type_lt=(3 if cfg.mcelog_error_type_filter is None else None),
    )
    if not sid_bounds:
        raise RuntimeError(f"mcelog 扫描无有效 sid 时间范围: {cfg.mcelog_path}")

    tix, by_sid_fail = load_ticket_failures(cfg.tickets_path, cfg.failure_types, time_epoch_offset)
    n_ticket_no_mcelog = sum(1 for s in by_sid_fail if s not in sid_bounds)

    raw_pos = _positive_candidates(by_sid_fail, cfg, sid_bounds)
    pos_df, near_neg, ticket_df = _label_split_candidates(
        raw_pos, cfg, time_epoch_offset, by_sid_fail=by_sid_fail
    )

    n_raw_candidates = int(len(raw_pos))
    n_pos = len(pos_df)
    ratio = float(cfg.target_positive_ratio)
    ratio = min(max(ratio, 0.05), 0.95)
    if n_pos == 0:
        raise RuntimeError(
            "未生成任何正样本锚点；请检查 label_horizon_seconds、positive_prefault_seconds、"
            "failure_types 与 mcelog 时间范围是否重叠。"
        )

    max_pos_cap = cfg.max_positive_candidates
    if max_pos_cap is None and cfg.max_rows is not None and ratio < 1.0:
        max_pos_cap = max(cfg.min_positives, int(round(int(cfg.max_rows) * ratio / (1.0 - ratio))))
    if max_pos_cap is not None and n_pos > int(max_pos_cap):
        pos_df = pos_df.sample(n=int(max_pos_cap), random_state=cfg.seed).reset_index(drop=True)
        n_pos = len(pos_df)

    exclude = set(zip(ticket_df["sid"].astype(str), ticket_df["predict_time"].astype(str)))
    n_neg_target = int(round(n_pos * (1.0 - ratio) / ratio))
    n_neg_target = max(n_neg_target, n_pos)
    if cfg.max_rows is not None:
        n_neg_target = max(n_neg_target, int(cfg.max_rows) - n_pos)
    neg_df = _negative_candidates(sid_bounds, by_sid_fail, cfg, exclude, n_neg_target)

    if len(near_neg) > 0:
        need = max(0, n_neg_target - len(neg_df))
        if need > 0:
            extra = near_neg.sample(n=min(need, len(near_neg)), random_state=cfg.seed)
            neg_df = pd.concat([neg_df, extra], ignore_index=True)
    neg_df = neg_df.drop_duplicates(subset=["sid", "predict_time"]).reset_index(drop=True)
    n_neg_raw = len(neg_df)
    if n_neg_raw < n_neg_target:
        print(
            f"警告：负样本仅 {n_neg_raw}/{n_neg_target}，正负比可能低于 target_positive_ratio={ratio:.2f}",
            flush=True,
        )

    out = pd.concat([pos_df, neg_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    out = _truncate_to_max_rows(out, cfg, time_epoch_offset, by_sid_fail=by_sid_fail)

    ticket_sid_coverage = len(set(pos_df["sid"].astype(str))) / max(1, len(by_sid_fail))
    stats: dict[str, Any] = {
        "time_axis": "relative_seconds_since_global_min",
        "time_epoch_offset_absolute": float(time_epoch_offset),
        "mcelog_scan_rows": cfg.mcelog_scan_rows,
        "n_sids_in_mcelog": len(sid_bounds),
        "n_ticket_sids": len(by_sid_fail),
        "n_ticket_sids_without_mcelog": int(n_ticket_no_mcelog),
        "ticket_sid_coverage_in_pos": float(ticket_sid_coverage),
        "n_neg_sids": int(neg_df["sid"].nunique()) if len(neg_df) else 0,
        "n_tickets_used": int(len(tix)),
        "n_raw_prefault_candidates": n_raw_candidates,
        "n_near_neg_labeled": int(len(near_neg)),
        "positive_candidates": int(n_pos),
        "negative_candidates_before_balance": int(n_neg_raw),
        "n_rows_output": int(len(out)),
        "neg_sample_strategy": cfg.neg_sample_strategy,
        "config": asdict(cfg),
    }
    stats["config"]["mcelog_path"] = _repo_relative(cfg.mcelog_path)
    stats["config"]["tickets_path"] = _repo_relative(cfg.tickets_path)
    return out, stats


def validate_and_write_index(
    df: pd.DataFrame,
    out_path: Path,
    cfg: IndexBuildConfig,
    *,
    time_epoch_offset: float | None = None,
) -> dict[str, Any]:
    """安全写入索引 CSV（先写临时文件再替换），并用 `merge_ticket_labels` 校验正负样本数是否达标。"""
    if time_epoch_offset is None:
        time_epoch_offset = compute_global_time_epoch_offset(
            cfg.mcelog_path,
            cfg.tickets_path,
            mcelog_chunk_rows=cfg.mcelog_chunk_rows,
            mcelog_max_rows=cfg.mcelog_scan_rows,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f".{out_path.name}.tmp"
    df.to_csv(tmp, index=False)
    idx_df = load_index_table(tmp, max_rows=None, time_epoch_offset=time_epoch_offset)
    y = merge_ticket_labels(
        idx_df,
        cfg.tickets_path,
        cfg.failure_types,
        cfg.label_horizon_seconds,
        time_epoch_offset=time_epoch_offset,
    )
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < cfg.min_positives:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"索引校验失败：正样本 {n_pos} < min_positives={cfg.min_positives}。"
            "请检查 positive_sample_interval_seconds、positive_prefault_seconds、"
            "label_horizon_seconds 与 mcelog 时间范围是否重叠。"
        )
    if len(y) < cfg.min_total_rows:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"索引校验失败：总行数 {len(y)} < min_total_rows={cfg.min_total_rows}")
    tmp.replace(out_path)
    return {
        "path": _repo_relative(out_path),
        "n_rows": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": float(n_pos / len(y)) if len(y) else 0.0,
        "label_horizon_seconds": cfg.label_horizon_seconds,
        "window_seconds": cfg.window_seconds,
        "sample_interval_seconds": cfg.sample_interval_seconds,
        "time_epoch_offset_absolute": float(time_epoch_offset),
        "time_axis": "relative_seconds_since_global_min",
        "mcelog_scan_rows": cfg.mcelog_scan_rows,
        "mcelog_error_type_filter": cfg.mcelog_error_type_filter,
        "grid_representation": cfg.grid_representation,
    }


def build_index_to_csv(
    cfg: IndexBuildConfig,
    out_path: Path,
    write_stats_json: bool = True,
) -> dict[str, Any]:
    """统一入口：构建索引 → 校验 → 写入 CSV（可选 `.stats.json`  sidecar）。"""
    df, build_stats = build_index_dataframe(cfg)
    write_meta = validate_and_write_index(
        df,
        out_path,
        cfg,
        time_epoch_offset=float(build_stats["time_epoch_offset_absolute"]),
    )
    summary = {**build_stats, **write_meta}
    if write_stats_json:
        stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
        stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
