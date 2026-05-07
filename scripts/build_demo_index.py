#!/usr/bin/env python3
"""从真实工单与 mcelog 生成 `data/index_stubs/features_5m_index_demo.csv`。

保证混入若干正样本（predict_time 在 failed_time 之前且落在 label_backtrace 内），
避免演示/冒烟流程中 y 全 0 → 验证集无正例 → 指标全 nan、训练看似「失败」。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dram2cnn.data.tianchi_build import load_index_table, merge_ticket_labels
from dram2cnn.utils.config import load_yaml


def _csv_datetime(ts: pd.Timestamp) -> str:
    """公元元年等低年份：strftime %%Y 在部分环境非四位，ISO 更稳。"""
    t = pd.Timestamp(ts)
    return t.isoformat(sep=" ", timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/index_stubs/features_5m_index_demo.csv"),
        help="输出 CSV（sid, predict_time）",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_cfg = load_yaml(root / "configs/data.yaml")
    tickets_path = Path(data_cfg["tickets_path"])
    if not tickets_path.is_absolute():
        tickets_path = root / tickets_path
    mcelog_path = Path(data_cfg["mcelog_path"])
    if not mcelog_path.is_absolute():
        mcelog_path = root / mcelog_path
    backtrace = float(data_cfg.get("label_backtrace_seconds", 2592000))
    fail_types = list(data_cfg.get("label_failure_types", [1, 2, 3]))

    tix = pd.read_csv(tickets_path)
    tix = tix[tix["failure_type"].isin(fail_types)].copy()
    tix["failed_time"] = pd.to_datetime(tix["failed_time"], errors="coerce", format="mixed")
    tix = tix.dropna(subset=["failed_time"])
    tix = tix.sort_values("failed_time").reset_index(drop=True)
    if len(tix) < 3:
        raise SystemExit("工单表有效行过少，无法构造演示正样本")

    n_pos = min(18, len(tix))
    ix = np.unique(np.linspace(0, len(tix) - 1, n_pos, dtype=int))
    pos = tix.iloc[ix].copy()
    # 用「几小时」而非「几天」，避免公元 0001 年附近减 5 日落到 year 0 导致 predict_time 无法解析
    pos["predict_time"] = pos["failed_time"] - pd.Timedelta(hours=12)
    pos = pos.dropna(subset=["predict_time"])
    pos_rows = pd.DataFrame({"sid": pos["sid"].astype(str), "predict_time": [_csv_datetime(x) for x in pos["predict_time"]]})

    m = pd.read_csv(mcelog_path, nrows=400_000, usecols=["sid", "error_time"], low_memory=False)
    m["t"] = pd.to_datetime(m["error_time"], errors="coerce", format="mixed")
    m = m.dropna(subset=["t"])
    last = m.groupby("sid", sort=False)["t"].max().reset_index()
    last = last.sample(min(150, len(last)), random_state=args.seed).reset_index(drop=True)
    last["predict_time"] = last["t"] - pd.Timedelta(hours=2)
    last = last.dropna(subset=["predict_time"])
    neg_cand = pd.DataFrame(
        {"sid": last["sid"].astype(str), "predict_time": [_csv_datetime(x) for x in last["predict_time"]]}
    )

    comb = pd.concat([pos_rows, neg_cand], ignore_index=True)
    comb = comb.dropna().drop_duplicates(subset=["sid", "predict_time"]).reset_index(drop=True)
    comb = comb[comb["predict_time"].astype(str).str.len() > 0]
    _pts = pd.to_datetime(comb["predict_time"], errors="coerce", format="mixed")
    comb = comb.loc[_pts.notna()].reset_index(drop=True)
    if len(comb) < 5:
        raise SystemExit("过滤非法 predict_time 后样本过少")

    tmp = args.out.parent / "._tmp_demo_index.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    comb.to_csv(tmp, index=False)
    idx_df = load_index_table(tmp, max_rows=None)
    y = merge_ticket_labels(idx_df, tickets_path, fail_types, backtrace)
    tmp.unlink(missing_ok=True)

    pos_mask = y == 1
    neg_mask = y == 0
    if pos_mask.sum() < 3:
        raise SystemExit(f"构造失败：正样本仅 {pos_mask.sum()}，请检查工单与 backtrace 配置")

    # 保留全部验证过的正样本 + 至多 30 条负样本
    pos_final = comb.loc[pos_mask].copy()
    neg_pool = comb.loc[neg_mask]
    neg_final = neg_pool.sample(min(30, len(neg_pool)), random_state=args.seed)
    out_df = pd.concat([pos_final, neg_final], ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    idx2 = load_index_table(args.out, max_rows=None)
    y2 = merge_ticket_labels(idx2, tickets_path, fail_types, backtrace)
    print(
        f"Wrote {args.out} rows={len(out_df)} pos={int(y2.sum())} neg={int(len(y2) - y2.sum())} "
        f"(label_backtrace_seconds={backtrace:g})"
    )


if __name__ == "__main__":
    main()
