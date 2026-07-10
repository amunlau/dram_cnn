"""部署阶段评价指标汇总：从 runs/ 读取 metrics.json，合并多策略对比表。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dram2cnn.utils.config import project_root
from dram2cnn.utils.run_layout import read_pointer_dir

KPI_COLUMNS = (
    "strategy",
    "run_name",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "threshold",
    "metrics_path",
)

RUN_NAME_STRATEGY: dict[str, str] = {
    "exp_tianchi_full_f1_ddp": "F1/time",
    "exp_tianchi_full_f1_sid": "F1/sid",
    "exp_tianchi_full_recall_ddp": "recall/time",
    "exp_tianchi_full_recall_sid": "recall/sid",
    "exp_tianchi_full_precision_ddp": "precision/time",
    "exp_tianchi_full_precision_sid": "precision/sid",
    "exp_tianchi_full_recall_ddp_balanced": "recall_balanced/time",
    "exp_tianchi_full_recall_sid_balanced": "recall_balanced/sid",
}

F1_KPI_PAIRS: tuple[tuple[str, str], ...] = (
    ("exp_tianchi_full_f1_ddp", "F1/time"),
    ("exp_tianchi_full_f1_sid", "F1/sid"),
)

RECALL_PRECISION_KPI_PAIRS: tuple[tuple[str, str], ...] = (
    ("exp_tianchi_full_recall_ddp", "recall/time"),
    ("exp_tianchi_full_recall_sid", "recall/sid"),
    ("exp_tianchi_full_precision_ddp", "precision/time"),
    ("exp_tianchi_full_precision_sid", "precision/sid"),
)


def load_deployment_metrics(
    run_name: str,
    *,
    project_root_path: Path | None = None,
    strategy_label: str | None = None,
) -> dict[str, Any] | None:
    """读取某 run 最近一次 infer 的部署阶段评价 metrics。"""
    root = project_root_path or project_root()
    run_root = root / "runs" / run_name
    infer_dir = read_pointer_dir(run_root, ".latest_infer_session", "latest_infer")
    if infer_dir is None:
        return None
    mp = infer_dir / "metrics.json"
    if not mp.is_file():
        return None
    payload = json.loads(mp.read_text(encoding="utf-8"))
    dep = payload.get("deployment") or {}
    m = dep.get("metrics") or payload
    strategy = strategy_label or RUN_NAME_STRATEGY.get(run_name, run_name)
    return {
        "strategy": strategy,
        "run_name": run_name,
        "threshold": dep.get("threshold"),
        "f1": m.get("f1"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "roc_auc": m.get("roc_auc"),
        "metrics_path": str(mp),
    }


def collect_kpi_rows(
    pairs: tuple[tuple[str, str], ...],
    *,
    project_root_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, strategy in pairs:
        row = load_deployment_metrics(
            run_name,
            project_root_path=project_root_path,
            strategy_label=strategy,
        )
        if row:
            rows.append(row)
        else:
            print(f"提示：尚无 {run_name} 的 metrics.json，请先执行训练、推理与评估")
    return rows


def write_kpi_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[list(KPI_COLUMNS)].to_csv(out_path, index=False)


def merge_all_strategies_kpi(
    *,
    tables_dir: Path | None = None,
    out_name: str = "full_all_strategies_deployment_kpi.csv",
) -> Path | None:
    """合并 F1 与工程实验评价指标子表为一张总表。"""
    root = project_root()
    td = tables_dir or (root / "outputs" / "tables")
    parts: list[pd.DataFrame] = []
    for fname in (
        "full_f1_time_vs_sid_deployment_kpi.csv",
        "full_recall_vs_precision_deployment_kpi.csv",
    ):
        p = td / fname
        if p.is_file():
            parts.append(pd.read_csv(p))
    if not parts:
        return None
    out = td / out_name
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    return out


def main() -> None:
    """CLI：合并 outputs/tables 下子表为 full_all_strategies_deployment_kpi.csv。"""
    out = merge_all_strategies_kpi()
    if out is None:
        print("未找到可合并的评价指标子表（需先跑 Notebook 04/05 或等价流水线）")
        raise SystemExit(1)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
