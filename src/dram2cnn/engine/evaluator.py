from __future__ import annotations

"""评估预测 CSV 并导出分析产物。

主 KPI（``metrics.json`` 顶层字段）= **部署阶段评价**：infer 使用的验证集阈值下的 ``y_pred``。
另写入 ``deployment`` / ``oracle_on_test`` 两段结构化说明，避免与测试集事后最优阈值混淆。
``best_threshold_on_eval.json`` 仅保留 oracle 阈值摘要（与训练 ``best_threshold.json`` 区分）。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dram2cnn.engine.metrics import build_confusion, build_pr_curve, classification_metrics, search_threshold
from dram2cnn.utils.io import dump_json


def evaluate(pred_csv: str, output_json: str) -> None:
    """计算指标、阈值分析、PR 数据与混淆矩阵。"""
    pred_path = Path(pred_csv)
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"未找到预测文件: {pred_path}。请先运行 `bash scripts/infer.sh <实验yaml>`，"
            f"或检查 `configs/base.yaml` 中 evaluate_run_id / 推理输出目录。"
        )
    run_dir = pred_path.parent
    df = pd.read_csv(pred_path)
    for col in ("y_true", "y_pred"):
        if col not in df.columns:
            raise ValueError(
                f"predictions.csv 缺少必需列 `{col}`（当前列: {list(df.columns)})。"
                "请确认 infer 步骤完整执行且 checkpoint 与 splits 匹配。"
            )

    if "y_prob" in df.columns:
        y_prob_raw = df["y_prob"].to_numpy(dtype=float, copy=False)
        if len(y_prob_raw) > 0 and not np.isfinite(y_prob_raw).all():
            n_bad = int(np.sum(~np.isfinite(y_prob_raw)))
            raise ValueError(
                f"predictions.csv 中 y_prob 含 {n_bad}/{len(y_prob_raw)} 个非有限值（NaN/Inf）。"
                "这通常表示训练 checkpoint 已数值崩溃（train log 中 loss=nan）。"
                "请检查 train 配置（GridCNN-Deep 建议 focal_gamma=0），删除 train/infer marker 后重训。"
            )

    y_true = df["y_true"]
    y_prob_series = df["y_prob"] if "y_prob" in df.columns else None
    deploy_threshold = None
    if "threshold" in df.columns and len(df) > 0:
        try:
            deploy_threshold = float(df["threshold"].iloc[0])
        except (TypeError, ValueError):
            deploy_threshold = None

    # 部署阶段评价：infer 写出的 y_pred（验证集搜阈后的阈值）
    metrics_deploy = classification_metrics(y_true, df["y_pred"], y_prob_series)
    payload: dict = {
        **metrics_deploy,
        "kpi_scope": "deployment",
        "deployment": {
            "threshold": deploy_threshold,
            "threshold_source": "validation best_threshold.json（infer 使用；汇报主 KPI）",
            "metrics": dict(metrics_deploy),
        },
    }

    if y_prob_series is not None:
        y_prob_np = y_prob_series.to_numpy(dtype=float, copy=False)
        oracle = search_threshold(
            y_true.to_numpy(), y_prob_np, 0.1, 0.9, 0.01, objective="f1"
        )
        y_pred_oracle = (y_prob_np >= float(oracle["threshold"])).astype(int)
        metrics_oracle = classification_metrics(y_true, y_pred_oracle, y_prob_series)
        payload["oracle_on_test"] = {
            "threshold": float(oracle["threshold"]),
            "threshold_source": "测试集标签事后网格搜索（仅分析，不可作部署 KPI）",
            "metrics": dict(metrics_oracle),
        }
        dump_json(run_dir / "best_threshold_on_eval.json", oracle)
        pr = build_pr_curve(y_true, y_prob_np)
    else:
        payload["oracle_on_test"] = {
            "threshold": None,
            "threshold_source": "y_prob 缺失，跳过测试集 oracle 阈值搜索",
            "metrics": None,
        }
        dump_json(
            run_dir / "best_threshold_on_eval.json",
            {"threshold": None, "f1": None, "note": "y_prob column missing"},
        )
        pr = {"precision": [], "recall": [], "thresholds": []}

    dump_json(output_json, payload)

    if pr["precision"] and pr["recall"]:
        pd.DataFrame({"precision": pr["precision"], "recall": pr["recall"]}).to_csv(
            run_dir / "pr_curve.csv", index=False
        )
    else:
        pd.DataFrame({"precision": [], "recall": []}).to_csv(run_dir / "pr_curve.csv", index=False)

    cm = build_confusion(y_true, df["y_pred"])
    dump_json(run_dir / "confusion_matrix.json", cm)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if pr["precision"] and pr["recall"]:
        plt.figure(figsize=(6, 5))
        plt.plot(pr["recall"], pr["precision"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve (test)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "pr_curve.png", dpi=140)
        plt.close()
