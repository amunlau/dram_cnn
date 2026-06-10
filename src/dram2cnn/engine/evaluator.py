from __future__ import annotations

"""评估预测 CSV 并导出分析产物。

默认读取与 predictions.csv 同目录写入 metrics.json；若有 y_prob 列则额外做阈值网格搜索与 PR 曲线，
得到的后验阈值文件名为 best_threshold_on_eval.json（与训练阶段 best_threshold.json 区分）。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dram2cnn.engine.metrics import build_confusion, build_pr_curve, classification_metrics, search_best_threshold
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
                f"predictions.csv 缺少必需列 `{col}`（当前列: {list(df.columns)}）。"
                "请确认 infer 步骤完整执行且 checkpoint 与 splits 匹配。"
            )

    # 当前预测标签对应阈值下的指标。
    y_prob_series = df["y_prob"] if "y_prob" in df.columns else None
    metrics = classification_metrics(df["y_true"], df["y_pred"], y_prob_series)
    dump_json(output_json, metrics)

    # 阈值搜索与 PR 曲线依赖概率列；手工 CSV 可能只有硬标签。
    # 此处网格步长 0.01 仅为测试集事后分析；训练验证阶段的阈值搜索步长见 configs/train.yaml → threshold_search。
    if y_prob_series is not None:
        y_prob_np = y_prob_series.to_numpy(dtype=float, copy=False)
        best_t = search_best_threshold(df["y_true"].to_numpy(), y_prob_np, 0.1, 0.9, 0.01)
        dump_json(run_dir / "best_threshold_on_eval.json", best_t)

        pr = build_pr_curve(df["y_true"], y_prob_np)
    else:
        best_t = {"threshold": None, "f1": None, "note": "y_prob column missing; skipped threshold search and PR curve"}
        dump_json(run_dir / "best_threshold_on_eval.json", best_t)
        pr = {"precision": [], "recall": [], "thresholds": []}
    if pr["precision"] and pr["recall"]:
        pd.DataFrame({"precision": pr["precision"], "recall": pr["recall"]}).to_csv(run_dir / "pr_curve.csv", index=False)
    else:
        pd.DataFrame({"precision": [], "recall": []}).to_csv(run_dir / "pr_curve.csv", index=False)

    cm = build_confusion(df["y_true"], df["y_pred"])
    dump_json(run_dir / "confusion_matrix.json", cm)

    # 保存轻量 PR 曲线图，便于快速查看模型行为。
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if pr["precision"] and pr["recall"]:
        plt.figure(figsize=(6, 5))
        plt.plot(pr["recall"], pr["precision"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "pr_curve.png", dpi=140)
        plt.close()
