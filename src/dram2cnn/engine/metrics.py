from __future__ import annotations

"""二分类指标：准确率、F1、精确率、召回率；若提供 **`y_prob`** 则计算 ROC-AUC、PR-AUC。

另含混淆矩阵、PR 曲线构造与阈值搜索辅助函数，供 `evaluator` / Notebook 使用。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_prob=None):
    """计算核心分类指标；若提供 ``y_prob`` 则额外计算 ROC-AUC 与 Average Precision（PR-AUC）。"""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        yt = np.asarray(y_true)
        y_score = np.asarray(y_prob, dtype=float)
        if len(np.unique(yt)) < 2:
            out["roc_auc"] = None
            out["avg_precision"] = None
        else:
            try:
                out["roc_auc"] = float(roc_auc_score(yt, y_score))
            except ValueError:
                out["roc_auc"] = None
            # PR-AUC（Average Precision）：极不平衡下往往比 ROC-AUC 更能反映「检出正类」的排序能力。
            try:
                out["avg_precision"] = float(average_precision_score(yt, y_score))
            except ValueError:
                out["avg_precision"] = None
    return out


def search_best_threshold(y_true, y_prob, t_min=0.1, t_max=0.9, step=0.02):
    """在验证集上对概率做阈值网格搜索，使 F1 最大。

    用于极度不平衡场景下替代固定 0.5；搜索区间与步长由 configs/train.yaml 的
    threshold_search 配置。返回的最优阈值会写入训练产物并在推理阶段用于二分类决策。
    """
    best = {"threshold": 0.5, "f1": -1.0}
    t = t_min
    while t <= t_max + 1e-8:
        pred = (np.asarray(y_prob) >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(round(t, 4)), "f1": float(f1)}
        t += step
    return best


def build_pr_curve(y_true, y_prob):
    """构建 precision-recall 曲线数组用于绘图/导出。"""
    p, r, thr = precision_recall_curve(y_true, y_prob)
    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "thresholds": thr.tolist(),
    }


def build_confusion(y_true, y_pred):
    """以 JSON 友好格式返回混淆矩阵条目。"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
