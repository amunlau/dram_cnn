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
            try:
                out["avg_precision"] = float(average_precision_score(yt, y_score))
            except ValueError:
                out["avg_precision"] = None
    return out


def _f_beta(precision: float, recall: float, beta: float) -> float:
    if precision + recall <= 0:
        return 0.0
    b2 = beta * beta
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


def _score_at_threshold(y_true, y_prob, threshold: float) -> dict:
    pred = (np.asarray(y_prob) >= float(threshold)).astype(int)
    yt = np.asarray(y_true)
    p = float(precision_score(yt, pred, zero_division=0))
    r = float(recall_score(yt, pred, zero_division=0))
    f1 = float(f1_score(yt, pred, zero_division=0))
    return {"threshold": float(round(threshold, 4)), "precision": p, "recall": r, "f1": f1}


def search_threshold(
    y_true,
    y_prob,
    t_min: float = 0.1,
    t_max: float = 0.9,
    step: float = 0.02,
    *,
    objective: str = "f1",
    min_precision: float = 0.0,
    min_recall: float = 0.0,
    f_beta: float = 1.0,
) -> dict:
    """在验证集上对概率做阈值网格搜索。

    ``objective`` 可选：
    - ``f1``：最大化 F1（默认，历史行为）
    - ``f_beta``：最大化 F_beta（beta>1 偏向 recall，宁报勿漏可用 beta=2）
    - ``recall_first``：在 precision >= min_precision 前提下最大化 recall
    - ``min_recall``：在 recall >= min_recall 且 precision >= min_precision 前提下最大化 precision；
      若无满足者则先放宽 precision 约束，再取全局最高 recall（宁报勿漏）
    """
    obj = str(objective or "f1").lower().strip()
    candidates: list[dict] = []
    t = float(t_min)
    while t <= float(t_max) + 1e-8:
        row = _score_at_threshold(y_true, y_prob, t)
        candidates.append(row)
        t += float(step)

    if not candidates:
        return {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0, "objective": obj}

    def _pick_best(rows: list[dict], key_fn) -> dict:
        return max(rows, key=key_fn)

    if obj == "f1":
        best = _pick_best(candidates, lambda r: (r["f1"], r["recall"], -r["threshold"]))
    elif obj == "f_beta":
        beta = max(float(f_beta), 1e-6)

        def fb_key(r):
            return (_f_beta(r["precision"], r["recall"], beta), r["recall"], r["precision"])

        best = _pick_best(candidates, fb_key)
        best["f_beta"] = _f_beta(best["precision"], best["recall"], beta)
    elif obj == "recall_first":
        mp = float(min_precision)
        ok = [r for r in candidates if r["precision"] >= mp - 1e-12]
        pool = ok if ok else candidates
        best = _pick_best(pool, lambda r: (r["recall"], r["precision"], -r["threshold"]))
    elif obj == "min_recall":
        mr = float(min_recall)
        mp = float(min_precision)
        ok = [
            r for r in candidates
            if r["recall"] >= mr - 1e-12 and r["precision"] >= mp - 1e-12
        ]
        if ok:
            best = _pick_best(ok, lambda r: (r["precision"], r["recall"], -r["threshold"]))
        else:
            recall_ok = [r for r in candidates if r["recall"] >= mr - 1e-12]
            if recall_ok:
                best = _pick_best(recall_ok, lambda r: (r["precision"], r["recall"], -r["threshold"]))
            else:
                best = _pick_best(candidates, lambda r: (r["recall"], r["precision"], -r["threshold"]))
    else:
        raise ValueError(
            f"threshold_search.objective must be one of: f1, f_beta, recall_first, min_recall (got {objective!r})"
        )

    best["objective"] = obj
    if obj == "min_recall":
        best["min_recall_target"] = mr
        best["min_precision_floor"] = float(min_precision)
    elif obj == "recall_first":
        best["min_precision_floor"] = float(min_precision)
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
