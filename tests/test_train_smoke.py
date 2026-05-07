import numpy as np

from dram2cnn.engine.metrics import classification_metrics, search_best_threshold


def test_metrics_and_threshold_search_smoke():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.55, 0.4])
    y_pred = (y_prob >= 0.5).astype(int)

    m = classification_metrics(y_true, y_pred, y_prob)
    assert "f1" in m and 0 <= m["f1"] <= 1
    assert m.get("roc_auc") is not None and 0.0 <= m["roc_auc"] <= 1.0

    m_one = classification_metrics(np.zeros(6, dtype=int), np.zeros(6, dtype=int), np.zeros(6))
    assert m_one.get("roc_auc") is None
    assert m_one.get("avg_precision") is None

    m_ap = classification_metrics(y_true, y_pred, y_prob)
    assert m_ap.get("avg_precision") is not None and 0.0 <= m_ap["avg_precision"] <= 1.0
    best = search_best_threshold(y_true, y_prob, 0.1, 0.9, 0.1)
    assert 0.1 <= best["threshold"] <= 0.9
