"""阈值搜索 objective 与工程向 min_recall。"""
from __future__ import annotations

import numpy as np

from dram2cnn.engine.metrics import search_threshold


def test_min_recall_prefers_high_recall_then_precision() -> None:
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02])
    best = search_threshold(y, p, 0.05, 0.95, 0.05, objective="min_recall", min_recall=0.75)
    assert best["recall"] >= 0.75
    assert best["objective"] == "min_recall"


def test_recall_first_respects_min_precision() -> None:
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.55, 0.4, 0.1])
    best = search_threshold(y, p, 0.1, 0.9, 0.1, objective="recall_first", min_precision=0.5)
    assert best["precision"] >= 0.5 - 1e-9


def test_min_recall_respects_min_precision_floor() -> None:
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.95, 0.85, 0.55, 0.45, 0.5, 0.35, 0.2, 0.1])
    best = search_threshold(
        y, p, 0.1, 0.95, 0.05,
        objective="min_recall", min_recall=0.75, min_precision=0.6,
    )
    assert best["recall"] >= 0.75 - 1e-9
    assert best["precision"] >= 0.6 - 1e-9
