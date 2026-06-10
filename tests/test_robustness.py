"""入口与评估阶段的鲁棒性回归测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dram2cnn.engine.evaluator import evaluate
from dram2cnn.engine.predictor import _resolve_threshold


def test_resolve_threshold_rejects_corrupt_json(tmp_path: Path) -> None:
    bad = tmp_path / "best_threshold.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="无法解析阈值文件"):
        _resolve_threshold(tmp_path)


def test_resolve_threshold_missing_file_uses_default(tmp_path: Path) -> None:
    assert _resolve_threshold(tmp_path, default=0.42) == 0.42


def test_resolve_threshold_reads_valid_json(tmp_path: Path) -> None:
    (tmp_path / "best_threshold.json").write_text(
        json.dumps({"threshold": 0.37}), encoding="utf-8"
    )
    assert _resolve_threshold(tmp_path) == 0.37


def test_evaluate_requires_prediction_columns(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.csv"
    pred.write_text("y_true\n0\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="y_pred"):
        evaluate(str(pred), str(tmp_path / "metrics.json"))
