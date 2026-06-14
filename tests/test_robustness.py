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


def test_evaluate_writes_deployment_and_oracle_metrics(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "y_true": [0, 1, 1, 0],
            "y_pred": [0, 1, 0, 0],
            "y_prob": [0.2, 0.9, 0.55, 0.1],
            "threshold": [0.78, 0.78, 0.78, 0.78],
        }
    ).to_csv(pred, index=False)
    metrics_path = tmp_path / "metrics.json"
    evaluate(str(pred), str(metrics_path))
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["kpi_scope"] == "deployment"
    assert "f1" in payload
    assert "deployment" in payload
    assert payload["deployment"]["threshold"] == 0.78
    assert "oracle_on_test" in payload
    assert payload["oracle_on_test"]["metrics"] is not None
    assert (tmp_path / "best_threshold_on_eval.json").is_file()
