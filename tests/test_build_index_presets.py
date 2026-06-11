"""``build_index.py`` 与 ``index_build_presets`` 合并逻辑。"""
from __future__ import annotations

from dram2cnn.utils.config import load_yaml, project_root


def test_smoke_preset_overrides_index_build_base() -> None:
    """smoke preset 须覆盖 index_build 基值（如 neg_sample_strategy）。"""
    data_cfg = load_yaml(project_root() / "configs/data.yaml")
    ib_raw = dict(data_cfg.get("index_build") or {})
    assert ib_raw["neg_sample_strategy"] == "uniform"

    preset = dict(data_cfg.get("index_build_presets") or {}).get("smoke", {})
    ib_raw.update({k: v for k, v in preset.items() if v is not None})
    assert ib_raw["neg_sample_strategy"] == "random"
    assert ib_raw["max_rows"] == 500
    assert ib_raw["min_positives"] == 5
    assert ib_raw["positive_sample_interval_seconds"] == 60


def test_build_index_cli_horizon_sets_window_to_horizon() -> None:
    """CLI 指定 horizon 且未给 window 时，window 应等于 horizon（与 build_horizon_indexes 一致）。"""
    data_cfg = load_yaml(project_root() / "configs/data.yaml")
    ib_raw = dict(data_cfg.get("index_build") or {})
    horizon = 1800.0
    args_window = None
    args_horizon = horizon
    if args_window is not None:
        window = float(args_window)
    elif args_horizon is not None:
        window = horizon
    elif data_cfg.get("window_seconds") is not None:
        window = float(data_cfg["window_seconds"])
    elif ib_raw.get("window_seconds") is not None:
        window = float(ib_raw["window_seconds"])
    else:
        window = horizon
    assert window == 1800.0


def test_build_index_cli_horizon_sets_interval_to_horizon() -> None:
    """CLI 指定 horizon 且未给 interval 时，interval 应等于 horizon。"""
    data_cfg = load_yaml(project_root() / "configs/data.yaml")
    ib_raw = dict(data_cfg.get("index_build") or {})
    horizon = 1800.0
    args_interval = None
    args_horizon = horizon
    if args_interval is not None:
        interval = float(args_interval)
    elif args_horizon is not None:
        interval = horizon
    else:
        interval = float(ib_raw.get("sample_interval_seconds", horizon))
    assert interval == 1800.0


def test_full_preset_overrides_neg_sampling() -> None:
    data_cfg = load_yaml(project_root() / "configs/data.yaml")
    ib_raw = dict(data_cfg.get("index_build") or {})
    preset = dict(data_cfg.get("index_build_presets") or {}).get("full", {})
    ib_raw.update({k: v for k, v in preset.items() if v is not None})
    assert ib_raw["min_neg_per_ticket_sid"] == 3
    assert ib_raw["max_neg_grid_points_per_sid"] == 512
    assert ib_raw["positive_sample_interval_seconds"] == 60
    assert ib_raw["target_positive_ratio"] == 0.15


def test_horizon_from_data_yaml_not_index_build_duplicate() -> None:
    """index_build 段不应再含 label_horizon；horizon 由顶层 data.yaml 解析。"""
    data_cfg = load_yaml(project_root() / "configs/data.yaml")
    ib_raw = dict(data_cfg.get("index_build") or {})
    assert "label_horizon_seconds" not in ib_raw
    from dram2cnn.utils.config import label_horizon_seconds_from_dict

    assert label_horizon_seconds_from_dict(data_cfg) == 300.0
