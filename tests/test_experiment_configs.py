"""所有实验 YAML 须能经 `load_experiment_config` 加载（含 validate_config）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from dram2cnn.utils.config import load_experiment_config
from dram2cnn.utils.gpu_launch import merged_scope_section, resolve_ddp_enabled

EXP_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"


@pytest.mark.parametrize(
    "yaml_name",
    sorted(p.name for p in EXP_DIR.glob("*.yaml")),
)
def test_experiment_yaml_loads(yaml_name: str) -> None:
    cfg = load_experiment_config(EXP_DIR / yaml_name)
    assert "data" in cfg and "model" in cfg and "train" in cfg


@pytest.mark.parametrize(
    "yaml_name",
    ["exp_baseline.yaml", "exp_horizon_30m.yaml", "exp_horizon_1h.yaml"],
)
def test_smoke_experiments_disable_ddp_and_auto_batch(yaml_name: str) -> None:
    """冒烟/对比实验须显式关闭 train/infer 的 DDP 与显存 batch 探测，避免 CI 意外多卡启动。"""
    cfg = load_experiment_config(EXP_DIR / yaml_name)
    for scope in ("train", "infer"):
        ddp_cfg = merged_scope_section(cfg, "ddp", scope)
        gpu_cfg = merged_scope_section(cfg, "gpu", scope)
        assert resolve_ddp_enabled(ddp_cfg.get("enabled", "auto"), "cuda", 3) is False
        assert gpu_cfg.get("auto_batch_size") is False


@pytest.mark.parametrize(
    "yaml_name",
    ["exp_full_recall_ddp.yaml", "exp_optuna_smoke.yaml", "exp_ddp_verify.yaml"],
)
def test_production_experiments_inherit_ddp_auto(yaml_name: str) -> None:
    """全量/Optuna/DDP 验证实验须继承 base 的多卡 auto 策略。"""
    cfg = load_experiment_config(EXP_DIR / yaml_name)
    for scope in ("train", "infer"):
        ddp_cfg = merged_scope_section(cfg, "ddp", scope)
        assert resolve_ddp_enabled(ddp_cfg.get("enabled", "auto"), "cuda", 3) is True


def test_full_recall_ddp_low_memory_preprocess_settings() -> None:
    """全量实验须启用 50% 系统 RAM 预算（继承 data.yaml）。"""
    cfg = load_experiment_config(EXP_DIR / "exp_full_recall_ddp.yaml")
    data = cfg["data"]
    assert data.get("ram_budget_frac") == 0.5
    assert data.get("ram_total_gib") is None
    assert cfg["train"].get("num_workers") == 2
    assert cfg["train"].get("prefetch_factor") == 2


def test_full_recall_sid_uses_sid_split_and_separate_splits_dir() -> None:
    cfg = load_experiment_config(EXP_DIR / "exp_full_recall_sid.yaml")
    assert cfg["data"]["split_strategy"] == "sid"
    assert cfg["data"]["splits_dir"].endswith("splits_sid")
    assert "full" in str(cfg["data"]["index_features_path"])


def test_baseline_uses_isolated_smoke_data_dirs() -> None:
    cfg = load_experiment_config(EXP_DIR / "exp_baseline.yaml")
    assert cfg["data"]["processed_dir"].endswith("processed_smoke")
    assert cfg["data"]["splits_dir"].endswith("splits_smoke")


def test_full_recall_ddp_uses_recall_guard_threshold_profile() -> None:
    cfg = load_experiment_config(EXP_DIR / "exp_full_recall_ddp.yaml")
    ts = cfg["train"]["threshold_search"]
    assert cfg["base"]["run_name"] == "exp_tianchi_full_recall_ddp"
    assert cfg["train"].get("threshold_profile") == "recall_guard"
    assert ts["objective"] == "min_recall"
    assert float(ts["min_recall"]) == 0.88
    assert float(ts["min_precision"]) == 0.55


def test_full_recall_ddp_balanced_threshold_profile() -> None:
    cfg = load_experiment_config(EXP_DIR / "exp_full_recall_ddp_balanced.yaml")
    ts = cfg["train"]["threshold_search"]
    assert cfg["train"].get("threshold_profile") == "balanced"
    assert float(ts["min_recall"]) == 0.82
    assert float(ts["min_precision"]) == 0.58


def test_full_f1_ddp_uses_f1_early_stop_and_threshold() -> None:
    cfg = load_experiment_config(EXP_DIR / "exp_full_f1_ddp.yaml")
    assert cfg["base"]["run_name"] == "exp_tianchi_full_f1_ddp"
    assert cfg["train"]["early_stop_metric"] == "best_f1"
    assert cfg["train"]["threshold_search"]["objective"] == "f1"


def test_full_precision_ddp_uses_precision_guard_profile() -> None:
    cfg = load_experiment_config(EXP_DIR / "exp_full_precision_ddp.yaml")
    assert cfg["train"].get("threshold_profile") == "precision_guard"
    assert cfg["base"]["run_name"] == "exp_tianchi_full_precision_ddp"


def test_baseline_sid_smoke_settings() -> None:
    cfg = load_experiment_config(EXP_DIR / "exp_baseline_sid.yaml")
    assert cfg["data"]["split_strategy"] == "sid"
    assert resolve_ddp_enabled(merged_scope_section(cfg, "ddp", "train").get("enabled", "auto"), "cuda", 3) is False
