import pytest

from dram2cnn.utils.gpu_launch import (
    LaunchPlan,
    merged_scope_section,
    resolve_ddp_enabled,
    resolve_nproc_per_node,
)


@pytest.mark.parametrize(
    "raw,cuda_count,expected",
    [
        ("auto", 0, False),
        ("auto", 1, False),
        ("auto", 3, True),
        (True, 2, True),
        (False, 4, False),
        ("false", 4, False),
        ("true", 1, True),
    ],
)
def test_resolve_ddp_enabled(raw, cuda_count, expected):
    assert resolve_ddp_enabled(raw, "cuda", cuda_count) is expected
    assert resolve_ddp_enabled(raw, "cpu", cuda_count) is False


def test_resolve_ddp_enabled_rejects_unknown():
    with pytest.raises(ValueError, match="enabled"):
        resolve_ddp_enabled("maybe", "cuda", 2)


@pytest.mark.parametrize(
    "raw,cuda_count,expected",
    [
        ("auto", 0, 1),
        ("auto", 4, 4),
        (2, 4, 2),
        (0, 3, 1),
    ],
)
def test_resolve_nproc_per_node(raw, cuda_count, expected):
    assert resolve_nproc_per_node(raw, cuda_count) == expected


def test_launch_plan_validate_nproc_exceeds_gpus():
    plan = LaunchPlan(
        mode="train",
        device="cuda",
        configured_device="auto",
        ddp_enabled=True,
        nproc_per_node=4,
        cuda_count=2,
        using_all_visible_gpus=False,
        batch_size=64,
        batch_size_auto=False,
        gpu_info=[],
        vram_budget_frac=0.8,
    )
    with pytest.raises(RuntimeError, match="nproc_per_node"):
        plan.validate()


def test_merged_scope_section_prefers_scope_override():
    cfg = {
        "base": {"gpu": {"vram_budget_frac": 0.8, "batch_size_max": 512}, "ddp": {"enabled": "auto"}},
        "infer": {"gpu": {"batch_size_max": 2048}, "ddp": {"enabled": False}},
    }
    gpu = merged_scope_section(cfg, "gpu", "infer")
    ddp = merged_scope_section(cfg, "ddp", "infer")
    assert gpu["vram_budget_frac"] == 0.8
    assert gpu["batch_size_max"] == 2048
    assert ddp["enabled"] is False
