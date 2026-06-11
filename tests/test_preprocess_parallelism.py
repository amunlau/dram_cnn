"""``dram2cnn.utils.preprocess_parallelism``：CPU 并行与 RAM 预算。"""

from dram2cnn.utils.preprocess_parallelism import (
    DEFAULT_PREPROCESS_PARALLEL_FRAC,
    DEFAULT_RAM_BUDGET_FRAC,
    estimate_preprocess_peak_gib,
    resolve_preprocess_worker_count,
)


def test_default_frac_is_point_six():
    assert DEFAULT_PREPROCESS_PARALLEL_FRAC == 0.6


def test_default_ram_budget_frac_is_half():
    assert DEFAULT_RAM_BUDGET_FRAC == 0.5


def test_explicit_preprocess_workers_wins():
    cfg = {"preprocess_workers": 3, "preprocess_parallel_frac": 0.1, "ram_budget_frac": None}
    assert resolve_preprocess_worker_count(cfg) == 3


def test_null_workers_uses_cpu_times_frac(monkeypatch):
    monkeypatch.setattr("dram2cnn.utils.preprocess_parallelism.os.cpu_count", lambda: 10)
    cfg = {"preprocess_workers": None, "preprocess_parallel_frac": 0.6, "ram_budget_frac": None}
    assert resolve_preprocess_worker_count(cfg) == 6
    cfg2 = {"preprocess_parallel_frac": 0.6, "ram_budget_frac": None}
    assert resolve_preprocess_worker_count(cfg2) == 6


def test_frac_clamped_and_at_least_one_worker(monkeypatch):
    monkeypatch.setattr("dram2cnn.utils.preprocess_parallelism.os.cpu_count", lambda: 2)
    assert resolve_preprocess_worker_count({"preprocess_parallel_frac": 0.01, "ram_budget_frac": None}) == 1
    assert resolve_preprocess_worker_count({"preprocess_parallel_frac": 99.0, "ram_budget_frac": None}) == 2


def test_ram_budget_caps_workers_on_64gb_full_index(monkeypatch):
    """66k×31×128×32 全量 preprocess：高 CPU worker 应被 RAM 50% 预算压至 1。"""
    monkeypatch.setattr("dram2cnn.utils.preprocess_parallelism.os.cpu_count", lambda: 112)
    cfg = {
        "preprocess_parallel_frac": 0.85,
        "ram_total_gib": 64,
        "ram_budget_frac": 0.5,
    }
    n = 66320
    shape = (31, 128, 32)
    w = resolve_preprocess_worker_count(cfg, n_samples=n, grid_shape=shape, mcelog_path=None)
    assert w == 1
    peak = estimate_preprocess_peak_gib(n, shape, n_workers=w)
    assert peak <= 64 * 0.5 + 8  # X 数组本身可接近 50%；允许估算余量


def test_mcelog_max_rows_reduces_events_estimate():
    """mcelog_max_rows 截断扫描时不应按整文件体积估算 events。"""
    from dram2cnn.utils import preprocess_parallelism as pp

    # 800k 行 × 120B ≈ 91MiB
    capped = pp._estimate_mcelog_events_gib(__file__, mcelog_max_rows=800_000)
    full = pp._estimate_mcelog_events_gib(__file__, mcelog_max_rows=None)
    assert capped <= full
    assert capped < 0.2


def test_preprocess_ignore_ram_budget(monkeypatch):
    monkeypatch.setattr("dram2cnn.utils.preprocess_parallelism.os.cpu_count", lambda: 10)
    cfg = {
        "preprocess_parallel_frac": 0.6,
        "ram_total_gib": 64,
        "ram_budget_frac": 0.5,
        "preprocess_ignore_ram_budget": True,
    }
    assert (
        resolve_preprocess_worker_count(
            cfg, n_samples=66320, grid_shape=(31, 128, 32)
        )
        == 6
    )
