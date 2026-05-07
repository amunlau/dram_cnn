"""``dram2cnn.utils.preprocess_parallelism``：默认 ~60% 逻辑核与显式覆盖。"""

from dram2cnn.utils.preprocess_parallelism import (
    DEFAULT_PREPROCESS_PARALLEL_FRAC,
    resolve_preprocess_worker_count,
)


def test_default_frac_is_point_six():
    assert DEFAULT_PREPROCESS_PARALLEL_FRAC == 0.6


def test_explicit_preprocess_workers_wins():
    cfg = {"preprocess_workers": 3, "preprocess_parallel_frac": 0.1}
    assert resolve_preprocess_worker_count(cfg) == 3


def test_null_workers_uses_cpu_times_frac(monkeypatch):
    monkeypatch.setattr("dram2cnn.utils.preprocess_parallelism.os.cpu_count", lambda: 10)
    cfg = {"preprocess_workers": None, "preprocess_parallel_frac": 0.6}
    assert resolve_preprocess_worker_count(cfg) == 6
    cfg2 = {"preprocess_parallel_frac": 0.6}  # missing preprocess_workers key
    assert resolve_preprocess_worker_count(cfg2) == 6


def test_frac_clamped_and_at_least_one_worker(monkeypatch):
    monkeypatch.setattr("dram2cnn.utils.preprocess_parallelism.os.cpu_count", lambda: 2)
    assert resolve_preprocess_worker_count({"preprocess_parallel_frac": 0.01}) == 1
    assert resolve_preprocess_worker_count({"preprocess_parallel_frac": 99.0}) == 2
