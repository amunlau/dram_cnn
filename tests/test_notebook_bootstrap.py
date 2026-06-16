"""notebook_bootstrap 与 preflight 辅助测试。"""

from __future__ import annotations

from pathlib import Path

from dram2cnn.utils.config import project_root
from dram2cnn.utils.notebook_bootstrap import (
    clear_dram2cnn_modules,
    infer_project_root,
    prepare_notebook_kernel,
)
from dram2cnn.utils.pipeline_helpers import preflight_full_scale_index


def test_infer_project_root_from_notebooks_dir(tmp_path: Path) -> None:
    nb = tmp_path / "notebooks"
    nb.mkdir()
    assert infer_project_root(cwd=nb) == tmp_path


def test_notebook_setup_returns_project_root() -> None:
    from dram2cnn.utils.config import project_root
    from dram2cnn.utils.notebook_bootstrap import notebook_setup

    root = notebook_setup()
    assert root == project_root()


def test_clear_dram2cnn_modules_allows_fresh_import() -> None:
    import dram2cnn.utils.experiment_guards as eg

    assert hasattr(eg, "assert_full_scale_index_not_stub")
    n = clear_dram2cnn_modules()
    assert n >= 1
    from dram2cnn.utils.experiment_guards import assert_full_scale_index_not_stub

    assert callable(assert_full_scale_index_not_stub)


def test_preflight_full_scale_index_accepts_f1_config() -> None:
    root = project_root()
    exp = root / "configs/experiments/exp_full_f1_ddp.yaml"
    if not (root / "data/raw/features_index_h300s_step300s_full.csv").is_file():
        return
    cfg = preflight_full_scale_index(exp, root=root, context="test")
    assert "full" in str(cfg["data"]["index_features_path"]).lower()
