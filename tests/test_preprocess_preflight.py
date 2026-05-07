from pathlib import Path

import pytest

from dram2cnn.utils.preprocess_preflight import check_preprocess_raw_inputs, experiment_run_name


def test_experiment_run_name_reads_baseline(tmp_path, monkeypatch):
    """使用最小实验 yaml，不触碰 data 原始文件。"""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "configs").mkdir()
    (root / "configs" / "base.yaml").write_text("run_name: my_exp\nseed: 1\n", encoding="utf-8")
    # 实验 yaml 放在项目根目录，保证 `configs/data.yaml` 相对路径解析到 tmp 内文件，而非真实仓库。
    exp = root / "exp.yaml"
    exp.write_text(
        "base: configs/base.yaml\n"
        "data: configs/data.yaml\n"
        "model: configs/model.yaml\n"
        "train: configs/train.yaml\n"
        "infer: configs/infer.yaml\n"
        "overrides:\n"
        "  base:\n"
        "    run_name: overridden\n",
        encoding="utf-8",
    )
    for name, content in [
        ("configs/data.yaml", "mcelog_path: x\ntickets_path: x\nindex_features_path: x\nwindow_seconds: 300\n"),
        ("configs/model.yaml", "arch: grid_cnn\n"),
        ("configs/train.yaml", "epochs: 1\n"),
        ("configs/infer.yaml", "mode: full\n"),
    ]:
        (root / name).write_text(content, encoding="utf-8")

    monkeypatch.chdir(root)
    from dram2cnn.utils import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "project_root", lambda: root)

    assert experiment_run_name(exp) == "overridden"


def test_check_preprocess_raw_inputs_missing_index(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "mcelog.csv").write_text("a\n", encoding="utf-8")
    (raw / "trouble_tickets.csv").write_text("a\n", encoding="utf-8")
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "base.yaml").write_text("run_name: e\nversioned_runs: false\ndevice: cpu\n", encoding="utf-8")
    (root / "configs" / "data.yaml").write_text(
        f"mcelog_path: {raw / 'mcelog.csv'}\n"
        f"tickets_path: {raw / 'trouble_tickets.csv'}\n"
        f"index_features_path: {raw / 'missing_index.csv'}\n"
        "window_seconds: 300\n",
        encoding="utf-8",
    )
    for stub in ("model.yaml", "train.yaml", "infer.yaml"):
        (root / "configs" / stub).write_text("x: 1\n", encoding="utf-8")
    exp = root / "exp.yaml"
    exp.write_text(
        "base: configs/base.yaml\n"
        "data: configs/data.yaml\n"
        "model: configs/model.yaml\n"
        "train: configs/train.yaml\n"
        "infer: configs/infer.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    from dram2cnn.utils import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "project_root", lambda: root)

    with pytest.raises(FileNotFoundError, match="index_features_path"):
        check_preprocess_raw_inputs(exp)
