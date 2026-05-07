from pathlib import Path

from dram2cnn.utils.run_layout import (
    publish_infer_session_pointer,
    publish_train_session_pointer,
    resolve_checkpoint_dir,
    resolve_evaluate_artifact_dir,
    resolve_infer_output_dir,
    resolve_train_run_dir,
    train_log_path,
)


def test_resolve_train_run_dir_versioned_creates_timestamped_child(tmp_path):
    base = {"run_name": "demo_exp", "versioned_runs": True}
    rd = resolve_train_run_dir(tmp_path, base, world_size=1)
    assert rd.parent == tmp_path / "runs" / "demo_exp"
    assert len(rd.name) >= 15  # YYYYMMDD_HHMMSS_ffffff
    assert (rd.parent / rd.name).is_dir()


def test_resolve_train_run_dir_flat(tmp_path):
    base = {"run_name": "flat_exp", "versioned_runs": False}
    rd = resolve_train_run_dir(tmp_path, base, world_size=1)
    assert rd == tmp_path / "runs" / "flat_exp"


def test_versioned_false_ignores_run_id(tmp_path):
    base = {"run_name": "flat2", "versioned_runs": False, "run_id": "ignored"}
    rd = resolve_train_run_dir(tmp_path, base, world_size=1)
    assert rd == tmp_path / "runs" / "flat2"


def test_versioned_false_infer_ignores_infer_run_id(tmp_path):
    base = {"run_name": "flat3", "versioned_runs": False, "infer_run_id": "ignored"}
    inf = resolve_infer_output_dir(tmp_path, base)
    assert inf == tmp_path / "runs" / "flat3"


def test_checkpoint_pointer_roundtrip(tmp_path):
    base = {"run_name": "t", "versioned_runs": True}
    sess = tmp_path / "runs" / "t" / "sess_a"
    (sess / "checkpoints").mkdir(parents=True)
    (sess / "checkpoints" / "grid_cnn.pt").write_bytes(b"x")
    publish_train_session_pointer(tmp_path, base, sess)
    ck = resolve_checkpoint_dir(tmp_path, base)
    assert ck.resolve() == sess.resolve()


def test_infer_output_and_evaluate_pointer(tmp_path):
    base = {"run_name": "t2", "versioned_runs": True}
    inf = resolve_infer_output_dir(tmp_path, base)
    assert inf.name.startswith("infer_")
    (inf / "predictions.csv").write_text("y_true,y_pred\n0,0\n", encoding="utf-8")
    publish_infer_session_pointer(tmp_path, base, inf)
    art = resolve_evaluate_artifact_dir(tmp_path, base)
    assert art.resolve() == inf.resolve()
    assert (art / "predictions.csv").is_file()


def test_train_log_path_flat_vs_session(tmp_path):
    base = {"run_name": "rn"}
    rr = tmp_path / "runs" / "rn"
    rr.mkdir(parents=True)
    assert train_log_path(tmp_path, base, rr).name == "train.log"
    sub = rr / "20260101_120000_000000"
    sub.mkdir()
    assert "train_" in train_log_path(tmp_path, base, sub).name
