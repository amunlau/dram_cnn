from dram2cnn.pipelines.run_optuna import (
    _apply_trial_params,
    _categorical_choices_for_optuna,
    _decode_categorical_value,
)


def test_categorical_choices_converts_lists_to_json_strings():
    assert _categorical_choices_for_optuna([[64, 128], 3, True]) == ['[64,128]', 3, True]


def test_decode_categorical_backbone_depth():
    assert _decode_categorical_value("backbone_depth", "[64,128,256]") == [64, 128, 256]
    assert _decode_categorical_value("backbone_depth", (64, 128, 256)) == [64, 128, 256]
    assert _decode_categorical_value("lr", 0.001) == 0.001


def test_apply_trial_params_backbone_depth_from_list():
    cfg = {"model": {}, "train": {}}
    out = _apply_trial_params(cfg, {"backbone_depth": [64, 128, 256], "lr": 0.001})
    assert out["model"]["backbone_channels"] == [64, 128, 256]
    assert out["train"]["lr"] == 0.001
