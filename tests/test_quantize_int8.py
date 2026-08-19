"""W8A8 PTQ / packed INT8 参考推理单元测试（小模型、CPU）。

当前方案：逐层（per-tensor）对称 signed INT8，zero_point=0。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dram2cnn.engine.infer_int8 import load_packed_int8, predict_int8, quantize_activation_int8
from dram2cnn.engine.quantize_ptq import (
    FoldedConvReLU,
    QuantGridCNN,
    fold_batchnorm,
    iter_quant_layers,
    pack_int8_weights,
    qparams_per_tensor_symmetric,
    qparams_weight_per_tensor_symmetric,
    fake_quant_act_per_tensor,
    fake_quant_weight_per_tensor,
    fake_quant_symmetric,
)
from dram2cnn.models.grid_cnn import GridCNN, _ConvBNReLU
from dataclasses import asdict


def _tiny_cfg(**overrides) -> dict:
    cfg = {
        "arch": "grid_cnn",
        "in_channels": 4,
        "grid_height": 16,
        "grid_width": 8,
        "num_classes": 2,
        "stem_channels": 8,
        "backbone_channels": [8, 16],
        "blocks_per_stage": 1,
        "head_hidden": 16,
        "dropout": 0.0,
        "tabular_dim": 0,
        "use_batch_norm": True,
        "conv_kernel_size": 3,
        "head_bias": False,
    }
    cfg.update(overrides)
    return cfg


def test_qparams_per_layer_symmetric_int8():
    w = torch.tensor([[-2.0, 1.0], [0.5, -0.5]])
    scale, zp = qparams_weight_per_tensor_symmetric(w)
    assert scale.ndim == 0 or scale.numel() == 1
    assert int(zp.item()) == 0
    assert abs(float(scale) - (2.0 / 127.0)) < 1e-6
    s, z = qparams_per_tensor_symmetric(3.0)
    assert s > 0 and z == 0
    x = torch.linspace(-3, 3, 16)
    xq = fake_quant_symmetric(x, s)
    assert xq.shape == x.shape
    assert float(xq.abs().max()) <= 3.0 + 1e-5
    wq = fake_quant_weight_per_tensor(w, scale)
    assert wq.shape == w.shape
    # zp=0 路径
    xq2 = fake_quant_act_per_tensor(x, s, 0)
    assert torch.allclose(xq, xq2)
    q = quantize_activation_int8(x, s, 0)
    assert float(q.min()) >= -128
    assert float(q.max()) <= 127


def test_fold_batchnorm_with_bn_and_identity():
    cfg = _tiny_cfg(use_batch_norm=True)
    m = GridCNN(cfg)
    with torch.no_grad():
        _ = m(torch.randn(2, 4, 16, 8))
    fold_batchnorm(m, force_zero_bias=True)
    assert type(m.stem).__name__ == "FoldedConvReLU"
    assert isinstance(m.stem, FoldedConvReLU)
    for mod in m.modules():
        assert not isinstance(mod, _ConvBNReLU)

    cfg2 = _tiny_cfg(use_batch_norm=False)
    m2 = GridCNN(cfg2)
    fold_batchnorm(m2, force_zero_bias=True)
    assert isinstance(m2.stem, FoldedConvReLU)
    y = m2(torch.randn(2, 4, 16, 8))
    assert y.shape == (2, 2)


def test_pack_and_int8_infer_smoke(tmp_path: Path):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = GridCNN(cfg)
    with torch.no_grad():
        _ = model(torch.randn(4, 4, 16, 8))
    fold_batchnorm(model, force_zero_bias=True)
    qmodel = QuantGridCNN(model)
    for name, kind, mod in iter_quant_layers(qmodel):
        w = mod.conv.weight if kind == "conv" else mod.weight  # type: ignore[attr-defined]
        scale, _ = qparams_weight_per_tensor_symmetric(w)
        from dram2cnn.engine.quantize_ptq import LayerQuantSpec

        qmodel.set_spec(
            LayerQuantSpec(
                name=name,
                kind=kind,
                weight_scale=[float(scale.item())],
                act_in_scale=0.05,
                act_in_zp=0,
                keep_fp32=False,
            )
        )
    qmodel.use_hard_round = True
    qmodel.enable_act_quant = True
    qmodel.enable_weight_quant = True

    export = {
        "format": "test",
        "threshold": 0.5,
        "protect_stem": False,
        "protect_head_out": False,
        "layer_specs": {k: asdict(v) for k, v in qmodel.layer_specs.items()},
        "state_dict": {
            k: v.detach().cpu() for k, v in qmodel.state_dict().items() if not k.startswith("adaround")
        },
    }
    pack_path = tmp_path / "packed.pt"
    stats = pack_int8_weights(qmodel, export, pack_path, model_cfg=cfg)
    assert pack_path.is_file()
    assert stats["theoretical_compression_ratio"] >= 1.0

    int8_m, meta = load_packed_int8(pack_path, model_cfg=cfg, map_location="cpu")
    assert meta["format"].startswith("gridcnn_w8a8_packed")
    x = torch.randn(3, 4, 16, 8)
    with torch.no_grad():
        logits_fq = qmodel(x)
        logits_i8 = int8_m(x)
    assert logits_i8.shape == (3, 2)
    assert torch.allclose(logits_fq, logits_i8, atol=1e-4, rtol=1e-3)

    n = 8
    x_path = tmp_path / "X_test.npy"
    y_path = tmp_path / "y_test.npy"
    np.save(x_path, np.random.randn(n, 4, 16, 8).astype(np.float32))
    np.save(y_path, np.array([0, 1] * (n // 2), dtype=np.int64))
    out_csv = tmp_path / "predictions.csv"
    result = predict_int8(
        model_cfg=cfg,
        pack_path=pack_path,
        splits_dir=tmp_path,
        output_csv=out_csv,
        device="cpu",
        batch_size=4,
        threshold=0.5,
    )
    assert out_csv.is_file()
    assert "f1" in result["metrics"]
    assert result["metrics"]["threshold"] == 0.5


def test_quant_grid_cnn_rejects_tabular():
    cfg = _tiny_cfg(tabular_dim=15, tabular_fuse_dim=8)
    m = GridCNN(cfg)
    with torch.no_grad():
        _ = m(torch.randn(2, 4, 16, 8), torch.randn(2, 15))
    fold_batchnorm(m)
    try:
        QuantGridCNN(m)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "tabular_dim" in str(exc)
