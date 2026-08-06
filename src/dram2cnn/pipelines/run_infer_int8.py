"""W8A8 packed INT8 参考推理入口。

默认读取训练会话下 ``.latest_quant_session`` 中的 ``grid_cnn_w8a8_packed.pt``，
对测试集写出 ``predictions_int8.csv``（列格式与 FP32 ``predictions.csv`` 兼容）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from dram2cnn.engine.infer_int8 import predict_int8
from dram2cnn.utils.config import load_experiment_config, project_root
from dram2cnn.utils.logger import get_logger
from dram2cnn.utils.run_layout import new_timestamp_id, read_pointer_dir, write_pointer


def _resolve_train_dir(root: Path, base: dict, train_dir: str | None) -> Path:
    if train_dir:
        p = Path(train_dir)
        return p if p.is_absolute() else (root / p)
    run_name = str(base.get("run_name", "exp_tianchi_full_f1_ddp"))
    run_root = root / "runs" / run_name
    latest = read_pointer_dir(run_root, ".latest_train_session", "latest_train")
    if latest is None:
        raise FileNotFoundError(
            f"未找到最新训练会话指针：{run_root}/.latest_train_session；请传 --train-dir"
        )
    return latest


def _resolve_pack(train_dir: Path, quant_dir: str | None, pack: str | None) -> tuple[Path, Path]:
    """返回 (pack_path, quant_session_dir)。"""
    if pack:
        p = Path(pack)
        pack_path = p if p.is_absolute() else (Path.cwd() / p)
        # 量化会话目录取 pack 父目录
        return pack_path, pack_path.parent

    qdir: Path | None = None
    if quant_dir:
        q = Path(quant_dir)
        qdir = q if q.is_absolute() else (train_dir / q if not q.exists() else q)
        if not qdir.is_absolute():
            qdir = (train_dir / quant_dir).resolve()
    else:
        qdir = read_pointer_dir(train_dir, ".latest_quant_session", "latest_quant")

    if qdir is None or not qdir.is_dir():
        raise FileNotFoundError(
            f"未找到量化会话（{train_dir}/.latest_quant_session）；"
            "请先 bash scripts/quantize.sh，或传 --quant-dir / --pack"
        )
    pack_path = qdir / "grid_cnn_w8a8_packed.pt"
    if not pack_path.is_file():
        raise FileNotFoundError(f"packed 模型不存在: {pack_path}")
    return pack_path, qdir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GridCNN-Deep W8A8 packed INT8 参考推理",
        epilog="示例: python -m dram2cnn.pipelines.run_infer_int8 configs/experiments/exp_full_f1_ddp.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_full_f1_ddp.yaml",
        help="实验 YAML（提供 model / splits_dir / run_name）",
    )
    parser.add_argument("--train-dir", default=None, help="训练会话目录（默认 latest_train）")
    parser.add_argument("--quant-dir", default=None, help="量化会话目录（默认 latest_quant）")
    parser.add_argument("--pack", default=None, help="grid_cnn_w8a8_packed.pt 路径")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=None, help="覆盖 packed 内阈值")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="推理设备（单卡；非 DDP）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="输出目录（默认在 run 下新建 infer_int8_<ts>）",
    )
    args = parser.parse_args(argv)

    root = project_root()
    cfg = load_experiment_config(args.exp_config)
    train_dir = _resolve_train_dir(root, cfg["base"], args.train_dir)
    pack_path, quant_dir = _resolve_pack(train_dir, args.quant_dir, args.pack)

    run_name = str(cfg["base"].get("run_name", "exp_tianchi_full_f1_ddp"))
    run_root = root / "runs" / run_name
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    else:
        out_name = f"infer_int8_{new_timestamp_id()}"
        out_dir = run_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pointer(run_root, ".latest_infer_int8_session", out_dir.name)

    logger = get_logger(out_dir / "infer_int8.log")
    logger.info("exp_config=%s", args.exp_config)
    logger.info("train_dir=%s", train_dir.resolve())
    logger.info("quant_dir=%s", quant_dir.resolve())
    logger.info("pack=%s", pack_path.resolve())
    logger.info("out_dir=%s", out_dir.resolve())
    logger.info("device=%s", args.device)

    splits = Path(cfg["data"].get("splits_dir", "data/splits"))
    if not splits.is_absolute():
        splits = root / splits

    thr = args.threshold
    if thr is None:
        thr_path = train_dir / "best_threshold.json"
        if thr_path.is_file():
            thr = float(json.loads(thr_path.read_text(encoding="utf-8"))["threshold"])

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    result = predict_int8(
        model_cfg=cfg["model"],
        pack_path=pack_path,
        splits_dir=splits,
        output_csv=out_dir / "predictions.csv",
        device=str(device),
        batch_size=int(args.batch_size),
        threshold=thr,
    )
    metrics = result["metrics"]
    logger.info(
        "INT8 deployment: F1=%.4f P=%.4f R=%.4f thr=%.4f",
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        result["threshold"],
    )
    # 便于与 quant 会话对照
    (out_dir / "metrics_int8.json").write_text(
        json.dumps(
            {
                "backend": "int8_packed_reference",
                "pack_path": str(pack_path.resolve()),
                "quant_dir": str(quant_dir.resolve()),
                "threshold": result["threshold"],
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"INT8 infer done: {out_dir}")
    print(
        f"  F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"threshold={result['threshold']:.4f}"
    )
    print(f"  predictions: {result['predictions_csv']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
