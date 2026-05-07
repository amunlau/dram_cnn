from __future__ import annotations

"""评估命令行入口。

默认读取 **`infer` 会话目录**下的 `predictions.csv`（路径由 `base.evaluate_run_id` /
`.latest_infer_session` 解析），计算分类指标并写入同目录的 **`metrics.json`**，
可选生成 PR 曲线数据与混淆矩阵（见 `engine.evaluator`）。
"""

import argparse
from pathlib import Path

from dram2cnn.engine.evaluator import evaluate
from dram2cnn.utils.config import load_experiment_config
from dram2cnn.utils.run_layout import resolve_evaluate_artifact_dir


def _project_root() -> Path:
    """返回仓库根目录路径。"""
    return Path(__file__).resolve().parents[3]


def main(exp_config: str = "configs/experiments/exp_baseline.yaml"):
    """加载配置，解析预测 CSV 路径，调用 `evaluate` 写出 metrics 等。"""
    cfg = load_experiment_config(exp_config)
    art_dir = resolve_evaluate_artifact_dir(_project_root(), cfg["base"])
    metrics_path = art_dir / "metrics.json"
    evaluate(str(art_dir / "predictions.csv"), str(metrics_path))
    print(f"Evaluate done: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="读取 predictions.csv 计算 metrics.json（及 PR/混淆矩阵等）。",
        epilog="示例: python -m dram2cnn.pipelines.run_evaluate configs/experiments/exp_baseline.yaml",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_baseline.yaml",
        help="实验 YAML",
    )
    args = parser.parse_args()
    main(args.exp_config)
