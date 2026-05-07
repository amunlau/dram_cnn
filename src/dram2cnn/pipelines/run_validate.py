from __future__ import annotations

"""实验预检命令行入口。

默认 **强制检查** `data/splits/*.npy` 是否与 `model.yaml` 的 C/H/W（及 tabular）一致。
尚未预处理时，请传 **`--no-require-splits`**，只做配置与原始路径校验。
"""

import argparse
import json

from dram2cnn.engine.validator import validate_experiment
from dram2cnn.utils.config import load_experiment_config


def main(exp_config: str = "configs/experiments/exp_baseline.yaml", require_splits: bool = True):
    """加载 YAML，调用 `validate_experiment`，向 stdout 打印 JSON 报告。"""
    cfg = load_experiment_config(exp_config)
    report = validate_experiment(cfg, require_splits=require_splits)
    print(json.dumps({"status": "ok", "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="校验实验配置与数据 shape（推荐在 train 前执行）。",
        epilog="无切分时: python -m dram2cnn.pipelines.run_validate <yaml> --no-require-splits",
    )
    parser.add_argument(
        "exp_config",
        nargs="?",
        default="configs/experiments/exp_baseline.yaml",
        help="实验 YAML",
    )
    parser.add_argument(
        "--no-require-splits",
        action="store_true",
        help="不检查 data/splits（仅当尚未 preprocess 时使用）",
    )
    args = parser.parse_args()
    main(args.exp_config, require_splits=not args.no_require_splits)

