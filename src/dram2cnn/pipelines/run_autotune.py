from __future__ import annotations

"""顺序执行 preprocess → validate → train → infer → evaluate。

用途：本地/regression 一键冒烟。**实验列表 `EXPS`** 请在运行前按需求修改
（默认仅 `exp_baseline`）。失败时打印 traceback 并抛出异常。"""

import traceback

from dram2cnn.engine.validator import validate_experiment
from dram2cnn.pipelines.run_evaluate import main as eval_main
from dram2cnn.pipelines.run_infer import main as infer_main
from dram2cnn.pipelines.run_preprocess import main as preprocess_main
from dram2cnn.pipelines.run_train import main as train_main
from dram2cnn.utils.config import load_experiment_config

EXPS = [
    # 在此列出要串行跑通的实验 YAML（可复制configs/experiments/下的路径）
    "configs/experiments/exp_baseline.yaml",
]


def main():
    """依次对 `EXPS` 中每个 YAML 跑完整流水线（适合冒烟 / CI）。"""
    for exp in EXPS:
        print(f"===== Running {exp} =====", flush=True)
        try:
            print(f"[{exp}] step=preprocess", flush=True)
            preprocess_main(exp)
            print(f"[{exp}] step=validate", flush=True)
            validate_experiment(load_experiment_config(exp), require_splits=True)
            print(f"[{exp}] step=train", flush=True)
            train_main(exp)
            print(f"[{exp}] step=infer", flush=True)
            infer_main(exp)
            print(f"[{exp}] step=evaluate", flush=True)
            eval_main(exp)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
