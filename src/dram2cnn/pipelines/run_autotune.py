from __future__ import annotations

"""顺序执行 预处理（含校验）→ train → infer → evaluate。

`preprocess_main` 结束时已内嵌 `validate_experiment`（与 `scripts/preprocess.sh` 一致）。

用途：本地/regression 端到端链路检查。**实验列表 `EXPS`** 请在运行前按需求修改
（默认 `exp_baseline`，已关闭 DDP 与显存 batch 探测）。若索引过期可先：`bash scripts/build_index.sh smoke`。
"""

import subprocess
import traceback

from dram2cnn.pipelines.run_evaluate import main as eval_main
from dram2cnn.pipelines.run_preprocess import main as preprocess_main
from dram2cnn.utils.config import project_root

EXPS = [
    "configs/experiments/exp_baseline.yaml",
]

PROJECT_ROOT = project_root()


def main():
    """依次对 `EXPS` 中每个 YAML 跑完整流水线（适合冒烟 / CI）。"""
    for exp in EXPS:
        print(f"===== Running {exp} =====", flush=True)
        try:
            print(f"[{exp}] step=preprocess (includes validate)", flush=True)
            preprocess_main(exp)
            print(f"[{exp}] step=train (via scripts/train.sh)", flush=True)
            subprocess.run(
                ["bash", str(PROJECT_ROOT / "scripts/train.sh"), exp],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            print(f"[{exp}] step=infer (via scripts/infer.sh)", flush=True)
            subprocess.run(
                ["bash", str(PROJECT_ROOT / "scripts/infer.sh"), exp],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            print(f"[{exp}] step=evaluate", flush=True)
            eval_main(exp)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
