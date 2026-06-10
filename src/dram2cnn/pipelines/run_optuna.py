from __future__ import annotations

"""Optuna 驱动的 GridCNN-Deep 超参搜索。

前置：已对目标实验完成 preprocess + validate（存在 `data/splits/`）。
每个 trial 在 `runs/optuna/<study>/<trial_id>/` 落盘；训练经 subprocess 调用 `run_train`（与 `train.sh` 相同 DDP / 显存策略）。

可搜结构超参（见 `configs/hpo.yaml`）：
- `backbone_depth` → 写入 `model.backbone_channels`（stage 数 = 列表长度）；
- `blocks_per_stage` / `use_residual`：控制每个 stage 内卷积块数与是否残差；
- `conv_kernel_size`、`head_hidden`、`dropout`：卷积核与 MLP 头；
- `lr` / `weight_decay`：优化器。

用法：
  PYTHONPATH=src python3 -m dram2cnn.pipelines.run_optuna \\
    configs/experiments/exp_optuna_smoke.yaml configs/hpo.yaml
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from dram2cnn.engine.validator import validate_experiment
from dram2cnn.utils.config import deep_update, load_experiment_config, load_yaml, project_root
from dram2cnn.utils.gpu_launch import resolve_launch_plan_from_cfg
from dram2cnn.utils.io import dump_json
from dram2cnn.utils.logger import get_logger
from dram2cnn.utils.run_layout import resolve_train_run_dir
from dram2cnn.utils.seed import set_seed


def _run_trial_training(root: Path, exp_config: str, trial_cfg: dict, run_dir: Path) -> None:
    """按 GPU 启动计划 subprocess 训练（支持 DDP + 显存 batch 探测）。"""
    dump_json(run_dir / "config_snapshot.json", trial_cfg)
    plan = resolve_launch_plan_from_cfg(trial_cfg, "train")
    env = os.environ.copy()
    env["DRAM2CNN_CONFIG_SNAPSHOT"] = str(run_dir / "config_snapshot.json")
    if plan.batch_size_auto:
        env["DRAM2CNN_BATCH_SIZE"] = str(plan.batch_size)

    if plan.ddp_enabled and plan.nproc_per_node > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={plan.nproc_per_node}",
            "-m",
            "dram2cnn.pipelines.run_train",
            exp_config,
        ]
    else:
        cmd = [sys.executable, "-m", "dram2cnn.pipelines.run_train", exp_config]

    subprocess.run(cmd, env=env, cwd=str(root), check=True)


def _categorical_choices_for_optuna(choices: list[Any]) -> list[Any]:
    """Optuna categorical 须为 hashable 标量；YAML 中的 backbone 列表编码为 JSON 字符串。"""
    out: list[Any] = []
    for c in choices:
        if isinstance(c, list):
            out.append(json.dumps(c, separators=(",", ":")))
        else:
            out.append(c)
    return out


def _decode_categorical_value(name: str, value: Any) -> Any:
    """将 suggest 返回值还原为配置可用类型（兼容历史 tuple 与 JSON 字符串）。"""
    if name != "backbone_depth":
        return value
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _suggest_param(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """按 hpo.yaml 中 `type` 字段调用 Optuna suggest API。"""
    kind = str(spec.get("type", "")).lower()
    if kind == "categorical":
        raw = trial.suggest_categorical(name, _categorical_choices_for_optuna(list(spec["choices"])))
        return _decode_categorical_value(name, raw)
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
        )
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=spec.get("step"),
        )
    if kind == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    raise ValueError(f"Unknown search_space type for {name}: {kind}")


def _apply_trial_params(cfg: dict[str, Any], trial_params: dict[str, Any]) -> dict[str, Any]:
    """将 trial 采样结果映射到实验配置的 `model` / `train` 字段。"""
    out = copy.deepcopy(cfg)
    model = out.setdefault("model", {})
    train = out.setdefault("train", {})
    # backbone_depth 为 categorical 列表，直接作为 backbone_channels
    if "backbone_depth" in trial_params:
        model["backbone_channels"] = list(trial_params["backbone_depth"])
    for k in ("conv_kernel_size", "head_hidden", "dropout", "blocks_per_stage"):
        if k in trial_params:
            model[k] = trial_params[k]
    if "use_residual" in trial_params:
        model["use_residual"] = bool(trial_params["use_residual"])
    if "lr" in trial_params:
        train["lr"] = float(trial_params["lr"])
    if "weight_decay" in trial_params:
        train["weight_decay"] = float(trial_params["weight_decay"])
    return out


def _read_best_score(run_dir: Path, metric: str) -> float:
    """从 trial 训练目录读取最优 epoch 分数；列名优先用 HPO `metric`，与 `train.early_stop_metric` 对齐。"""
    ep_path = run_dir / "epoch_metrics.csv"
    if not ep_path.is_file():
        return float("-inf")
    df = pd.read_csv(ep_path)
    if df.empty:
        return float("-inf")
    for col in (metric, "early_stop_score", "avg_precision"):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                return float(series.max())
    return float("-inf")


def run_study(exp_config: str, hpo_config: str) -> optuna.Study:
    """运行 Optuna study：每个 trial 训练 GridCNN-Deep 并返回 HPO `metric` 列上的最优验证分数。

    前置：`data/splits/` 已存在且通过 `validate_experiment`。
    产物：`runs/optuna/<study_name>/best_trial.json` 与各 trial 的 `runs/<run_name>/` 目录。
    """
    root = project_root()
    base_cfg = load_experiment_config(exp_config)
    validate_experiment(base_cfg, require_splits=True)

    hpo_path = Path(hpo_config)
    if not hpo_path.is_absolute():
        hpo_path = root / hpo_path
    hpo = load_yaml(hpo_path)
    search_space = dict(hpo.get("search_space") or {})
    trial_train = dict(hpo.get("trial_train") or {})
    metric = str(hpo.get("metric", base_cfg.get("train", {}).get("early_stop_metric", "avg_precision")))
    study_name = str(hpo.get("study_name", "grid_cnn_hpo"))
    storage = hpo.get("storage")
    direction = str(hpo.get("direction", "maximize"))
    seed = int(hpo.get("seed", base_cfg.get("base", {}).get("seed", 42)))

    optuna_dir = root / "runs" / "optuna" / study_name
    optuna_dir.mkdir(parents=True, exist_ok=True)
    dump_json(optuna_dir / "base_config_snapshot.json", base_cfg)
    dump_json(optuna_dir / "hpo_config_snapshot.json", hpo)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        for pname, pspec in search_space.items():
            params[pname] = _suggest_param(trial, pname, pspec)

        trial_cfg = _apply_trial_params(base_cfg, params)
        trial_cfg = deep_update(
            trial_cfg,
            {
                "base": {
                    "run_name": f"{study_name}_trial{trial.number:04d}",
                    "seed": seed + trial.number,
                },
                "train": {
                    "epochs": int(trial_train.get("epochs", 40)),
                    "patience": int(trial_train.get("patience", 8)),
                    "batch_size": int(
                        trial_train.get(
                            "batch_size",
                            trial_cfg["train"].get("batch_size", 256),
                        )
                    ),
                    "early_stop_metric": metric,
                },
            },
        )

        set_seed(int(trial_cfg["base"]["seed"]))
        run_dir = resolve_train_run_dir(root, trial_cfg["base"], world_size=1)
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_json(run_dir / "optuna_trial_params.json", params)

        logger = get_logger(run_dir / "train.log")
        logger.info("Optuna trial %d params=%s", trial.number, json.dumps(params, ensure_ascii=False))

        _run_trial_training(root, exp_config, trial_cfg, run_dir)
        score = _read_best_score(run_dir, metric)
        trial.set_user_attr("run_dir", str(run_dir))
        logger.info("Optuna trial %d finished %s=%.6f", trial.number, metric, score)
        if not np.isfinite(score):
            raise optuna.TrialPruned(f"trial failed: {metric} is not finite (no valid epoch_metrics)")
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=bool(storage),
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=int(hpo.get("n_trials", 20)),
        timeout=hpo.get("timeout_seconds"),
        show_progress_bar=True,
    )

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(float(t.value))
    ]
    if not completed:
        raise RuntimeError(
            f"Optuna study '{study_name}' 无成功 trial（全部失败或被 prune）。"
            "请检查 data/splits、train 日志与 hpo.yaml。"
        )

    best = study.best_trial
    summary = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_run_dir": best.user_attrs.get("run_dir"),
        "metric": metric,
    }
    (optuna_dir / "best_trial.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return study


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna GridCNN 超参搜索")
    ap.add_argument("exp_config", nargs="?", default="configs/experiments/exp_optuna_smoke.yaml")
    ap.add_argument("hpo_config", nargs="?", default="configs/hpo.yaml")
    args = ap.parse_args()
    run_study(args.exp_config, args.hpo_config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
