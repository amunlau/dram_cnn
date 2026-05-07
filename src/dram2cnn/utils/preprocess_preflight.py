from __future__ import annotations

"""Notebook / 长耗时预处理前的轻量检查：原始文件是否存在、`window_seconds` 是否合理。

与 `validate_config` 中 data 路径规则一致，但不加载 model/train/infer，便于仅解析 `run_name` 等场景。
"""

from pathlib import Path

from dram2cnn.utils.config import _resolve_config_include, deep_update, load_yaml, project_root


def experiment_run_name(exp_yaml: str | Path) -> str:
    """从实验 yaml 解析 `run_name`（合并 `overrides.base`），无需校验 data 原始文件是否存在。"""
    exp_file = Path(exp_yaml).resolve()
    exp = load_yaml(exp_file)
    exp_parent = exp_file.parent
    root = project_root()
    base = load_yaml(_resolve_config_include(exp["base"], exp_parent, root))
    base = deep_update(base, (exp.get("overrides") or {}).get("base") or {})
    return str(base.get("run_name", "exp_tianchi_baseline"))


def check_preprocess_raw_inputs(exp_yaml: str | Path) -> None:
    """在 `materialize_dataset` / `run_preprocess` 前调用：避免缺索引表时长时间读 mcelog 后才失败。"""
    exp_file = Path(exp_yaml).resolve()
    exp = load_yaml(exp_file)
    exp_parent = exp_file.parent
    root = project_root()
    data = load_yaml(_resolve_config_include(exp["data"], exp_parent, root))
    data = deep_update(data, (exp.get("overrides") or {}).get("data") or {})
    for key in ("mcelog_path", "tickets_path", "index_features_path"):
        p = Path(data.get(key, ""))
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"无法进行预处理：{key} 文件不存在:\n  {p}\n\n"
                "请放置索引表（至少含 sid, predict_time）或修改 configs/data.yaml 中的路径。\n"
                "说明见 docs/data_dictionary.md。"
            )
    inv = data.get("inventory_path")
    if inv:
        ip = Path(inv)
        if not ip.is_absolute():
            ip = (root / ip).resolve()
        if not ip.is_file():
            raise FileNotFoundError(
                f"无法进行预处理：inventory_path 不存在:\n  {ip}\n\n"
                "可放置 inventory.csv，或在 data.yaml 中将 inventory_path 设为 null（tabular 全 0）。"
            )
    if int(data.get("window_seconds", 0)) <= 0:
        raise ValueError(
            "configs/data.yaml 中 window_seconds 必须 > 0。"
            "本工程不使用 window_days；请使用 window_seconds（秒）。"
        )
