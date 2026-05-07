from __future__ import annotations

"""实验 YAML 加载与合并：`configs/experiments/*.yaml` 中的 base/data/model/train/infer

通过 `include` 指向各片段文件，再用 `overrides` 深度合并。最终以 **绝对路径** 规范化所有 `*_path` / `*_dir`，
便于在不同工作目录下运行脚本。

**校验**：`validate_config` 会检查原始数据文件是否存在、`tabular_dim` 与 `TABULAR_DIM` 是否匹配等。
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from dram2cnn.data.inventory_encode import TABULAR_DIM


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(ref: str | Path, base_dir: Path) -> Path:
    p = Path(ref)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_config_include(ref: str | Path, exp_parent: Path, root: Path) -> Path:
    p = Path(ref)
    if p.is_absolute():
        return p
    next_to_exp = (exp_parent / p).resolve()
    if next_to_exp.is_file():
        return next_to_exp
    at_root = (root / p).resolve()
    if at_root.is_file():
        return at_root
    return next_to_exp


def _normalize_io_paths(obj: Any, project_root: Path) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out[k] = _normalize_io_paths(v, project_root)
            elif isinstance(v, str) and (k.endswith("_path") or k.endswith("_dir")):
                vp = Path(v)
                out[k] = str(vp if vp.is_absolute() else (project_root / vp).resolve())
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_normalize_io_paths(x, project_root) for x in obj]
    return obj


def validate_config(cfg: Dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    for k in ("mcelog_path", "tickets_path", "index_features_path"):
        p = Path(data_cfg.get(k, ""))
        if not p.exists():
            raise FileNotFoundError(f"Config error: `{k}` does not exist: {p}")

    inv = data_cfg.get("inventory_path")
    if inv:
        ip = Path(inv)
        if not ip.exists():
            raise FileNotFoundError(f"Config error: `inventory_path` does not exist: {ip}")

    if int(data_cfg.get("window_seconds", 0)) <= 0:
        raise ValueError("data.window_seconds must be > 0")

    split_strategy = data_cfg.get("split_strategy", "time")
    if split_strategy not in {"time", "sid"}:
        raise ValueError("Config error: `split_strategy` must be one of: time, sid")

    infer_mode = cfg.get("infer", {}).get("mode", "full")
    if infer_mode != "full":
        raise ValueError("Config error: `infer.mode` must be `full` for grid_cnn")

    m_tab = int(cfg.get("model", {}).get("tabular_dim", 0))
    if m_tab > 0 and m_tab != TABULAR_DIM:
        raise ValueError(
            f"Config error: model.tabular_dim={m_tab} 必须与 inventory 编码维度 TABULAR_DIM={TABULAR_DIM} 一致"
        )


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_experiment_config(exp_path: str | Path) -> Dict[str, Any]:
    exp_file = Path(exp_path).resolve()
    exp = load_yaml(exp_file)
    exp_parent = exp_file.parent
    root = project_root()

    cfg = {
        "base": load_yaml(_resolve_config_include(exp["base"], exp_parent, root)),
        "data": load_yaml(_resolve_config_include(exp["data"], exp_parent, root)),
        "model": load_yaml(_resolve_config_include(exp["model"], exp_parent, root)),
        "train": load_yaml(_resolve_config_include(exp["train"], exp_parent, root)),
        "infer": load_yaml(_resolve_config_include(exp["infer"], exp_parent, root)),
    }
    merged = deep_update(cfg, exp.get("overrides", {}))
    merged = _normalize_io_paths(merged, root)
    validate_config(merged)
    return merged
