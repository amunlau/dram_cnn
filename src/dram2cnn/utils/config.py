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
    """读取单个 YAML 文件；空文件或仅注释时返回空 dict，避免下游对 None 判空。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def project_root() -> Path:
    """仓库根目录：本文件位于 src/dram2cnn/utils/config.py，向上 3 级到含 configs/、data/ 的根。"""
    return Path(__file__).resolve().parents[3]


def label_horizon_seconds_from_dict(raw: dict[str, Any], *, default: float = 300.0) -> float:
    """从 data 配置 dict 解析 label horizon（``label_horizon_seconds`` 优先，``label_backtrace_seconds`` 为别名）。"""
    if raw.get("label_horizon_seconds") is not None:
        return float(raw["label_horizon_seconds"])
    if raw.get("label_backtrace_seconds") is not None:
        return float(raw["label_backtrace_seconds"])
    return float(default)


def sync_label_horizon_aliases(data_cfg: dict[str, Any], *, default: float = 300.0) -> float:
    """双向同步 horizon 别名字段，返回最终秒数。"""
    horizon = label_horizon_seconds_from_dict(data_cfg, default=default)
    data_cfg["label_horizon_seconds"] = horizon
    data_cfg["label_backtrace_seconds"] = horizon
    return horizon


def _resolve_config_include(ref: str | Path, exp_parent: Path, root: Path) -> Path:
    """解析实验 yaml 里 `base:` / `data:` 等指向的片段配置文件路径。

    顺序：绝对路径 → 与实验 yaml 同目录的相对路径 → 仓库根下的相对路径。
    若前两步都不存在文件，仍返回「实验目录/相对路径」供 load_yaml 报错时路径可读。
    """
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
    """递归遍历配置树：凡键名以 `_path` 或 `_dir` 结尾的字符串值，若为相对路径则改为「项目根下的绝对路径字符串」。

    这样 train/preprocess 里 `Path(cfg["data"]["mcelog_path"])` 不依赖进程 cwd。
    注意形参名与外层变量重名易混，调用处传入的即 `project_root()` 的返回值。
    """
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
    """在返回合并配置前做强校验；失败抛异常，避免训练中途才发现路径或维度错误。

    与 Notebook 里 `check_preprocess_raw_inputs` 分工不同：此处要求 model/infer 等完整字段存在且合法。
    `run_preprocess` 在加载完整配置前会先调用 `check_preprocess_raw_inputs` 做原始文件预检。
    """
    data_cfg = cfg.get("data", {})
    for k in ("mcelog_path", "tickets_path", "index_features_path"):
        p = Path(data_cfg.get(k, ""))
        if not p.is_file():
            raise FileNotFoundError(f"Config error: `{k}` is not a readable file: {p}")

    inv = data_cfg.get("inventory_path")
    if inv:
        ip = Path(inv)
        if not ip.is_file():
            raise FileNotFoundError(f"Config error: `inventory_path` is not a readable file: {ip}")

    if int(data_cfg.get("window_seconds", 0)) <= 0:
        raise ValueError("data.window_seconds must be > 0")

    # label_horizon_seconds 为首选字段；label_backtrace_seconds 为同义别名
    horizon = sync_label_horizon_aliases(data_cfg)
    if horizon <= 0:
        raise ValueError("data.label_horizon_seconds (或 label_backtrace_seconds) must be > 0")

    split_strategy = data_cfg.get("split_strategy", "time_stratified")
    if split_strategy not in {"time", "time_stratified", "sid"}:
        raise ValueError(
            "Config error: `split_strategy` must be one of: time, time_stratified, sid"
        )
    norm_method = str(data_cfg.get("normalization_method", "robust")).lower().strip()
    if norm_method not in {"zscore", "robust"}:
        raise ValueError("Config error: data.normalization_method must be zscore or robust")
    clip_raw = data_cfg.get("normalization_clip", 20.0)
    if clip_raw not in (None, "null", False):
        if float(clip_raw) <= 0:
            raise ValueError("Config error: data.normalization_clip must be > 0 or null")
    if split_strategy in {"time", "time_stratified"}:
        ts = data_cfg.get("time_split") or {}
        tr = float(ts.get("train_ratio", 0.7))
        vr = float(ts.get("val_ratio", 0.15))
        te = float(ts.get("test_ratio", 0.15))
        if round(tr + vr + te, 5) != 1.0:
            raise ValueError("Config error: data.time_split train/val/test ratios must sum to 1.0")
        min_pos = int(ts.get("min_pos_per_split", 1))
        if min_pos < 0:
            raise ValueError("Config error: data.time_split.min_pos_per_split must be >= 0")

    infer_mode = cfg.get("infer", {}).get("mode", "full")
    if infer_mode != "full":
        raise ValueError("Config error: `infer.mode` must be `full` for grid_cnn")

    m_tab = int(cfg.get("model", {}).get("tabular_dim", 0))
    if m_tab > 0 and m_tab != TABULAR_DIM:
        raise ValueError(
            f"Config error: model.tabular_dim={m_tab} 必须与 inventory 编码维度 TABULAR_DIM={TABULAR_DIM} 一致"
        )

    model_cfg = cfg.get("model", {})
    rep = str(data_cfg.get("grid_representation", "spatial_heatmap")).lower().strip()
    if rep == "spatial_heatmap":
        inv_as_grid = bool(data_cfg.get("inventory_as_grid_channels", True))
        n_banks = int(data_cfg.get("num_bank_channels", 16))
        expect_in = n_banks + (TABULAR_DIM if inv_as_grid else 0)
        cfg_in = int(model_cfg.get("in_channels", expect_in))
        if cfg_in != expect_in:
            raise ValueError(
                f"Config error: spatial_heatmap 期望 model.in_channels={expect_in} "
                f"(={n_banks} bank + {TABULAR_DIM if inv_as_grid else 0} inventory)，"
                f"当前为 {cfg_in}"
            )
        if not inv_as_grid and m_tab != TABULAR_DIM:
            raise ValueError(
                "Config error: inventory_as_grid_channels=false 时 model.tabular_dim 应为 "
                f"{TABULAR_DIM}（inventory 独立分支）"
            )
        if inv_as_grid and m_tab != 0:
            raise ValueError(
                "Config error: inventory_as_grid_channels=true 时 model.tabular_dim 应为 0"
            )

    train_cfg = cfg.get("train", {})
    esm = str(train_cfg.get("early_stop_metric", "avg_precision")).lower()
    if esm not in {"best_f1", "roc_auc", "f1_at_0p5", "avg_precision"}:
        raise ValueError(
            "Config error: train.early_stop_metric must be one of: "
            "best_f1, roc_auc, f1_at_0p5, avg_precision"
        )
    imb_cfg = train_cfg.get("imbalance") or {}
    if imb_cfg:
        scheme = str(imb_cfg.get("weight_scheme", "inverse_freq")).lower().strip()
        if scheme not in {"inverse_freq", "sqrt"}:
            raise ValueError(
                "Config error: train.imbalance.weight_scheme must be 'inverse_freq' or 'sqrt'"
            )
        cap = imb_cfg.get("positive_weight_cap")
        if cap is not None and float(cap) <= 0:
            raise ValueError("Config error: train.imbalance.positive_weight_cap must be > 0 or null")

    _validate_merged_gpu_ddp(cfg, "train")
    _validate_merged_gpu_ddp(cfg, "infer")


def _validate_ddp_enabled_value(raw: Any, label: str) -> None:
    if isinstance(raw, bool):
        return
    if str(raw).strip().lower() not in {"true", "false", "auto", "1", "0", "yes", "no"}:
        raise ValueError(f"Config error: {label}.ddp.enabled must be true, false, or auto")


def _validate_gpu_section(gpu_cfg: dict, label: str) -> None:
    if not gpu_cfg:
        return
    frac = gpu_cfg.get("vram_budget_frac")
    if frac is not None and not (0.0 < float(frac) <= 1.0):
        raise ValueError(f"Config error: {label}.gpu.vram_budget_frac must be in (0, 1]")
    bmin = gpu_cfg.get("batch_size_min")
    bmax = gpu_cfg.get("batch_size_max")
    if bmin is not None and int(bmin) < 1:
        raise ValueError(f"Config error: {label}.gpu.batch_size_min must be >= 1")
    if bmin is not None and bmax is not None and int(bmin) > int(bmax):
        raise ValueError(f"Config error: {label}.gpu.batch_size_min must be <= batch_size_max")


def _validate_merged_gpu_ddp(cfg: Dict[str, Any], scope: str) -> None:
    from dram2cnn.utils.gpu_launch import merged_scope_section

    gpu_cfg = merged_scope_section(cfg, "gpu", scope)
    ddp_cfg = merged_scope_section(cfg, "ddp", scope)
    _validate_gpu_section(gpu_cfg, f"merged {scope}")
    _validate_ddp_enabled_value(ddp_cfg.get("enabled", "auto"), f"merged {scope}")


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并：patch 与 base 同键且双方均为 dict 时递归合并；否则整键覆盖（含 list/标量替换）。"""
    out = deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_experiment_config(exp_path: str | Path) -> Dict[str, Any]:
    """加载实验 yaml：合并五段片段 + overrides，绝对化 IO 路径，再 validate_config。

    返回 dict 结构为 cfg["base"|"data"|"model"|"train"|"infer"]，供 pipelines 直接使用。
    """
    exp_file = Path(exp_path).resolve()
    exp = load_yaml(exp_file)
    missing = [k for k in ("base", "data", "model", "train", "infer") if k not in exp]
    if missing:
        raise ValueError(f"实验 YAML {exp_file} 缺少必需键: {', '.join(missing)}")
    exp_parent = exp_file.parent
    root = project_root()

    # 先各自 load 片段 yaml，尚未应用实验级 overrides
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


def load_pipeline_config(exp_path: str | Path) -> Dict[str, Any]:
    """加载 pipeline 运行时配置：优先 `DRAM2CNN_CONFIG_SNAPSHOT`（Optuna trial 等）。"""
    import os

    snap = os.environ.get("DRAM2CNN_CONFIG_SNAPSHOT")
    if snap:
        from dram2cnn.utils.io import load_json

        cfg = load_json(snap)
        validate_config(cfg)
        return cfg
    return load_experiment_config(exp_path)
