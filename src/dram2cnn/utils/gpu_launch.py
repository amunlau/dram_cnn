from __future__ import annotations

"""GPU 启动规划：DDP、显存检测、batch 自动探测（训练 / 推理等 pipeline 共用）。

供 `scripts/train.sh`、`scripts/infer.sh` 等在 torchrun 启动前调用；
探测结果经 `DRAM2CNN_BATCH_SIZE` 传入对应 `run_*` 模块。
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from dram2cnn.engine.imbalance import compute_class_weights_tensor, focal_gamma_from_cfg
from dram2cnn.models.factory import build_model
from dram2cnn.models.losses import classification_loss
from dram2cnn.utils.config import deep_update, load_experiment_config, project_root
from dram2cnn.utils.device import resolve_device
from dram2cnn.utils.model_shape import validate_model_input_shape
from dram2cnn.utils.run_layout import resolve_checkpoint_dir

PipelineMode = Literal["train", "infer"]


def merged_scope_section(cfg: dict, section: str, scope: str) -> dict:
    """合并 `base.<section>` 与 `<scope>.<section>`（scope 优先）。"""
    base_part = dict((cfg.get("base") or {}).get(section) or {})
    scope_part = dict((cfg.get(scope) or {}).get(section) or {})
    return deep_update(base_part, scope_part)


def resolve_ddp_enabled(raw: Any, device: str, cuda_count: int) -> bool:
    """是否允许以 DDP 多进程启动（仍须 nproc>1 且 shell 脚本调用 torchrun）。"""
    if not str(device).startswith("cuda") or cuda_count <= 0:
        return False
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in {"false", "0", "no"}:
        return False
    if s in {"true", "1", "yes"}:
        return True
    if s == "auto":
        return cuda_count >= 2
    raise ValueError(f"ddp.enabled must be true/false/auto, got {raw!r}")


def resolve_nproc_per_node(raw: Any, cuda_count: int) -> int:
    """解析每节点进程数；`auto` 表示使用当前可见的全部 GPU。"""
    if cuda_count <= 0:
        return 1
    if str(raw).strip().lower() == "auto":
        return cuda_count
    return max(1, int(raw))


def get_cuda_device_info() -> list[dict[str, Any]]:
    """返回可见 CUDA 设备列表（名称、总显存 bytes）。"""
    if not torch.cuda.is_available():
        return []
    out: list[dict[str, Any]] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        out.append(
            {
                "index": i,
                "name": props.name,
                "total_bytes": int(props.total_memory),
                "total_gib": round(props.total_memory / (1024**3), 2),
            }
        )
    return out


def _min_gpu_total_bytes() -> int | None:
    info = get_cuda_device_info()
    if not info:
        return None
    return min(d["total_bytes"] for d in info)


def _vram_budget_bytes(gpu_cfg: dict) -> int | None:
    total = _min_gpu_total_bytes()
    if total is None:
        return None
    frac = float(gpu_cfg.get("vram_budget_frac", 0.8))
    reserve = int(gpu_cfg.get("vram_reserve_mib", 512)) * 1024 * 1024
    return max(0, int(total * frac) - reserve)


def _load_probe_batch_from_split(
    splits_dir: Path,
    split_prefix: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    x_np = np.load(splits_dir / f"X_{split_prefix}.npy", mmap_mode="r")
    y_np = np.load(splits_dir / f"y_{split_prefix}.npy", mmap_mode="r")
    n = min(batch_size, x_np.shape[0])
    x = torch.from_numpy(np.array(x_np[:n], copy=True)).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(np.array(y_np[:n], dtype=np.int64, copy=True)).to(device=device, dtype=torch.long)
    tab_path = splits_dir / f"tabular_{split_prefix}.npy"
    tab: torch.Tensor | None = None
    if tab_path.is_file():
        tab_np = np.load(tab_path, mmap_mode="r")
        tab = torch.from_numpy(np.array(tab_np[:n], copy=True)).to(device=device, dtype=torch.float32)
    return x, tab, y


def _batch_fits_vram_budget_train(
    batch_size: int,
    model_cfg: dict,
    train_cfg: dict,
    splits_dir: Path,
    device: torch.device,
    budget_bytes: int,
) -> bool:
    x: torch.Tensor | None = None
    tab: torch.Tensor | None = None
    y: torch.Tensor | None = None
    model = None
    optim = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = build_model(model_cfg).to(device)
        optim = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        x, tab, y = _load_probe_batch_from_split(splits_dir, "train", batch_size, device)

        imb_cfg = train_cfg.get("imbalance") or {}
        use_weight = bool(imb_cfg.get("use_class_weight", True))
        class_w = None
        if use_weight:
            y_train_np = np.load(splits_dir / "y_train.npy")
            class_w = compute_class_weights_tensor(y_train_np, imb_cfg).to(device)
        focal_gamma = focal_gamma_from_cfg(imb_cfg)

        if tab is not None:
            logits = model(x, tab)
        else:
            logits = model(x)
        loss = classification_loss(logits, y, class_weight=class_w, focal_gamma=focal_gamma)
        loss.backward()
        optim.step()

        peak = int(torch.cuda.max_memory_allocated(device))
        return peak <= budget_bytes
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return False
        raise
    finally:
        del x, tab, y, model, optim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _batch_fits_vram_budget_infer(
    batch_size: int,
    model_cfg: dict,
    splits_dir: Path,
    device: torch.device,
    budget_bytes: int,
    ckpt_path: Path | None,
) -> bool:
    x: torch.Tensor | None = None
    tab: torch.Tensor | None = None
    model = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = build_model(model_cfg).to(device)
        if ckpt_path is not None and ckpt_path.is_file():
            state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            model.load_state_dict(state)
        model.eval()

        x, tab, _y = _load_probe_batch_from_split(splits_dir, "test", batch_size, device)
        with torch.no_grad():
            if tab is not None:
                logits = model(x, tab)
            else:
                logits = model(x)
            _ = torch.softmax(logits, dim=1)

        peak = int(torch.cuda.max_memory_allocated(device))
        return peak <= budget_bytes
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return False
        raise
    finally:
        del x, tab, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _binary_search_batch_size(
    batch_min: int,
    batch_max: int,
    fits_fn,
) -> int:
    best = batch_min
    if not fits_fn(batch_min):
        raise RuntimeError(
            f"GPU batch probe failed: batch_size={batch_min} exceeds VRAM budget. "
            "Lower model size or gpu.vram_budget_frac / batch_size_min."
        )
    lo, hi = batch_min, batch_max
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits_fn(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return max(batch_min, best)


def probe_batch_size_for_training(
    model_cfg: dict,
    train_cfg: dict,
    splits_dir: str | Path,
    gpu_cfg: dict,
) -> int:
    """按显存预算二分搜索训练 per-GPU batch_size（含 forward+backward）。"""
    if not torch.cuda.is_available():
        return int(train_cfg.get("batch_size", 256))

    splits_p = Path(splits_dir)
    x_path = splits_p / "X_train.npy"
    if not x_path.is_file():
        raise FileNotFoundError(f"Cannot probe batch_size: missing {x_path}")

    budget = _vram_budget_bytes(gpu_cfg)
    if budget is None or budget <= 0:
        return int(train_cfg.get("batch_size", 256))

    x_np = np.load(x_path, mmap_mode="r")
    tab_path = splits_p / "tabular_train.npy"
    tab_shape = tuple(np.load(tab_path, mmap_mode="r").shape[1:]) if tab_path.is_file() else None
    validate_model_input_shape(model_cfg, tuple(x_np.shape), tab_shape)

    batch_min = max(1, int(gpu_cfg.get("batch_size_min", 8)))
    batch_max = min(int(gpu_cfg.get("batch_size_max", 512)), int(x_np.shape[0]))
    if batch_max < batch_min:
        return max(1, batch_max)

    device = torch.device("cuda:0")
    return _binary_search_batch_size(
        batch_min,
        batch_max,
        lambda bs: _batch_fits_vram_budget_train(bs, model_cfg, train_cfg, splits_p, device, budget),
    )


def probe_batch_size_for_inference(
    model_cfg: dict,
    infer_cfg: dict,
    splits_dir: str | Path,
    gpu_cfg: dict,
    ckpt_path: str | Path | None = None,
) -> int:
    """按显存预算二分搜索推理 per-GPU batch_size（仅 forward）。"""
    if not torch.cuda.is_available():
        return int(infer_cfg.get("batch_size", 512))

    splits_p = Path(splits_dir)
    x_path = splits_p / "X_test.npy"
    if not x_path.is_file():
        raise FileNotFoundError(f"Cannot probe batch_size: missing {x_path}")

    budget = _vram_budget_bytes(gpu_cfg)
    if budget is None or budget <= 0:
        return int(infer_cfg.get("batch_size", 512))

    x_np = np.load(x_path, mmap_mode="r")
    tab_path = splits_p / "tabular_test.npy"
    tab_shape = tuple(np.load(tab_path, mmap_mode="r").shape[1:]) if tab_path.is_file() else None
    validate_model_input_shape(model_cfg, tuple(x_np.shape), tab_shape)

    batch_min = max(1, int(gpu_cfg.get("batch_size_min", 8)))
    batch_max = min(int(gpu_cfg.get("batch_size_max", 2048)), int(x_np.shape[0]))
    if batch_max < batch_min:
        return max(1, batch_max)

    ckpt = Path(ckpt_path) if ckpt_path else None
    device = torch.device("cuda:0")
    return _binary_search_batch_size(
        batch_min,
        batch_max,
        lambda bs: _batch_fits_vram_budget_infer(bs, model_cfg, splits_p, device, budget, ckpt),
    )


@dataclass
class LaunchPlan:
    mode: PipelineMode
    device: str
    configured_device: str
    ddp_enabled: bool
    nproc_per_node: int
    cuda_count: int
    using_all_visible_gpus: bool
    batch_size: int
    batch_size_auto: bool
    gpu_info: list[dict[str, Any]]
    vram_budget_frac: float | None

    def validate(self) -> None:
        if self.device.startswith("cuda") and self.cuda_count == 0:
            raise RuntimeError(
                "device=cuda but no GPU visible (torch.cuda.device_count()==0)"
            )
        if self.ddp_enabled and self.nproc_per_node > 1 and self.cuda_count > 0:
            if self.nproc_per_node > self.cuda_count:
                raise RuntimeError(
                    f"ddp.nproc_per_node={self.nproc_per_node} > visible GPU count="
                    f"{self.cuda_count}. Adjust nproc_per_node or CUDA_VISIBLE_DEVICES."
                )


def resolve_launch_plan_from_cfg(cfg: dict, mode: PipelineMode) -> LaunchPlan:
    """从已合并实验配置解析启动计划。"""
    scope = mode
    configured_device = str(cfg.get("base", {}).get("device", "auto"))
    device = resolve_device(configured_device)

    ddp_cfg = merged_scope_section(cfg, "ddp", scope)
    gpu_cfg = merged_scope_section(cfg, "gpu", scope)
    scope_cfg = cfg.get(scope) or {}

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    ddp_enabled = resolve_ddp_enabled(ddp_cfg.get("enabled", "auto"), device, cuda_count)
    nproc = resolve_nproc_per_node(ddp_cfg.get("nproc_per_node", "auto"), cuda_count)

    default_batch = 256 if mode == "train" else 512
    config_batch = int(scope_cfg.get("batch_size", default_batch))
    auto_batch = bool(gpu_cfg.get("auto_batch_size", True))
    batch_size = config_batch
    batch_size_auto = False

    root = project_root()
    splits_dir = Path(cfg["data"]["splits_dir"])
    if not splits_dir.is_absolute():
        splits_dir = root / splits_dir

    if auto_batch and device.startswith("cuda") and cuda_count > 0:
        if mode == "train":
            batch_size = probe_batch_size_for_training(
                cfg["model"], scope_cfg, splits_dir, gpu_cfg
            )
        else:
            ckpt_path: Path | None = None
            try:
                ckpt_dir = resolve_checkpoint_dir(root, cfg["base"])
                candidate = ckpt_dir / "checkpoints" / "grid_cnn.pt"
                if candidate.is_file():
                    ckpt_path = candidate
            except FileNotFoundError:
                ckpt_path = None
            batch_size = probe_batch_size_for_inference(
                cfg["model"],
                scope_cfg,
                splits_dir,
                gpu_cfg,
                ckpt_path=ckpt_path,
            )
        batch_size_auto = True

    using_all = bool(
        device.startswith("cuda")
        and ddp_enabled
        and nproc > 1
        and cuda_count > 0
        and nproc == cuda_count
    )

    frac_raw = gpu_cfg.get("vram_budget_frac")
    vram_frac = float(frac_raw) if frac_raw is not None else None

    plan = LaunchPlan(
        mode=mode,
        device=device,
        configured_device=configured_device,
        ddp_enabled=ddp_enabled,
        nproc_per_node=max(1, nproc),
        cuda_count=cuda_count,
        using_all_visible_gpus=using_all,
        batch_size=batch_size,
        batch_size_auto=batch_size_auto,
        gpu_info=get_cuda_device_info(),
        vram_budget_frac=vram_frac,
    )
    plan.validate()
    return plan


def resolve_launch_plan(exp_config: str | Path, mode: PipelineMode) -> LaunchPlan:
    cfg = load_experiment_config(str(exp_config))
    return resolve_launch_plan_from_cfg(cfg, mode)


def print_launch_plan_for_shell(mode: PipelineMode, exp_config: str) -> None:
    """输出供 bash `readarray` 解析的多行字段；stderr 打印 GPU / batch 摘要。"""
    plan = resolve_launch_plan(exp_config, mode)
    tag = f"gpu_launch:{mode}"
    if plan.gpu_info:
        gpu_lines = ", ".join(
            f"cuda:{g['index']} {g['name']} {g['total_gib']}GiB" for g in plan.gpu_info
        )
        print(f"[{tag}] visible GPUs ({plan.cuda_count}): {gpu_lines}", file=sys.stderr)
    if plan.batch_size_auto:
        print(
            f"[{tag}] auto batch_size={plan.batch_size} "
            f"(vram_budget_frac={plan.vram_budget_frac})",
            file=sys.stderr,
        )
    print(plan.device)
    print("1" if plan.ddp_enabled else "0")
    print(plan.nproc_per_node)
    print(plan.configured_device)
    print(plan.cuda_count)
    print("1" if plan.using_all_visible_gpus else "0")
    print(plan.batch_size)
    print("1" if plan.batch_size_auto else "0")


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python -m dram2cnn.utils.gpu_launch <train|infer> <exp_config>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    mode = sys.argv[1].strip().lower()
    if mode not in {"train", "infer"}:
        raise SystemExit(f"Unknown mode: {mode!r}")
    print_launch_plan_for_shell(mode, sys.argv[2])  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
