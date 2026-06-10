from __future__ import annotations

"""校验 `model.yaml` 与磁盘上 `X` / `tabular` 张量形状是否一致。"""


def validate_model_input_shape(
    model_cfg: dict,
    sample_shape: tuple[int, ...],
    tabular_shape: tuple[int, ...] | None,
) -> None:
    """检查单条样本的 C/H/W 及可选 tabular 维是否与 `model_cfg` 匹配。

    Args:
        model_cfg: 合并后的 `cfg["model"]`。
        sample_shape: `X` 的完整 shape，须为 4D `[N,C,H,W]`。
        tabular_shape: 若存在 tabular，为 `[N,F]`；否则 ``None``。

    Raises:
        ValueError: 任一维度与配置不一致，或 tabular 与 `tabular_dim` 语义冲突。
    """
    if len(sample_shape) != 4:
        raise ValueError(f"Expected 4D tensor [N,C,H,W], got shape: {sample_shape}")
    _, c, h, w = sample_shape
    mismatch = []
    if int(model_cfg.get("in_channels", c)) != c:
        mismatch.append(f"in_channels={model_cfg.get('in_channels')} != tensor C={c}")
    if int(model_cfg.get("grid_height", h)) != h:
        mismatch.append(f"grid_height={model_cfg.get('grid_height')} != tensor H={h}")
    if int(model_cfg.get("grid_width", w)) != w:
        mismatch.append(f"grid_width={model_cfg.get('grid_width')} != tensor W={w}")
    if mismatch:
        raise ValueError("Model/data shape mismatch: " + "; ".join(mismatch))
    td = int(model_cfg.get("tabular_dim", 0))
    if td > 0:
        if tabular_shape is None or len(tabular_shape) != 2:
            raise ValueError("model.tabular_dim>0 需要 splits 中的 tabular_*.npy [N,F]")
        if int(tabular_shape[1]) != td:
            raise ValueError(f"tabular F={tabular_shape[1]} != model.tabular_dim={td}")
    elif tabular_shape is not None:
        raise ValueError(
            "存在 tabular 数据但 model.tabular_dim=0，请在 model.yaml 中设置 tabular_dim 与之一致"
        )
