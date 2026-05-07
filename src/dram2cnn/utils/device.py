from __future__ import annotations

"""设备解析工具：统一处理 auto/cpu/cuda/cuda:N。"""

import torch


def resolve_device(device: str | None) -> str:
    """将配置设备解析为可执行设备字符串。

    支持：
    - auto: 有 CUDA 则返回 cuda，否则返回 cpu
    - cpu
    - cuda / cuda:N
    """
    d = str(device or "auto").strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cpu":
        return "cpu"
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Configured device is CUDA but torch.cuda.is_available() == False")
        if ":" in d:
            idx = int(d.split(":", 1)[1])
            if idx < 0 or idx >= torch.cuda.device_count():
                raise RuntimeError(f"Configured CUDA index out of range: {d}, available={torch.cuda.device_count()}")
        return d
    raise ValueError(f"Unsupported device value: `{device}`. Expected auto/cpu/cuda[:index].")
