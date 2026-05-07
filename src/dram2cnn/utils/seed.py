"""随机种子：固定 Python / NumPy / PyTorch（含 CUDA）以便结果可复现。

DDP 训练时各 rank 会对 base seed 加上 rank 偏移，避免各进程完全相同采样。"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """设置 Python/NumPy/PyTorch 的随机种子（含当前可见 GPU 的 CUDA 种子）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
