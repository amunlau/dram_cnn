from __future__ import annotations

"""NumPy 网格 **[N,C,H,W]** 与可选 **`tabular` [N,F]** 的 `Dataset`。

若切分目录无 **`tabular_*.npy`**（spatial 且 inventory 已拼入 **X**），构造时不传入 tabular 路径，迭代返回 **`(x, y)`**。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class IndexMappedDataset(Dataset):
    """按 index_map 过采样/重排训练索引；可选同步 tabular 行。"""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        index_map: np.ndarray,
        tab: torch.Tensor | None = None,
    ):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y length mismatch")
        if tab is not None and tab.shape[0] != x.shape[0]:
            raise ValueError("tabular length mismatch vs x")
        self.x = x
        self.y = y
        self.tab = tab
        self.index_map = torch.as_tensor(np.asarray(index_map, dtype=np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.index_map.shape[0])

    def __getitem__(self, idx: int):
        j = int(self.index_map[idx])
        if self.tab is not None:
            return self.x[j], self.tab[j], self.y[j]
        return self.x[j], self.y[j]


class NpyTensorDataset(Dataset):
    """加载 X.npy / y.npy，形状 [N,C,H,W] 与 [N]；若存在同目录 tabular_*.npy 则三值返回。"""

    def __init__(self, x_path: str, y_path: str, tabular_path: str | None = None):
        # mmap 降低大矩阵常驻 RAM（训练/推理共用）；torch.from_numpy 与 mmap 数组共享只读内存，输入不参与梯度写回
        x_np = np.load(x_path, mmap_mode="r")
        y_np = np.load(y_path, mmap_mode="r")
        if x_np.dtype != np.float32:
            x_np = x_np.astype(np.float32, copy=False)
        self.x = torch.from_numpy(x_np)
        self.y = torch.from_numpy(np.asarray(y_np, dtype=np.int64)).long()
        self.tab: torch.Tensor | None = None
        if tabular_path is not None and Path(tabular_path).is_file():
            t_np = np.load(tabular_path, mmap_mode="r")
            if t_np.dtype != np.float32:
                t_np = t_np.astype(np.float32, copy=False)
            if t_np.shape[0] != self.x.shape[0]:
                raise ValueError(f"tabular N={t_np.shape[0]} vs X N={self.x.shape[0]}")
            self.tab = torch.from_numpy(t_np)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        if self.tab is not None:
            return self.x[idx], self.tab[idx], self.y[idx]
        return self.x[idx], self.y[idx]
