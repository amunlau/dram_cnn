from __future__ import annotations

"""NumPy 网格 **[N,C,H,W]** 与可选 **`tabular` [N,F]** 的 `Dataset`。

若切分目录无 **`tabular_*.npy`**（spatial 且 inventory 已拼入 **X**），构造时不传入 tabular 路径，迭代返回 **`(x, y)`**。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class IndexMappedDataset(Dataset):
    """按 index_map 过采样/重排训练索引；包装 `NpyTensorDataset` 或内存张量。"""

    def __init__(
        self,
        base: NpyTensorDataset | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        index_map: np.ndarray,
    ):
        """Args:
            base: 底层数据集或 `(x, y, tab)` 内存张量。
            index_map: 长度 `train_virtual_len` 的索引映射，用于过采样正样本。
        """
        self.index_map = torch.as_tensor(np.asarray(index_map, dtype=np.int64), dtype=torch.long)
        if isinstance(base, NpyTensorDataset):
            self._base = base
            self._inline = None
        else:
            x, y, tab = base
            if x.shape[0] != y.shape[0]:
                raise ValueError("x and y length mismatch")
            if tab is not None and tab.shape[0] != x.shape[0]:
                raise ValueError("tabular length mismatch vs x")
            self._base = None
            self._inline = (x, y, tab)

    def __len__(self) -> int:
        return int(self.index_map.shape[0])

    def __getitem__(self, idx: int):
        j = int(self.index_map[idx])
        if self._base is not None:
            return self._base[j]
        x, y, tab = self._inline
        if tab is not None:
            return x[j], tab[j], y[j]
        return x[j], y[j]


class NpyTensorDataset(Dataset):
    """加载 X.npy / y.npy，形状 [N,C,H,W] 与 [N]；若存在同目录 tabular_*.npy 则三值返回。"""

    def __init__(self, x_path: str, y_path: str, tabular_path: str | None = None):
        """Args:
            x_path: `X_*.npy`，形状 `[N,C,H,W]`。
            y_path: `y_*.npy`，形状 `[N]`。
            tabular_path: 可选 `tabular_*.npy`；缺省则 `__getitem__` 返回 `(x, y)`。
        """
        self._x_np = np.load(x_path, mmap_mode="r")
        y_np = np.load(y_path, mmap_mode="r")
        self.y = torch.from_numpy(np.array(y_np, dtype=np.int64, copy=True)).long()
        self._tab_np: np.ndarray | None = None
        if tabular_path is not None and Path(tabular_path).is_file():
            self._tab_np = np.load(tabular_path, mmap_mode="r")
            if self._tab_np.shape[0] != self._x_np.shape[0]:
                raise ValueError(f"tabular N={self._tab_np.shape[0]} vs X N={self._x_np.shape[0]}")

    @property
    def x_shape(self) -> tuple[int, ...]:
        return tuple(self._x_np.shape)

    @property
    def tab_shape(self) -> tuple[int, ...] | None:
        return tuple(self._tab_np.shape) if self._tab_np is not None else None

    def __len__(self) -> int:
        return int(self._x_np.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self._x_np[idx], dtype=np.float32, copy=True))
        y = self.y[idx]
        if self._tab_np is not None:
            tab = torch.from_numpy(np.array(self._tab_np[idx], dtype=np.float32, copy=True))
            return x, tab, y
        return x, y
