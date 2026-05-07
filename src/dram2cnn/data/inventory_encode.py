from __future__ import annotations

"""`inventory.csv` → **固定 15 维 one-hot**（与天池 `one_hot_encoding` 类别集合对齐）。

三组互斥类别：**DRAM 型号**（A1–C2）、**DIMM 条数档位**（8/12/16/24）、**服务器厂商**（M1–M4）。
未知取值在该组内全 0；同一行至多每组一个为 1。"""

from pathlib import Path

import numpy as np
import pandas as pd

# 与竞赛侧 `one_hot_encoding_features` 类别一致；未知取值在该组全 0。
DRAM_MODEL_ORDER = ("A1", "A2", "B1", "B2", "B3", "C1", "C2")
DIMM_ORDER = (8, 12, 16, 24)
MANUFACTURER_ORDER = ("M1", "M2", "M3", "M4")

TABULAR_FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"model_{m}" for m in DRAM_MODEL_ORDER]
    + [f"dimm_num_{d}" for d in DIMM_ORDER]
    + [f"manufacturer_{m}" for m in MANUFACTURER_ORDER]
)

TABULAR_DIM = len(TABULAR_FEATURE_NAMES)


def encode_inventory_row(dram_model: str, dimm_number, manufacturer: str) -> np.ndarray:
    """单行 → float32 向量 [TABULAR_DIM]，多热组内至多一个为 1。"""
    v = np.zeros(TABULAR_DIM, dtype=np.float32)
    dm = str(dram_model).strip() if dram_model is not None and str(dram_model) != "nan" else ""
    if dm in DRAM_MODEL_ORDER:
        v[DRAM_MODEL_ORDER.index(dm)] = 1.0
    try:
        dn = int(float(dimm_number))
    except (TypeError, ValueError):
        dn = None
    if dn in DIMM_ORDER:
        v[len(DRAM_MODEL_ORDER) + DIMM_ORDER.index(dn)] = 1.0
    mf = str(manufacturer).strip() if manufacturer is not None and str(manufacturer) != "nan" else ""
    base = len(DRAM_MODEL_ORDER) + len(DIMM_ORDER)
    if mf in MANUFACTURER_ORDER:
        v[base + MANUFACTURER_ORDER.index(mf)] = 1.0
    return v


def load_inventory_by_sid(path: Path | str) -> dict[str, np.ndarray]:
    """读取 inventory.csv，返回 sid(str) → one-hot 向量。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"inventory 文件不存在: {p}")
    df = pd.read_csv(p, low_memory=False)
    for col in ("sid", "server_manufacturer", "DRAM_model", "DIMM_number"):
        if col not in df.columns:
            raise ValueError(f"inventory 需含列 `{col}`，当前: {list(df.columns)}")
    df["sid"] = df["sid"].astype(str)
    out: dict[str, np.ndarray] = {}
    # 按列名属性访问，避免 CSV 列顺序与 itertuples 位置不一致；比 iterrows 快约一个数量级。
    for t in df.itertuples(index=False):
        sid = str(t.sid)
        out[sid] = encode_inventory_row(t.DRAM_model, t.DIMM_number, t.server_manufacturer)
    return out


def stack_tabular_for_index(sids: np.ndarray, inv_by_sid: dict[str, np.ndarray]) -> np.ndarray:
    """按索引表 sid 顺序堆叠 [N, TABULAR_DIM]；缺失 sid 用全 0（T 已预置零，仅覆盖命中项）。"""
    n = len(sids)
    T = np.zeros((n, TABULAR_DIM), dtype=np.float32)
    for i in range(n):
        vec = inv_by_sid.get(str(sids[i]))
        if vec is not None:
            T[i] = vec
    return T
