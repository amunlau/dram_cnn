"""切分时若 processed 无 tabular.npy，应删除 splits 下残留的 tabular_*.npy（legacy → spatial 切换）。"""

from pathlib import Path

import numpy as np
import pandas as pd

from dram2cnn.data.splitter import split_by_time


def test_split_by_time_removes_orphan_tabular_when_processed_has_no_tabular(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    spl = tmp_path / "splits"
    proc.mkdir(parents=True)
    spl.mkdir(parents=True)

    n = 12
    X = np.zeros((n, 2, 2, 2), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    meta = pd.DataFrame(
        {
            "sid": [f"S{i}" for i in range(n)],
            "predict_time": [f"2000-01-01 {i:02d}:00:00" for i in range(n)],
            "label": y,
            "predict_ts": np.arange(n, dtype=np.float64),
        }
    )
    np.save(proc / "X.npy", X)
    np.save(proc / "y.npy", y)
    meta.to_csv(proc / "meta.csv", index=False)

    # 模拟旧 legacy 切分遗留
    np.save(spl / "tabular_train.npy", np.zeros((3, 15), dtype=np.float32))
    np.save(spl / "tabular_val.npy", np.zeros((2, 15), dtype=np.float32))
    np.save(spl / "tabular_test.npy", np.zeros((2, 15), dtype=np.float32))

    split_by_time(str(proc), str(spl), 0.5, 0.25, 0.25)

    assert not (spl / "tabular_train.npy").is_file()
    assert not (spl / "tabular_val.npy").is_file()
    assert not (spl / "tabular_test.npy").is_file()
    assert (spl / "X_train.npy").is_file()
