import numpy as np

from dram2cnn.data.inventory_encode import TABULAR_DIM, encode_inventory_row, load_inventory_by_sid


def test_encode_inventory_row_known_values():
    v = encode_inventory_row("B1", 12, "M2")
    assert v.shape == (TABULAR_DIM,)
    assert int(v[2]) == 1  # B1（A1,A2,B1,…）
    assert int(v[7 + 1]) == 1  # dimm 12
    assert int(v[7 + 4 + 1]) == 1  # M2


def test_encode_inventory_unknowns_are_zero():
    v = encode_inventory_row("ZZ", 99, "MX")
    assert np.allclose(v, 0.0)


def test_load_inventory_by_sid(tmp_path):
    p = tmp_path / "inv.csv"
    p.write_text(
        "sid,server_manufacturer,DRAM_model,DIMM_number\n"
        "S9,M3,A1,8\n"
        "S9,M3,C2,24\n",
        encoding="utf-8",
    )
    m = load_inventory_by_sid(p)
    assert "S9" in m
    # 重复 sid：后行覆盖
    v = m["S9"]
    assert int(v[6]) == 1  # C2
    assert int(v[10]) == 1  # DIMM 24
    assert int(v[13]) == 1  # M3
