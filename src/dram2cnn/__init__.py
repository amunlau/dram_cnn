"""dram2cnn — DRAM 故障二分类（spatial 热图 GridCNN-Deep）。

包内含：`data`（`tianchi_build` 预处理、`index_builder` 索引生成、切分）、
`models`（GridCNN-Deep）、`engine`（训练/推理/指标/W8A8 PTQ/INT8 参考推理/剪枝/先剪枝后量化；剪枝主指标为 val 重搜 τ）、
`pipelines`（preprocess/train/infer/quantize/infer_int8/prune/prune_quantize/optuna 等）、`utils`（配置与路径）。

默认预测 **未来 5 分钟内是否故障**（`label_horizon_seconds=300`）。
使用前请 **`export PYTHONPATH=src`**，详见根目录 `README.md`。
"""

__all__ = ["data", "models", "engine", "pipelines", "utils"]
