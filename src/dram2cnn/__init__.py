"""dram2cnn — 天池式 DRAM 故障二分类。

包内含：`data`（预处理与切分）、`models`（GridCNN）、`engine`（训练/推理/指标）、
`pipelines`（命令行入口）、`utils`（配置与路径）。

使用前请在仓库根目录设置 **`export PYTHONPATH=src`**，详见根目录 `README.md` 与 `docs/`。
"""

__all__ = ["data", "models", "engine", "pipelines", "utils"]
