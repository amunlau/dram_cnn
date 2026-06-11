#!/usr/bin/env bash
# 清理中间产物、训练/推理输出、缓存与 Notebook 检查点（保留 data/raw 与 index_stubs）
#
# 用法：
#   bash scripts/clean.sh              # 默认：runs、processed、splits、outputs 生成物、缓存
#   bash scripts/clean.sh --notebooks  # 额外清除 notebooks/*.ipynb 单元格输出
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CLEAR_NOTEBOOKS=0
for arg in "$@"; do
  case "${arg}" in
    --notebooks) CLEAR_NOTEBOOKS=1 ;;
    -h|--help)
      echo "用法: bash scripts/clean.sh [--notebooks]"
      echo "  --notebooks  清除 notebooks/*.ipynb 的 cell 输出（不含 checkpoint 目录）"
      exit 0
      ;;
    *) echo "未知参数: ${arg}" >&2; exit 1 ;;
  esac
done

echo "=== 清理 runs/（训练/推理/流水线日志）==="
find runs -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true
mkdir -p runs/pipeline-logs/markers

echo "=== 清理 data/processed/ 与 data/splits/ ==="
rm -rf data/processed/* data/splits/* 2>/dev/null || true

echo "=== 清理 outputs/ 生成物（保留 .gitkeep）==="
find outputs/figures outputs/tables outputs/reports outputs/notebooks -mindepth 1 \
  ! -name '.gitkeep' -delete 2>/dev/null || true

echo "=== 清理 Python / 测试 / Notebook 缓存 ==="
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf .pytest_cache .hypothesis .mypy_cache .ruff_cache 2>/dev/null || true
find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name '*.bak' -o -name '*~' -o -name '.DS_Store' \) -delete 2>/dev/null || true

if [[ "${CLEAR_NOTEBOOKS}" -eq 1 ]]; then
  echo "=== 清除 Notebook 单元格输出 ==="
  export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
  python3 << 'PY'
from pathlib import Path
import json

root = Path("notebooks")
for nb_path in sorted(root.glob("*.ipynb")):
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = False
    for cell in data.get("cells", []):
        if cell.get("outputs"):
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"  cleared: {nb_path.name}")
PY
fi

echo "Done. 重建冒烟数据: bash scripts/build_index.sh smoke && bash scripts/preprocess.sh configs/experiments/exp_baseline.yaml"
