#!/usr/bin/env python3
"""Execute all notebooks under notebooks/ and write logs."""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
LOG_DIR = ROOT / "notebooks_run_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
os.environ.setdefault("DRAM2CNN_RUN_FULL_NOTEBOOKS", "0")  # smoke path
os.chdir(ROOT)

NOTEBOOKS = [
    "01_eda.ipynb",
    "04_run_pipelines.ipynb",  # smoke first so 02/03 have data
    "02_feature_analysis.ipynb",
    "03_error_analysis.ipynb",
    "05_full_recall_vs_precision.ipynb",
]

def run_one(name: str) -> tuple[bool, str]:
    path = NB_DIR / name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{path.stem}__{stamp}.log"
    print(f"\n======== RUN {name} ========", flush=True)
    print(f"log -> {log_path}", flush=True)
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None,  # no per-cell timeout; smoke train can take a while
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    ok = True
    err = ""
    try:
        client.execute(cwd=str(ROOT))
    except CellExecutionError as e:
        ok = False
        err = str(e)
        print(f"FAIL {name}: {e}", flush=True)
    except Exception:
        ok = False
        err = traceback.format_exc()
        print(f"FAIL {name}:\n{err}", flush=True)

    # persist executed notebook with outputs
    out_nb = LOG_DIR / f"{path.stem}__executed.ipynb"
    nbformat.write(nb, out_nb)

    # extract text streams into log
    lines = [f"# {name} ok={ok} at {stamp}", f"# DRAM2CNN_RUN_FULL_NOTEBOOKS={os.environ.get('DRAM2CNN_RUN_FULL_NOTEBOOKS')}", ""]
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        lines.append(f"===== cell {i} =====")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = out.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                lines.append(text.rstrip())
            elif out.get("output_type") == "error":
                lines.append("ERROR: " + "\n".join(out.get("traceback", [])))
            elif out.get("output_type") in {"execute_result", "display_data"}:
                data = out.get("data", {})
                if "text/plain" in data:
                    t = data["text/plain"]
                    if isinstance(t, list):
                        t = "".join(t)
                    lines.append(t.rstrip())
        lines.append("")
    if err:
        lines.append("===== EXECUTION ERROR =====")
        lines.append(err)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # also write latest symlink-like copy
    (LOG_DIR / f"{path.stem}__latest.log").write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"DONE {name} ok={ok}", flush=True)
    return ok, str(log_path)

results = []
for name in NOTEBOOKS:
    ok, logp = run_one(name)
    results.append((name, ok, logp))

print("\n======== SUMMARY ========")
all_ok = True
for name, ok, logp in results:
    print(f"{'PASS' if ok else 'FAIL'}: {name}  log={logp}")
    all_ok = all_ok and ok
sys.exit(0 if all_ok else 1)
