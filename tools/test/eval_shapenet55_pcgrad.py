#!/usr/bin/env python
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if "--dataset" not in sys.argv:
    sys.argv.extend(["--dataset", "55"])
if "--mode" not in sys.argv:
    sys.argv.extend(["--mode", "pcgrad_ours"])

runpy.run_path(
    str(ROOT / "reproduce/ShapeNet/ShapeNet55/eval_table2.py"),
    run_name="__main__",
)
