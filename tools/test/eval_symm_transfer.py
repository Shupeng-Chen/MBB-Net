#!/usr/bin/env python
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if "--dataset" not in sys.argv:
    sys.argv.extend(["--dataset", "shapenet55"])

runpy.run_path(
    str(ROOT / "reproduce/SymmCompletion/eval_transfer.py"),
    run_name="__main__",
)
