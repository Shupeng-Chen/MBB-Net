#!/usr/bin/env python
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

runpy.run_path(
    str(ROOT / "reproduce/Analysis/Table4/verify_table4.py"),
    run_name="__main__",
)
