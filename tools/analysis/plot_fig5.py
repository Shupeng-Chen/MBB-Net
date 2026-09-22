#!/usr/bin/env python
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "reproduce/Analysis/Fig5/plot_fig5.py"), run_name="__main__")
