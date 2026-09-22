#!/usr/bin/env python
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def add_default(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])

add_default("--dataset", "shapenet55")
add_default("--epochs", 20)
add_default("--batch_size", 16)
add_default("--eval_batch_size", 8)
add_default("--num_workers", 8)
add_default("--alpha", 0.0004)
add_default("--learning_rate", 0.0001)
add_default("--backbone_lr_scale", 0.1)
add_default("--weight_decay", 0.0005)
add_default("--warmup_epochs", 5)
add_default("--minimum_lr_ratio", 0.05)
add_default("--freeze_backbone_epochs", 5)
add_default("--gradient_clip", 1.0)
add_default("--val_freq", 5)
add_default("--dense_val_last", 20)
add_default("--seed", 42)

runpy.run_path(
    str(ROOT / "reproduce/SymmCompletion/train_transfer.py"),
    run_name="__main__",
)
