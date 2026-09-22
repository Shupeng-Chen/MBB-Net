#!/usr/bin/env python
from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TASKS = {
    "pcn-train": (
        ROOT / "tools/train/train_pcn.py",
        [],
    ),
    "pcn-test": (
        ROOT / "tools/test/eval_pcn.py",
        [],
    ),
    "shapenet55-test": (
        ROOT / "tools/test/eval_shapenet55.py",
        [],
    ),
    "shapenet55-train": (
        ROOT / "tools/train/train_shapenet55.py",
        [],
    ),
    "shapenet34-test": (
        ROOT / "tools/test/eval_shapenet34_21.py",
        ["--dataset", "34"],
    ),
    "shapenet21-test": (
        ROOT / "tools/test/eval_shapenet34_21.py",
        ["--dataset", "21"],
    ),
    "shapenet55-pcgrad-train": (
        ROOT / "tools/train/train_shapenet55_pcgrad.py",
        [],
    ),
    "shapenet55-pcgrad-test": (
        ROOT / "tools/test/eval_shapenet55_pcgrad.py",
        [],
    ),
    "shapenet34-train": (
        ROOT / "tools/train/train_shapenet34.py",
        ["--dataset", "34"],
    ),
    "symm-train": (
        ROOT / "tools/train/train_symm_transfer.py",
        [],
    ),
    "symm-test": (
        ROOT / "tools/test/eval_symm_transfer.py",
        [],
    ),
    "kitti-infer": (
        ROOT / "tools/test/infer_kitti.py",
        [],
    ),
    "kitti-eval": (
        ROOT / "tools/test/eval_kitti.py",
        [],
    ),
    "table4": (
        ROOT / "tools/analysis/verify_table4.py",
        [],
    ),
    "fig5": (
        ROOT / "tools/analysis/verify_fig5.py",
        [],
    ),
    "fig5-plot": (
        ROOT / "tools/analysis/plot_fig5.py",
        [],
    ),
}


def print_help() -> None:
    print("MBB-Net public launcher")
    print()
    print("Usage:")
    print("  python main.py <task> [task arguments]")
    print()
    print("Tasks:")
    for task in TASKS:
        print(f"  {task}")
    print()
    print("Examples:")
    print("  python main.py pcn-train --data_root /path/to/PCN")
    print("  python main.py shapenet55-train --mode ours --data-root /path/to/ShapeNet55")
    print("  python main.py shapenet55-test --dataset 55 --mode ours --data_root /path/to/ShapeNet55")
    print("  python main.py pcn-test --data_root /path/to/PCN --checkpoint checkpoints/PCN/pcn_mbb_ablation_ours_best.pth")
    print("  python main.py shapenet55-pcgrad-train --mbb_mode ours --data_root /path/to/ShapeNet55")
    print("  python main.py shapenet34-train --mode ours --data_root /path/to/ShapeNet")
    print("  python main.py symm-train --variant ours --pretrained checkpoints/SymmCompletion/ShapeNet55/baseline/ckpt-best.pth")
    print("  python main.py table4")
    print("  python main.py fig5")
    print()
    print("Use the corresponding script under tools/ with --help for full options.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_help()
        return

    task = sys.argv[1]
    if task not in TASKS:
        print(f"Unknown task: {task}", file=sys.stderr)
        print_help()
        raise SystemExit(2)

    target, injected_args = TASKS[task]
    if not target.is_file():
        raise FileNotFoundError(target)

    sys.path.insert(0, str(ROOT))
    sys.argv = [str(target), *injected_args, *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
