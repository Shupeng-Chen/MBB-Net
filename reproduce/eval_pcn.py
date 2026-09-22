#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate paper-facing MBB-Net PCN checkpoints with a deterministic
SnowflakeNet-style PCN test protocol.

Protocol
--------
1. Use PCN test rendering 00 for every object.
2. Convert each raw partial to 2048 points with SnowflakeNet UpSamplePoints.
3. Fix the per-object seed, so every checkpoint receives identical inputs.
4. Report per-category and overall CD-L1 x1000, F1@1%, and accuracy.
5. Load every checkpoint strictly; no suffix/fuzzy parameter matching.

This script only evaluates. It never trains and never overwrites checkpoints.

Recommended audit:
    python -u scripts/PCN/eval_existing_ckpts_official.py \
        --mode completionOnly full ours \
        --audit_per_class 10 --batch_size 8 --num_workers 4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

def find_project_root(start: Path) -> Path:
    """Locate the public MBB-Net repository root without assuming script depth."""
    start = start.resolve()

    for candidate in (start, *start.parents):
        if (
            (candidate / "main.py").is_file()
            and (candidate / "models" / "pcn_model.py").is_file()
            and (candidate / "cfgs" / "PCN_config.yaml").is_file()
        ):
            return candidate

    raise RuntimeError(
        "Cannot locate the MBB-Net public repository root."
    )


PROJECT_ROOT = str(
    find_project_root(Path(__file__).resolve().parent)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the local evaluation dataset before adding the SnowflakeNet path, so
# the project's datasets package cannot be shadowed by any backbone package.
from data.pcn_official_eval import (  # noqa: E402
    PCN_TAXONOMY,
    PCNSnowflakeOfficialEvalDataset,
    resolve_pcn_data_root,
)

SNOW_ROOT = os.path.join(PROJECT_ROOT, "backbones", "SnowflakeNet-main")
if SNOW_ROOT not in sys.path:
    sys.path.append(SNOW_ROOT)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from models.pcn_modes import (  # noqa: E402
    build_pcn_model,
    normalize_pcn_mode,
)
from metrics.pcn_metrics import (  # noqa: E402
    calc_fscore,
    calc_pcn_cd,
    compute_squared_dists,
)

VALID_MODES = (
    "completionOnly",
    "full",
    "ours",
)

DEFAULT_CHECKPOINTS = {
    "completionOnly": (
        "checkpoints/PCN/"
        "pcn_mbb_ablation_comp_only_best.pth"
    ),
    "full": (
        "checkpoints/PCN/"
        "pcn_mbb_ablation_full_best.pth"
    ),
    "ours": (
        "checkpoints/PCN/"
        "pcn_mbb_ablation_ours_best.pth"
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(
            f"Checkpoint must be a dict/state_dict, got {type(raw).__name__}."
        )

    state_dict = raw
    for candidate in (
        "model_state_dict",
        "state_dict",
        "model",
        "base_model",
        "net",
    ):
        value = raw.get(candidate)
        if isinstance(value, dict) and value:
            state_dict = value
            break

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key
        while new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key in cleaned:
            raise RuntimeError(f"Duplicate checkpoint key after cleaning: {new_key}")
        cleaned[new_key] = value

    if not cleaned:
        raise RuntimeError(f"No tensor state_dict found in {checkpoint_path}")
    return cleaned


def load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: str) -> None:
    state_dict = parse_state_dict(checkpoint_path)
    model_state = model.state_dict()

    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    shape_mismatch = sorted(
        key
        for key in set(model_state).intersection(state_dict)
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape)
    )

    matched_numel = sum(
        model_state[key].numel()
        for key in set(model_state).intersection(state_dict)
        if tuple(model_state[key].shape) == tuple(state_dict[key].shape)
    )
    total_numel = sum(value.numel() for value in model_state.values())
    coverage = 100.0 * matched_numel / max(total_numel, 1)

    print(
        f"[Checkpoint] {checkpoint_path}\n"
        f"  model tensors:      {len(model_state)}\n"
        f"  checkpoint tensors: {len(state_dict)}\n"
        f"  parameter coverage: {coverage:.4f}%"
    )

    if missing or unexpected or shape_mismatch:
        if missing:
            print("  Missing keys:")
            for key in missing[:30]:
                print(f"    - {key}")
        if unexpected:
            print("  Unexpected keys:")
            for key in unexpected[:30]:
                print(f"    - {key}")
        if shape_mismatch:
            print("  Shape mismatches:")
            for key in shape_mismatch[:30]:
                print(
                    f"    - {key}: model={tuple(model_state[key].shape)}, "
                    f"ckpt={tuple(state_dict[key].shape)}"
                )
        raise RuntimeError(
            "Checkpoint does not exactly match the selected PCN architecture."
        )

    model.load_state_dict(state_dict, strict=True)
    print("  strict load: OK")


def build_model(cfg: dict, mode: str):
    """Build one paper-facing PCN model from the shared public definition."""
    return build_pcn_model(
        cfg,
        normalize_pcn_mode(mode),
    )


def default_checkpoint(mode: str) -> str:
    return os.path.join(PROJECT_ROOT, DEFAULT_CHECKPOINTS[mode])


def make_audit_indices(dataset: PCNSnowflakeOfficialEvalDataset, per_class: int) -> List[int]:
    if per_class <= 0:
        return list(range(len(dataset)))

    counts = defaultdict(int)
    selected: List[int] = []
    for index, sample in enumerate(dataset.samples):
        taxonomy_id = sample["taxonomy_id"]
        if counts[taxonomy_id] < per_class:
            selected.append(index)
            counts[taxonomy_id] += 1
    print(
        f"[Audit subset] per_class={per_class}, categories={len(counts)}, "
        f"objects={len(selected)}"
    )
    return selected


def evaluate_one_mode(
    args,
    cfg: dict,
    loader: DataLoader,
    mode: str,
    checkpoint_path: str,
) -> dict:
    print("\n" + "=" * 92)
    print(
        f"PCN SnowflakeNet-protocol evaluation | mode={mode} | "
        f"checkpoint={checkpoint_path}"
    )
    print("=" * 92)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    model = build_model(cfg, mode).cuda()
    load_checkpoint_strict(model, checkpoint_path)
    model.eval()

    has_completion = mode != "cls_only"
    has_classification = normalize_pcn_mode(mode) != "completionOnly"

    per_class = {
        taxonomy_id: {
            "name": class_name,
            "cd": [],
            "f1": [],
            "correct": 0,
            "count": 0,
        }
        for taxonomy_id, class_name in PCN_TAXONOMY
    }

    start = time.time()
    with torch.inference_mode():
        progress = tqdm(loader, desc=f"PCN-{mode}", leave=True)
        for partial, gt, labels, taxonomy_ids, _model_ids in progress:
            partial = partial.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            coarse, fine, logits = model(partial)

            if has_completion:
                if fine.ndim != 3 or fine.shape[-1] != 3:
                    raise RuntimeError(
                        f"Invalid final prediction shape: {tuple(fine.shape)}"
                    )
                dist_pred_to_gt, dist_gt_to_pred = compute_squared_dists(fine, gt)
                cd_batch = calc_pcn_cd(dist_pred_to_gt, dist_gt_to_pred)
                f1_batch = calc_fscore(
                    dist_pred_to_gt,
                    dist_gt_to_pred,
                    threshold=args.f1_threshold,
                )
            else:
                cd_batch = None
                f1_batch = None

            if has_classification:
                if logits is None:
                    raise RuntimeError(f"Mode {mode} returned logits=None.")
                predictions = logits.argmax(dim=1)
            else:
                predictions = None

            for batch_index, taxonomy_id in enumerate(taxonomy_ids):
                entry = per_class[str(taxonomy_id)]
                entry["count"] += 1
                if has_completion:
                    entry["cd"].append(float(cd_batch[batch_index].item()))
                    entry["f1"].append(float(f1_batch[batch_index].item()))
                if has_classification:
                    entry["correct"] += int(
                        predictions[batch_index].item() == labels[batch_index].item()
                    )

            postfix = {}
            if has_completion:
                all_cd = [value for entry in per_class.values() for value in entry["cd"]]
                if all_cd:
                    postfix["CD"] = f"{np.mean(all_cd):.4f}"
            if has_classification:
                correct = sum(entry["correct"] for entry in per_class.values())
                count = sum(entry["count"] for entry in per_class.values())
                if count:
                    postfix["Acc"] = f"{100.0 * correct / count:.2f}%"
            if postfix:
                progress.set_postfix(postfix)

    elapsed = time.time() - start

    class_rows = []
    for taxonomy_id, class_name in PCN_TAXONOMY:
        entry = per_class[taxonomy_id]
        row = {
            "taxonomy_id": taxonomy_id,
            "class_name": class_name,
            "count": entry["count"],
            "cd_l1_x1000": (
                float(np.mean(entry["cd"])) if has_completion else None
            ),
            "f1_at_1pct": (
                float(np.mean(entry["f1"])) if has_completion else None
            ),
            "accuracy": (
                float(entry["correct"] / entry["count"])
                if has_classification and entry["count"] > 0
                else None
            ),
        }
        class_rows.append(row)

    result = {
        "protocol": (
            "SnowflakeNet UpSamplePoints: random n_valid in [512,1024], "
            "NumPy FPS, repeat real points to 2048; deterministic per-object seed"
        ),
        "seed": args.seed,
        "mode": mode,
        "checkpoint": checkpoint_path,
        "objects": sum(row["count"] for row in class_rows),
        "elapsed_seconds": elapsed,
        "per_class": class_rows,
    }

    if has_completion:
        # PCN has exactly 150 test objects per taxonomy. Report both macro and
        # micro for transparency; on the full balanced split they should agree.
        result["cd_l1_x1000_macro"] = float(
            np.mean([row["cd_l1_x1000"] for row in class_rows])
        )
        result["f1_at_1pct_macro"] = float(
            np.mean([row["f1_at_1pct"] for row in class_rows])
        )
        result["cd_l1_x1000_micro"] = float(
            np.mean([value for entry in per_class.values() for value in entry["cd"]])
        )
        result["f1_at_1pct_micro"] = float(
            np.mean([value for entry in per_class.values() for value in entry["f1"]])
        )

    if has_classification:
        total_correct = sum(entry["correct"] for entry in per_class.values())
        total_count = sum(entry["count"] for entry in per_class.values())
        result["accuracy_micro"] = float(total_correct / total_count)
        result["accuracy_macro"] = float(
            np.mean([row["accuracy"] for row in class_rows])
        )

    print("\n--- PCN Standard Evaluation Result ---")
    print(f"Mode:       {mode}")
    print(f"Objects:    {result['objects']}")
    print(f"Time:       {elapsed / 60.0:.2f} min")
    print(f"Input seed: {args.seed}")
    print("Protocol:   SnowflakeNet official UpSamplePoints, fixed per-object inputs")

    header = (
        f"{'Taxonomy':<10} {'Class':<12} {'#':>5} "
        f"{'CD-L1':>10} {'F1@1%':>10} {'Acc':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in class_rows:
        cd_text = (
            f"{row['cd_l1_x1000']:.4f}"
            if row["cd_l1_x1000"] is not None
            else "-"
        )
        f1_text = (
            f"{row['f1_at_1pct']:.4f}"
            if row["f1_at_1pct"] is not None
            else "-"
        )
        acc_text = (
            f"{100.0 * row['accuracy']:.2f}%"
            if row["accuracy"] is not None
            else "-"
        )
        print(
            f"{row['taxonomy_id']:<10} {row['class_name']:<12} "
            f"{row['count']:>5d} {cd_text:>10} {f1_text:>10} {acc_text:>10}"
        )

    print("-" * len(header))
    if has_completion:
        print(
            f"Overall macro | CD-L1={result['cd_l1_x1000_macro']:.4f} | "
            f"F1={result['f1_at_1pct_macro']:.4f}"
        )
        print(
            f"Overall micro | CD-L1={result['cd_l1_x1000_micro']:.4f} | "
            f"F1={result['f1_at_1pct_micro']:.4f}"
        )
    if has_classification:
        print(
            f"Accuracy      | micro={100.0*result['accuracy_micro']:.2f}% | "
            f"macro={100.0*result['accuracy_macro']:.2f}%"
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing PCN MBB checkpoints with SnowflakeNet's official "
            "UpSamplePoints test transform."
        )
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        required=True,
        choices=VALID_MODES,
    )
    parser.add_argument(
        "--checkpoint",
        nargs="*",
        default=None,
        help="Optional paths corresponding one-to-one with --mode.",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "cfgs", "PCN_config.yaml"),
    )
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--audit_per_class", type=int, default=0)
    parser.add_argument("--f1_threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable transformed input/GT caching.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "PCN", "evaluation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model inference and Chamfer metrics.")

    if args.checkpoint is not None and len(args.checkpoint) != len(args.mode):
        raise ValueError(
            "--checkpoint must contain exactly the same number of paths as --mode."
        )

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["NUM_CLASSES"] = 8
    cfg["AUGMENT"] = False

    # Public release path resolution:
    # command-line override > config value.
    # Do not depend on the historical resolver's positional signature.
    configured_root = (
        cfg.get("DATASET_ROOT")
        or cfg.get("DATA_ROOT")
    )

    requested_root = (
        args.data_root
        if args.data_root is not None
        else configured_root
    )

    if requested_root is None:
        raise RuntimeError(
            "PCN data root is not configured. "
            "Use --data_root /path/to/ShapeNetCompletion."
        )

    data_root_path = Path(
        requested_root
    ).expanduser()

    if not data_root_path.is_absolute():
        data_root_path = (
            Path(PROJECT_ROOT)
            / data_root_path
        )

    data_root_path = data_root_path.resolve()

    if not data_root_path.is_dir():
        raise FileNotFoundError(
            f"PCN data root does not exist: {data_root_path}"
        )

    data_root = str(data_root_path)

    print(
        f"[PCN data] root={data_root}"
    )
    dataset = PCNSnowflakeOfficialEvalDataset(
        data_root=data_root,
        num_partial=int(cfg.get("NUM_PARTIAL_POINTS", 2048)),
        num_complete=int(cfg.get("NUM_COMPLETE_POINTS", 16384)),
        seed=args.seed,
        cache=not args.no_cache,
    )

    indices = make_audit_indices(dataset, args.audit_per_class)
    eval_dataset: Dataset = (
        dataset if len(indices) == len(dataset) else Subset(dataset, indices)
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(args.num_workers > 0),
    )

    checkpoint_paths = (
        [default_checkpoint(mode) for mode in args.mode]
        if args.checkpoint is None
        else args.checkpoint
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = (
        f"audit{args.audit_per_class}perclass"
        if args.audit_per_class > 0
        else "full"
    )
    all_results = []
    for mode, checkpoint_path in zip(args.mode, checkpoint_paths):
        result = evaluate_one_mode(
            args=args,
            cfg=cfg,
            loader=loader,
            mode=mode,
            checkpoint_path=checkpoint_path,
        )
        all_results.append(result)
        result_path = output_dir / f"PCN_{mode}_{suffix}_seed{args.seed}.json"
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"[Saved] {result_path}")

    summary_path = output_dir / f"PCN_summary_{suffix}_seed{args.seed}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)
    print(f"\n[Summary saved] {summary_path}")


if __name__ == "__main__":
    main()
