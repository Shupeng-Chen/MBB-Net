#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Official-protocol evaluation for SymmCompletion + MBB transfer checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Subset,
)
from tqdm import tqdm


# ---------------------------------------------------------------------
# Locate the public MBB-Net root and vendored SymmCompletion code.
# This file lives at:
# Public entry under reproduce/SymmCompletion/.
# ---------------------------------------------------------------------
import sys

CURRENT_DIR = Path(__file__).resolve().parent
MBB_ROOT = CURRENT_DIR.parents[1]

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

_BOOTSTRAP_PARSER = argparse.ArgumentParser(add_help=False)
_BOOTSTRAP_PARSER.add_argument(
    "--symm_root",
    default=os.environ.get(
        "SYMM_ROOT",
        str(MBB_ROOT / "third_party" / "symmcompletion"),
    ),
)
_BOOTSTRAP_ARGS, _ = _BOOTSTRAP_PARSER.parse_known_args()
SYMM_ROOT = Path(_BOOTSTRAP_ARGS.symm_root).expanduser().resolve()

if str(SYMM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYMM_ROOT))

from symm_mbb.checkpoint import (
    load_model_strict,
)
from symm_mbb.data import (
    TaxonomyLabelMap,
    build_dataset_bundle,
    stratified_indices,
)
from symm_mbb.metrics import (
    AccuracyAccumulator,
    CompletionMetricAccumulator,
    FIXED_CROP_DIRECTIONS,
    SHAPENET_DIFFICULTY_CROPS,
    per_sample_completion_metrics,
)
from symm_mbb.model import (
    SymmMBBTransfer,
)
from utils import misc


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = (
        True
    )
    torch.backends.cudnn.benchmark = (
        False
    )


def taxonomy_list(
    taxonomy_ids: Sequence,
):
    return [
        str(
            item.item()
            if isinstance(
                item,
                torch.Tensor,
            )
            else item
        )
        for item in taxonomy_ids
    ]


@torch.no_grad()
def evaluate_pcn(
    model: SymmMBBTransfer,
    loader: DataLoader,
    label_map: TaxonomyLabelMap,
    device: torch.device,
) -> Dict:
    model.eval()

    completion = (
        CompletionMetricAccumulator()
    )
    accuracy = AccuracyAccumulator()

    for taxonomy_ids, _, data in tqdm(
        loader,
        desc="[PCN official test]",
    ):
        taxonomies = taxonomy_list(
            taxonomy_ids
        )
        partial = data[0].to(
            device,
            non_blocking=True,
        ).float()
        ground_truth = data[1].to(
            device,
            non_blocking=True,
        ).float()
        labels = label_map.encode_batch(
            taxonomy_ids,
            device,
        )

        outputs, logits = model(
            partial
        )
        metrics = (
            per_sample_completion_metrics(
                outputs[-1],
                ground_truth,
            )
        )
        completion.update(
            taxonomies,
            metrics,
        )
        accuracy.update_logits(
            logits,
            labels,
            taxonomies,
        )

    return {
        "dataset": "pcn",
        "completion_taxonomy_macro": (
            completion.taxonomy_macro()
        ),
        "completion_object_average": (
            completion.object_average()
        ),
        "completion_per_category": (
            completion.per_category()
        ),
        "accuracy": (
            accuracy.state_dict()
        ),
        "bridge": model.bridge_state(),
    }


@torch.no_grad()
def evaluate_shapenet55(
    model: SymmMBBTransfer,
    loader: DataLoader,
    label_map: TaxonomyLabelMap,
    device: torch.device,
    difficulties,
) -> Dict:
    model.eval()

    completion = {
        difficulty: (
            CompletionMetricAccumulator()
        )
        for difficulty in difficulties
    }
    object_accuracy = {
        difficulty: AccuracyAccumulator()
        for difficulty in difficulties
    }
    view_accuracy = {
        difficulty: AccuracyAccumulator()
        for difficulty in difficulties
    }

    for taxonomy_ids, _, data in tqdm(
        loader,
        desc="[ShapeNet-55 official 8-view test]",
    ):
        taxonomies = taxonomy_list(
            taxonomy_ids
        )
        ground_truth = data.to(
            device,
            non_blocking=True,
        ).float()
        labels = label_map.encode_batch(
            taxonomy_ids,
            device,
        )

        for difficulty in difficulties:
            crop_count = (
                SHAPENET_DIFFICULTY_CROPS[
                    difficulty
                ]
            )
            logits_sum = None

            for direction_values in (
                FIXED_CROP_DIRECTIONS
            ):
                direction = torch.tensor(
                    direction_values,
                    dtype=ground_truth.dtype,
                    device=device,
                )
                partial, _ = (
                    misc.seprate_point_cloud(
                        ground_truth,
                        8192,
                        crop_count,
                        fixed_points=direction,
                    )
                )
                partial = misc.fps(
                    partial,
                    2048,
                )

                outputs, logits = model(
                    partial
                )
                metrics = (
                    per_sample_completion_metrics(
                        outputs[-1],
                        ground_truth,
                    )
                )
                completion[
                    difficulty
                ].update(
                    taxonomies,
                    metrics,
                )
                view_accuracy[
                    difficulty
                ].update_logits(
                    logits,
                    labels,
                    taxonomies,
                )
                logits_sum = (
                    logits
                    if logits_sum is None
                    else logits_sum + logits
                )

            mean_logits = (
                logits_sum
                / len(
                    FIXED_CROP_DIRECTIONS
                )
            )
            object_accuracy[
                difficulty
            ].update_logits(
                mean_logits,
                labels,
                taxonomies,
            )

    difficulty_results = {}
    for difficulty in difficulties:
        difficulty_results[
            difficulty
        ] = {
            "completion_taxonomy_macro": (
                completion[
                    difficulty
                ].taxonomy_macro()
            ),
            "completion_object_view_average": (
                completion[
                    difficulty
                ].object_average()
            ),
            "completion_per_category": (
                completion[
                    difficulty
                ].per_category()
            ),
            "accuracy_object_mean_logits": (
                object_accuracy[
                    difficulty
                ].state_dict()
            ),
            "accuracy_view_level": (
                view_accuracy[
                    difficulty
                ].state_dict()
            ),
        }

    average_macro = {}
    for metric_name in (
        "F-Score",
        "CDL1",
        "CDL2",
    ):
        average_macro[
            metric_name
        ] = sum(
            difficulty_results[
                difficulty
            ][
                "completion_taxonomy_macro"
            ][metric_name]
            for difficulty in difficulties
        ) / len(difficulties)

    return {
        "dataset": "shapenet55",
        "views_per_difficulty": 8,
        "difficulties": (
            difficulty_results
        ),
        "average_completion_taxonomy_macro": (
            average_macro
        ),
        "bridge": model.bridge_state(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symm_root",
        default=str(SYMM_ROOT),
        help=(
            "Path to the external official SymmCompletion repository. "
            "It must contain models/, datasets/, utils/, and extensions/."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "pcn",
            "shapenet55",
        ),
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "no_bridge",
            "full",
            "ours",
            "s2g",
            "g2s",
        ),
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help=(
            "The same official SymmCompletion checkpoint used to initialize training."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Transfer ckpt-best.pth to evaluate."
        ),
    )
    parser.add_argument(
        "--tune_policy",
        choices=(
            "full",
            "adapter",
        ),
        default="full",
    )
    parser.add_argument(
        "--dataset_config",
        default=None,
    )
    parser.add_argument(
        "--pcn_category_file",
        default=None,
    )
    parser.add_argument(
        "--pcn_partial_pattern",
        default=None,
    )
    parser.add_argument(
        "--pcn_complete_pattern",
        default=None,
    )
    parser.add_argument(
        "--shapenet_index_root",
        default=None,
    )
    parser.add_argument(
        "--shapenet_pc_root",
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--audit_per_class",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--difficulty",
        choices=(
            "all",
            "simple",
            "moderate",
            "hard",
        ),
        default="all",
    )
    parser.add_argument(
        "--output_json",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )
    set_seed(
        args.seed
    )

    symm_root = Path(
        args.symm_root
    ).expanduser().resolve()
    required_symm_files = (
        symm_root / "models" / "SymmCompletion.py",
        symm_root / "datasets" / "PCNDataset.py",
        symm_root / "datasets" / "ShapeNet55Dataset.py",
        symm_root / "utils" / "misc.py",
    )
    missing_symm_files = [
        str(path)
        for path in required_symm_files
        if not path.is_file()
    ]
    if missing_symm_files:
        raise RuntimeError(
            "Invalid --symm_root. Missing required official files:\n"
            + "\n".join(missing_symm_files)
        )
    bundle = build_dataset_bundle(
        dataset=args.dataset,
        repository_root=str(
            symm_root
        ),
        dataset_config_path=(
            args.dataset_config
        ),
        pcn_category_file=(
            args.pcn_category_file
        ),
        pcn_partial_pattern=(
            args.pcn_partial_pattern
        ),
        pcn_complete_pattern=(
            args.pcn_complete_pattern
        ),
        shapenet_index_root=(
            args.shapenet_index_root
        ),
        shapenet_pc_root=(
            args.shapenet_pc_root
        ),
    )

    checkpoint_header = torch.load(
        args.checkpoint,
        map_location="cpu",
    )
    metadata = checkpoint_header.get(
        "metadata",
        {},
    )
    stored_label_map = metadata.get(
        "label_map"
    )
    if stored_label_map is not None:
        label_map = (
            TaxonomyLabelMap.from_state_dict(
                stored_label_map
            )
        )
        if (
            label_map.label_to_taxonomy
            != bundle.label_map.label_to_taxonomy
        ):
            raise RuntimeError(
                "Checkpoint taxonomy mapping differs from the current dataset mapping."
            )
    else:
        label_map = bundle.label_map

    dataset = bundle.test_dataset
    if args.audit_per_class > 0:
        indices = stratified_indices(
            dataset,
            args.audit_per_class,
        )
        dataset = Subset(
            dataset,
            indices,
        )
        print(
            f"[Audit] selected {len(indices)} samples."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    device = torch.device(
        "cuda"
    )
    model = SymmMBBTransfer(
        dataset=args.dataset,
        num_classes=bundle.num_classes,
        official_checkpoint=(
            args.pretrained
        ),
        variant=args.variant,
        tune_policy=(
            args.tune_policy
        ),
    ).to(device)
    loaded = load_model_strict(
        model,
        args.checkpoint,
    )

    print("\n" + "=" * 96)
    print("SymmCompletion + MBB official evaluation")
    print("=" * 96)
    print(f"MBB root:            {MBB_ROOT}")
    print(f"Symm root:           {symm_root}")
    print(f"Dataset:             {args.dataset}")
    print(f"Variant:             {args.variant}")
    print(f"Transfer checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Saved epoch:         {loaded.get('epoch', -1)}")
    print(f"Samples:             {len(dataset)}")
    print("=" * 96)

    if args.dataset == "pcn":
        result = evaluate_pcn(
            model,
            loader,
            label_map,
            device,
        )
        macro = result[
            "completion_taxonomy_macro"
        ]
        print(
            "\nPCN taxonomy-macro results\n"
            f"F-Score: {macro['F-Score']:.6f}\n"
            f"CDL1:    {macro['CDL1']:.6f}\n"
            f"CDL2:    {macro['CDL2']:.6f}\n"
            f"Acc micro: "
            f"{100.0*result['accuracy']['micro']:.2f}%\n"
            f"Acc macro: "
            f"{100.0*result['accuracy']['taxonomy_macro']:.2f}%"
        )
    else:
        difficulties = (
            (
                "simple",
                "moderate",
                "hard",
            )
            if args.difficulty == "all"
            else (
                args.difficulty,
            )
        )
        result = (
            evaluate_shapenet55(
                model,
                loader,
                label_map,
                device,
                difficulties,
            )
        )

        print(
            "\nShapeNet-55 taxonomy-macro results"
        )
        for difficulty in difficulties:
            block = result[
                "difficulties"
            ][difficulty]
            macro = block[
                "completion_taxonomy_macro"
            ]
            object_acc = block[
                "accuracy_object_mean_logits"
            ]
            view_acc = block[
                "accuracy_view_level"
            ]
            print(
                f"{difficulty:8s} | "
                f"CDL2={macro['CDL2']:.6f} | "
                f"CDL1={macro['CDL1']:.6f} | "
                f"F1={macro['F-Score']:.6f} | "
                f"AccObj={100.0*object_acc['micro']:.2f}% | "
                f"AccView={100.0*view_acc['micro']:.2f}%"
            )

        average = result[
            "average_completion_taxonomy_macro"
        ]
        print(
            f"Average  | "
            f"CDL2={average['CDL2']:.6f} | "
            f"CDL1={average['CDL1']:.6f} | "
            f"F1={average['F-Score']:.6f}"
        )

    output_path = (
        Path(args.output_json)
        if args.output_json
        else Path(args.checkpoint).with_name(
            "official_results.json"
        )
    )
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(
        f"\nSaved JSON results to {output_path}"
    )


if __name__ == "__main__":
    main()
