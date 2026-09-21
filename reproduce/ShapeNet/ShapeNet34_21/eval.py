#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Official ShapeNet-34 / ShapeNet-Unseen21 evaluation entry.

Important:
- Models are trained on ShapeNet-34.
- The same ShapeNet-34 checkpoints are evaluated on ShapeNet-34
  and ShapeNet-Unseen21.
- ShapeNet-21 is completion-only evaluation. Classification
  accuracy is intentionally not reported because its categories
  do not belong to the ShapeNet-34 classifier label space.
- The actual metric/model evaluation core is shared with the
  already verified ShapeNet-55 release evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data.shapenet34_21_official_eval import (
    ShapeNet34OfficialEvalDataset,
)

# Reuse the already verified release evaluation core.
from reproduce.ShapeNet.ShapeNet55.eval_table2 import (
    evaluate_one_mode,
    set_seed,
    stratified_subset_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALID_MODES = (
    "baseline",
    "no_bridge",
    "ours",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Official ShapeNet-34 / ShapeNet-Unseen21 "
            "evaluation for released MBB-Net checkpoints."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=("34", "21"),
    )

    parser.add_argument(
        "--mode",
        nargs="+",
        default=[
            "baseline",
            "no_bridge",
            "ours",
        ],
        choices=VALID_MODES,
    )

    parser.add_argument(
        "--checkpoint",
        nargs="*",
        default=None,
        help=(
            "Optional checkpoint paths corresponding "
            "one-to-one with --mode."
        ),
    )

    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "ShapeNet34_21_config.yaml"
        ),
    )

    parser.add_argument(
        "--data_root",
        required=True,
        help=(
            "ShapeNet data root containing shapenet_pc/."
        ),
    )

    parser.add_argument(
        "--split_root",
        default=str(
            PROJECT_ROOT
            / "data"
            / "splits"
        ),
        help=(
            "Directory containing ShapeNet-34/ and "
            "ShapeNet-Unseen21/ split files."
        ),
    )

    parser.add_argument(
        "--audit_per_class",
        type=int,
        default=0,
        help=(
            "0 evaluates the full set; "
            "1 selects one deterministic object per category."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--metric_chunk",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--f1_threshold",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output_dir",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "shapenet34_21"
        ),
    )

    return parser.parse_args()


def default_checkpoint(
    mode: str,
) -> Path:

    if mode not in VALID_MODES:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    return (
        PROJECT_ROOT
        / "checkpoints"
        / "ShapeNet"
        / "ShapeNet34"
        / mode
        / "best_model.pth"
    )


def build_dataset_and_cfg(args):
    config_path = Path(
        args.config
    ).expanduser().resolve()

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    # Network is ALWAYS the ShapeNet-34-trained
    # 34-class model, even for unseen-21 evaluation.
    cfg["NUM_CLASSES"] = 34

    cfg["DATA_ROOT"] = str(
        Path(args.data_root)
        .expanduser()
        .resolve()
    )

    cfg["SPLIT_ROOT"] = str(
        Path(args.split_root)
        .expanduser()
        .resolve()
    )

    dataset_cfg = dict(cfg)
    dataset_cfg["DATASET_TYPE"] = args.dataset

    dataset = ShapeNet34OfficialEvalDataset(
        dataset_cfg,
        subset="test",
    )

    print(
        f"[Config] {config_path}"
    )
    print(
        f"[Data root] {cfg['DATA_ROOT']}"
    )
    print(
        f"[Split root] {cfg['SPLIT_ROOT']}"
    )
    print(
        "[Model classes] 34"
    )

    if args.dataset == "21":
        print(
            "[Protocol] ShapeNet-21 is unseen-category "
            "completion evaluation; classification accuracy "
            "is not reported."
        )

    return dataset, cfg


def main():
    args = parse_args()

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for pointnet2 FPS "
            "and official evaluation."
        )

    if (
        args.checkpoint is not None
        and len(args.checkpoint) != len(args.mode)
    ):
        raise ValueError(
            "--checkpoint must contain exactly the same "
            "number of paths as --mode."
        )

    dataset, cfg = build_dataset_and_cfg(
        args
    )

    subset_indices = stratified_subset_indices(
        dataset,
        args.audit_per_class,
    )

    eval_dataset = (
        dataset
        if len(subset_indices) == len(dataset)
        else Subset(
            dataset,
            subset_indices,
        )
    )

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    if args.checkpoint is None:
        checkpoint_paths = [
            str(default_checkpoint(mode))
            for mode in args.mode
        ]
    else:
        checkpoint_paths = [
            str(
                Path(p)
                .expanduser()
                .resolve()
            )
            for p in args.checkpoint
        ]

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for mode, checkpoint in zip(
        args.mode,
        checkpoint_paths,
    ):
        result = evaluate_one_mode(
            args=args,
            cfg=cfg,
            loader=loader,
            mode=mode,
            checkpoint_path=checkpoint,
        )

        results.append(
            result
        )

        suffix = (
            f"audit{args.audit_per_class}perclass"
            if args.audit_per_class > 0
            else "full"
        )

        result_path = (
            output_dir
            / (
                f"ShapeNet{args.dataset}_"
                f"{mode}_{suffix}.json"
            )
        )

        with result_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[Saved] {result_path}"
        )

    suffix = (
        f"audit{args.audit_per_class}perclass"
        if args.audit_per_class > 0
        else "full"
    )

    summary_path = (
        output_dir
        / (
            f"ShapeNet{args.dataset}_"
            f"summary_{suffix}.json"
        )
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\n[Summary saved] {summary_path}"
    )


if __name__ == "__main__":
    main()
