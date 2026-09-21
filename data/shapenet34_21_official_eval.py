#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Official-evaluation datasets for ShapeNet-34 seen categories and
ShapeNet-21 unseen categories.

Partial point clouds are not generated here. Use:
    from datasets.ShapeNet_official_eval import generate_official_partial
inside the CUDA evaluation loop.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from data.shapenet55_official_eval import normalize_unit_sphere


class ShapeNet34OfficialEvalDataset(Dataset):
    """
    ShapeNet-34 / ShapeNet-Unseen21 deterministic evaluation dataset.

    DATASET_TYPE='34':
        label is in [0, 33].

    DATASET_TYPE='21':
        label is -1 because unseen categories do not belong to the
        ShapeNet-34 classifier label space. Completion metrics are still valid.

    Returns:
        {
            "gt": FloatTensor (8192, 3),
            "label": LongTensor scalar,
            "model_id": str,
            "cat_id": str
        }
    """

    def __init__(self, cfg: dict, subset: str = "test") -> None:
        super().__init__()

        if subset != "test":
            raise ValueError(
                "ShapeNet34OfficialEvalDataset is eval-only; subset must be 'test'."
            )

        self.subset = subset
        self.data_root = cfg["DATA_ROOT"]
        self.pc_path = os.path.join(self.data_root, "shapenet_pc")
        self.split_root = cfg["SPLIT_ROOT"]
        self.dataset_type = str(cfg.get("DATASET_TYPE", "34"))
        self.gt_npoints = int(cfg.get("NUM_GT_POINTS", 8192))

        if self.dataset_type not in {"34", "21"}:
            raise ValueError(
                f"DATASET_TYPE must be '34' or '21', got '{self.dataset_type}'."
            )

        train_34_file = os.path.join(
            self.split_root,
            "ShapeNet-34",
            "train.txt",
        )
        if not os.path.isfile(train_34_file):
            raise FileNotFoundError(train_34_file)

        with open(train_34_file, "r", encoding="utf-8") as handle:
            train_ids = [
                line.strip().replace(".npy", "")
                for line in handle
                if line.strip()
            ]

        self.classes_34: List[str] = sorted(
            {model_id.split("-")[0] for model_id in train_ids}
        )
        if len(self.classes_34) != 34:
            raise RuntimeError(
                f"Expected 34 seen classes, found {len(self.classes_34)} "
                f"in {train_34_file}."
            )

        self.cat2label = {
            cat_id: class_idx
            for class_idx, cat_id in enumerate(self.classes_34)
        }

        if self.dataset_type == "34":
            split_file = os.path.join(
                self.split_root,
                "ShapeNet-34",
                "test.txt",
            )
        else:
            split_file = os.path.join(
                self.split_root,
                "ShapeNet-Unseen21",
                "test.txt",
            )

        if not os.path.isfile(split_file):
            raise FileNotFoundError(split_file)
        if not os.path.isdir(self.pc_path):
            raise FileNotFoundError(self.pc_path)

        with open(split_file, "r", encoding="utf-8") as handle:
            self.file_list = [
                line.strip().replace(".npy", "")
                for line in handle
                if line.strip()
            ]

        if not self.file_list:
            raise RuntimeError(f"No samples found in {split_file}.")

        if self.dataset_type == "34":
            invalid = sorted(
                {
                    model_id.split("-")[0]
                    for model_id in self.file_list
                    if model_id.split("-")[0] not in self.cat2label
                }
            )
            if invalid:
                raise RuntimeError(
                    f"ShapeNet-34 test split contains unknown categories: {invalid}"
                )

        categories = sorted(
            {model_id.split("-")[0] for model_id in self.file_list}
        )

        print(
            f"[ShapeNet{self.dataset_type}OfficialEvalDataset] "
            f"samples={len(self.file_list)}, categories={len(categories)}, "
            f"GT={self.gt_npoints}"
        )

    def __getitem__(self, index: int) -> dict:
        model_id = self.file_list[index]
        cat_id = model_id.split("-")[0]
        file_path = os.path.join(self.pc_path, model_id + ".npy")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        gt = np.load(file_path).astype(np.float32, copy=False)
        if gt.ndim != 2 or gt.shape[1] != 3:
            raise ValueError(f"{file_path}: expected (N, 3), got {gt.shape}.")
        if gt.shape[0] != self.gt_npoints:
            raise ValueError(
                f"{file_path}: expected {self.gt_npoints} points, got {gt.shape[0]}."
            )

        gt = normalize_unit_sphere(gt)
        label = self.cat2label[cat_id] if self.dataset_type == "34" else -1

        return {
            "gt": torch.from_numpy(gt).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "model_id": model_id,
            "cat_id": cat_id,
        }

    def __len__(self) -> int:
        return len(self.file_list)