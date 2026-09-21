#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ShapeNet-55 official-evaluation dataset and PoinTr-style partial generation.

This file is ONLY for evaluating existing checkpoints. It does not change or
resume the original training protocol.

Official evaluation pipeline:
    normalized GT (8192 points)
    -> remove the points nearest to one of eight fixed crop centers
    -> FPS to 2048 points
    -> evaluate all 8 views for Simple / Moderate / Hard

The eight fixed crop centers and crop direction match the PoinTr runner:
    partial, _ = misc.seprate_point_cloud(gt, 8192, num_crop, fixed_points=item)
    partial = misc.fps(partial, 2048)
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from pointnet2_ops import pointnet2_utils
except ImportError as exc:
    raise ImportError(
        "Cannot import pointnet2_ops. Activate the environment used by "
        "SnowflakeNet/PoinTr and make sure pointnet2_ops is installed."
    ) from exc


SHAPENET55_CLASSES: List[str] = [
    '02691156', '02747177', '02773838', '02801938', '02808440', '02818832',
    '02828884', '02843684', '02871439', '02876657', '02880940', '02924116',
    '02933112', '02942699', '02946921', '02954340', '02958343', '02992529',
    '03001627', '03046257', '03085013', '03207941', '03211117', '03261776',
    '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
    '03636649', '03642806', '03691459', '03710193', '03759954', '03761084',
    '03790512', '03797390', '03928116', '03938244', '03948459', '03991062',
    '04004475', '04074963', '04090263', '04099429', '04225987', '04256520',
    '04330267', '04379243', '04401088', '04460130', '04468005', '04530566',
    '04554684'
]

# Exact order used by the supplied PoinTr tools/runner.py.
FIXED_CROP_CENTERS = torch.tensor(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
    ],
    dtype=torch.float32,
)

DIFFICULTY_CROPS: Dict[str, int] = {
    "simple": 2048,
    "moderate": 4096,
    "hard": 6144,
}


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Apply the same unit-sphere normalization used by the old training loader."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {points.shape}.")

    centroid = np.mean(points, axis=0, keepdims=True)
    points = points - centroid
    radius = float(np.max(np.sqrt(np.sum(points ** 2, axis=1))))

    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError("Degenerate point cloud: invalid normalization radius.")

    return np.ascontiguousarray(points / radius, dtype=np.float32)


@torch.no_grad()
def fps_cuda(points: torch.Tensor, number: int) -> torch.Tensor:
    """
    Farthest-point sample a CUDA tensor.

    Args:
        points: (B, N, 3), CUDA float tensor.
        number: number of output points.

    Returns:
        (B, number, 3)
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected (B, N, 3), got {tuple(points.shape)}.")
    if not points.is_cuda:
        raise RuntimeError("fps_cuda must run on a CUDA tensor.")
    if points.shape[1] < number:
        raise ValueError(
            f"Cannot FPS {points.shape[1]} input points to {number} points."
        )

    points = points.contiguous()
    fps_idx = pointnet2_utils.furthest_point_sample(points, number)
    sampled = pointnet2_utils.gather_operation(
        points.transpose(1, 2).contiguous(),
        fps_idx,
    )
    return sampled.transpose(1, 2).contiguous()


@torch.no_grad()
def generate_official_partial(
    gt: torch.Tensor,
    difficulty: str,
    view_idx: int,
    num_partial: int = 2048,
) -> torch.Tensor:
    """
    Generate one PoinTr-style ShapeNet test partial.

    The implementation exactly follows the supplied PoinTr logic:
      1. compute distance to fixed_points;
      2. sort ascending;
      3. remove idx[:num_crop], i.e. points nearest to the crop center;
      4. FPS the remaining point cloud to 2048 points.
    """
    if difficulty not in DIFFICULTY_CROPS:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from {list(DIFFICULTY_CROPS)}."
        )
    if view_idx < 0 or view_idx >= len(FIXED_CROP_CENTERS):
        raise ValueError(f"view_idx must be in [0, 7], got {view_idx}.")
    if gt.ndim != 3 or gt.shape[-1] != 3:
        raise ValueError(f"Expected GT shape (B, N, 3), got {tuple(gt.shape)}.")
    if not gt.is_cuda:
        raise RuntimeError("GT must be moved to CUDA before generating partials.")

    num_crop = DIFFICULTY_CROPS[difficulty]
    remaining = gt.shape[1] - num_crop
    if remaining < num_partial:
        raise ValueError(
            f"GT has {gt.shape[1]} points. After cropping {num_crop}, only "
            f"{remaining} remain, fewer than num_partial={num_partial}."
        )

    center = FIXED_CROP_CENTERS[view_idx].to(
        device=gt.device,
        dtype=gt.dtype,
    ).view(1, 1, 3)

    distances = torch.norm(gt - center, p=2, dim=-1)
    sorted_idx = torch.argsort(distances, dim=1, descending=False)
    keep_idx = sorted_idx[:, num_crop:]

    observed = torch.gather(
        gt,
        dim=1,
        index=keep_idx.unsqueeze(-1).expand(-1, -1, 3),
    ).contiguous()

    return fps_cuda(observed, num_partial)


class ShapeNet55OfficialEvalDataset(Dataset):
    """
    Deterministic ShapeNet-55 evaluation dataset.

    Returns a dictionary:
        gt:       FloatTensor (8192, 3)
        label:    LongTensor scalar
        model_id: string
        cat_id:   string

    It intentionally does not generate partials inside DataLoader workers.
    Partials are generated on GPU by generate_official_partial().
    """

    def __init__(self, cfg: dict, subset: str = "test") -> None:
        super().__init__()

        if subset != "test":
            raise ValueError(
                "ShapeNet55OfficialEvalDataset is eval-only; subset must be 'test'."
            )

        self.subset = subset
        self.data_root = cfg["DATA_ROOT"]
        self.pc_path = os.path.join(self.data_root, "shapenet_pc")
        self.gt_npoints = int(cfg.get("NUM_GT_POINTS", 8192))

        self.classes = list(SHAPENET55_CLASSES)
        self.cat2label = {
            cat_id: class_idx for class_idx, cat_id in enumerate(self.classes)
        }

        split_file = os.path.join(self.data_root, "test.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Cannot find split file: {split_file}")
        if not os.path.isdir(self.pc_path):
            raise FileNotFoundError(f"Cannot find point-cloud folder: {self.pc_path}")

        with open(split_file, "r", encoding="utf-8") as handle:
            raw_ids = [
                line.strip().replace(".npy", "")
                for line in handle
                if line.strip()
            ]

        self.file_list = [
            model_id
            for model_id in raw_ids
            if model_id.split("-")[0] in self.cat2label
        ]

        if not self.file_list:
            raise RuntimeError(f"No valid samples found in {split_file}.")

        print(
            f"[ShapeNet55OfficialEvalDataset] samples={len(self.file_list)}, "
            f"classes={len(self.classes)}, GT={self.gt_npoints}"
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

        return {
            "gt": torch.from_numpy(gt).float(),
            "label": torch.tensor(self.cat2label[cat_id], dtype=torch.long),
            "model_id": model_id,
            "cat_id": cat_id,
        }

    def __len__(self) -> int:
        return len(self.file_list)