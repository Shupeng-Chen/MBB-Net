#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Historical ShapeNet-55 evaluation protocol used by the
SymmCompletion transfer experiments.

Important:
    This module intentionally uses the vendored historical
    SymmCompletion PointNet2 FPS implementation.

    Do NOT replace this FPS backend with the repository-wide
    pointnet2_ops backend: the two CUDA implementations can
    select different points in tie / near-tie cases.
"""

from __future__ import annotations

from typing import Dict

import torch

from third_party.symmcompletion.pointnet2_legacy import (
    pointnet2_utils as legacy_pointnet2,
)


FIXED_CROP_DIRECTIONS = torch.tensor(
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


SHAPENET55_DIFFICULTY_CROPS: Dict[str, int] = {
    "simple": 2048,
    "moderate": 4096,
    "hard": 6144,
}


@torch.no_grad()
def legacy_fps(
    points: torch.Tensor,
    number: int,
) -> torch.Tensor:
    """
    Historical SymmCompletion FPS.

    Args:
        points: CUDA tensor (B, N, 3).
        number: requested number of points.

    Returns:
        CUDA tensor (B, number, 3).
    """

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(
            f"Expected (B, N, 3), got {tuple(points.shape)}."
        )

    if not points.is_cuda:
        raise RuntimeError(
            "Historical SymmCompletion FPS requires CUDA."
        )

    if points.shape[1] < number:
        raise ValueError(
            f"Cannot sample {number} from "
            f"{points.shape[1]} points."
        )

    points = points.contiguous()

    fps_idx = legacy_pointnet2.furthest_point_sample(
        points,
        number,
    )

    sampled = legacy_pointnet2.gather_operation(
        points.transpose(1, 2).contiguous(),
        fps_idx,
    )

    return (
        sampled
        .transpose(1, 2)
        .contiguous()
    )


@torch.no_grad()
def generate_symm_partial(
    gt: torch.Tensor,
    difficulty: str,
    view_idx: int,
    num_partial: int = 2048,
) -> torch.Tensor:
    """
    Reproduce the historical SymmCompletion transfer
    ShapeNet-55 partial generation:

        fixed-direction crop
        -> remove nearest points
        -> historical Symm PointNet2 FPS to 2048.
    """

    if difficulty not in SHAPENET55_DIFFICULTY_CROPS:
        raise ValueError(
            f"Unknown difficulty: {difficulty}"
        )

    if not 0 <= view_idx < 8:
        raise ValueError(
            f"view_idx must be in [0, 7], got {view_idx}"
        )

    if gt.ndim != 3 or gt.shape[-1] != 3:
        raise ValueError(
            f"Expected GT (B, N, 3), got {tuple(gt.shape)}"
        )

    if not gt.is_cuda:
        raise RuntimeError(
            "GT must be CUDA tensor."
        )

    num_crop = (
        SHAPENET55_DIFFICULTY_CROPS[
            difficulty
        ]
    )

    center = (
        FIXED_CROP_DIRECTIONS[
            view_idx
        ]
        .to(
            device=gt.device,
            dtype=gt.dtype,
        )
        .view(1, 1, 3)
    )

    distances = torch.norm(
        gt - center,
        p=2,
        dim=-1,
    )

    sorted_idx = torch.argsort(
        distances,
        dim=1,
        descending=False,
    )

    keep_idx = sorted_idx[
        :,
        num_crop:
    ]

    observed = torch.gather(
        gt,
        dim=1,
        index=keep_idx
        .unsqueeze(-1)
        .expand(-1, -1, 3),
    ).contiguous()

    return legacy_fps(
        observed,
        num_partial,
    )
