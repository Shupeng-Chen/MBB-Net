#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Historical ShapeNet-55 training dataset used by the canonical Table-2
ablation runs.

IMPORTANT
---------
This intentionally preserves the original global NumPy RNG behavior.

Do NOT replace np.random.* with a per-dataset RandomState/Generator.
Do NOT add deterministic per-sample seeding.

The historical training trajectory depended on DataLoader worker RNG
and global RNG state.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SHAPENET55_CLASSES = (
    "02691156", "02747177", "02773838", "02801938",
    "02808440", "02818832", "02828884", "02843684",
    "02871439", "02876657", "02880940", "02924116",
    "02933112", "02942699", "02946921", "02954340",
    "02958343", "02992529", "03001627", "03046257",
    "03085013", "03207941", "03211117", "03261776",
    "03325088", "03337140", "03467517", "03513137",
    "03593526", "03624134", "03636649", "03642806",
    "03691459", "03710193", "03759954", "03761084",
    "03790512", "03797390", "03928116", "03938244",
    "03948459", "03991062", "04004475", "04074963",
    "04090263", "04099429", "04225987", "04256520",
    "04330267", "04379243", "04401088", "04460130",
    "04468005", "04530566", "04554684",
)


class ShapeNet55HistoricalTrainDataset(Dataset):
    """
    Faithful portable copy of the historical ShapeNetDataset used for
    canonical ShapeNet-55 training/checkpoint selection.
    """

    def __init__(
        self,
        data_root,
        subset="train",
        num_classes=55,
        num_partial_points=2048,
        num_gt_points=8192,
    ):
        super().__init__()

        if subset not in {"train", "test"}:
            raise ValueError(
                f"subset must be 'train' or 'test', got {subset!r}"
            )

        self.subset = subset
        self.data_root = Path(data_root).expanduser().resolve()
        self.pc_path = self.data_root / "shapenet_pc"

        self.npoints = int(num_partial_points)
        self.gt_npoints = int(num_gt_points)

        classes = list(SHAPENET55_CLASSES)

        if num_classes == 34:
            classes = classes[:34]
        elif num_classes == 21:
            classes = classes[34:]
        elif num_classes != 55:
            raise ValueError(
                f"Unsupported NUM_CLASSES={num_classes}"
            )

        self.classes = classes
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self.classes)
        }

        split_file = self.data_root / f"{subset}.txt"

        if not split_file.is_file():
            raise FileNotFoundError(
                f"Missing ShapeNet split file: {split_file}"
            )

        if not self.pc_path.is_dir():
            raise FileNotFoundError(
                f"Missing ShapeNet point-cloud directory: {self.pc_path}"
            )

        with split_file.open("r", encoding="utf-8") as f:
            self.file_list = [
                line.strip().replace(".npy", "")
                for line in f
                if line.strip()
            ]

        self.file_list = [
            item
            for item in self.file_list
            if item.split("-")[0] in self.cat2label
        ]

        self.fixed_viewpoints = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        )

        self.fixed_viewpoints /= (
            np.linalg.norm(
                self.fixed_viewpoints,
                axis=1,
                keepdims=True,
            )
            + 1e-8
        )

    @staticmethod
    def pc_norm(pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid

        scale = np.max(
            np.sqrt(
                np.sum(pc ** 2, axis=1)
            )
        )

        return pc / (scale + 1e-9)

    @staticmethod
    def crop_pc(
        pc,
        viewpoint,
        n_remove,
    ):
        vp_pos = viewpoint * 2.0

        dist = np.sum(
            (pc - vp_pos) ** 2,
            axis=1,
        )

        idx = np.argsort(dist)

        keep_idx = idx[
            : pc.shape[0] - n_remove
        ]

        return pc[keep_idx]

    @staticmethod
    def random_sample(
        pc,
        n,
    ):
        # Historical behavior:
        # intentionally uses global np.random.
        idx = np.random.permutation(
            pc.shape[0]
        )

        if pc.shape[0] < n:
            res_idx = np.concatenate(
                [
                    idx,
                    np.random.randint(
                        pc.shape[0],
                        size=n - pc.shape[0],
                    ),
                ]
            )
        else:
            res_idx = idx[:n]

        return pc[res_idx]

    def __getitem__(
        self,
        idx,
    ):
        model_id = self.file_list[idx]
        cat_id = model_id.split("-")[0]

        path = (
            self.pc_path
            / f"{model_id}.npy"
        )

        gt_pc = np.load(
            path
        ).astype(
            np.float32
        )

        gt_pc = self.pc_norm(gt_pc)

        gt_pc = self.random_sample(
            gt_pc,
            self.gt_npoints,
        )

        label = self.cat2label.get(
            cat_id,
            0,
        )

        if self.subset == "train":

            viewpoint = np.random.randn(3)

            viewpoint /= (
                np.linalg.norm(viewpoint)
                + 1e-8
            )

            n_remove = np.random.randint(
                2048,
                6144,
            )

            partial_pc = self.crop_pc(
                gt_pc,
                viewpoint,
                n_remove,
            )

            partial_pc = self.random_sample(
                partial_pc,
                self.npoints,
            )

            theta = np.random.uniform(
                0,
                2 * np.pi,
            )

            rot_mat = np.array(
                [
                    [
                        np.cos(theta),
                        -np.sin(theta),
                        0,
                    ],
                    [
                        np.sin(theta),
                        np.cos(theta),
                        0,
                    ],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            gt_pc = gt_pc @ rot_mat
            partial_pc = partial_pc @ rot_mat

            return (
                torch.from_numpy(
                    partial_pc
                ).float(),
                torch.from_numpy(
                    gt_pc
                ).float(),
                torch.tensor(
                    label,
                    dtype=torch.long,
                ),
            )

        viewpoint = self.fixed_viewpoints[
            idx % 8
        ]

        result = {}

        for difficulty, n_remove in zip(
            (
                "simple",
                "moderate",
                "hard",
            ),
            (
                2048,
                4096,
                6144,
            ),
        ):
            partial = self.crop_pc(
                gt_pc,
                viewpoint,
                n_remove,
            )

            result[difficulty] = (
                torch.from_numpy(
                    self.random_sample(
                        partial,
                        self.npoints,
                    )
                ).float()
            )

        return (
            result,
            torch.from_numpy(
                gt_pc
            ).float(),
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )

    def __len__(self):
        return len(self.file_list)
