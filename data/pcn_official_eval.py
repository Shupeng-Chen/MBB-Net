from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


PCN_TAXONOMY: Tuple[Tuple[str, str], ...] = (
    ("02691156", "plane"),
    ("02933112", "cabinet"),
    ("02958343", "car"),
    ("03001627", "chair"),
    ("03636649", "lamp"),
    ("04256520", "couch"),
    ("04379243", "table"),
    ("04530566", "watercraft"),
)

PCN_TAXONOMY_TO_LABEL = {
    taxonomy_id: index
    for index, (taxonomy_id, _) in enumerate(PCN_TAXONOMY)
}


def resolve_pcn_data_root(
    explicit_root: Optional[str] = None,
) -> str:
    candidates: List[str] = []

    if explicit_root:
        candidates.append(explicit_root)

    env_root = os.environ.get("PCN_DATA_ROOT")
    if env_root:
        candidates.append(env_root)

    checked = []

    for candidate in candidates:
        candidate = os.path.abspath(
            os.path.expanduser(candidate)
        )
        checked.append(candidate)

        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "PCN data root was not found. "
        "Pass --data_root or set PCN_DATA_ROOT.\n"
        + "\n".join(f"  - {p}" for p in checked)
    )


def read_pcd(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for official PCN evaluation."
        ) from exc

    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)

    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] == 0
    ):
        raise RuntimeError(
            f"Invalid PCD point cloud: {path}, "
            f"shape={points.shape}"
        )

    return np.ascontiguousarray(
        points,
        dtype=np.float32,
    )


def farthest_point_sample_numpy(
    points: np.ndarray,
    npoint: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            f"Expected (N, >=3), got {points.shape}"
        )

    if points.shape[0] == 0:
        raise ValueError(
            "Cannot FPS an empty point cloud."
        )

    if npoint <= 0:
        raise ValueError(
            f"npoint must be positive, got {npoint}"
        )

    n_points = points.shape[0]
    xyz = points[:, :3]

    centroids = np.zeros(
        (npoint,),
        dtype=np.int64,
    )

    distance = np.full(
        (n_points,),
        1e10,
        dtype=np.float64,
    )

    farthest = int(
        rng.randint(0, n_points)
    )

    for i in range(npoint):
        centroids[i] = farthest

        centroid = xyz[farthest]

        dist = np.sum(
            (xyz - centroid) ** 2,
            axis=-1,
        )

        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = int(
            np.argmax(distance)
        )

    return points[centroids]


def snowflake_upsample_points(
    points: np.ndarray,
    n_points: int = 2048,
    seed: int = 42,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected raw partial (N,3), got {points.shape}"
        )

    py_rng = random.Random(
        int(seed)
    )

    np_rng = np.random.RandomState(
        int(seed)
    )

    n_valid = py_rng.randint(
        512,
        1024,
    )

    sampled = farthest_point_sample_numpy(
        points,
        n_valid,
        np_rng,
    )

    current = sampled.shape[0]
    need = n_points - current

    if need < 0:
        choice = np_rng.permutation(
            current
        )[:n_points]

        sampled = sampled[choice]

    else:
        while current <= need:
            sampled = np.tile(
                sampled,
                (2, 1),
            )

            need -= current
            current *= 2

        if need > 0:
            choice = np_rng.permutation(
                need
            )

            sampled = np.concatenate(
                (
                    sampled,
                    sampled[choice],
                ),
                axis=0,
            )

    if sampled.shape != (
        n_points,
        3,
    ):
        raise RuntimeError(
            f"UpSamplePoints produced "
            f"{sampled.shape}, expected "
            f"({n_points}, 3)"
        )

    return np.ascontiguousarray(
        sampled,
        dtype=np.float32,
    )


class PCNSnowflakeOfficialEvalDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        num_partial: int = 2048,
        num_complete: int = 16384,
        seed: int = 42,
        cache: bool = True,
    ):
        super().__init__()

        self.data_root = os.path.abspath(
            data_root
        )

        self.num_partial = int(
            num_partial
        )

        self.num_complete = int(
            num_complete
        )

        self.seed = int(seed)
        self.use_cache = bool(cache)

        self.cache: Dict[int, tuple] = {}

        self.samples = (
            self._collect_samples()
        )

    def _find_partial(
        self,
        taxonomy_id: str,
        model_id: str,
    ) -> str:
        candidates = [
            os.path.join(
                self.data_root,
                "test",
                "partial",
                taxonomy_id,
                model_id,
                "00.pcd",
            ),
            os.path.join(
                self.data_root,
                "test",
                "partial",
                taxonomy_id,
                f"{model_id}.pcd",
            ),
        ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(
            f"No test partial found for "
            f"{taxonomy_id}/{model_id}"
        )

    def _collect_samples(
        self,
    ) -> List[dict]:
        samples: List[dict] = []

        for taxonomy_id, class_name in PCN_TAXONOMY:

            complete_dir = os.path.join(
                self.data_root,
                "test",
                "complete",
                taxonomy_id,
            )

            if not os.path.isdir(
                complete_dir
            ):
                raise FileNotFoundError(
                    complete_dir
                )

            gt_files = sorted(
                os.path.join(
                    complete_dir,
                    filename,
                )
                for filename
                in os.listdir(
                    complete_dir
                )
                if filename.lower().endswith(
                    ".pcd"
                )
            )

            if not gt_files:
                raise RuntimeError(
                    f"No GT PCD files in "
                    f"{complete_dir}"
                )

            for gt_path in gt_files:

                model_id = Path(
                    gt_path
                ).stem

                samples.append(
                    {
                        "taxonomy_id":
                            taxonomy_id,
                        "class_name":
                            class_name,
                        "label":
                            PCN_TAXONOMY_TO_LABEL[
                                taxonomy_id
                            ],
                        "model_id":
                            model_id,
                        "partial_path":
                            self._find_partial(
                                taxonomy_id,
                                model_id,
                            ),
                        "gt_path":
                            gt_path,
                    }
                )

        print(
            "[PCN official dataset] "
            f"objects={len(samples)}, "
            f"classes={len(PCN_TAXONOMY)}, "
            f"partial={self.num_partial}, "
            f"complete={self.num_complete}, "
            f"seed={self.seed}"
        )

        return samples

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ):
        if (
            self.use_cache
            and index in self.cache
        ):
            return self.cache[index]

        sample = self.samples[index]

        partial_raw = read_pcd(
            sample["partial_path"]
        )

        complete = read_pcd(
            sample["gt_path"]
        )

        if complete.shape != (
            self.num_complete,
            3,
        ):
            raise RuntimeError(
                f"GT shape mismatch for "
                f"{sample['taxonomy_id']}/"
                f"{sample['model_id']}: "
                f"{complete.shape}"
            )

        partial = (
            snowflake_upsample_points(
                partial_raw,
                n_points=self.num_partial,
                seed=self.seed + index,
            )
        )

        item = (
            torch.from_numpy(
                partial
            ).float(),

            torch.from_numpy(
                complete
            ).float(),

            int(
                sample["label"]
            ),

            str(
                sample["taxonomy_id"]
            ),

            str(
                sample["model_id"]
            ),
        )

        if self.use_cache:
            self.cache[index] = item

        return item
