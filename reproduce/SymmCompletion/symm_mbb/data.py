#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset and taxonomy helpers for SymmCompletion MBB transfer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import yaml

from datasets.PCNDataset import PCN
from datasets.ShapeNet55Dataset import ShapeNet


def _resolve_path(
    repository_root: Path,
    value: str,
) -> str:
    expanded = os.path.expanduser(
        str(value)
    )
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str(
        (repository_root / path).resolve()
    )


@dataclass
class TaxonomyLabelMap:
    taxonomy_to_label: Dict[str, int]
    label_to_taxonomy: List[str]

    @classmethod
    def from_pcn_json(
        cls,
        json_path: str,
    ) -> "TaxonomyLabelMap":
        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as handle:
            categories = json.load(handle)

        taxonomy_ids = [
            str(item["taxonomy_id"])
            for item in categories
        ]
        if len(taxonomy_ids) != 8:
            raise RuntimeError(
                "Expected 8 PCN taxonomies, "
                f"found {len(taxonomy_ids)}."
            )
        return cls(
            taxonomy_to_label={
                taxonomy: index
                for index, taxonomy in enumerate(
                    taxonomy_ids
                )
            },
            label_to_taxonomy=taxonomy_ids,
        )

    @classmethod
    def from_shapenet_lists(
        cls,
        train_list_path: str,
        test_list_path: str,
    ) -> "TaxonomyLabelMap":
        taxonomy_ids = set()

        for list_path in (
            train_list_path,
            test_list_path,
        ):
            with open(
                list_path,
                "r",
                encoding="utf-8",
            ) as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    taxonomy_ids.add(
                        stripped.split(
                            "-",
                            1,
                        )[0]
                    )

        ordered = sorted(
            taxonomy_ids
        )
        if len(ordered) != 55:
            raise RuntimeError(
                "Expected 55 ShapeNet taxonomies, "
                f"found {len(ordered)}."
            )

        return cls(
            taxonomy_to_label={
                taxonomy: index
                for index, taxonomy in enumerate(
                    ordered
                )
            },
            label_to_taxonomy=ordered,
        )

    @classmethod
    def from_state_dict(
        cls,
        state: Dict,
    ) -> "TaxonomyLabelMap":
        label_to_taxonomy = [
            str(item)
            for item in state[
                "label_to_taxonomy"
            ]
        ]
        taxonomy_to_label = {
            taxonomy: index
            for index, taxonomy in enumerate(
                label_to_taxonomy
            )
        }
        return cls(
            taxonomy_to_label=taxonomy_to_label,
            label_to_taxonomy=label_to_taxonomy,
        )

    def state_dict(
        self,
    ) -> Dict:
        return {
            "label_to_taxonomy": list(
                self.label_to_taxonomy
            )
        }

    def encode_batch(
        self,
        taxonomy_ids: Sequence,
        device: torch.device,
    ) -> torch.Tensor:
        labels = []
        for taxonomy in taxonomy_ids:
            if isinstance(
                taxonomy,
                torch.Tensor,
            ):
                taxonomy = taxonomy.item()
            taxonomy = str(taxonomy)
            if taxonomy not in (
                self.taxonomy_to_label
            ):
                raise KeyError(
                    f"Unknown taxonomy ID: {taxonomy}"
                )
            labels.append(
                self.taxonomy_to_label[
                    taxonomy
                ]
            )

        return torch.tensor(
            labels,
            dtype=torch.long,
            device=device,
        )


@dataclass
class DatasetBundle:
    train_dataset: object
    test_dataset: object
    label_map: TaxonomyLabelMap
    npoints: int
    num_classes: int
    dataset_config: Dict


def build_dataset_bundle(
    dataset: str,
    repository_root: str,
    dataset_config_path: Optional[str] = None,
    pcn_category_file: Optional[str] = None,
    pcn_partial_pattern: Optional[str] = None,
    pcn_complete_pattern: Optional[str] = None,
    shapenet_index_root: Optional[str] = None,
    shapenet_pc_root: Optional[str] = None,
) -> DatasetBundle:
    dataset = dataset.lower()
    repo_root = Path(
        repository_root
    ).resolve()

    if dataset == "pcn":
        config_path = (
            Path(dataset_config_path)
            if dataset_config_path
            else repo_root
            / "cfgs"
            / "dataset_configs"
            / "PCN.yaml"
        )
    elif dataset == "shapenet55":
        config_path = (
            Path(dataset_config_path)
            if dataset_config_path
            else repo_root
            / "cfgs"
            / "dataset_configs"
            / "ShapeNet-55.yaml"
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}."
        )

    if not config_path.is_absolute():
        config_path = (
            repo_root / config_path
        ).resolve()

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as handle:
        raw_config = yaml.safe_load(
            handle
        )

    if dataset == "pcn":
        category_file = (
            pcn_category_file
            or raw_config[
                "CATEGORY_FILE_PATH"
            ]
        )
        partial_pattern = (
            pcn_partial_pattern
            or raw_config[
                "PARTIAL_POINTS_PATH"
            ]
        )
        complete_pattern = (
            pcn_complete_pattern
            or raw_config[
                "COMPLETE_POINTS_PATH"
            ]
        )

        category_file = _resolve_path(
            repo_root,
            category_file,
        )

        base_values = {
            "PARTIAL_POINTS_PATH": str(
                partial_pattern
            ),
            "COMPLETE_POINTS_PATH": str(
                complete_pattern
            ),
            "CATEGORY_FILE_PATH": (
                category_file
            ),
            "N_POINTS": int(
                raw_config.get(
                    "N_POINTS",
                    16384,
                )
            ),
            "CARS": bool(
                raw_config.get(
                    "CARS",
                    False,
                )
            ),
        }

        train_config = SimpleNamespace(
            **base_values,
            subset="train",
        )
        test_config = SimpleNamespace(
            **base_values,
            subset="test",
        )

        train_dataset = PCN(
            train_config
        )
        test_dataset = PCN(
            test_config
        )
        label_map = (
            TaxonomyLabelMap.from_pcn_json(
                category_file
            )
        )
        num_classes = 8

    else:
        index_root = (
            shapenet_index_root
            or raw_config[
                "DATA_PATH"
            ]
        )
        pc_root = (
            shapenet_pc_root
            or raw_config[
                "PC_PATH"
            ]
        )

        index_root = _resolve_path(
            repo_root,
            index_root,
        )
        pc_root = _resolve_path(
            repo_root,
            pc_root,
        )

        base_values = {
            "DATA_PATH": index_root,
            "PC_PATH": pc_root,
            "N_POINTS": int(
                raw_config.get(
                    "N_POINTS",
                    8192,
                )
            ),
        }

        train_config = SimpleNamespace(
            **base_values,
            subset="train",
        )
        test_config = SimpleNamespace(
            **base_values,
            subset="test",
        )

        train_dataset = ShapeNet(
            train_config
        )
        test_dataset = ShapeNet(
            test_config
        )

        label_map = (
            TaxonomyLabelMap.from_shapenet_lists(
                os.path.join(
                    index_root,
                    "train.txt",
                ),
                os.path.join(
                    index_root,
                    "test.txt",
                ),
            )
        )
        num_classes = 55

    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        label_map=label_map,
        npoints=int(
            raw_config["N_POINTS"]
        ),
        num_classes=num_classes,
        dataset_config=raw_config,
    )


def stratified_indices(
    dataset,
    per_class: int,
) -> List[int]:
    """Return a deterministic first-N-per-taxonomy audit subset."""
    if per_class <= 0:
        return list(
            range(len(dataset))
        )

    selected: List[int] = []
    counts: Dict[str, int] = {}

    if hasattr(
        dataset,
        "file_list",
    ):
        file_list = dataset.file_list
    else:
        raise RuntimeError(
            "Dataset does not expose file_list; "
            "cannot create a stratified audit subset."
        )

    for index, sample in enumerate(
        file_list
    ):
        taxonomy = str(
            sample["taxonomy_id"]
        )
        current = counts.get(
            taxonomy,
            0,
        )
        if current < per_class:
            selected.append(index)
            counts[taxonomy] = (
                current + 1
            )

    return selected
