#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-sample completion and classification metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from extensions.chamfer_dist import ChamferFunction


FIXED_CROP_DIRECTIONS = (
    (1.0, 1.0, 1.0),
    (1.0, 1.0, -1.0),
    (1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, -1.0, -1.0),
)

SHAPENET_DIFFICULTY_CROPS = {
    "simple": 2048,
    "moderate": 4096,
    "hard": 6144,
}


@torch.no_grad()
def per_sample_completion_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    fscore_threshold: float = 0.01,
) -> Dict[str, torch.Tensor]:
    prediction = prediction.contiguous().float()
    ground_truth = ground_truth.contiguous().float()

    dist_prediction, dist_ground_truth = (
        ChamferFunction.apply(
            prediction,
            ground_truth,
        )
    )

    cd_l1 = (
        torch.sqrt(
            dist_prediction.clamp_min(
                1e-12
            )
        ).mean(dim=1)
        + torch.sqrt(
            dist_ground_truth.clamp_min(
                1e-12
            )
        ).mean(dim=1)
    ) * 0.5 * 1000.0

    cd_l2 = (
        dist_prediction.mean(dim=1)
        + dist_ground_truth.mean(dim=1)
    ) * 1000.0

    precision = (
        torch.sqrt(
            dist_prediction.clamp_min(
                1e-12
            )
        )
        < fscore_threshold
    ).float().mean(dim=1)

    recall = (
        torch.sqrt(
            dist_ground_truth.clamp_min(
                1e-12
            )
        )
        < fscore_threshold
    ).float().mean(dim=1)

    fscore = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-8
        )
    )

    return {
        "F-Score": fscore,
        "CDL1": cd_l1,
        "CDL2": cd_l2,
    }


class CompletionMetricAccumulator:
    """Accumulate object metrics and taxonomy-macro averages."""

    def __init__(self) -> None:
        self.object_sums = {
            "F-Score": 0.0,
            "CDL1": 0.0,
            "CDL2": 0.0,
        }
        self.object_count = 0
        self.category_sums = defaultdict(
            lambda: {
                "F-Score": 0.0,
                "CDL1": 0.0,
                "CDL2": 0.0,
            }
        )
        self.category_counts = defaultdict(
            int
        )

    def update(
        self,
        taxonomy_ids: Sequence,
        metrics: Dict[str, torch.Tensor],
    ) -> None:
        batch_size = len(
            taxonomy_ids
        )
        for name in self.object_sums:
            values = metrics[
                name
            ].detach().cpu()
            self.object_sums[name] += float(
                values.sum().item()
            )

        self.object_count += batch_size

        for index, taxonomy in enumerate(
            taxonomy_ids
        ):
            taxonomy = str(taxonomy)
            self.category_counts[
                taxonomy
            ] += 1
            for name in self.object_sums:
                self.category_sums[
                    taxonomy
                ][name] += float(
                    metrics[name][
                        index
                    ]
                    .detach()
                    .cpu()
                    .item()
                )

    def object_average(
        self,
    ) -> Dict[str, float]:
        if self.object_count == 0:
            return {
                name: float("nan")
                for name in self.object_sums
            }
        return {
            name: value
            / self.object_count
            for name, value in (
                self.object_sums.items()
            )
        }

    def taxonomy_macro(
        self,
    ) -> Dict[str, float]:
        if not self.category_counts:
            return {
                name: float("nan")
                for name in self.object_sums
            }

        category_means = []
        for taxonomy, count in (
            self.category_counts.items()
        ):
            category_means.append(
                {
                    name: (
                        self.category_sums[
                            taxonomy
                        ][name]
                        / count
                    )
                    for name in self.object_sums
                }
            )

        return {
            name: sum(
                item[name]
                for item in category_means
            )
            / len(category_means)
            for name in self.object_sums
        }

    def per_category(
        self,
    ) -> Dict[str, Dict[str, float]]:
        return {
            taxonomy: {
                name: (
                    self.category_sums[
                        taxonomy
                    ][name]
                    / count
                )
                for name in self.object_sums
            }
            for taxonomy, count in sorted(
                self.category_counts.items()
            )
        }


class AccuracyAccumulator:
    def __init__(self) -> None:
        self.correct = 0
        self.count = 0
        self.category_correct = defaultdict(
            int
        )
        self.category_count = defaultdict(
            int
        )

    def update_predictions(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        taxonomy_ids: Sequence,
    ) -> None:
        predictions = (
            predictions.detach().cpu()
        )
        labels = labels.detach().cpu()

        matches = (
            predictions == labels
        )
        self.correct += int(
            matches.sum().item()
        )
        self.count += int(
            labels.numel()
        )

        for index, taxonomy in enumerate(
            taxonomy_ids
        ):
            taxonomy = str(taxonomy)
            self.category_count[
                taxonomy
            ] += 1
            self.category_correct[
                taxonomy
            ] += int(
                matches[index].item()
            )

    def update_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        taxonomy_ids: Sequence,
    ) -> None:
        self.update_predictions(
            logits.argmax(dim=1),
            labels,
            taxonomy_ids,
        )

    @property
    def micro(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.correct / self.count

    @property
    def taxonomy_macro(self) -> float:
        if not self.category_count:
            return float("nan")
        return sum(
            self.category_correct[
                taxonomy
            ]
            / count
            for taxonomy, count in (
                self.category_count.items()
            )
        ) / len(
            self.category_count
        )

    def state_dict(
        self,
    ) -> Dict[str, float]:
        return {
            "micro": self.micro,
            "taxonomy_macro": (
                self.taxonomy_macro
            ),
            "correct": self.correct,
            "count": self.count,
        }
