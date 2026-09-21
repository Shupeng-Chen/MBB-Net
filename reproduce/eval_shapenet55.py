#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.shapenet55_official_eval import (
    DIFFICULTY_CROPS,
    ShapeNet55OfficialEvalDataset,
    generate_official_partial,
)
from models.shapenet_model import MBB_Model_ShapeNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ScalarMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update_sum(self, value: float, count: int = 1):
        self.total += float(value)
        self.count += int(count)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.total / self.count


def load_checkpoint_strict(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    try:
        raw = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        raw = torch.load(
            str(checkpoint_path),
            map_location="cpu",
        )

    state = raw

    if isinstance(raw, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "base_model",
        ):
            value = raw.get(key)
            if isinstance(value, dict):
                state = value
                break

    cleaned = {}

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue

        while key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    model_state = model.state_dict()

    missing = sorted(set(model_state) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(model_state))

    shape_bad = sorted(
        key
        for key in set(model_state) & set(cleaned)
        if tuple(model_state[key].shape)
        != tuple(cleaned[key].shape)
    )

    print(
        f"[Checkpoint]\n"
        f"  tensors in model:      {len(model_state)}\n"
        f"  tensors in checkpoint: {len(cleaned)}"
    )

    if missing or unexpected or shape_bad:
        raise RuntimeError(
            "Checkpoint does not exactly match release model.\n"
            f"missing={missing[:10]}\n"
            f"unexpected={unexpected[:10]}\n"
            f"shape_bad={shape_bad[:10]}"
        )

    model.load_state_dict(
        cleaned,
        strict=True,
    )

    print("  strict load: OK")


def stratified_subset_indices(
    dataset,
    per_class: int,
):
    if per_class <= 0:
        return list(range(len(dataset)))

    selected = []
    counts: Dict[str, int] = defaultdict(int)

    for index, model_id in enumerate(dataset.file_list):
        cat_id = model_id.split("-")[0]

        if counts[cat_id] < per_class:
            selected.append(index)
            counts[cat_id] += 1

    print(
        f"[Audit subset] per_class={per_class}, "
        f"categories={len(counts)}, "
        f"samples={len(selected)}"
    )

    return selected


def _try_import_raw_chamfer():
    candidates = []

    try:
        from extensions.chamfer_dist import ChamferFunction
        candidates.append(
            (
                "extensions.chamfer_dist.ChamferFunction",
                ChamferFunction,
            )
        )
    except Exception:
        pass

    try:
        from extensions.chamfer_dist.chamfer import ChamferFunction
        candidates.append(
            (
                "extensions.chamfer_dist.chamfer.ChamferFunction",
                ChamferFunction,
            )
        )
    except Exception:
        pass

    return candidates


class NearestDistanceBackend:
    def __init__(
        self,
        chunk_size: int = 256,
    ):
        self.chunk_size = int(chunk_size)
        self.raw_name = None
        self.raw_fn = None

        for name, candidate in _try_import_raw_chamfer():
            self.raw_name = name
            self.raw_fn = candidate
            break

        if self.raw_fn is not None:
            print(
                "[Metric backend] raw CUDA Chamfer:",
                self.raw_name,
            )
        else:
            print(
                "[Metric backend] raw ChamferFunction "
                "not found; using torch.cdist."
            )

    def _raw_distances(
        self,
        pred,
        gt,
    ):
        if self.raw_fn is None:
            return None

        try:
            if hasattr(self.raw_fn, "apply"):
                result = self.raw_fn.apply(
                    pred.contiguous().float(),
                    gt.contiguous().float(),
                )
            else:
                result = self.raw_fn(
                    pred.contiguous().float(),
                    gt.contiguous().float(),
                )

            if isinstance(result, (tuple, list)) and len(result) >= 2:
                dist1, dist2 = result[0], result[1]

                if (
                    isinstance(dist1, torch.Tensor)
                    and isinstance(dist2, torch.Tensor)
                    and dist1.ndim == 2
                    and dist2.ndim == 2
                ):
                    return dist1, dist2

        except Exception as exc:
            print(
                "[Metric backend] raw Chamfer failed:",
                exc,
            )
            print("Falling back to torch.cdist.")
            self.raw_fn = None

        return None

    def _one_direction(
        self,
        source,
        target,
    ):
        chunks: List[torch.Tensor] = []

        for start in range(
            0,
            source.shape[1],
            self.chunk_size,
        ):
            end = min(
                start + self.chunk_size,
                source.shape[1],
            )

            distances = torch.cdist(
                source[:, start:end].float(),
                target.float(),
                p=2,
            )

            chunks.append(
                distances
                .min(dim=2)
                .values
                .square()
            )

        return torch.cat(
            chunks,
            dim=1,
        )

    @torch.no_grad()
    def distances(
        self,
        pred,
        gt,
    ):
        raw = self._raw_distances(
            pred,
            gt,
        )

        if raw is not None:
            return raw

        return (
            self._one_direction(pred, gt),
            self._one_direction(gt, pred),
        )


@torch.no_grad()
def per_sample_cd_f1(
    pred,
    gt,
    backend,
    f1_threshold,
):
    dist_pred_to_gt, dist_gt_to_pred = (
        backend.distances(
            pred,
            gt,
        )
    )

    cd = (
        dist_pred_to_gt.mean(dim=1)
        + dist_gt_to_pred.mean(dim=1)
    ) * 1000.0

    precision = (
        torch.sqrt(
            dist_pred_to_gt.clamp_min(0.0)
            + 1e-12
        )
        < f1_threshold
    ).float().mean(dim=1)

    recall = (
        torch.sqrt(
            dist_gt_to_pred.clamp_min(0.0)
            + 1e-12
        )
        < f1_threshold
    ).float().mean(dim=1)

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-8
        )
    )

    return cd, f1


def taxonomy_macro(
    meters_by_category,
):
    values = [
        meter.avg
        for meter in meters_by_category.values()
        if meter.count > 0
    ]

    if not values:
        return float("nan")

    return float(
        np.mean(values)
    )


def evaluate(args):
    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    cfg["DATA_ROOT"] = str(
        Path(args.data_root).resolve()
    )
    cfg["NUM_CLASSES"] = 55

    dataset = ShapeNet55OfficialEvalDataset(
        cfg,
        subset="test",
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

    model = MBB_Model_ShapeNet(
        cfg,
        use_mbb=True,
    ).cuda()

    load_checkpoint_strict(
        model,
        Path(args.checkpoint),
    )

    model.eval()

    backend = NearestDistanceBackend(
        args.metric_chunk
    )

    cd_meters = {
        d: ScalarMeter()
        for d in DIFFICULTY_CROPS
    }

    f1_meters = {
        d: ScalarMeter()
        for d in DIFFICULTY_CROPS
    }

    category_cd_meters = {
        d: defaultdict(ScalarMeter)
        for d in DIFFICULTY_CROPS
    }

    category_f1_meters = {
        d: defaultdict(ScalarMeter)
        for d in DIFFICULTY_CROPS
    }

    object_correct = 0
    object_total = 0

    view_correct = 0
    view_total = 0

    start_time = time.time()

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc="55-ours",
            leave=True,
        )

        for batch in progress:
            gt = batch["gt"].cuda(
                non_blocking=True
            )
            labels = batch["label"].cuda(
                non_blocking=True
            )
            cat_ids = list(
                batch["cat_id"]
            )

            batch_size = gt.shape[0]

            moderate_logits_sum = None

            for difficulty in (
                "simple",
                "moderate",
                "hard",
            ):
                for view_idx in range(8):
                    partial = generate_official_partial(
                        gt=gt,
                        difficulty=difficulty,
                        view_idx=view_idx,
                        num_partial=int(
                            cfg.get(
                                "NUM_PARTIAL_POINTS",
                                2048,
                            )
                        ),
                    )

                    out_list, logits = model(
                        partial
                    )

                    pred = out_list[-1]

                    cd_batch, f1_batch = (
                        per_sample_cd_f1(
                            pred=pred,
                            gt=gt,
                            backend=backend,
                            f1_threshold=(
                                args.f1_threshold
                            ),
                        )
                    )

                    cd_meters[difficulty].update_sum(
                        cd_batch.sum().item(),
                        batch_size,
                    )

                    f1_meters[difficulty].update_sum(
                        f1_batch.sum().item(),
                        batch_size,
                    )

                    for sample_idx, cat_id in enumerate(
                        cat_ids
                    ):
                        category_cd_meters[
                            difficulty
                        ][cat_id].update_sum(
                            cd_batch[
                                sample_idx
                            ].item(),
                            1,
                        )

                        category_f1_meters[
                            difficulty
                        ][cat_id].update_sum(
                            f1_batch[
                                sample_idx
                            ].item(),
                            1,
                        )

                    if difficulty == "moderate":
                        moderate_logits_sum = (
                            logits
                            if moderate_logits_sum is None
                            else moderate_logits_sum
                            + logits
                        )

                        view_pred = logits.argmax(
                            dim=1
                        )

                        view_correct += int(
                            (
                                view_pred
                                == labels
                            )
                            .sum()
                            .item()
                        )

                        view_total += batch_size

            object_logits = (
                moderate_logits_sum
                / 8.0
            )

            object_pred = (
                object_logits.argmax(
                    dim=1
                )
            )

            object_correct += int(
                (
                    object_pred
                    == labels
                )
                .sum()
                .item()
            )

            object_total += batch_size

            progress.set_postfix(
                {
                    "CD-H": (
                        f"{cd_meters['hard'].avg:.3f}"
                    ),
                    "Acc-obj": (
                        f"{100.0*object_correct/object_total:.2f}%"
                    ),
                }
            )

    elapsed = time.time() - start_time

    result = {
        "dataset": "55",
        "mode": "ours",
        "checkpoint": str(
            Path(args.checkpoint).resolve()
        ),
        "num_objects": len(eval_dataset),
        "views_per_difficulty": 8,
        "elapsed_seconds": elapsed,
    }

    result["cd_l2_x1000"] = {
        difficulty: taxonomy_macro(
            category_cd_meters[
                difficulty
            ]
        )
        for difficulty in (
            "simple",
            "moderate",
            "hard",
        )
    }

    result["f1_at_1pct"] = {
        difficulty: taxonomy_macro(
            category_f1_meters[
                difficulty
            ]
        )
        for difficulty in (
            "simple",
            "moderate",
            "hard",
        )
    }

    result["cd_l2_x1000"]["average"] = float(
        np.mean(
            [
                result[
                    "cd_l2_x1000"
                ]["simple"],
                result[
                    "cd_l2_x1000"
                ]["moderate"],
                result[
                    "cd_l2_x1000"
                ]["hard"],
            ]
        )
    )

    result["f1_at_1pct"]["average"] = float(
        np.mean(
            [
                result[
                    "f1_at_1pct"
                ]["simple"],
                result[
                    "f1_at_1pct"
                ]["moderate"],
                result[
                    "f1_at_1pct"
                ]["hard"],
            ]
        )
    )

    result["cd_l2_x1000_micro"] = {
        d: cd_meters[d].avg
        for d in (
            "simple",
            "moderate",
            "hard",
        )
    }

    result["f1_at_1pct_micro"] = {
        d: f1_meters[d].avg
        for d in (
            "simple",
            "moderate",
            "hard",
        )
    }

    result["cd_l2_x1000_micro"]["average"] = float(
        np.mean(
            list(
                result[
                    "cd_l2_x1000_micro"
                ].values()
            )
        )
    )

    result["f1_at_1pct_micro"]["average"] = float(
        np.mean(
            list(
                result[
                    "f1_at_1pct_micro"
                ].values()
            )
        )
    )

    result["accuracy_object_mean_logits"] = (
        object_correct / object_total
    )

    result["accuracy_view_average"] = (
        view_correct / view_total
    )

    print()
    print("--- Official Evaluation Result ---")

    print(
        "Objects:",
        result["num_objects"],
    )

    cd = result["cd_l2_x1000"]
    f1 = result["f1_at_1pct"]

    print(
        "CD-L2 x1000 | "
        f"S={cd['simple']:.4f} | "
        f"M={cd['moderate']:.4f} | "
        f"H={cd['hard']:.4f} | "
        f"Avg={cd['average']:.4f}"
    )

    print(
        "F1 @1%     | "
        f"S={f1['simple']:.4f} | "
        f"M={f1['moderate']:.4f} | "
        f"H={f1['hard']:.4f} | "
        f"Avg={f1['average']:.4f}"
    )

    print(
        "Accuracy    | "
        f"Object={100.0*result['accuracy_object_mean_logits']:.2f}% | "
        f"View={100.0*result['accuracy_view_average']:.2f}%"
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        f"audit{args.audit_per_class}perclass"
        if args.audit_per_class > 0
        else "full"
    )

    output_path = (
        output_dir
        / f"ShapeNet55_ours_{suffix}.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[Saved]",
        output_path,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_root",
        required=True,
    )

    p.add_argument(
        "--checkpoint",
        required=True,
    )

    p.add_argument(
        "--config",
        default=(
            "configs/ShapeNet55_config.yaml"
        ),
    )

    p.add_argument(
        "--audit_per_class",
        type=int,
        default=0,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--metric_chunk",
        type=int,
        default=256,
    )

    p.add_argument(
        "--f1_threshold",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--output_dir",
        default=(
            "expected_results/"
            "shapenet55_release"
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_seed(
        args.seed
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required."
        )

    evaluate(args)
