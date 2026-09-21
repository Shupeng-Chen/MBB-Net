#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate existing MBB-Net ShapeNet checkpoints with the official PoinTr-style
ShapeNet test protocol.

Important:
- This script NEVER trains and NEVER overwrites checkpoints.
- It uses 8 fixed crop views for every object and every difficulty.
- It removes points nearest to the fixed crop center, then FPS samples 2048.
- It supports a stratified audit subset before a full evaluation.
- Full-set completion metrics use PoinTr-style taxonomy macro averaging.
- It uses exact model architectures for baseline/full/ours/g2s/s2g.
- no_bridge and cls_only reproduce the forward behavior used by the supplied
  dynamic-ablation code.
- pcgrad_s2g reproduces the exact monkey-patched S2G forward used in training,
  including bypassing sem_norm on the semantic token.

Recommended first audit:
    python -u scripts/ShapeNet/ShapeNet55/eval_existing_ckpts_official.py \
        --dataset 55 \
        --mode baseline full ours \
        --audit_per_class 10 \
        --batch_size 8

Then run the full set only if the core ranking is stable.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the local evaluation dataset before adding the backbone path, so the
# project's datasets package cannot be shadowed by a backbone package.
from data.shapenet55_official_eval import (  # noqa: E402
    DIFFICULTY_CROPS,
    ShapeNet55OfficialEvalDataset,
    generate_official_partial,
)

SNOW_ROOT = os.path.join(PROJECT_ROOT, "backbones", "SnowflakeNet-main")
if SNOW_ROOT not in sys.path:
    sys.path.append(SNOW_ROOT)

from models.shapenet_table2_model import ShapeNet55Table2Model  # noqa: E402


VALID_MODES = (
    "baseline",
    "cls_only",
    "no_bridge",
    "full",
    "g2s",
    "s2g",
    "ours",
    "pcgrad_ours",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Evaluation contains no dropout and BatchNorm is in eval mode. Keeping these
    # flags improves reproducibility, although some CUDA extensions may still have
    # implementation-specific behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ScalarMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update_sum(self, value_sum: float, count: int) -> None:
        self.total += float(value_sum)
        self.count += int(count)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.total / self.count


def parse_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Read a checkpoint without heuristic suffix matching."""
    raw = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(raw, dict):
        raise TypeError(
            f"Checkpoint must be a dict/state_dict, got {type(raw).__name__}."
        )

    state_dict = raw
    for candidate in ("model_state_dict", "state_dict", "model", "base_model"):
        value = raw.get(candidate)
        if isinstance(value, dict) and value:
            state_dict = value
            break

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        new_key = key
        # Controlled DataParallel prefix removal only. No fuzzy matching.
        while new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key in cleaned:
            raise RuntimeError(f"Duplicate key after prefix cleaning: {new_key}")
        cleaned[new_key] = value

    if not cleaned:
        raise RuntimeError(f"No tensors found in checkpoint: {checkpoint_path}")

    return cleaned


def load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: str) -> None:
    state_dict = parse_state_dict(checkpoint_path)
    model_state = model.state_dict()

    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    shape_mismatch = sorted(
        key
        for key in set(model_state).intersection(state_dict)
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape)
    )

    matched_numel = sum(
        model_state[key].numel()
        for key in set(model_state).intersection(state_dict)
        if tuple(model_state[key].shape) == tuple(state_dict[key].shape)
    )
    total_numel = sum(tensor.numel() for tensor in model_state.values())
    coverage = 100.0 * matched_numel / max(total_numel, 1)

    print(
        f"[Checkpoint] {checkpoint_path}\n"
        f"  tensors in model:      {len(model_state)}\n"
        f"  tensors in checkpoint: {len(state_dict)}\n"
        f"  parameter coverage:    {coverage:.4f}%"
    )

    if missing:
        print("  Missing keys:")
        for key in missing[:30]:
            print(f"    - {key}")
        if len(missing) > 30:
            print(f"    ... and {len(missing) - 30} more")

    if unexpected:
        print("  Unexpected keys:")
        for key in unexpected[:30]:
            print(f"    - {key}")
        if len(unexpected) > 30:
            print(f"    ... and {len(unexpected) - 30} more")

    if shape_mismatch:
        print("  Shape mismatches:")
        for key in shape_mismatch[:30]:
            print(
                f"    - {key}: model={tuple(model_state[key].shape)}, "
                f"ckpt={tuple(state_dict[key].shape)}"
            )
        if len(shape_mismatch) > 30:
            print(f"    ... and {len(shape_mismatch) - 30} more")

    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Checkpoint does not exactly match the selected model. "
            "Evaluation stopped instead of silently loading partial weights."
        )

    model.load_state_dict(state_dict, strict=True)
    print("  strict load: OK")


def build_model(cfg: dict, mode: str):
    normalized = mode.lower()

    if normalized not in VALID_MODES:
        raise ValueError(
            f"Unsupported mode: {mode}. "
            f"Choose from {VALID_MODES}."
        )

    # PCGrad changes optimization only.
    # Its released checkpoint uses the canonical OURS
    # inference architecture.
    inference_mode = (
        "ours"
        if normalized == "pcgrad_ours"
        else normalized
    )

    return ShapeNet55Table2Model(
        cfg,
        mode=inference_mode,
    )


def default_checkpoint(dataset: str, mode: str) -> str:
    if dataset != "55":
        raise ValueError("This evaluator supports ShapeNet-55 only.")

    relative_paths = {
        "baseline": "checkpoints/ShapeNet/ShapeNet55/baseline/best_model.pth",
        "cls_only": "checkpoints/ShapeNet/ShapeNet55/cls_only/best_model.pth",
        "full": "checkpoints/ShapeNet/ShapeNet55/full/best_model.pth",
        "g2s": "checkpoints/ShapeNet/ShapeNet55/g2s/best_model.pth",
        "no_bridge": "checkpoints/ShapeNet/ShapeNet55/no_bridge/best_model.pth",
        "ours": "checkpoints/ShapeNet/ShapeNet55/ours/best_model.pth",
        "s2g": "checkpoints/ShapeNet/ShapeNet55/s2g/best_model.pth",
        "pcgrad_ours": (
            "checkpoints/ShapeNet/ShapeNet55/"
            "pcgrad_ours_aligned/best_model.pth"
        ),
        "pcgrad_s2g": (
            "checkpoints/ShapeNet/ShapeNet55/"
            "pcgrad_s2g_aligned/best_model.pth"
        ),
    }
    return os.path.join(PROJECT_ROOT, relative_paths[mode])


def stratified_subset_indices(dataset, per_class: int) -> List[int]:
    """Select the first N deterministic samples from each taxonomy category."""
    if per_class <= 0:
        return list(range(len(dataset)))

    selected: List[int] = []
    counts: Dict[str, int] = defaultdict(int)

    for index, model_id in enumerate(dataset.file_list):
        cat_id = model_id.split("-")[0]
        if counts[cat_id] < per_class:
            selected.append(index)
            counts[cat_id] += 1

    category_count = len(counts)
    print(
        f"[Audit subset] per_class={per_class}, "
        f"categories={category_count}, samples={len(selected)}"
    )
    return selected


def _try_import_raw_chamfer():
    candidates = []

    try:
        from extensions.chamfer_dist import ChamferFunction  # type: ignore
        candidates.append(("extensions.chamfer_dist.ChamferFunction", ChamferFunction))
    except Exception:
        pass

    try:
        from extensions.chamfer_dist.chamfer import ChamferFunction  # type: ignore
        candidates.append(
            ("extensions.chamfer_dist.chamfer.ChamferFunction", ChamferFunction)
        )
    except Exception:
        pass

    return candidates


class NearestDistanceBackend:
    """
    Return per-point squared nearest-neighbor distances.

    It first tries the raw Chamfer CUDA function. If the local extension exposes
    only scalar Chamfer modules, it falls back to chunked torch.cdist.
    """

    def __init__(self, chunk_size: int = 256) -> None:
        self.chunk_size = int(chunk_size)
        self.raw_name: Optional[str] = None
        self.raw_fn = None

        for name, candidate in _try_import_raw_chamfer():
            self.raw_name = name
            self.raw_fn = candidate
            break

        if self.raw_fn is not None:
            print(f"[Metric backend] raw CUDA Chamfer: {self.raw_name}")
        else:
            print(
                "[Metric backend] raw ChamferFunction not found; "
                f"using chunked torch.cdist (chunk={self.chunk_size})."
            )

    def _raw_distances(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
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
                f"[Metric backend] raw Chamfer failed once: {exc}\n"
                "Falling back to chunked torch.cdist."
            )
            self.raw_fn = None

        return None

    def _one_direction(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for start in range(0, source.shape[1], self.chunk_size):
            end = min(start + self.chunk_size, source.shape[1])
            # torch.cdist returns Euclidean distance; square it for CD-L2.
            distances = torch.cdist(
                source[:, start:end].float(),
                target.float(),
                p=2,
            )
            chunks.append(distances.min(dim=2).values.square())
        return torch.cat(chunks, dim=1)

    @torch.no_grad()
    def distances(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self._raw_distances(pred, gt)
        if raw is not None:
            return raw

        return (
            self._one_direction(pred, gt),
            self._one_direction(gt, pred),
        )


@torch.no_grad()
def per_sample_cd_f1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    backend: NearestDistanceBackend,
    f1_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dist_pred_to_gt, dist_gt_to_pred = backend.distances(pred, gt)

    cd = (
        dist_pred_to_gt.mean(dim=1)
        + dist_gt_to_pred.mean(dim=1)
    ) * 1000.0

    precision = (
        torch.sqrt(dist_pred_to_gt.clamp_min(0.0) + 1e-12)
        < f1_threshold
    ).float().mean(dim=1)

    recall = (
        torch.sqrt(dist_gt_to_pred.clamp_min(0.0) + 1e-12)
        < f1_threshold
    ).float().mean(dim=1)

    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return cd, f1


def create_dataset_and_cfg(args) -> Tuple[object, dict]:
    if args.dataset != "55":
        raise ValueError("This evaluator supports ShapeNet-55 only.")

    cfg_path = args.config or os.path.join(
        PROJECT_ROOT,
        "configs",
        "ShapeNet55_config.yaml",
    )
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg["DATA_ROOT"] = str(
        Path(args.data_root).resolve()
    )
    cfg["NUM_CLASSES"] = 55
    dataset = ShapeNet55OfficialEvalDataset(cfg, subset="test")
    print(f"[Config] {cfg_path}")
    return dataset, cfg


def evaluate_one_mode(
    args,
    cfg: dict,
    loader: DataLoader,
    mode: str,
    checkpoint_path: str,
) -> dict:
    print("\n" + "=" * 88)
    print(
        f"Official evaluation | Dataset={args.dataset} | "
        f"Mode={mode} | Checkpoint={checkpoint_path}"
    )
    print("=" * 88)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    model = build_model(cfg, mode).cuda()
    load_checkpoint_strict(model, checkpoint_path)
    model.eval()

    is_cls_only = mode == "cls_only"
    has_completion = not is_cls_only
    has_classification = mode != "baseline" and args.dataset != "21"

    backend = NearestDistanceBackend(args.metric_chunk)

    cd_meters = {
        difficulty: ScalarMeter()
        for difficulty in DIFFICULTY_CROPS
    }
    # Micro meters are sample-weighted and are kept as diagnostics.
    f1_meters = {
        difficulty: ScalarMeter()
        for difficulty in DIFFICULTY_CROPS
    }

    # PoinTr's reported ShapeNet "Overall" first averages within each taxonomy
    # and then averages the taxonomy means. These category meters therefore
    # provide the official macro-average used as the main reported result.
    category_cd_meters = {
        difficulty: defaultdict(ScalarMeter)
        for difficulty in DIFFICULTY_CROPS
    }
    category_f1_meters = {
        difficulty: defaultdict(ScalarMeter)
        for difficulty in DIFFICULTY_CROPS
    }

    object_correct = 0
    object_total = 0
    view_correct = 0
    view_total = 0

    start_time = time.time()

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=f"{args.dataset}-{mode}",
            leave=True,
        )

        for batch in progress:
            gt = batch["gt"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)
            cat_ids = list(batch["cat_id"])
            batch_size = gt.shape[0]

            if len(cat_ids) != batch_size:
                raise RuntimeError(
                    f"Category metadata count {len(cat_ids)} does not match "
                    f"batch size {batch_size}."
                )

            moderate_logits_sum: Optional[torch.Tensor] = None

            # CLS_ONLY has no meaningful completion decoder. It needs only the
            # 8 moderate views for object-level and view-level classification.
            difficulties: Sequence[str]
            if is_cls_only:
                difficulties = ("moderate",)
            else:
                difficulties = ("simple", "moderate", "hard")

            for difficulty in difficulties:
                for view_idx in range(8):
                    partial = generate_official_partial(
                        gt=gt,
                        difficulty=difficulty,
                        view_idx=view_idx,
                        num_partial=int(
                            cfg.get("NUM_PARTIAL_POINTS", 2048)
                        ),
                    )

                    out_list, logits = model(partial)

                    if has_completion:
                        if not isinstance(out_list, (list, tuple)) or not out_list:
                            raise RuntimeError(
                                f"Mode {mode} returned no completion outputs."
                            )

                        pred = out_list[-1]
                        if pred.ndim != 3 or pred.shape[-1] != 3:
                            raise RuntimeError(
                                f"Invalid final point cloud shape: {tuple(pred.shape)}"
                            )

                        cd_batch, f1_batch = per_sample_cd_f1(
                            pred=pred,
                            gt=gt,
                            backend=backend,
                            f1_threshold=args.f1_threshold,
                        )

                        cd_meters[difficulty].update_sum(
                            cd_batch.sum().item(),
                            batch_size,
                        )
                        f1_meters[difficulty].update_sum(
                            f1_batch.sum().item(),
                            batch_size,
                        )

                        # Per-taxonomy accumulation for official PoinTr-style
                        # macro averaging. Each object contributes all 8 views.
                        for sample_idx, cat_id in enumerate(cat_ids):
                            category_cd_meters[difficulty][cat_id].update_sum(
                                cd_batch[sample_idx].item(),
                                1,
                            )
                            category_f1_meters[difficulty][cat_id].update_sum(
                                f1_batch[sample_idx].item(),
                                1,
                            )

                    if (
                        difficulty == "moderate"
                        and has_classification
                    ):
                        if logits is None:
                            raise RuntimeError(
                                f"Mode {mode} should return logits but returned None."
                            )

                        moderate_logits_sum = (
                            logits
                            if moderate_logits_sum is None
                            else moderate_logits_sum + logits
                        )

                        view_pred = logits.argmax(dim=1)
                        view_correct += int(
                            (view_pred == labels).sum().item()
                        )
                        view_total += batch_size

            if has_classification:
                if moderate_logits_sum is None:
                    raise RuntimeError("Moderate logits were not accumulated.")

                object_logits = moderate_logits_sum / 8.0
                object_pred = object_logits.argmax(dim=1)
                object_correct += int(
                    (object_pred == labels).sum().item()
                )
                object_total += batch_size

            postfix = {}
            if has_completion:
                postfix["CD-H"] = f"{cd_meters['hard'].avg:.3f}"
            if has_classification and object_total > 0:
                postfix["Acc-obj"] = f"{100.0*object_correct/object_total:.2f}%"
            if postfix:
                progress.set_postfix(postfix)

    elapsed = time.time() - start_time

    result = {
        "dataset": args.dataset,
        "mode": mode,
        "checkpoint": checkpoint_path,
        "num_objects": len(loader.dataset),
        "views_per_difficulty": 8,
        "elapsed_seconds": elapsed,
    }

    if has_completion:
        def taxonomy_macro(meters_by_category) -> float:
            values = [
                meter.avg
                for meter in meters_by_category.values()
                if meter.count > 0
            ]
            if not values:
                return float("nan")
            return float(np.mean(values))

        # Main results: official PoinTr-style taxonomy macro average.
        result["cd_l2_x1000"] = {
            difficulty: taxonomy_macro(category_cd_meters[difficulty])
            for difficulty in ("simple", "moderate", "hard")
        }
        result["f1_at_1pct"] = {
            difficulty: taxonomy_macro(category_f1_meters[difficulty])
            for difficulty in ("simple", "moderate", "hard")
        }
        result["cd_l2_x1000"]["average"] = float(
            np.mean(
                [
                    result["cd_l2_x1000"]["simple"],
                    result["cd_l2_x1000"]["moderate"],
                    result["cd_l2_x1000"]["hard"],
                ]
            )
        )
        result["f1_at_1pct"]["average"] = float(
            np.mean(
                [
                    result["f1_at_1pct"]["simple"],
                    result["f1_at_1pct"]["moderate"],
                    result["f1_at_1pct"]["hard"],
                ]
            )
        )

        # Diagnostics: sample-weighted micro average. On balanced audit subsets,
        # macro and micro are equal; on the full imbalanced split, use macro in
        # the paper and retain micro only for auditing.
        result["cd_l2_x1000_micro"] = {
            difficulty: cd_meters[difficulty].avg
            for difficulty in ("simple", "moderate", "hard")
        }
        result["f1_at_1pct_micro"] = {
            difficulty: f1_meters[difficulty].avg
            for difficulty in ("simple", "moderate", "hard")
        }
        result["cd_l2_x1000_micro"]["average"] = float(
            np.mean(list(result["cd_l2_x1000_micro"].values()))
        )
        result["f1_at_1pct_micro"]["average"] = float(
            np.mean(list(result["f1_at_1pct_micro"].values()))
        )

    if has_classification:
        result["accuracy_object_mean_logits"] = (
            object_correct / object_total if object_total else float("nan")
        )
        result["accuracy_view_average"] = (
            view_correct / view_total if view_total else float("nan")
        )

    print("\n--- Official Evaluation Result ---")
    print(f"Dataset: {args.dataset}")
    print(f"Mode:    {mode}")
    print(f"Objects: {result['num_objects']}")
    print(f"Time:    {elapsed / 60.0:.2f} min")

    if has_completion:
        cd = result["cd_l2_x1000"]
        f1 = result["f1_at_1pct"]
        print(
            "CD-L2 ×1000 (taxonomy macro) | "
            f"S={cd['simple']:.4f} | "
            f"M={cd['moderate']:.4f} | "
            f"H={cd['hard']:.4f} | "
            f"Avg={cd['average']:.4f}"
        )
        print(
            "F1 @1% (taxonomy macro)      | "
            f"S={f1['simple']:.4f} | "
            f"M={f1['moderate']:.4f} | "
            f"H={f1['hard']:.4f} | "
            f"Avg={f1['average']:.4f}"
        )

    if has_classification:
        print(
            "Accuracy     | "
            f"Object(mean 8 logits)="
            f"{100.0*result['accuracy_object_mean_logits']:.2f}% | "
            f"View-average="
            f"{100.0*result['accuracy_view_average']:.2f}%"
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Official ShapeNet-55 evaluation for existing MBB checkpoints."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("55",),
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["baseline", "full", "ours"],
        choices=VALID_MODES,
        help="One or more modes. First audit recommendation: baseline full ours.",
    )
    parser.add_argument(
        "--checkpoint",
        nargs="*",
        default=None,
        help=(
            "Optional checkpoint paths corresponding one-to-one with --mode. "
            "If omitted, standard project paths are used."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config YAML path.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help=(
            "Path to the ShapeNet-55 dataset root "
            "containing test.txt and the GT point clouds."
        ),
    )
    parser.add_argument(
        "--audit_per_class",
        type=int,
        default=0,
        help="0 evaluates the full set; e.g. 10 selects 10 objects per category.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--metric_chunk", type=int, default=256)
    parser.add_argument("--f1_threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            PROJECT_ROOT,
            "outputs",
            "shapenet55",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pointnet2 FPS and model inference.")

    if args.checkpoint is not None and len(args.checkpoint) != len(args.mode):
        raise ValueError(
            "--checkpoint must contain exactly the same number of paths as --mode."
        )

    dataset, cfg = create_dataset_and_cfg(args)
    subset_indices = stratified_subset_indices(
        dataset,
        args.audit_per_class,
    )

    eval_dataset = (
        dataset
        if len(subset_indices) == len(dataset)
        else Subset(dataset, subset_indices)
    )

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    if args.checkpoint is None:
        checkpoint_paths = [
            default_checkpoint(args.dataset, mode)
            for mode in args.mode
        ]
    else:
        checkpoint_paths = args.checkpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for mode, checkpoint_path in zip(args.mode, checkpoint_paths):
        result = evaluate_one_mode(
            args=args,
            cfg=cfg,
            loader=loader,
            mode=mode,
            checkpoint_path=checkpoint_path,
        )
        all_results.append(result)

        suffix = (
            f"audit{args.audit_per_class}perclass"
            if args.audit_per_class > 0
            else "full"
        )
        result_path = output_dir / (
            f"ShapeNet{args.dataset}_{mode}_{suffix}.json"
        )
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"[Saved] {result_path}")

    summary_path = output_dir / (
        f"ShapeNet{args.dataset}_summary_"
        + (
            f"audit{args.audit_per_class}perclass.json"
            if args.audit_per_class > 0
            else "full.json"
        )
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)

    print(f"\n[Summary saved] {summary_path}")


if __name__ == "__main__":
    main()
