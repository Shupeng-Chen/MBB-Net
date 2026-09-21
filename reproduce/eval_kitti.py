#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Official-equivalent KITTI FD/MMD evaluator for existing prediction files.

This script evaluates existing:
    *_pred.npy
    *_input.npy

It does NOT retrain or rerun inference.

Protocol
--------
1. Fidelity (FD-L2):
       mean_{x in input} min_{y in prediction} ||x-y||_2^2

2. MMD-CD-L2:
       for each KITTI prediction P_i,
       min over every ShapeNet/PCN complete car G_j of
       CD-L2(P_i, G_j),
       followed by averaging over KITTI samples.

The exact-batch implementation preserves one CD value per GT shape before
taking the minimum. It does NOT average a whole GT batch first. All-zero
prediction padding rows are removed before MMD, matching ignore_zeros=True.

Default result directories
--------------------------
- MBB_CompOnly_PCN
- MBB_Full_PCN
- MBB_G2S_PCN
- MBB_S2G_PCN
- PoinTr_Official
- SeedFormer_Official
- SnowflakeNet_Official
- Symm_Official

Recommended use
---------------
1. First verify the exact-batch implementation against the official serial
   implementation on 3 samples:

   CUDA_VISIBLE_DEVICES=0 python -u \
     scripts/KITTI/metric_official_exact.py \
     --verify_samples 3 \
     --verify_method PoinTr_Official \
     --only_verify

2. Then evaluate all methods from existing predictions:

   CUDA_VISIBLE_DEVICES=0 python -u \
     scripts/KITTI/metric_official_exact.py \
     --backend exact_batch \
     --gt_batch_size 64 \
     --resume \
     2>&1 | tee paper/KITTI_inference_result/KITTI_official_metrics.log

Notes
-----
- All compared methods must use identical KITTI input files.
- All rows in the main paper table should use PCN-trained checkpoints with no
  KITTI/ShapeNetCars fine-tuning or target-domain adaptation.
- The reference library follows the existing PoinTr-style setup:
  PCN Cars train + test + val.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict
from tqdm import tqdm


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataset_from_cfg  # noqa: E402
from extensions.chamfer_dist import (  # noqa: E402
    ChamferDistanceL2,
    ChamferDistanceL2_split,
)

try:
    from extensions.chamfer_dist import ChamferFunction  # noqa: E402
except Exception:
    ChamferFunction = None


DEFAULT_METHODS = (
    "MBB_CompOnly_PCN",
    "MBB_Full_PCN",
    "MBB_G2S_PCN",
    "MBB_S2G_PCN",
    "PoinTr_Official",
    "SeedFormer_Official",
    "SnowflakeNet_Official",
    "Symm_Official",
)

PRED_SUFFIXES = ("_pred.npy", "_fine.npy")


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_hash(path: Path) -> str:
    """
    Hash numeric input content rather than the raw .npy container bytes.
    """
    array = np.load(str(path))
    array = np.asarray(array, dtype=np.float32)
    array = np.ascontiguousarray(array)

    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def ensure_cloud(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape [N, 3], got {array.shape}."
        )

    if array.shape[0] == 0:
        raise ValueError(f"{name} contains zero points.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")

    return np.ascontiguousarray(array)


def contains_zero_rows(tensor: torch.Tensor) -> bool:
    return bool(torch.all(tensor == 0, dim=-1).any().item())


def strip_zero_rows_single(
    cloud: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    Remove all-zero padding rows from one point cloud.

    PoinTr's ignore_zeros=True implementation removes zero rows before
    Chamfer evaluation. For the exact-batch backend, the KITTI prediction
    is a single cloud [1, N, 3], so we can perform the same operation once
    before expanding it against a GT batch.

    Returns:
        cleaned_cloud: [1, N_valid, 3]
        removed_rows: number of removed zero-padding rows
    """
    if cloud.ndim != 3 or cloud.shape[0] != 1 or cloud.shape[-1] != 3:
        raise ValueError(
            "strip_zero_rows_single expects [1, N, 3], got "
            f"{tuple(cloud.shape)}"
        )

    valid_mask = ~torch.all(cloud[0] == 0, dim=-1)
    removed_rows = int((~valid_mask).sum().item())

    if not bool(valid_mask.any().item()):
        raise RuntimeError(
            "The point cloud contains only zero-padding rows."
        )

    cleaned = cloud[:, valid_mask, :].contiguous()
    return cleaned, removed_rows


# ---------------------------------------------------------------------
# Prediction directory discovery
# ---------------------------------------------------------------------
def sample_id_from_prediction(path: Path) -> str:
    name = path.name

    for suffix in PRED_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]

    raise ValueError(f"Unsupported prediction filename: {path}")


def discover_prediction_pairs(
    method_dir: Path,
) -> Dict[str, Tuple[Path, Path]]:
    if not method_dir.is_dir():
        raise FileNotFoundError(
            f"Result directory does not exist: {method_dir}"
        )

    prediction_paths: List[Path] = []

    for suffix in PRED_SUFFIXES:
        prediction_paths.extend(method_dir.glob(f"*{suffix}"))

    prediction_paths = sorted(set(prediction_paths))

    if not prediction_paths:
        raise FileNotFoundError(
            f"No *_pred.npy or *_fine.npy files found in {method_dir}"
        )

    pairs: Dict[str, Tuple[Path, Path]] = {}

    for pred_path in prediction_paths:
        sample_id = sample_id_from_prediction(pred_path)
        input_path = method_dir / f"{sample_id}_input.npy"

        if not input_path.is_file():
            raise FileNotFoundError(
                f"Missing input paired with {pred_path.name}: "
                f"{input_path}"
            )

        if sample_id in pairs:
            raise RuntimeError(
                f"Duplicate prediction for sample '{sample_id}' "
                f"in {method_dir}"
            )

        pairs[sample_id] = (pred_path, input_path)

    return pairs


def verify_identical_inputs(
    all_pairs: Dict[str, Dict[str, Tuple[Path, Path]]],
    methods: Sequence[str],
    policy: str,
) -> Dict[str, object]:
    """
    Verify that every method uses the same sample IDs and identical input data.
    """
    if policy == "off":
        return {
            "policy": "off",
            "reference_method": None,
            "num_samples": None,
            "identical": None,
        }

    reference_method = methods[0]
    reference_pairs = all_pairs[reference_method]
    reference_ids = set(reference_pairs)

    mismatches: List[str] = []

    reference_hashes = {
        sample_id: canonical_array_hash(input_path)
        for sample_id, (_, input_path) in tqdm(
            reference_pairs.items(),
            desc=f"Hashing reference inputs ({reference_method})",
        )
    }

    for method in methods[1:]:
        current_pairs = all_pairs[method]
        current_ids = set(current_pairs)

        missing = sorted(reference_ids - current_ids)
        extra = sorted(current_ids - reference_ids)

        if missing:
            mismatches.append(
                f"{method}: missing {len(missing)} sample IDs; "
                f"examples={missing[:5]}"
            )

        if extra:
            mismatches.append(
                f"{method}: extra {len(extra)} sample IDs; "
                f"examples={extra[:5]}"
            )

        for sample_id in sorted(reference_ids.intersection(current_ids)):
            current_hash = canonical_array_hash(
                current_pairs[sample_id][1]
            )

            if current_hash != reference_hashes[sample_id]:
                mismatches.append(
                    f"{method}: input mismatch for {sample_id}"
                )

                if len(mismatches) >= 30:
                    break

        if len(mismatches) >= 30:
            break

    identical = len(mismatches) == 0

    if mismatches:
        message = (
            "KITTI input consistency check failed:\n  "
            + "\n  ".join(mismatches)
        )

        if policy == "strict":
            raise RuntimeError(message)

        print("[WARNING]", message)

    else:
        print(
            f"[Input audit] All {len(reference_ids)} KITTI inputs are "
            f"identical across {len(methods)} methods."
        )

    return {
        "policy": policy,
        "reference_method": reference_method,
        "num_samples": len(reference_ids),
        "identical": identical,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------
# ShapeNet/PCN car reference library
# ---------------------------------------------------------------------
def resolve_pcn_cars_config(
    config_path: Path,
    root_override: Optional[str],
) -> EasyDict:
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)

    if root_override:
        raw_cfg["DATA_ROOT"] = str(
            Path(root_override).expanduser().resolve()
        )

    data_root = Path(raw_cfg["DATA_ROOT"]).expanduser()

    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    raw_cfg["DATA_ROOT"] = str(data_root)

    pcn_json = data_root / "PCN.json"
    if not pcn_json.is_file():
        raise FileNotFoundError(
            "PCN Cars root is invalid.\n"
            f"Expected: {pcn_json}\n"
            "Pass the correct path with --pcn_cars_root."
        )

    for split in ("train", "test", "val"):
        if not (data_root / split).is_dir():
            raise FileNotFoundError(
                f"Missing split directory: {data_root / split}"
            )

    return EasyDict(raw_cfg)


def build_shapenet_cars_dataset(cfg: EasyDict):
    """
    Preserve the existing PoinTr-style reference composition:
    train + test + val.
    """
    train_set = build_dataset_from_cfg(
        cfg,
        EasyDict(subset="train"),
    )
    test_set = build_dataset_from_cfg(
        cfg,
        EasyDict(subset="test"),
    )
    val_set = build_dataset_from_cfg(
        cfg,
        EasyDict(subset="val"),
    )

    dataset = train_set + test_set + val_set

    print(
        "[Reference library] "
        f"train={len(train_set)}, "
        f"test={len(test_set)}, "
        f"val={len(val_set)}, "
        f"total={len(dataset)}"
    )

    return dataset


def extract_gt_tensor(item) -> torch.Tensor:
    """
    Expected current dataset output:
        taxonomy_id, model_id, (dummy_partial, gt_tensor)
    """
    try:
        gt = item[-1][1]
    except Exception as exc:
        raise RuntimeError(
            "Unable to extract GT with dataset[i][-1][1]. "
            f"Item type: {type(item)}"
        ) from exc

    if not torch.is_tensor(gt):
        gt = torch.as_tensor(gt)

    gt = gt.float().cpu()

    if gt.ndim == 3 and gt.shape[0] == 1:
        gt = gt[0]

    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(
            f"Reference GT must be [N, 3], got {tuple(gt.shape)}"
        )

    if not torch.isfinite(gt).all():
        raise ValueError("Reference GT contains NaN or Inf.")

    return gt.contiguous()


def load_or_build_gt_cache(
    dataset,
    cache_path: Path,
    rebuild: bool,
    seed: int,
    data_root: str,
) -> Tuple[torch.Tensor, dict]:
    """
    Cache all complete car references once so every method is evaluated against
    exactly the same sampled GT tensors.
    """
    if cache_path.is_file() and not rebuild:
        print(f"[GT cache] Loading: {cache_path}")
        payload = torch.load(str(cache_path), map_location="cpu")

        if not isinstance(payload, dict) or "gt" not in payload:
            raise RuntimeError(
                f"Invalid GT cache format: {cache_path}"
            )

        gt = payload["gt"].float().contiguous()
        meta = payload.get("meta", {})

        print(
            f"[GT cache] shape={tuple(gt.shape)}, "
            f"source={meta.get('data_root')}"
        )

        return gt, meta

    print("[GT cache] Building fixed ShapeNet/PCN car reference cache.")
    set_seed(seed)

    gt_list: List[torch.Tensor] = []
    expected_points: Optional[int] = None

    for index in tqdm(
        range(len(dataset)),
        desc="Loading complete ShapeNet cars",
    ):
        gt = extract_gt_tensor(dataset[index])

        if expected_points is None:
            expected_points = int(gt.shape[0])

        if gt.shape[0] != expected_points:
            raise RuntimeError(
                "Reference clouds have inconsistent point counts: "
                f"expected {expected_points}, got {gt.shape[0]} "
                f"at index {index}."
            )

        gt_list.append(gt.unsqueeze(0))

    gt = torch.cat(gt_list, dim=0).contiguous()

    meta = {
        "data_root": data_root,
        "num_references": int(gt.shape[0]),
        "num_points": int(gt.shape[1]),
        "composition": ["train", "test", "val"],
        "seed": seed,
        "dtype": str(gt.dtype),
        "contains_zero_rows": contains_zero_rows(gt),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gt": gt,
            "meta": meta,
        },
        str(cache_path),
    )

    print(f"[GT cache] Saved: {cache_path}")
    print(f"[GT cache] shape={tuple(gt.shape)}")

    return gt, meta


# ---------------------------------------------------------------------
# Chamfer metric backends
# ---------------------------------------------------------------------
def invoke_raw_chamfer(
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Call the project's raw ChamferFunction and retain per-point distances.
    """
    if ChamferFunction is None:
        raise RuntimeError(
            "ChamferFunction is not importable from "
            "extensions.chamfer_dist. Use --backend serial."
        )

    if hasattr(ChamferFunction, "apply"):
        output = ChamferFunction.apply(x, y)
    else:
        instance = ChamferFunction()
        output = instance(x, y)

    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise RuntimeError(
            "Unexpected ChamferFunction output. "
            f"Type={type(output)}"
        )

    dist1 = output[0]
    dist2 = output[1]

    if dist1.ndim < 2 or dist2.ndim < 2:
        raise RuntimeError(
            "Raw ChamferFunction did not return per-point distances. "
            f"Shapes: {tuple(dist1.shape)}, {tuple(dist2.shape)}"
        )

    return dist1, dist2


@torch.inference_mode()
def compute_fd(
    input_cloud: torch.Tensor,
    prediction: torch.Tensor,
    split_metric: torch.nn.Module,
) -> float:
    """
    PoinTr-style one-way fidelity: input -> prediction.
    """
    dist_input_to_pred, _ = split_metric(
        input_cloud,
        prediction,
    )

    return float(dist_input_to_pred.item())


@torch.inference_mode()
def compute_mmd_serial(
    prediction: torch.Tensor,
    gt_cpu: torch.Tensor,
    serial_metric: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Official serial definition: compute one CD per GT, then take the minimum.
    """
    best = float("inf")

    for index in range(gt_cpu.shape[0]):
        gt = gt_cpu[index : index + 1].to(
            device,
            non_blocking=True,
        )

        value = serial_metric(
            gt,
            prediction,
        )

        scalar = float(value.item())
        if scalar < best:
            best = scalar

    return best


@torch.inference_mode()
def compute_mmd_exact_batch(
    prediction: torch.Tensor,
    gt_cpu: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """
    Exact batched equivalent of the serial minimum.

    Crucially:
      pair_cd has shape [GT_batch].
    We take the minimum only after obtaining one CD for every GT.
    """
    best = float("inf")

    for start in range(0, gt_cpu.shape[0], batch_size):
        end = min(start + batch_size, gt_cpu.shape[0])

        gt = gt_cpu[start:end].to(
            device,
            non_blocking=True,
        )
        current_batch = gt.shape[0]

        pred_expand = prediction.expand(
            current_batch,
            -1,
            -1,
        ).contiguous()

        dist_pred_to_gt, dist_gt_to_pred = invoke_raw_chamfer(
            pred_expand,
            gt,
        )

        pair_cd = (
            dist_pred_to_gt.reshape(current_batch, -1).mean(dim=1)
            + dist_gt_to_pred.reshape(current_batch, -1).mean(dim=1)
        )

        batch_best = float(pair_cd.min().item())
        if batch_best < best:
            best = batch_best

    return best


def relative_gap(reference: float, candidate: float) -> float:
    return (
        (candidate - reference)
        / max(abs(reference), 1e-12)
        * 100.0
    )


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------
def verify_exact_batch_against_serial(
    method: str,
    pairs: Dict[str, Tuple[Path, Path]],
    gt_cpu: torch.Tensor,
    num_samples: int,
    gt_batch_size: int,
    device: torch.device,
    tolerance_percent: float,
) -> List[dict]:
    if num_samples <= 0:
        return []

    serial_metric = ChamferDistanceL2(
        ignore_zeros=True,
    ).to(device)

    records: List[dict] = []

    selected_ids = sorted(pairs)[:num_samples]
    gt_has_zero_rows = contains_zero_rows(gt_cpu)

    print("\n" + "=" * 88)
    print(
        f"Verifying exact-batch MMD against official serial MMD "
        f"on {len(selected_ids)} samples from {method}"
    )
    print("=" * 88)

    for sample_id in selected_ids:
        pred_path, _ = pairs[sample_id]

        pred_np = ensure_cloud(
            np.load(str(pred_path)),
            f"prediction:{sample_id}",
        )
        pred = torch.from_numpy(pred_np).unsqueeze(0).to(device)

        # Official ChamferDistanceL2(ignore_zeros=True) removes all-zero
        # padding rows. The exact-batch backend reproduces this by stripping
        # zero rows from the single prediction before batch expansion.
        pred_for_mmd, removed_zero_rows = strip_zero_rows_single(pred)

        # The cached complete ShapeNet/PCN car references should not contain
        # zero-padding rows. If they do, variable-length masking across a GT
        # batch is no longer equivalent, so fall back to the serial backend.
        if gt_has_zero_rows:
            raise RuntimeError(
                "The GT cache contains zero-padding rows. Exact-batch MMD "
                "cannot preserve ignore_zeros=True for variable-length GT "
                "clouds. Rebuild the GT cache or use --backend serial."
            )

        if removed_zero_rows > 0:
            print(
                f"[Zero padding] {sample_id}: removed "
                f"{removed_zero_rows} prediction rows before MMD."
            )

        serial_value = compute_mmd_serial(
            prediction=pred,
            gt_cpu=gt_cpu,
            serial_metric=serial_metric,
            device=device,
        )
        batch_value = compute_mmd_exact_batch(
            prediction=pred_for_mmd,
            gt_cpu=gt_cpu,
            batch_size=gt_batch_size,
            device=device,
        )
        gap = relative_gap(serial_value, batch_value)

        record = {
            "sample_id": sample_id,
            "serial_x1000": serial_value * 1000.0,
            "exact_batch_x1000": batch_value * 1000.0,
            "relative_gap_percent": gap,
        }
        records.append(record)

        print(f"\n{sample_id}")
        print(f"Official serial : {serial_value * 1000.0:.9f}")
        print(f"Exact batch     : {batch_value * 1000.0:.9f}")
        print(f"Relative gap    : {gap:+.8f}%")

        if abs(gap) > tolerance_percent:
            raise RuntimeError(
                "Exact-batch implementation failed equivalence verification.\n"
                f"Sample: {sample_id}\n"
                f"Gap: {gap:+.8f}%\n"
                f"Tolerance: {tolerance_percent}%\n"
                "Use --backend serial."
            )

        del pred
        torch.cuda.empty_cache()

    print(
        "\n[Verification] Exact-batch MMD matches the official serial "
        f"definition within {tolerance_percent}%."
    )

    return records


# ---------------------------------------------------------------------
# Per-method evaluation with resume
# ---------------------------------------------------------------------
def load_progress(path: Path) -> dict:
    if not path.is_file():
        return {
            "per_sample": {},
        }

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "per_sample" not in payload:
        raise RuntimeError(f"Invalid progress file: {path}")

    return payload


def save_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(str(temporary), str(path))


def summarize_per_sample(per_sample: Dict[str, dict]) -> dict:
    if not per_sample:
        raise RuntimeError("No per-sample metrics were computed.")

    fd_values = [
        float(record["fd_l2"])
        for record in per_sample.values()
    ]
    mmd_values = [
        float(record["mmd_cd_l2"])
        for record in per_sample.values()
    ]
    output_points = sorted(
        set(
            int(record["output_points"])
            for record in per_sample.values()
        )
    )

    return {
        "num_samples": len(per_sample),
        "fd_l2": float(np.mean(fd_values)),
        "fd_l2_x1000": float(np.mean(fd_values) * 1000.0),
        "mmd_cd_l2": float(np.mean(mmd_values)),
        "mmd_cd_l2_x1000": float(np.mean(mmd_values) * 1000.0),
        "fd_std_x1000": float(np.std(fd_values) * 1000.0),
        "mmd_std_x1000": float(np.std(mmd_values) * 1000.0),
        "output_points": output_points,
        "all_fd_zero_within_1e_12": bool(
            all(abs(value) <= 1e-12 for value in fd_values)
        ),
    }


def evaluate_method(
    method: str,
    method_dir: Path,
    pairs: Dict[str, Tuple[Path, Path]],
    gt_cpu: torch.Tensor,
    backend: str,
    gt_batch_size: int,
    device: torch.device,
    resume: bool,
    save_every: int,
    evaluator_version: str,
) -> dict:
    split_metric = ChamferDistanceL2_split(
        ignore_zeros=True,
    ).to(device)

    serial_metric = None
    if backend == "serial":
        serial_metric = ChamferDistanceL2(
            ignore_zeros=True,
        ).to(device)

    progress_path = (
        method_dir
        / "official_pointr_metric_progress.json"
    )
    final_path = (
        method_dir
        / "official_pointr_metric_result.json"
    )
    sample_csv_path = (
        method_dir
        / "official_pointr_metric_per_sample.csv"
    )

    if resume:
        progress = load_progress(progress_path)
    else:
        progress = {"per_sample": {}}

    per_sample: Dict[str, dict] = progress["per_sample"]

    ordered_ids = sorted(pairs)
    pending_ids = [
        sample_id
        for sample_id in ordered_ids
        if sample_id not in per_sample
    ]

    print("\n" + "=" * 88)
    print(f"Method:       {method}")
    print(f"Directory:    {method_dir}")
    print(f"Samples:      {len(ordered_ids)}")
    print(f"Already done: {len(per_sample)}")
    print(f"Pending:      {len(pending_ids)}")
    print(f"MMD backend:  {backend}")
    print("=" * 88)

    start_time = time.time()
    gt_has_zero_rows = contains_zero_rows(gt_cpu)

    if gt_has_zero_rows and backend == "exact_batch":
        print(
            "[MMD backend] GT cache contains zero-padding rows; "
            "this method will use the official serial backend."
        )

    progress_bar = tqdm(
        pending_ids,
        desc=f"Evaluating {method}",
    )

    for local_index, sample_id in enumerate(progress_bar, start=1):
        pred_path, input_path = pairs[sample_id]

        pred_np = ensure_cloud(
            np.load(str(pred_path)),
            f"prediction:{sample_id}",
        )
        input_np = ensure_cloud(
            np.load(str(input_path)),
            f"input:{sample_id}",
        )

        prediction = (
            torch.from_numpy(pred_np)
            .unsqueeze(0)
            .to(device)
        )
        input_cloud = (
            torch.from_numpy(input_np)
            .unsqueeze(0)
            .to(device)
        )

        fd_value = compute_fd(
            input_cloud=input_cloud,
            prediction=prediction,
            split_metric=split_metric,
        )

        # FD continues to use the official split metric with
        # ignore_zeros=True. For exact-batch MMD, remove zero-padding rows
        # from the single prediction before expanding it against GT batches.
        prediction_for_mmd, removed_zero_rows = strip_zero_rows_single(
            prediction
        )

        must_use_serial = (
            backend == "serial"
            or gt_has_zero_rows
        )

        if must_use_serial:
            if serial_metric is None:
                serial_metric = ChamferDistanceL2(
                    ignore_zeros=True,
                ).to(device)

            mmd_value = compute_mmd_serial(
                prediction=prediction,
                gt_cpu=gt_cpu,
                serial_metric=serial_metric,
                device=device,
            )
            actual_backend = "serial"
        else:
            mmd_value = compute_mmd_exact_batch(
                prediction=prediction_for_mmd,
                gt_cpu=gt_cpu,
                batch_size=gt_batch_size,
                device=device,
            )
            actual_backend = "exact_batch"

        per_sample[sample_id] = {
            "fd_l2": fd_value,
            "fd_l2_x1000": fd_value * 1000.0,
            "mmd_cd_l2": mmd_value,
            "mmd_cd_l2_x1000": mmd_value * 1000.0,
            "input_points": int(input_np.shape[0]),
            "output_points": int(pred_np.shape[0]),
            "prediction_zero_rows_removed_for_mmd": removed_zero_rows,
            "mmd_backend": actual_backend,
            "prediction_file": pred_path.name,
            "input_file": input_path.name,
        }

        progress_bar.set_postfix(
            fd=f"{fd_value * 1000.0:.4f}",
            mmd=f"{mmd_value * 1000.0:.4f}",
        )

        if (
            local_index % save_every == 0
            or local_index == len(pending_ids)
        ):
            payload = {
                "method": method,
                "method_dir": str(method_dir),
                "evaluator_version": evaluator_version,
                "requested_backend": backend,
                "gt_batch_size": gt_batch_size,
                "per_sample": per_sample,
            }
            save_json_atomic(payload, progress_path)

        del prediction
        del input_cloud

        if local_index % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    summary = summarize_per_sample(per_sample)
    elapsed_minutes = (time.time() - start_time) / 60.0

    final_payload = {
        "method": method,
        "method_dir": str(method_dir),
        "protocol": {
            "source_training": "PCN",
            "kitti_adaptation": "None",
            "shapenet_cars_finetuning": "None",
            "fd_definition": "one-way CD-L2: input -> prediction",
            "mmd_definition": (
                "minimum symmetric CD-L2 to every complete PCN car "
                "reference, averaged over KITTI samples"
            ),
            "reference_composition": ["train", "test", "val"],
            "scale": "x1000 for reporting",
        },
        "evaluator_version": evaluator_version,
        "requested_backend": backend,
        "gt_batch_size": gt_batch_size,
        "elapsed_minutes_this_run": elapsed_minutes,
        "summary": summary,
        "per_sample": per_sample,
    }

    save_json_atomic(final_payload, final_path)

    with sample_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "fd_l2_x1000",
                "mmd_cd_l2_x1000",
                "input_points",
                "output_points",
                "mmd_backend",
            ]
        )

        for sample_id in sorted(per_sample):
            record = per_sample[sample_id]
            writer.writerow(
                [
                    sample_id,
                    f"{record['fd_l2_x1000']:.9f}",
                    f"{record['mmd_cd_l2_x1000']:.9f}",
                    record["input_points"],
                    record["output_points"],
                    record["mmd_backend"],
                ]
            )

    print("\n--- KITTI Official-Equivalent Result ---")
    print(f"Method:                    {method}")
    print(f"Samples:                   {summary['num_samples']}")
    print(
        f"FD-L2 ×1000:              "
        f"{summary['fd_l2_x1000']:.6f}"
    )
    print(
        f"MMD-CD-L2 ×1000:          "
        f"{summary['mmd_cd_l2_x1000']:.6f}"
    )
    print(
        f"Output points:             "
        f"{summary['output_points']}"
    )
    print(
        f"All FD values effectively zero: "
        f"{summary['all_fd_zero_within_1e_12']}"
    )
    print(f"[Saved] {final_path}")
    print(f"[Saved] {sample_csv_path}")

    return final_payload


# ---------------------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------------------
def write_global_outputs(
    results: List[dict],
    output_root: Path,
    input_audit: dict,
    gt_meta: dict,
    evaluator_version: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    summary_json = (
        output_root
        / "KITTI_official_pointr_metrics_summary.json"
    )
    summary_csv = (
        output_root
        / "KITTI_official_pointr_metrics_summary.csv"
    )
    summary_md = (
        output_root
        / "KITTI_official_pointr_metrics_summary.md"
    )
    manifest_json = (
        output_root
        / "KITTI_protocol_manifest.json"
    )

    compact = []

    for payload in results:
        summary = payload["summary"]
        compact.append(
            {
                "method": payload["method"],
                "source_training": "PCN",
                "adaptation": "None",
                "num_samples": summary["num_samples"],
                "fd_l2_x1000": summary["fd_l2_x1000"],
                "mmd_cd_l2_x1000": summary[
                    "mmd_cd_l2_x1000"
                ],
                "output_points": summary["output_points"],
                "all_fd_zero": summary[
                    "all_fd_zero_within_1e_12"
                ],
                "evaluator_version": evaluator_version,
            }
        )

    save_json_atomic(
        {
            "evaluator_version": evaluator_version,
            "input_audit": input_audit,
            "gt_reference": gt_meta,
            "results": compact,
        },
        summary_json,
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "source_training",
                "adaptation",
                "num_samples",
                "fd_l2_x1000",
                "mmd_cd_l2_x1000",
                "output_points",
                "all_fd_zero",
                "evaluator_version",
            ],
        )
        writer.writeheader()
        writer.writerows(compact)

    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write(
            "| Method | Training | Adaptation | #Output points | "
            "FD-L2 ×1000 ↓ | MMD-CD-L2 ×1000 ↓ |\n"
        )
        handle.write(
            "|---|---|---|---:|---:|---:|\n"
        )

        for row in compact:
            points = ", ".join(
                str(value)
                for value in row["output_points"]
            )

            handle.write(
                f"| {row['method']} "
                f"| PCN "
                f"| None "
                f"| {points} "
                f"| {row['fd_l2_x1000']:.4f} "
                f"| {row['mmd_cd_l2_x1000']:.4f} |\n"
            )

        handle.write("\n")
        handle.write(
            "Protocol: PCN-trained checkpoints are directly evaluated "
            "on identical KITTI inputs without KITTI adaptation, "
            "ShapeNetCars fine-tuning, or test-time optimization. "
            "FD and MMD use CD-L2 and are reported at ×1000.\n"
        )

    manifest = {
        "evaluator_version": evaluator_version,
        "project_root": str(PROJECT_ROOT),
        "result_root": str(output_root),
        "input_audit": input_audit,
        "gt_reference": gt_meta,
        "methods": [
            {
                "method": payload["method"],
                "prediction_directory": payload["method_dir"],
                "source_training": "PCN",
                "kitti_adaptation": "None",
                "shapenet_cars_finetuning": "None",
                "metric_result_file": str(
                    Path(payload["method_dir"])
                    / "official_pointr_metric_result.json"
                ),
            }
            for payload in results
        ],
    }
    save_json_atomic(manifest, manifest_json)

    print(f"\n[Saved] {summary_json}")
    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {summary_md}")
    print(f"[Saved] {manifest_json}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing KITTI predictions with official-equivalent "
            "PoinTr FD and MMD definitions."
        )
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default=str(
            PROJECT_ROOT
            / "paper"
            / "KITTI_inference_result"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--pcn_cars_config",
        type=str,
        default=str(
            PROJECT_ROOT
            / "configs"
            / "PCNCars.yaml"
        ),
    )
    parser.add_argument(
        "--pcn_cars_root",
        type=str,
        default=None,
        help=(
            "Optional override for the PCN/ShapeNetCompletion root "
            "containing PCN.json and train/test/val."
        ),
    )
    parser.add_argument(
        "--gt_cache",
        type=str,
        default=str(
            PROJECT_ROOT
            / "paper"
            / "KITTI_metric_cache"
            / "ShapeNetCars_train_test_val_16384.pt"
        ),
    )
    parser.add_argument(
        "--rebuild_gt_cache",
        action="store_true",
    )
    parser.add_argument(
        "--backend",
        choices=("exact_batch", "serial"),
        default="exact_batch",
    )
    parser.add_argument(
        "--gt_batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--input_check",
        choices=("strict", "warn", "off"),
        default="strict",
    )
    parser.add_argument(
        "--verify_samples",
        type=int,
        default=0,
        help=(
            "Compare exact_batch against official serial MMD on this "
            "many samples before full evaluation."
        ),
    )
    parser.add_argument(
        "--verify_method",
        type=str,
        default="PoinTr_Official",
    )
    parser.add_argument(
        "--verify_tolerance_percent",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--only_verify",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this Chamfer implementation."
        )

    if args.gt_batch_size <= 0:
        raise ValueError("--gt_batch_size must be positive.")

    if args.save_every <= 0:
        raise ValueError("--save_every must be positive.")

    set_seed(args.seed)

    device = torch.device("cuda")
    results_root = Path(
        args.results_root
    ).expanduser().resolve()

    methods = list(args.methods)

    all_pairs: Dict[
        str,
        Dict[str, Tuple[Path, Path]],
    ] = {}

    for method in methods:
        method_dir = results_root / method
        all_pairs[method] = discover_prediction_pairs(
            method_dir
        )

    input_audit = verify_identical_inputs(
        all_pairs=all_pairs,
        methods=methods,
        policy=args.input_check,
    )

    config_path = Path(
        args.pcn_cars_config
    ).expanduser().resolve()

    cfg = resolve_pcn_cars_config(
        config_path=config_path,
        root_override=args.pcn_cars_root,
    )
    dataset = build_shapenet_cars_dataset(cfg)

    gt_cache_path = Path(
        args.gt_cache
    ).expanduser().resolve()

    gt_cpu, gt_meta = load_or_build_gt_cache(
        dataset=dataset,
        cache_path=gt_cache_path,
        rebuild=args.rebuild_gt_cache,
        seed=args.seed,
        data_root=str(cfg.DATA_ROOT),
    )

    evaluator_version = (
        "KITTI_PoinTr_official_equivalent_v3_"
        "per_GT_min_CD_L2_ignore_zero_padding"
    )

    verification_records: List[dict] = []

    if args.verify_samples > 0:
        if args.verify_method not in all_pairs:
            raise ValueError(
                f"--verify_method {args.verify_method} is not "
                f"in --methods: {methods}"
            )

        if args.backend != "exact_batch":
            print(
                "[Verification] Skipped because --backend is serial."
            )
        else:
            verification_records = (
                verify_exact_batch_against_serial(
                    method=args.verify_method,
                    pairs=all_pairs[args.verify_method],
                    gt_cpu=gt_cpu,
                    num_samples=args.verify_samples,
                    gt_batch_size=args.gt_batch_size,
                    device=device,
                    tolerance_percent=(
                        args.verify_tolerance_percent
                    ),
                )
            )

            verification_path = (
                results_root
                / "exact_batch_serial_verification.json"
            )
            save_json_atomic(
                {
                    "method": args.verify_method,
                    "records": verification_records,
                    "tolerance_percent": (
                        args.verify_tolerance_percent
                    ),
                },
                verification_path,
            )
            print(f"[Saved] {verification_path}")

    if args.only_verify:
        print("[Done] Verification-only mode.")
        return

    results: List[dict] = []

    for method in methods:
        method_dir = results_root / method

        payload = evaluate_method(
            method=method,
            method_dir=method_dir,
            pairs=all_pairs[method],
            gt_cpu=gt_cpu,
            backend=args.backend,
            gt_batch_size=args.gt_batch_size,
            device=device,
            resume=args.resume,
            save_every=args.save_every,
            evaluator_version=evaluator_version,
        )
        results.append(payload)

        gc.collect()
        torch.cuda.empty_cache()

    write_global_outputs(
        results=results,
        output_root=results_root,
        input_audit=input_audit,
        gt_meta=gt_meta,
        evaluator_version=evaluator_version,
    )

    print("\n" + "=" * 88)
    print("All KITTI metrics completed.")
    print("=" * 88)
    print(
        "Use only the generated official-equivalent summary in the paper. "
        "Keep the previous batch-averaged MMD logs as legacy records."
    )


if __name__ == "__main__":
    main()