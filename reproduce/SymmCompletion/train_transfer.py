#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train No-Bridge, Full, or MBB on official pretrained SymmCompletion.

Run this script from the official SymmCompletion repository root after copying
the ``mbb_transfer`` package and this script into that root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------
# Locate the public MBB-Net root and vendored SymmCompletion code.
# This file lives at:
# Public entry under reproduce/SymmCompletion/.
# ---------------------------------------------------------------------
import sys

CURRENT_DIR = Path(__file__).resolve().parent
MBB_ROOT = CURRENT_DIR.parents[1]

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

_BOOTSTRAP_PARSER = argparse.ArgumentParser(add_help=False)
_BOOTSTRAP_PARSER.add_argument(
    "--symm_root",
    default=os.environ.get(
        "SYMM_ROOT",
        str(MBB_ROOT / "third_party" / "symmcompletion"),
    ),
)
_BOOTSTRAP_ARGS, _ = _BOOTSTRAP_PARSER.parse_known_args()
SYMM_ROOT = Path(_BOOTSTRAP_ARGS.symm_root).expanduser().resolve()

if str(SYMM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYMM_ROOT))

from symm_mbb.checkpoint import (
    load_model_strict,
    save_best_model,
    save_resume_state,
)
from symm_mbb.data import (
    TaxonomyLabelMap,
    build_dataset_bundle,
)
from symm_mbb.metrics import (
    AccuracyAccumulator,
    CompletionMetricAccumulator,
    per_sample_completion_metrics,
)
from symm_mbb.model import (
    SymmMBBTransfer,
)
from utils import misc


def set_seed(
    seed: int,
    deterministic: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = (
        deterministic
    )
    torch.backends.cudnn.benchmark = (
        not deterministic
    )


def seed_worker(
    worker_id: int,
) -> None:
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def warmup_cosine_factor(
    epoch_index: int,
    warmup_epochs: int,
    total_epochs: int,
    minimum_ratio: float,
) -> float:
    if warmup_epochs > 0 and (
        epoch_index < warmup_epochs
    ):
        return float(
            epoch_index + 1
        ) / float(
            warmup_epochs
        )

    denominator = max(
        total_epochs - warmup_epochs,
        1,
    )
    progress = (
        epoch_index - warmup_epochs
    ) / denominator
    progress = min(
        max(progress, 0.0),
        1.0,
    )
    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )
    return (
        minimum_ratio
        + (
            1.0 - minimum_ratio
        )
        * cosine
    )


def prepare_batch(
    dataset: str,
    data,
    npoints: int,
    label_map: TaxonomyLabelMap,
    taxonomy_ids: Sequence,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if dataset == "pcn":
        partial = data[0].to(
            device,
            non_blocking=True,
        ).float()
        ground_truth = data[1].to(
            device,
            non_blocking=True,
        ).float()
    else:
        ground_truth = data.to(
            device,
            non_blocking=True,
        ).float()
        partial, _ = (
            misc.seprate_point_cloud(
                ground_truth,
                npoints,
                [
                    int(
                        npoints * 0.25
                    ),
                    int(
                        npoints * 0.75
                    ),
                ],
                fixed_points=None,
            )
        )

    labels = label_map.encode_batch(
        taxonomy_ids,
        device,
    )
    return (
        partial,
        ground_truth,
        labels,
    )


def build_optimizer(
    model: SymmMBBTransfer,
    learning_rate: float,
    backbone_lr_scale: float,
    weight_decay: float,
):
    transfer_parameters = list(
        model.transfer_parameters()
    )
    groups = [
        {
            "params": transfer_parameters,
            "lr": learning_rate,
            "name": "transfer",
        }
    ]

    if model.tune_policy == "full":
        groups.append(
            {
                "params": list(
                    model.backbone_parameters()
                ),
                "lr": (
                    learning_rate
                    * backbone_lr_scale
                ),
                "name": "backbone",
            }
        )

    return torch.optim.AdamW(
        groups,
        weight_decay=weight_decay,
    )


def _value_to_string(value) -> str:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return str(value.detach().cpu().item())
        return str(value.detach().cpu().tolist())
    return str(value)


def _sequence_to_strings(values: Sequence) -> list:
    return [
        _value_to_string(value)
        for value in values
    ]


def _tensor_state(tensor: Optional[torch.Tensor]) -> Dict:
    if tensor is None:
        return {
            "present": False,
        }

    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_count = int(
        finite_mask.sum().item()
    )
    total_count = int(
        detached.numel()
    )

    state = {
        "present": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": bool(
            finite_count == total_count
        ),
        "finite_count": finite_count,
        "total_count": total_count,
        "nan_count": int(
            torch.isnan(detached).sum().item()
        ),
        "posinf_count": int(
            torch.isposinf(detached).sum().item()
        ),
        "neginf_count": int(
            torch.isneginf(detached).sum().item()
        ),
    }

    if finite_count > 0:
        finite_values = detached[
            finite_mask
        ].float()
        state.update(
            {
                "finite_min": float(
                    finite_values.min().item()
                ),
                "finite_max": float(
                    finite_values.max().item()
                ),
                "finite_mean": float(
                    finite_values.mean().item()
                ),
            }
        )

    return state


def _outputs_state(outputs) -> list:
    if torch.is_tensor(outputs):
        return [_tensor_state(outputs)]

    states = []
    for output in outputs:
        if torch.is_tensor(output):
            states.append(
                _tensor_state(output)
            )
        else:
            states.append(
                {
                    "present": False,
                    "python_type": type(output).__name__,
                }
            )
    return states


def _all_outputs_finite(outputs) -> bool:
    if torch.is_tensor(outputs):
        return bool(
            torch.isfinite(outputs).all().item()
        )

    for output in outputs:
        if (
            torch.is_tensor(output)
            and not torch.isfinite(output).all()
        ):
            return False
    return True


def _append_nonfinite_event(
    path: Path,
    event: Dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        path,
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                event,
                ensure_ascii=False,
            )
            + "\n"
        )


def _find_nonfinite_gradients(
    model: nn.Module,
) -> list:
    bad_names = []
    for name, parameter in (
        model.named_parameters()
    ):
        if (
            parameter.grad is not None
            and not torch.isfinite(
                parameter.grad
            ).all()
        ):
            bad_names.append(name)
    return bad_names


def train_one_epoch(
    model: SymmMBBTransfer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    classification_criterion,
    dataset: str,
    npoints: int,
    label_map: TaxonomyLabelMap,
    alpha: float,
    device: torch.device,
    gradient_clip: float,
    epoch: int,
    output_dir: Path,
    nonfinite_policy: str,
    nonfinite_retries: int,
    max_nonfinite_batches: int,
) -> Dict:
    """Train one epoch with explicit NaN/Inf protection.

    A batch is retried with a freshly generated ShapeNet partial cloud when a
    non-finite input, output, loss, or gradient is detected. ``optimizer.step``
    is executed only after the complete gradient vector has a finite norm.

    Under ``nonfinite_policy=skip``, a batch is skipped only after all retries
    fail. Every event is recorded in ``nonfinite_events.jsonl``. Training aborts
    once the number of skipped batches exceeds ``max_nonfinite_batches``.
    """
    model.train()

    total_sum = 0.0
    completion_sum = 0.0
    classification_sum = 0.0
    sample_count = 0
    correct_count = 0

    retried_batch_count = 0
    skipped_batch_count = 0
    skipped_sample_count = 0
    nonfinite_event_count = 0

    diagnostic_path = (
        output_dir
        / "nonfinite_events.jsonl"
    )

    progress = tqdm(
        loader,
        desc=(
            f"[{dataset}][{model.variant}] "
            f"Epoch {epoch} Train"
        ),
    )

    for batch_index, (
        taxonomy_ids,
        model_ids,
        data,
    ) in enumerate(
        progress,
        start=1,
    ):
        batch_succeeded = False
        last_failure_message = None
        attempts_used = 0

        for attempt in range(
            1,
            nonfinite_retries + 2,
        ):
            attempts_used = attempt
            optimizer.zero_grad(
                set_to_none=True
            )

            partial = None
            ground_truth = None
            labels = None
            outputs = None
            logits = None
            completion_loss = None
            classification_loss = None
            total_loss = None
            gradient_norm = None
            failure_stage = None
            bad_gradient_names = []

            try:
                partial, ground_truth, labels = (
                    prepare_batch(
                        dataset,
                        data,
                        npoints,
                        label_map,
                        taxonomy_ids,
                        device,
                    )
                )

                if not torch.isfinite(
                    partial
                ).all():
                    failure_stage = (
                        "partial_input"
                    )
                    raise FloatingPointError(
                        "Partial input contains NaN/Inf."
                    )

                if not torch.isfinite(
                    ground_truth
                ).all():
                    failure_stage = (
                        "ground_truth"
                    )
                    raise FloatingPointError(
                        "Ground truth contains NaN/Inf."
                    )

                outputs, logits = model(
                    partial
                )

                if not _all_outputs_finite(
                    outputs
                ):
                    failure_stage = (
                        "completion_outputs"
                    )
                    raise FloatingPointError(
                        "Completion output contains NaN/Inf."
                    )

                if not torch.isfinite(
                    logits
                ).all():
                    failure_stage = "logits"
                    raise FloatingPointError(
                        "Classification logits contain NaN/Inf."
                    )

                if (
                    labels.min().item() < 0
                    or labels.max().item()
                    >= logits.shape[1]
                ):
                    failure_stage = "labels"
                    raise ValueError(
                        "Classification label is outside the logits range: "
                        f"min={labels.min().item()}, "
                        f"max={labels.max().item()}, "
                        f"classes={logits.shape[1]}."
                    )

                completion_tuple = (
                    model.completion_loss(
                        outputs,
                        ground_truth,
                    )
                )
                completion_loss = (
                    completion_tuple[0]
                )
                classification_loss = (
                    classification_criterion(
                        logits,
                        labels,
                    )
                )
                total_loss = (
                    completion_loss
                    + alpha
                    * classification_loss
                )

                losses_are_finite = all(
                    bool(
                        torch.isfinite(loss).item()
                    )
                    for loss in (
                        completion_loss,
                        classification_loss,
                        total_loss,
                    )
                )
                if not losses_are_finite:
                    failure_stage = "loss"
                    raise FloatingPointError(
                        "Loss contains NaN/Inf."
                    )

                total_loss.backward()

                if gradient_clip > 0:
                    gradient_norm = (
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=gradient_clip,
                            error_if_nonfinite=False,
                        )
                    )
                    gradient_is_finite = bool(
                        torch.isfinite(
                            gradient_norm
                        ).item()
                    )
                else:
                    bad_gradient_names = (
                        _find_nonfinite_gradients(
                            model
                        )
                    )
                    gradient_is_finite = (
                        len(bad_gradient_names)
                        == 0
                    )

                if not gradient_is_finite:
                    failure_stage = "gradient"
                    if not bad_gradient_names:
                        bad_gradient_names = (
                            _find_nonfinite_gradients(
                                model
                            )
                        )
                    raise FloatingPointError(
                        "Gradient contains NaN/Inf."
                    )

                # This is intentionally after every finite check. A bad
                # gradient can therefore never contaminate AdamW parameters or
                # optimizer moments.
                optimizer.step()
                batch_succeeded = True
                break

            except (
                FloatingPointError,
                ValueError,
            ) as exc:
                optimizer.zero_grad(
                    set_to_none=True
                )
                nonfinite_event_count += 1
                last_failure_message = str(
                    exc
                )

                event = {
                    "time": time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "dataset": dataset,
                    "variant": model.variant,
                    "epoch": int(epoch),
                    "batch_index": int(
                        batch_index
                    ),
                    "attempt": int(attempt),
                    "maximum_attempts": int(
                        nonfinite_retries + 1
                    ),
                    "stage": (
                        failure_stage
                        or "unknown"
                    ),
                    "message": str(exc),
                    "taxonomy_ids": (
                        _sequence_to_strings(
                            taxonomy_ids
                        )
                    ),
                    "model_ids": (
                        _sequence_to_strings(
                            model_ids
                        )
                    ),
                    "partial": (
                        _tensor_state(partial)
                    ),
                    "ground_truth": (
                        _tensor_state(
                            ground_truth
                        )
                    ),
                    "labels": (
                        _tensor_state(labels)
                    ),
                    "completion_outputs": (
                        []
                        if outputs is None
                        else _outputs_state(
                            outputs
                        )
                    ),
                    "logits": (
                        _tensor_state(logits)
                    ),
                    "losses": {
                        "completion": (
                            None
                            if completion_loss
                            is None
                            else float(
                                completion_loss
                                .detach()
                                .cpu()
                                .item()
                            )
                        ),
                        "classification": (
                            None
                            if classification_loss
                            is None
                            else float(
                                classification_loss
                                .detach()
                                .cpu()
                                .item()
                            )
                        ),
                        "total": (
                            None
                            if total_loss is None
                            else float(
                                total_loss
                                .detach()
                                .cpu()
                                .item()
                            )
                        ),
                    },
                    "gradient_norm": (
                        None
                        if gradient_norm is None
                        else float(
                            gradient_norm
                            .detach()
                            .cpu()
                            .item()
                        )
                    ),
                    "bad_gradient_names": (
                        bad_gradient_names[:50]
                    ),
                }
                _append_nonfinite_event(
                    diagnostic_path,
                    event,
                )

                tqdm.write(
                    "[NonFinite] "
                    f"epoch={epoch} "
                    f"batch={batch_index}/{len(loader)} "
                    f"attempt={attempt}/"
                    f"{nonfinite_retries + 1} "
                    f"stage={event['stage']} "
                    f"message={exc}"
                )

                if attempt <= nonfinite_retries:
                    # ShapeNet-55 regenerates a new random crop here. PCN uses
                    # the same partial input, but retrying still distinguishes
                    # transient CUDA failures from persistent model failures.
                    continue

        if not batch_succeeded:
            skipped_batch_count += 1

            inferred_batch_size = (
                int(labels.size(0))
                if labels is not None
                else len(taxonomy_ids)
            )
            skipped_sample_count += (
                inferred_batch_size
            )

            should_abort = (
                nonfinite_policy == "raise"
                or skipped_batch_count
                > max_nonfinite_batches
            )

            if should_abort:
                raise FloatingPointError(
                    "A training batch remained non-finite after all retries. "
                    f"epoch={epoch}, "
                    f"batch={batch_index}/{len(loader)}, "
                    f"attempts={attempts_used}, "
                    f"skipped_batches={skipped_batch_count}, "
                    f"last_error={last_failure_message}. "
                    f"Diagnostics: {diagnostic_path}"
                )

            tqdm.write(
                "[NonFinite][SKIP] "
                f"epoch={epoch} "
                f"batch={batch_index}/{len(loader)} "
                f"skipped_batches="
                f"{skipped_batch_count}/"
                f"{max_nonfinite_batches}. "
                f"Diagnostics: {diagnostic_path}"
            )
            continue

        if attempts_used > 1:
            retried_batch_count += 1

        batch_size = labels.size(0)
        predictions = logits.argmax(
            dim=1
        )
        correct_count += int(
            (
                predictions == labels
            ).sum().item()
        )
        sample_count += batch_size

        total_sum += float(
            total_loss.item()
        ) * batch_size
        completion_sum += float(
            completion_loss.item()
        ) * batch_size
        classification_sum += float(
            classification_loss.item()
        ) * batch_size

        gates = model.bridge_state()
        progress.set_postfix(
            {
                "Loss": (
                    f"{total_sum/sample_count:.4f}"
                ),
                "Comp": (
                    f"{completion_sum/sample_count:.4f}"
                ),
                "CE": (
                    f"{classification_sum/sample_count:.4f}"
                ),
                "Acc": (
                    f"{100.0*correct_count/sample_count:.2f}%"
                ),
                "Retry": retried_batch_count,
                "Skip": skipped_batch_count,
                "gS": (
                    "-"
                    if gates["sem_gate"] is None
                    else f"{gates['sem_gate']:.4f}"
                ),
                "gG": (
                    "-"
                    if gates["geo_gate"] is None
                    else f"{gates['geo_gate']:.4f}"
                ),
            }
        )

    if sample_count == 0:
        raise RuntimeError(
            "No finite training batch was completed in this epoch. "
            f"Diagnostics: {diagnostic_path}"
        )

    if nonfinite_event_count > 0:
        print(
            "[NonFinite Summary] "
            f"epoch={epoch} | "
            f"events={nonfinite_event_count} | "
            f"retried_batches={retried_batch_count} | "
            f"skipped_batches={skipped_batch_count} | "
            f"skipped_samples={skipped_sample_count} | "
            f"diagnostics={diagnostic_path}"
        )

    return {
        "total_loss": (
            total_sum / sample_count
        ),
        "completion_loss": (
            completion_sum
            / sample_count
        ),
        "classification_loss": (
            classification_sum
            / sample_count
        ),
        "accuracy": (
            correct_count
            / sample_count
        ),
        "bridge": model.bridge_state(),
        "nonfinite": {
            "event_count": (
                nonfinite_event_count
            ),
            "retried_batch_count": (
                retried_batch_count
            ),
            "skipped_batch_count": (
                skipped_batch_count
            ),
            "skipped_sample_count": (
                skipped_sample_count
            ),
            "diagnostic_path": str(
                diagnostic_path
            ),
        },
    }


@torch.no_grad()
def validate(
    model: SymmMBBTransfer,
    loader: DataLoader,
    dataset: str,
    npoints: int,
    label_map: TaxonomyLabelMap,
    device: torch.device,
    epoch: int,
) -> Dict:
    model.eval()

    completion = (
        CompletionMetricAccumulator()
    )
    accuracy = AccuracyAccumulator()

    progress = tqdm(
        loader,
        desc=(
            f"[{dataset}][{model.variant}] "
            f"Epoch {epoch} Eval"
        ),
    )

    for taxonomy_ids, _, data in progress:
        partial, ground_truth, labels = (
            prepare_batch(
                dataset,
                data,
                npoints,
                label_map,
                taxonomy_ids,
                device,
            )
        )

        outputs, logits = model(
            partial
        )
        metrics = (
            per_sample_completion_metrics(
                outputs[-1],
                ground_truth,
            )
        )

        completion.update(
            taxonomy_ids,
            metrics,
        )
        accuracy.update_logits(
            logits,
            labels,
            taxonomy_ids,
        )

    return {
        "completion_taxonomy_macro": (
            completion.taxonomy_macro()
        ),
        "completion_object_average": (
            completion.object_average()
        ),
        "accuracy": (
            accuracy.state_dict()
        ),
        "bridge": model.bridge_state(),
    }


def metadata_from_args(
    args,
    label_map: TaxonomyLabelMap,
    model: SymmMBBTransfer,
) -> Dict:
    return {
        "dataset": args.dataset,
        "variant": args.variant,
        "tune_policy": (
            args.tune_policy
        ),
        "alpha": float(args.alpha),
        "official_pretrained": str(
            Path(
                args.pretrained
            ).resolve()
        ),
        "official_epoch": (
            model.official_epoch
        ),
        "official_metrics": (
            model.official_metrics
        ),
        "up_factors": list(
            model.up_factors
        ),
        "include_input": (
            model.include_input
        ),
        "num_classes": (
            model.num_classes
        ),
        "label_map": (
            label_map.state_dict()
        ),
        "training_protocol": {
            "checkpoint_selection": (
                "minimum taxonomy-macro "
                + (
                    "CDL1"
                    if args.dataset == "pcn"
                    else "CDL2"
                )
                + " on the existing SymmCompletion test-as-validation pipeline"
            ),
            "final_reporting": (
                "separate official evaluation script"
            ),
            "completion_loss": (
                "official SymmCompletion multi-stage ChamferDistanceL1 sum"
            ),
            "classification_loss": (
                "CrossEntropyLoss"
            ),
        },
        "args": vars(args),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symm_root",
        default=str(SYMM_ROOT),
        help=(
            "Path to the external official SymmCompletion repository. "
            "It must contain models/, datasets/, utils/, and extensions/."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "pcn",
            "shapenet55",
        ),
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "no_bridge",
            "full",
            "ours",
            "s2g",
            "g2s",
        ),
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help=(
            "Official SymmCompletion ckpt-best.pth for the selected dataset."
        ),
    )
    parser.add_argument(
        "--dataset_config",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )

    parser.add_argument(
        "--pcn_category_file",
        default=None,
    )
    parser.add_argument(
        "--pcn_partial_pattern",
        default=None,
    )
    parser.add_argument(
        "--pcn_complete_pattern",
        default=None,
    )
    parser.add_argument(
        "--shapenet_index_root",
        default=None,
    )
    parser.add_argument(
        "--shapenet_pc_root",
        default=None,
    )

    parser.add_argument(
        "--tune_policy",
        choices=(
            "full",
            "adapter",
        ),
        default="full",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--minimum_lr_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--val_freq",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--dense_val_last",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--resume_save_interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Path to a resumable last.pth checkpoint."
        ),
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
    )

    parser.add_argument(
        "--nonfinite_policy",
        choices=(
            "skip",
            "raise",
        ),
        default="skip",
        help=(
            "How to handle a batch that remains non-finite after retries. "
            "'skip' records and skips a limited number of batches; "
            "'raise' stops immediately."
        ),
    )
    parser.add_argument(
        "--nonfinite_retries",
        type=int,
        default=2,
        help=(
            "Number of fresh attempts after the first non-finite attempt. "
            "For ShapeNet-55 each retry regenerates the random crop."
        ),
    )
    parser.add_argument(
        "--max_nonfinite_batches",
        type=int,
        default=5,
        help=(
            "Maximum number of fully failed batches that may be skipped in "
            "one epoch when --nonfinite_policy=skip."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.nonfinite_retries < 0:
        raise ValueError(
            "--nonfinite_retries must be non-negative."
        )
    if args.max_nonfinite_batches < 0:
        raise ValueError(
            "--max_nonfinite_batches must be non-negative."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the official SymmCompletion extensions."
        )

    set_seed(
        args.seed,
        args.deterministic,
    )

    symm_root = Path(
        args.symm_root
    ).expanduser().resolve()
    required_symm_files = (
        symm_root / "models" / "SymmCompletion.py",
        symm_root / "datasets" / "PCNDataset.py",
        symm_root / "datasets" / "ShapeNet55Dataset.py",
        symm_root / "utils" / "misc.py",
    )
    missing_symm_files = [
        str(path)
        for path in required_symm_files
        if not path.is_file()
    ]
    if missing_symm_files:
        raise RuntimeError(
            "Invalid --symm_root. Missing required official files:\n"
            + "\n".join(missing_symm_files)
        )
    bundle = build_dataset_bundle(
        dataset=args.dataset,
        repository_root=str(
            symm_root
        ),
        dataset_config_path=(
            args.dataset_config
        ),
        pcn_category_file=(
            args.pcn_category_file
        ),
        pcn_partial_pattern=(
            args.pcn_partial_pattern
        ),
        pcn_complete_pattern=(
            args.pcn_complete_pattern
        ),
        shapenet_index_root=(
            args.shapenet_index_root
        ),
        shapenet_pc_root=(
            args.shapenet_pc_root
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(
            args.num_workers > 0
        ),
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    device = torch.device(
        "cuda"
    )
    model = SymmMBBTransfer(
        dataset=args.dataset,
        num_classes=bundle.num_classes,
        official_checkpoint=(
            args.pretrained
        ),
        variant=args.variant,
        tune_policy=(
            args.tune_policy
        ),
    ).to(device)

    output_dir = Path(
        args.output_dir
        or (
            MBB_ROOT
            / "checkpoints"
            / "Symm_MBB_Transfer"
            / args.dataset
            / args.variant
        )
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=(
            args.learning_rate
        ),
        backbone_lr_scale=(
            args.backbone_lr_scale
        ),
        weight_decay=(
            args.weight_decay
        ),
    )
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch_index: (
                warmup_cosine_factor(
                    epoch_index,
                    args.warmup_epochs,
                    args.epochs,
                    args.minimum_lr_ratio,
                )
            ),
        )
    )
    classification_criterion = (
        nn.CrossEntropyLoss()
    )

    best_model_path = (
        output_dir / "ckpt-best.pth"
    )
    last_model_path = (
        output_dir / "ckpt-last.pth"
    )
    final_model_path = (
        output_dir / "ckpt-final.pth"
    )
    history_path = (
        output_dir / "history.jsonl"
    )

    start_epoch = 1
    best_value = float("inf")
    best_metrics = None

    resume_path = None
    if args.resume:
        resume_path = Path(
            args.resume
        )
    elif (
        args.auto_resume
        and last_model_path.is_file()
    ):
        resume_path = (
            last_model_path
        )

    if resume_path is not None:
        resume_checkpoint = (
            load_model_strict(
                model,
                str(resume_path),
            )
        )
        optimizer.load_state_dict(
            resume_checkpoint[
                "optimizer_state_dict"
            ]
        )
        scheduler.load_state_dict(
            resume_checkpoint[
                "scheduler_state_dict"
            ]
        )
        start_epoch = int(
            resume_checkpoint["epoch"]
        ) + 1
        best_value = float(
            resume_checkpoint.get(
                "best_value",
                float("inf"),
            )
        )
        best_metrics = (
            resume_checkpoint.get(
                "best_metrics"
            )
        )

    metadata = metadata_from_args(
        args,
        bundle.label_map,
        model,
    )

    report = model.parameter_report()
    print("\n" + "=" * 96)
    print("SymmCompletion + MBB transfer training")
    print("=" * 96)
    print(f"Dataset:                 {args.dataset}")
    print(f"Variant:                 {args.variant}")
    print(f"Tune policy:             {args.tune_policy}")
    print(f"Symm root:               {symm_root}")
    print(f"MBB root:                {MBB_ROOT}")
    print(f"Official checkpoint:     {Path(args.pretrained).resolve()}")
    print(f"Official epoch:          {model.official_epoch}")
    print(f"Official metrics:        {model.official_metrics}")
    print(f"Up-factors:              {model.up_factors}")
    print(f"Include input:           {model.include_input}")
    print(f"Train samples:           {len(bundle.train_dataset)}")
    print(f"Test/validation samples: {len(bundle.test_dataset)}")
    print(f"Batch size:              {args.batch_size}")
    print(f"Epochs:                  {args.epochs}")
    print(f"Alpha:                   {args.alpha}")
    print(f"New-module LR:           {args.learning_rate}")
    print(f"Backbone LR scale:       {args.backbone_lr_scale}")
    print(f"Freeze backbone epochs:  {args.freeze_backbone_epochs}")
    print(f"Non-finite policy:       {args.nonfinite_policy}")
    print(f"Non-finite retries:      {args.nonfinite_retries}")
    print(f"Max skipped batches:     {args.max_nonfinite_batches}")
    print(f"Parameters total:        {report['total']:,}")
    print(f"Parameters trainable:    {report['trainable']:,}")
    print(f"Output:                  {output_dir}")
    print("=" * 96)

    selection_name = (
        "CDL1"
        if args.dataset == "pcn"
        else "CDL2"
    )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        epoch_start = time.time()

        if args.tune_policy == "full":
            should_train_backbone = (
                epoch
                > args.freeze_backbone_epochs
            )
            model.set_backbone_trainable(
                should_train_backbone
            )

        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            classification_criterion=(
                classification_criterion
            ),
            dataset=args.dataset,
            npoints=bundle.npoints,
            label_map=bundle.label_map,
            alpha=args.alpha,
            device=device,
            gradient_clip=(
                args.gradient_clip
            ),
            epoch=epoch,
            output_dir=output_dir,
            nonfinite_policy=(
                args.nonfinite_policy
            ),
            nonfinite_retries=(
                args.nonfinite_retries
            ),
            max_nonfinite_batches=(
                args.max_nonfinite_batches
            ),
        )
        scheduler.step()

        should_validate = (
            epoch % args.val_freq == 0
            or (
                args.epochs - epoch
                < args.dense_val_last
            )
            or epoch == args.epochs
        )
        validation_result = None

        if should_validate:
            validation_result = validate(
                model=model,
                loader=test_loader,
                dataset=args.dataset,
                npoints=bundle.npoints,
                label_map=(
                    bundle.label_map
                ),
                device=device,
                epoch=epoch,
            )

            current_value = float(
                validation_result[
                    "completion_taxonomy_macro"
                ][selection_name]
            )

            print(
                f"\n[Eval {epoch}] "
                f"Macro F1="
                f"{validation_result['completion_taxonomy_macro']['F-Score']:.4f} | "
                f"Macro CDL1="
                f"{validation_result['completion_taxonomy_macro']['CDL1']:.4f} | "
                f"Macro CDL2="
                f"{validation_result['completion_taxonomy_macro']['CDL2']:.4f} | "
                f"Acc(micro)="
                f"{100.0*validation_result['accuracy']['micro']:.2f}% | "
                f"Acc(macro)="
                f"{100.0*validation_result['accuracy']['taxonomy_macro']:.2f}%"
            )

            if current_value < best_value:
                best_value = (
                    current_value
                )
                best_metrics = (
                    validation_result
                )
                save_best_model(
                    model=model,
                    epoch=epoch,
                    metrics=(
                        validation_result
                    ),
                    metadata=metadata,
                    path=best_model_path,
                )
                print(
                    f"[*] New best {selection_name}="
                    f"{best_value:.6f}; saved {best_model_path}"
                )

        if (
            epoch
            % args.resume_save_interval
            == 0
            or epoch == 1
            or epoch == args.epochs
        ):
            save_resume_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_value=best_value,
                best_metrics=best_metrics,
                metadata=metadata,
                path=last_model_path,
            )

        if epoch == args.epochs:
            save_best_model(
                model=model,
                epoch=epoch,
                metrics=(
                    validation_result
                    or {}
                ),
                metadata={
                    **metadata,
                    "selection_rule": (
                        "fixed_final_epoch"
                    ),
                },
                path=final_model_path,
            )

        history_row = {
            "epoch": epoch,
            "backbone_trainable": (
                model.backbone_trainable
            ),
            "learning_rates": [
                group["lr"]
                for group in (
                    optimizer.param_groups
                )
            ],
            "train": train_result,
            "validation": (
                validation_result
            ),
            "best_value": best_value,
            "elapsed_seconds": (
                time.time()
                - epoch_start
            ),
        }
        with open(
            history_path,
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    history_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\nTraining complete.")
    print(f"Best:  {best_model_path}")
    print(f"Final: {final_model_path}")
    print(f"Last:  {last_model_path}")


if __name__ == "__main__":
    main()
