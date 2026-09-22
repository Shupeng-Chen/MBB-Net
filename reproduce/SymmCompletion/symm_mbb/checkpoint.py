#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safe checkpoint utilities."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def atomic_torch_save(
    payload,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    try:
        torch.save(
            payload,
            temporary,
        )
        os.replace(
            temporary,
            path,
        )
    except Exception as exception:
        if temporary.exists():
            temporary.unlink()
        free_gib = (
            shutil.disk_usage(
                path.parent
            ).free
            / (1024 ** 3)
        )
        raise RuntimeError(
            f"Failed to save checkpoint: {path}\n"
            f"Free disk space: {free_gib:.2f} GiB"
        ) from exception


def clean_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        while new_key.startswith(
            "module."
        ):
            new_key = new_key[
                len("module.") :
            ]
        cleaned[new_key] = value
    return cleaned


def extract_transfer_state(
    checkpoint: Dict,
) -> Dict[str, torch.Tensor]:
    for key in (
        "model_state_dict",
        "base_model",
        "model",
        "state_dict",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return clean_state_dict(
                value
            )

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return clean_state_dict(
            checkpoint
        )

    raise RuntimeError(
        "No model state dict was found in the transfer checkpoint."
    )


def load_model_strict(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> Dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    state_dict = extract_transfer_state(
        checkpoint
    )

    model_state = model.state_dict()
    missing = sorted(
        set(model_state)
        - set(state_dict)
    )
    unexpected = sorted(
        set(state_dict)
        - set(model_state)
    )
    mismatched = sorted(
        key
        for key in (
            set(model_state)
            & set(state_dict)
        )
        if tuple(
            model_state[key].shape
        )
        != tuple(
            state_dict[key].shape
        )
    )

    if (
        missing
        or unexpected
        or mismatched
    ):
        lines = [
            "Transfer checkpoint does not exactly match the model."
        ]
        if missing:
            lines.append(
                "Missing: "
                + ", ".join(
                    missing[:20]
                )
            )
        if unexpected:
            lines.append(
                "Unexpected: "
                + ", ".join(
                    unexpected[:20]
                )
            )
        if mismatched:
            lines.append(
                "Shape mismatch: "
                + ", ".join(
                    mismatched[:20]
                )
            )
        raise RuntimeError(
            "\n".join(lines)
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )
    return checkpoint


def save_best_model(
    model: torch.nn.Module,
    epoch: int,
    metrics: Dict,
    metadata: Dict,
    path: Path,
) -> None:
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "model_state_dict": (
                model.state_dict()
            ),
            "metrics": metrics,
            "metadata": metadata,
        },
        path,
    )


def save_resume_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_value: float,
    best_metrics: Optional[Dict],
    metadata: Dict,
    path: Path,
) -> None:
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "best_value": float(
                best_value
            ),
            "best_metrics": (
                best_metrics
            ),
            "metadata": metadata,
        },
        path,
    )
