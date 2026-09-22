#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.pcn_modes import build_pcn_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KITTI_PAPER_MODES = (
    "completionOnly",
    "ours",
)

DEFAULT_CHECKPOINTS = {
    "completionOnly": (
        PROJECT_ROOT
        / "checkpoints"
        / "PCN"
        / "pcn_mbb_ablation_comp_only_best.pth"
    ),
    "ours": (
        PROJECT_ROOT
        / "checkpoints"
        / "PCN"
        / "pcn_mbb_ablation_ours_best.pth"
    ),
}

EXPECTED_CKPT_SHA256 = {
    "completionOnly": (
        "f29e541ff5b581bc99129f358bd64a9a"
        "ad74498d781eb89dbe160df88a7e2b33"
    ),
    "ours": (
        "b32d7a03dd759741ded8ca22b4fb48e"
        "60d7f5039cee120050229875058f36940"
    ),
}

CFG = {
    "NUM_CLASSES": 8,
    "NUM_PC": 256,
    "NUM_P0": 512,
    "UP_FACTORS": [1, 4, 8],
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_state_dict(path: Path):
    try:
        raw = torch.load(
            str(path),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        raw = torch.load(
            str(path),
            map_location="cpu",
        )

    state = raw

    if isinstance(raw, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "base_model",
            "net",
        ):
            value = raw.get(key)

            if isinstance(value, dict) and value:
                state = value
                break

    if not isinstance(state, dict):
        raise RuntimeError(
            "Unable to locate checkpoint state_dict."
        )

    cleaned = {}

    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue

        while key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    if not cleaned:
        raise RuntimeError(
            "No model tensors found in checkpoint."
        )

    return cleaned


def load_cloud(
    path: Path,
    expected_points: int,
) -> np.ndarray:
    x = np.load(str(path))

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    if x.ndim != 2 or x.shape != (
        expected_points,
        3,
    ):
        raise RuntimeError(
            f"{path}: expected "
            f"({expected_points}, 3), got {x.shape}"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            f"{path}: contains NaN/Inf."
        )

    return np.ascontiguousarray(x)


def extract_final_prediction(output):
    pcs = []

    def collect(obj):
        if torch.is_tensor(obj):
            if (
                obj.ndim == 3
                and obj.shape[-1] == 3
            ):
                pcs.append(obj)

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                collect(item)

    collect(output)

    if not pcs:
        raise RuntimeError(
            "No [B,N,3] point-cloud output found."
        )

    return max(
        pcs,
        key=lambda x: int(x.shape[1]),
    )


def atomic_save_npy(
    path: Path,
    array: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open("wb") as f:
        np.save(f, array)

    os.replace(
        str(tmp),
        str(path),
    )


def valid_output(
    path: Path,
    expected_points: int,
) -> bool:
    if not path.is_file():
        return False

    try:
        x = np.load(
            str(path),
            mmap_mode="r",
        )
    except Exception:
        return False

    return (
        x.shape == (expected_points, 3)
        and x.dtype == np.float32
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Portable zero-shot KITTI inference "
            "for released PCN CompletionOnly and MBB-Net modes."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=KITTI_PAPER_MODES,
        help=(
            "PCN-trained paper mode to evaluate zero-shot on KITTI. "
            "completionOnly = completion branch only; "
            "ours = canonical MBB-Net."
        ),
    )

    parser.add_argument(
        "--fixed_input_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint override. If omitted, the released "
            "checkpoint for --mode is used and its SHA256 is enforced."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected_samples",
        type=int,
        default=2401,
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Smoke-test only; omit for full evaluation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for KITTI inference."
        )

    input_dir = (
        args.fixed_input_dir
        .expanduser()
        .resolve()
    )

    mode = args.mode

    using_default_checkpoint = (
        args.checkpoint is None
    )

    checkpoint_arg = (
        DEFAULT_CHECKPOINTS[mode]
        if using_default_checkpoint
        else args.checkpoint
    )

    ckpt = (
        checkpoint_arg
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not input_dir.is_dir():
        raise FileNotFoundError(
            input_dir
        )

    if not ckpt.is_file():
        raise FileNotFoundError(
            ckpt
        )

    inputs = sorted(
        input_dir.glob("*_input.npy")
    )

    if len(inputs) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} "
            f"fixed KITTI inputs, "
            f"found {len(inputs)}."
        )

    if args.max_samples is not None:
        inputs = inputs[
            : args.max_samples
        ]

    ckpt_sha = sha256_file(
        ckpt
    )

    print(
        "checkpoint SHA256 =",
        ckpt_sha,
    )

    expected_ckpt_sha = (
        EXPECTED_CKPT_SHA256[mode]
    )

    matches_released_checkpoint = (
        ckpt_sha == expected_ckpt_sha
    )

    if (
        using_default_checkpoint
        and not matches_released_checkpoint
    ):
        raise RuntimeError(
            f"Released {mode} checkpoint SHA256 mismatch. "
            f"Expected {expected_ckpt_sha}, got {ckpt_sha}."
        )

    if using_default_checkpoint:
        print(
            f"[PASS] released {mode} checkpoint SHA256"
        )
    else:
        print(
            "[INFO] custom checkpoint supplied; "
            "released-checkpoint SHA enforcement disabled."
        )
        print(
            "released checkpoint SHA256 =",
            expected_ckpt_sha,
        )
        print(
            "matches released checkpoint =",
            matches_released_checkpoint,
        )

    seed_all(
        args.seed
    )

    model = build_pcn_model(
        CFG,
        mode,
    ).cuda()

    state = load_state_dict(
        ckpt
    )

    result = model.load_state_dict(
        state,
        strict=True,
    )

    if (
        result.missing_keys
        or result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint load failed."
        )

    model.eval()

    print(
        f"[PASS] checkpoint strict-load: "
        f"{len(state)} tensors"
    )

    print(
        f"KITTI samples to run = {len(inputs)}"
    )

    print(
        f"Mode: {mode}"
    )

    print(
        "Protocol: PCN checkpoint -> "
        "KITTI fixed inputs; "
        "no KITTI fine-tuning; "
        "no ShapeNetCars fine-tuning; "
        "no target adaptation."
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = 0
    skipped = 0

    with torch.inference_mode():

        for input_path in tqdm(
            inputs,
            desc="KITTI inference",
        ):
            sample_id = (
                input_path.name[
                    :-len("_input.npy")
                ]
            )

            pred_path = (
                output_dir
                / f"{sample_id}_pred.npy"
            )

            copied_input_path = (
                output_dir
                / f"{sample_id}_input.npy"
            )

            if (
                not args.overwrite
                and valid_output(
                    pred_path,
                    16384,
                )
                and valid_output(
                    copied_input_path,
                    2048,
                )
            ):
                skipped += 1
                continue

            input_np = load_cloud(
                input_path,
                2048,
            )

            x = (
                torch.from_numpy(
                    input_np
                )
                .unsqueeze(0)
                .cuda()
            )

            output = model(x)

            pred = (
                extract_final_prediction(
                    output
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            if pred.shape != (
                16384,
                3,
            ):
                raise RuntimeError(
                    f"{sample_id}: "
                    f"unexpected prediction shape "
                    f"{pred.shape}"
                )

            if not np.isfinite(
                pred
            ).all():
                raise RuntimeError(
                    f"{sample_id}: "
                    "prediction contains NaN/Inf."
                )

            atomic_save_npy(
                pred_path,
                pred,
            )

            atomic_save_npy(
                copied_input_path,
                input_np,
            )

            written += 1

    manifest = {
        "mode": mode,
        "display_name": (
            "CompletionOnly"
            if mode == "completionOnly"
            else "MBB-Net"
        ),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_source": (
            "released_default"
            if using_default_checkpoint
            else "custom_override"
        ),
        "released_checkpoint_sha256": (
            expected_ckpt_sha
        ),
        "matches_released_checkpoint": (
            matches_released_checkpoint
        ),
        "seed": args.seed,
        "num_selected_samples": len(inputs),
        "written": written,
        "skipped": skipped,
        "input_points": 2048,
        "output_points": 16384,
        "source_training": "PCN",
        "kitti_adaptation": "None",
        "shapenet_cars_finetuning": "None",
        "target_adaptation": "None",
    }

    with (
        output_dir
        / "inference_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print(
        f"written = {written}"
    )

    print(
        f"skipped = {skipped}"
    )

    print(
        "[PASS] KITTI inference completed."
    )


if __name__ == "__main__":
    main()
