
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Render the paper-facing PCN qualitative samples directly from
the released MBB-Net checkpoints.

Supported public modes
----------------------
completionOnly
    SnowflakeNet completion branch only.

full
    Historical PCN sequential/cascaded bidirectional bridge.

ours
    Final canonical MBB-Net.

By default the exact eight manuscript samples are rendered:

    3, 160, 309, 452, 602, 753, 909, 1061

Outputs are written under:

    outputs/PCN/qualitative/from_checkpoints/

This script provides the checkpoint -> inference -> render
reproduction path for the three internal PCN configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm


# ============================================================
# 1. Project paths
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SNOW_ROOT = PROJECT_ROOT / "backbones" / "SnowflakeNet-main"
if str(SNOW_ROOT) not in sys.path:
    sys.path.append(str(SNOW_ROOT))

from datasets.PCN_dataset import PCNDataset  # noqa: E402
from models.pcn_modes import (  # noqa: E402
    PAPER_PCN_MODES,
    build_pcn_model,
    normalize_pcn_mode,
)


# ============================================================
# 2. Constants
# ============================================================

TAXONOMY = [
    "plane",
    "cabinet",
    "car",
    "chair",
    "lamp",
    "couch",
    "table",
    "watercraft",
]

# These are the exact rows used by the existing generate_PCN_grid.py.
DEFAULT_PAPER_INDICES = [
    3,     # plane
    160,   # cabinet
    309,   # car
    452,   # chair
    602,   # lamp
    753,   # couch
    909,   # table
    1061,  # watercraft
]


# ============================================================
# 3. Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 4. Public mode / dataset configuration
# ============================================================

DISPLAY_NAMES = {
    "completionOnly": "CompletionOnly",
    "full": "Full",
    "ours": "MBB-Net",
}

DEFAULT_CHECKPOINTS = {
    "completionOnly": (
        PROJECT_ROOT
        / "checkpoints"
        / "PCN"
        / "pcn_mbb_ablation_comp_only_best.pth"
    ),
    "full": (
        PROJECT_ROOT
        / "checkpoints"
        / "PCN"
        / "pcn_mbb_ablation_full_best.pth"
    ),
    "ours": (
        PROJECT_ROOT
        / "checkpoints"
        / "PCN"
        / "pcn_mbb_ablation_ours_best.pth"
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_cfg(
    config_path: str,
    data_root: str = None,
) -> Dict:

    cfg_path = (
        Path(config_path)
        .expanduser()
        .resolve()
    )

    with cfg_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    if data_root is not None:
        cfg["DATASET_ROOT"] = str(
            Path(data_root)
            .expanduser()
            .resolve()
        )

    root = cfg.get("DATASET_ROOT")

    if not root:
        raise RuntimeError(
            "PCN DATASET_ROOT is not configured. "
            "Use --data_root."
        )

    root_path = Path(
        root
    ).expanduser()

    if not root_path.is_absolute():
        root_path = (
            PROJECT_ROOT
            / root_path
        )

    root_path = root_path.resolve()

    if not root_path.is_dir():
        raise FileNotFoundError(
            f"PCN dataset root not found: {root_path}"
        )

    cfg["DATASET_ROOT"] = str(
        root_path
    )
    cfg["NUM_CLASSES"] = 8

    return cfg


# ============================================================
# 6. Strict checkpoint loading
# ============================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in [
            "base_model",
            "state_dict",
            "model_state_dict",
            "model",
            "net",
        ]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def clean_key(key: str) -> str:
    prefixes = [
        "module.",
        "base_model.",
        "net.",
    ]

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def load_checkpoint_strict(
    model: torch.nn.Module,
    weight_path: str,
) -> None:
    print(f"[Checkpoint] Loading: {weight_path}")

    checkpoint = torch.load(
        weight_path,
        map_location="cpu",
    )
    raw_state = extract_state_dict(checkpoint)

    if not isinstance(raw_state, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(raw_state)}"
        )

    cleaned = {
        clean_key(k): v
        for k, v in raw_state.items()
        if torch.is_tensor(v)
    }

    model_state = model.state_dict()

    missing = []
    shape_mismatch = []
    final_state = {}

    for key, tensor in model_state.items():
        if key not in cleaned:
            missing.append(key)
            continue

        source_tensor = cleaned[key]

        if tuple(source_tensor.shape) != tuple(tensor.shape):
            shape_mismatch.append(
                (
                    key,
                    tuple(source_tensor.shape),
                    tuple(tensor.shape),
                )
            )
            continue

        final_state[key] = source_tensor

    unexpected = [
        key for key in cleaned.keys()
        if key not in model_state
    ]

    if missing or unexpected or shape_mismatch:
        print("[Checkpoint audit]")
        print(f"  missing: {len(missing)}")
        print(f"  unexpected: {len(unexpected)}")
        print(f"  shape mismatch: {len(shape_mismatch)}")

        if missing:
            print("  missing examples:", missing[:10])

        if unexpected:
            print(
                "  unexpected examples:",
                unexpected[:10],
            )

        if shape_mismatch:
            print(
                "  shape mismatch examples:",
                shape_mismatch[:10],
            )

        raise RuntimeError(
            "Strict checkpoint audit failed. "
            "Do not render paper figures with partial weights."
        )

    model.load_state_dict(
        final_state,
        strict=True,
    )

    print(
        f"[Checkpoint] strict load OK: "
        f"{len(final_state)} tensors."
    )


# ============================================================
# 7. Model
# ============================================================

def build_model(
    cfg: Dict,
    mode: str,
    weight_path: str,
):
    mode = normalize_pcn_mode(
        mode
    )

    model = build_pcn_model(
        cfg,
        mode,
    ).cuda()

    load_checkpoint_strict(
        model,
        weight_path,
    )

    model.eval()

    return model


# ============================================================
# 8. Rendering
# ============================================================

def normalize_output_shape(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Return N x 3.
    """
    x = tensor.squeeze(0)

    if x.ndim != 2:
        raise RuntimeError(
            f"Unexpected prediction shape: {tuple(x.shape)}"
        )

    if x.shape[1] == 3:
        return x

    if x.shape[0] == 3:
        return x.transpose(0, 1).contiguous()

    raise RuntimeError(
        f"Cannot convert prediction shape "
        f"{tuple(x.shape)} to N x 3."
    )


def render_three_panel(
    partial: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    save_path: Path,
    title_name: str,
    elev: float,
    azim: float,
    dpi: int,
    pad_inches: float,
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(
        figsize=(15, 5),
        facecolor="white",
    )

    panels = [
        (
            partial,
            "#B22222",
            "Input",
        ),
        (
            pred,
            "#0000CD",
            f"{title_name} Result",
        ),
        (
            gt,
            "#228B22",
            "GT",
        ),
    ]

    for i, (points, color, title) in enumerate(
        panels,
        start=1,
    ):
        ax = fig.add_subplot(
            1,
            3,
            i,
            projection="3d",
        )

        point_size = (
            1.2
            if points.shape[0] > 5000
            else 3.5
        )

        ax.scatter(
            points[:, 0],
            points[:, 2],
            points[:, 1],
            c=color,
            s=point_size,
            alpha=0.80,
            edgecolors="none",
            depthshade=False,
        )

        # Keep exactly the same coordinate convention as the old PCN figure.
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        ax.set_zlim(-0.35, 0.35)

        ax.set_axis_off()

        ax.view_init(
            elev=elev,
            azim=azim,
        )

        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
        )

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )

    plt.close(fig)


# ============================================================
# 9. Sample selection
# ============================================================

def parse_indices(text: str) -> List[int]:
    if not text.strip():
        return list(DEFAULT_PAPER_INDICES)

    return [
        int(x.strip())
        for x in text.split(",")
        if x.strip()
    ]


def validate_paper_indices(
    dataset,
    indices: List[int],
) -> List[Tuple[int, str]]:
    info = []

    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            raise IndexError(
                f"Index {idx} out of range "
                f"for dataset length {len(dataset)}."
            )

        partial, gt, label = dataset[idx]
        label_id = int(label)

        if label_id < 0 or label_id >= len(TAXONOMY):
            raise RuntimeError(
                f"Unexpected PCN class label {label_id} "
                f"at dataset index {idx}."
            )

        info.append(
            (
                idx,
                TAXONOMY[label_id],
            )
        )

    return info


# ============================================================
# 10. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render paper-facing PCN samples directly "
            "from released checkpoints."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=PAPER_PCN_MODES,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional checkpoint override. "
            "Otherwise the released checkpoint for --mode is used."
        ),
    )

    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "cfgs"
            / "PCN_config.yaml"
        ),
    )

    parser.add_argument(
        "--data_root",
        default=None,
    )

    parser.add_argument(
        "--indices",
        default="",
        help=(
            "Comma-separated PCN test indices. "
            "Empty uses exactly "
            "3,160,309,452,602,753,909,1061."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "PCN"
            / "qualitative"
            / "from_checkpoints"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--elev",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--azim",
        type=float,
        default=135.0,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=(
            "Optional DPI override. Paper-compatible defaults: "
            "200 for CompletionOnly/Full, 300 for MBB-Net."
        ),
    )

    parser.add_argument(
        "--pad_inches",
        type=float,
        default=None,
        help=(
            "Optional savefig padding override. "
            "Paper-compatible defaults: 0.10 for "
            "CompletionOnly/Full, 0.02 for MBB-Net."
        ),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the released PCN model."
        )

    mode = normalize_pcn_mode(
        args.mode
    )

    display_name = DISPLAY_NAMES[
        mode
    ]

    # Historical paper-facing rendering profiles.
    #
    # CompletionOnly / Full were produced by the original
    # PCN_visualize.py:
    #
    #   dpi=200
    #   bbox_inches="tight"
    #   default Matplotlib pad_inches=0.1
    #
    # The final canonical MBB-Net column was regenerated with:
    #
    #   dpi=300
    #   bbox_inches="tight"
    #   pad_inches=0.02
    #
    if mode in {
        "completionOnly",
        "full",
    }:
        paper_dpi = 200
        paper_pad_inches = 0.10
    else:
        paper_dpi = 300
        paper_pad_inches = 0.02

    render_dpi = (
        args.dpi
        if args.dpi is not None
        else paper_dpi
    )

    render_pad_inches = (
        args.pad_inches
        if args.pad_inches is not None
        else paper_pad_inches
    )

    set_seed(
        args.seed
    )

    cfg = load_cfg(
        args.config,
        args.data_root,
    )

    checkpoint_path = (
        Path(args.checkpoint)
        .expanduser()
        .resolve()
        if args.checkpoint is not None
        else DEFAULT_CHECKPOINTS[mode].resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint_sha = sha256_file(
        checkpoint_path
    )

    model = build_model(
        cfg,
        mode,
        str(checkpoint_path),
    )

    dataset = PCNDataset(
        cfg,
        subset="test",
    )

    indices = parse_indices(
        args.indices
    )

    sample_info = validate_paper_indices(
        dataset,
        indices,
    )

    output_root = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    output_dir = (
        output_root
        / display_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 88)
    print(
        f"PCN qualitative inference | "
        f"mode={mode}"
    )
    print(
        f"Display name: {display_name}"
    )
    print(
        f"Checkpoint:   {checkpoint_path}"
    )
    print(
        f"SHA256:       {checkpoint_sha}"
    )
    print(
        f"Output:       {output_dir}"
    )
    print(
        f"Render DPI:   {render_dpi}"
    )
    print(
        f"Pad inches:   {render_pad_inches}"
    )
    print("=" * 88)

    manifest = []

    with torch.no_grad():

        for idx, class_name in tqdm(
            sample_info,
            desc=f"Render/{display_name}",
        ):

            partial, gt, label = (
                dataset[idx]
            )

            label_id = int(
                label
            )

            inp = (
                partial
                .unsqueeze(0)
                .cuda(
                    non_blocking=True
                )
            )

            _coarse, fine, logits = (
                model(inp)
            )

            pred = normalize_output_shape(
                fine
            )

            partial_np = (
                partial
                .detach()
                .cpu()
                .numpy()
            )

            pred_np = (
                pred
                .detach()
                .cpu()
                .numpy()
            )

            gt_np = (
                gt
                .detach()
                .cpu()
                .numpy()
            )

            pred_cls = (
                None
                if logits is None
                else int(
                    logits
                    .argmax(dim=1)
                    .item()
                )
            )

            file_name = (
                f"{class_name}_idx{idx}.png"
            )

            save_path = (
                output_dir
                / file_name
            )

            render_three_panel(
                partial=partial_np,
                pred=pred_np,
                gt=gt_np,
                save_path=save_path,
                title_name=display_name,
                elev=args.elev,
                azim=args.azim,
                dpi=render_dpi,
                pad_inches=render_pad_inches,
            )

            manifest.append({
                "dataset_index": idx,
                "class_name": class_name,
                "gt_label": label_id,
                "pred_label": pred_cls,
                "file": str(
                    save_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "file_sha256": (
                    sha256_file(
                        save_path
                    )
                ),
            })

    manifest_data = {
        "dataset": "PCN test split",
        "mode": mode,
        "display_name": display_name,
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_sha256": (
            checkpoint_sha
        ),
        "seed": args.seed,
        "view_elev": args.elev,
        "view_azim": args.azim,
        "dpi": render_dpi,
        "pad_inches": render_pad_inches,
        "render_profile": (
            "historical_paper_compatible"
            if args.dpi is None
            and args.pad_inches is None
            else "user_override"
        ),
        "samples": manifest,
    }

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 88)
    print(
        f"[PASS] {display_name} "
        f"rendered from released checkpoint."
    )
    print(
        f"Samples:  {len(manifest)}"
    )
    print(
        f"Manifest: {manifest_path}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
