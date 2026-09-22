#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Safe ShapeNet-34 / Unseen-21 experiment runner.

Key fixes:
1. The final method uses mbb_mode='ours' rather than the historical
   mbb_mode='s2g' mapping.
2. The model always has a 34-class classifier. Unseen-21 only changes
   the dataset-reading configuration.
3. Training and evaluation outputs are physically separated.
4. There is no hidden need_resume trigger and no shape-matching partial load.
5. Every paper checkpoint is loaded with strict=True.
6. The final Ours checkpoint is saved to an isolated directory
   (default: checkpoints/ShapeNet/ShapeNet34/ours).
"""

import argparse
import copy
import datetime
import gc
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reproduce.ShapeNet.ShapeNet34_21.support.dataset import ShapeNet34Dataset  # noqa: E402
from models.shapenet_table2_model import ShapeNet55Table2Model  # noqa: E402
from reproduce.ShapeNet.ShapeNet34_21.support.shapenet34_trainer import (  # noqa: E402
    ShapeNet34Trainer,
)


VALID_MODES = ("baseline", "full", "ours", "g2s", "s2g")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep the same deterministic policy as the historical script.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------
def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """
    Support both:
      1. torch.save(model.state_dict(), path)
      2. torch.save({"model_state_dict": model.state_dict(), ...}, path)
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a state-dict or a dictionary containing one."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "base_model",
        "net",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value

    direct_state = {
        key: value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }

    if not direct_state:
        raise RuntimeError("No model tensors were found in the checkpoint.")

    return direct_state


def clean_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remove repeated DataParallel 'module.' prefixes."""
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key
        while new_key.startswith("module."):
            new_key = new_key[7:]

        if new_key in cleaned:
            raise RuntimeError(
                "Duplicate key after removing module prefix: "
                f"{new_key}"
            )

        cleaned[new_key] = value

    return cleaned


def load_checkpoint_strict(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> dict:
    """
    Strict paper-grade checkpoint loading.

    It refuses to continue when any parameter is missing, unexpected,
    or has an incompatible shape.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    raw_checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
    )

    state_dict = clean_state_dict(
        extract_state_dict(raw_checkpoint)
    )
    model_state = model.state_dict()

    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))

    shape_mismatch = sorted(
        key
        for key in set(model_state).intersection(state_dict)
        if tuple(model_state[key].shape)
        != tuple(state_dict[key].shape)
    )

    matched_numel = sum(
        model_state[key].numel()
        for key in set(model_state).intersection(state_dict)
        if tuple(model_state[key].shape)
        == tuple(state_dict[key].shape)
    )
    total_numel = sum(
        tensor.numel()
        for tensor in model_state.values()
    )
    coverage = 100.0 * matched_numel / max(total_numel, 1)

    print(f"[Checkpoint] {checkpoint_path}")
    print(f"  tensors in model:      {len(model_state)}")
    print(f"  tensors in checkpoint: {len(state_dict)}")
    print(f"  parameter coverage:    {coverage:.4f}%")

    if missing:
        print("  Missing keys:")
        for key in missing[:30]:
            print("   ", key)

    if unexpected:
        print("  Unexpected keys:")
        for key in unexpected[:30]:
            print("   ", key)

    if shape_mismatch:
        print("  Shape mismatches:")
        for key in shape_mismatch[:30]:
            print(
                f"    {key}: "
                f"model={tuple(model_state[key].shape)}, "
                f"checkpoint={tuple(state_dict[key].shape)}"
            )

    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Strict checkpoint loading failed. "
            "This checkpoint cannot be used for the paper."
        )

    model.load_state_dict(state_dict, strict=True)
    print("  strict load: OK")

    return raw_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


# ---------------------------------------------------------------------
# Model and data construction
# ---------------------------------------------------------------------
def build_model(
    model_cfg: dict,
    mode: str,
):
    """
    Build the released static ShapeNet architecture.

    The released ShapeNetTable2 model preserves the same
    baseline/full/G2S/S2G/Ours graph definitions used by the
    final checkpoints.
    """
    return ShapeNet55Table2Model(
        model_cfg,
        mode=mode,
    )


def build_configs(
    base_cfg: dict,
    dataset_name: str,
) -> Tuple[dict, dict, dict]:
    """
    Return three independent configurations.

    model_cfg:
        Always represents the ShapeNet-34-trained model and therefore
        always uses a 34-class classifier.

    dataset_cfg:
        Controls which split the dataset loader reads. Unseen-21 uses
        21 dataset labels, but does not alter the model classifier.

    trainer_cfg:
        Tells the trainer which evaluation split is active while keeping
        the model output dimension at 34.
    """
    model_cfg = copy.deepcopy(base_cfg)
    model_cfg["DATASET_TYPE"] = "34"
    model_cfg["NUM_CLASSES"] = 34

    dataset_cfg = copy.deepcopy(base_cfg)
    dataset_cfg["DATASET_TYPE"] = dataset_name
    dataset_cfg["NUM_CLASSES"] = (
        34 if dataset_name == "34" else 21
    )

    trainer_cfg = copy.deepcopy(base_cfg)
    trainer_cfg["DATASET_TYPE"] = dataset_name
    trainer_cfg["NUM_CLASSES"] = 34

    return model_cfg, dataset_cfg, trainer_cfg


def build_dataloaders(
    dataset_cfg: dict,
    trainer_cfg: dict,
    dataset_name: str,
    eval_only: bool,
):
    batch_size = int(trainer_cfg["BATCH_SIZE"])
    num_workers = int(trainer_cfg["NUM_WORKERS"])

    test_dataset = ShapeNet34Dataset(
        dataset_cfg,
        subset="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_loader = None

    if dataset_name == "34" and not eval_only:
        train_dataset = ShapeNet34Dataset(
            dataset_cfg,
            subset="train",
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    return train_loader, test_loader


# ---------------------------------------------------------------------
# Output safety and provenance
# ---------------------------------------------------------------------
def default_run_name(mode: str) -> str:
    # Isolate the final method from the historical "ours" directory.
    if mode == "ours":
        return "ours"
    return mode


def make_quarantine_dir(
    project_root: Path,
    run_name: str,
    dataset_name: str,
) -> Path:
    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        project_root
        / "checkpoints"
        / "ShapeNet"
        / "eval_trash_bin"
        / f"{run_name}_dataset{dataset_name}_{timestamp}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_training_manifest(
    checkpoint_dir: Path,
    script_path: Path,
    config_path: Path,
    args: argparse.Namespace,
    model_cfg: dict,
    dataset_cfg: dict,
    trainer_cfg: dict,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "script": str(script_path),
        "config": str(config_path),
        "arguments": vars(args),
        "model_cfg": model_cfg,
        "dataset_cfg": dataset_cfg,
        "trainer_cfg": trainer_cfg,
        "important_definition": {
            "ours_mode": "mbb_mode='ours'",
            "model_num_classes": 34,
            "hidden_auto_resume": False,
            "checkpoint_loading": "strict=True",
        },
    }

    manifest_path = checkpoint_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    shutil.copy2(
        str(script_path),
        str(checkpoint_dir / "script_snapshot.py"),
    )
    shutil.copy2(
        str(config_path),
        str(checkpoint_dir / "config_snapshot.yaml"),
    )

    print(f"[Provenance] {manifest_path}")
    print(
        "[Provenance] Script/config snapshots were saved "
        "with the checkpoint."
    )


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safe ShapeNet-34 training and ShapeNet-34/21 evaluation."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=VALID_MODES,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=("34", "21"),
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only load a checkpoint and evaluate it.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint used for eval_only or Unseen-21 evaluation. "
            "When omitted, the script uses the best checkpoint in "
            "the selected run directory."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Checkpoint directory name. Default: ours for "
            "mode=ours, otherwise the mode name."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow training when the target directory already "
            "contains best_model.pth. Use only intentionally."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    parser.add_argument(
        "--config",
        default=None,
        help="Optional ShapeNet34_21 YAML configuration.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="ShapeNet root containing shapenet_pc/.",
    )
    parser.add_argument(
        "--split_root",
        default=None,
        help=(
            "Split directory containing ShapeNet-34/ and "
            "ShapeNet-Unseen21/. Default: data/splits."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Retraining output directory. Default: "
            "outputs/ShapeNet34/<mode>/seed_<seed>."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.dataset == "21" and not args.eval_only:
        raise ValueError(
            "ShapeNet-Unseen21 is evaluation-only. "
            "Add --eval_only and load a ShapeNet-34 checkpoint."
        )

    if not args.eval_only and args.checkpoint is not None:
        raise ValueError(
            "--checkpoint is disabled during final training. "
            "The final Ours run must start from scratch."
        )

    set_seed(args.seed)

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else (
            PROJECT_ROOT
            / "configs"
            / "ShapeNet34_21_config.yaml"
        ).resolve()
    )

    with config_path.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    base_cfg["DATA_ROOT"] = str(
        Path(args.data_root).expanduser().resolve()
    )
    base_cfg["SPLIT_ROOT"] = str(
        Path(args.split_root).expanduser().resolve()
        if args.split_root
        else (PROJECT_ROOT / "data" / "splits").resolve()
    )

    model_cfg, dataset_cfg, trainer_cfg = build_configs(
        base_cfg,
        args.dataset,
    )

    run_name = (
        args.run_name
        if args.run_name
        else default_run_name(args.mode)
    )

    real_ckpt_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            PROJECT_ROOT
            / "outputs"
            / "ShapeNet34"
            / run_name
            / f"seed_{args.seed}"
        )
    )
    best_ckpt_path = real_ckpt_dir / "best_model.pth"

    if args.checkpoint:
        load_path = Path(args.checkpoint).expanduser()
        if not load_path.is_absolute():
            load_path = PROJECT_ROOT / load_path
        load_path = load_path.resolve()
    else:
        load_path = best_ckpt_path.resolve()

    print("\n" + "=" * 88)
    print("Safe ShapeNet-34 / Unseen-21 runner")
    print("=" * 88)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Dataset:          ShapeNet-{args.dataset}")
    print(f"Mode:             {args.mode}")
    print(f"Run name:         {run_name}")
    print(f"Evaluation only:  {args.eval_only}")
    print(f"Model classes:    {model_cfg['NUM_CLASSES']}")
    print(f"Dataset classes:  {dataset_cfg['NUM_CLASSES']}")
    print(f"Real ckpt dir:    {real_ckpt_dir}")
    print(f"Load checkpoint:  {load_path}")
    print(f"Seed:             {args.seed}")
    print("=" * 88 + "\n")

    # Training safety: do not silently overwrite an existing final run.
    if not args.eval_only:
        if best_ckpt_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Target checkpoint already exists:\n"
                f"{best_ckpt_path}\n"
                "Use a different --run_name, or pass --overwrite "
                "only after verifying the existing file."
            )

        real_ckpt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_training_manifest(
            checkpoint_dir=real_ckpt_dir,
            script_path=Path(__file__).resolve(),
            config_path=config_path,
            args=args,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            trainer_cfg=trainer_cfg,
        )

    train_loader, test_loader = build_dataloaders(
        dataset_cfg=dataset_cfg,
        trainer_cfg=trainer_cfg,
        dataset_name=args.dataset,
        eval_only=args.eval_only,
    )

    model = build_model(
        model_cfg=model_cfg,
        mode=args.mode,
    )

    trainer = ShapeNet34Trainer(
        model,
        train_loader,
        test_loader,
        trainer_cfg,
    )

    # Training writes only to the real directory.
    # Evaluation writes only to a quarantine directory.
    if args.eval_only:
        quarantine_dir = make_quarantine_dir(
            PROJECT_ROOT,
            run_name,
            args.dataset,
        )
        trainer.ckpt_dir = str(quarantine_dir)
        print(
            "[Safety] Evaluation output is redirected to: "
            f"{quarantine_dir}"
        )
    else:
        trainer.ckpt_dir = str(real_ckpt_dir)
        print(
            "[Safety] Training checkpoints will be saved to: "
            f"{real_ckpt_dir}"
        )

    try:
        # -------------------------------------------------------------
        # Final ShapeNet-34 training
        # -------------------------------------------------------------
        if not args.eval_only:
            max_epoch = int(trainer_cfg["MAX_EPOCH"])

            print(
                f"\n[Training] Starting ShapeNet-34 training "
                f"from epoch 1 to {max_epoch}."
            )
            print(
                "[Training] No hidden resume or partial checkpoint "
                "loading is enabled.\n"
            )

            for epoch in range(1, max_epoch + 1):
                trainer.train_epoch(epoch)

                if epoch % 5 == 0 or epoch >= 280:
                    trainer.validate(epoch)

            if not best_ckpt_path.is_file():
                raise FileNotFoundError(
                    "Training finished but best_model.pth was not found:\n"
                    f"{best_ckpt_path}"
                )

            print(
                "\n[Training] Best checkpoint saved successfully:"
            )
            print(best_ckpt_path)
            print(
                "[Training] SHA256:",
                sha256_file(best_ckpt_path),
            )

            # Prevent the final validation from overwriting the selected
            # best checkpoint.
            quarantine_dir = make_quarantine_dir(
                PROJECT_ROOT,
                run_name,
                "34_final",
            )
            trainer.ckpt_dir = str(quarantine_dir)

            print(
                "\n[Final validation] Strictly loading the selected "
                "best checkpoint."
            )
            load_checkpoint_strict(
                model,
                best_ckpt_path,
            )
            trainer.validate(
                epoch="Final Test 34",
            )

        # -------------------------------------------------------------
        # Eval-only ShapeNet-34 or Unseen-21 evaluation
        # -------------------------------------------------------------
        else:
            print(
                "\n[Evaluation] Strictly loading checkpoint:"
            )
            load_checkpoint_strict(
                model,
                load_path,
            )

            if args.dataset == "34":
                trainer.validate(
                    epoch="Eval Only Test 34",
                )
            else:
                print(
                    "[Evaluation] ShapeNet-21 is an unseen-category "
                    "completion test."
                )
                print(
                    "[Evaluation] Do not report its classification "
                    "accuracy because the 34-class training label space "
                    "does not correspond to the 21 unseen categories."
                )
                trainer.validate(
                    epoch="Zero-Shot Test 21",
                )

    finally:
        del model
        del trainer
        del test_loader

        if train_loader is not None:
            del train_loader

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n[Cleanup] GPU memory cleared.\n")


if __name__ == "__main__":
    main()
