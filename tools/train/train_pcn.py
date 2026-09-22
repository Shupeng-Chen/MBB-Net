#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train the paper-facing PCN configurations used by MBB-Net.

Public modes
------------
completionOnly
    SnowflakeNet geometry encoder + decoder only.
    No semantic encoder, classifier, or MBB.

full
    Historical PCN sequential/cascaded bidirectional bridge.
    No stop-gradient.

ours
    Final canonical MBB-Net bridge:
    parallel G2S/S2G, with stop-gradient only on the G2S geometry source.

Checkpoint selection
--------------------
All three modes select the canonical training checkpoint by the lowest
validation CD-L1 x1000, matching the historical PCN ablation protocol.

Important
---------
Retraining outputs are written under outputs/PCN/training by default and do
NOT overwrite the distributed paper checkpoints in checkpoints/PCN.
Final paper metrics should be recomputed with:

    python main.py pcn-test --mode completionOnly full ours ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (
            (candidate / "main.py").is_file()
            and (candidate / "cfgs" / "PCN_config.yaml").is_file()
            and (candidate / "models" / "pcn_model.py").is_file()
            and (candidate / "datasets" / "PCN_dataset.py").is_file()
        ):
            return candidate
    raise RuntimeError("Cannot locate the MBB-Net public repository root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

from datasets.PCN_dataset import PCNDataset  # noqa: E402
from extensions.chamfer_dist import ChamferDistanceL1  # noqa: E402
from models.pcn_modes import (  # noqa: E402
    PAPER_PCN_MODES,
    build_pcn_model,
    normalize_pcn_mode,
)
from utils.PCN_loss import PCNLoss  # noqa: E402
from utils.PCN_metrics import AverageMeter, calc_acc  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def atomic_torch_save(payload: Any, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, save_path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        free_gib = shutil.disk_usage(save_path.parent).free / (1024 ** 3)
        raise RuntimeError(
            f"Checkpoint write failed: {save_path}\n"
            f"Free space in target filesystem: {free_gib:.2f} GiB"
        ) from exc


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cd_l1_x1000(
    metric: torch.nn.Module,
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    result = metric(pred.float(), gt.float())

    if isinstance(result, (tuple, list)) and len(result) >= 2:
        d1, d2 = result[0], result[1]
        if (
            isinstance(d1, torch.Tensor)
            and isinstance(d2, torch.Tensor)
            and d1.ndim == 2
            and d2.ndim == 2
        ):
            per_sample = 0.5 * (
                torch.sqrt(d1.clamp_min(0.0) + 1e-8).mean(dim=1)
                + torch.sqrt(d2.clamp_min(0.0) + 1e-8).mean(dim=1)
            )
            return per_sample.mean() * 1000.0

    if not isinstance(result, torch.Tensor):
        raise TypeError(
            "ChamferDistanceL1 returned unsupported type: "
            f"{type(result).__name__}"
        )

    return result.mean() * 1000.0


def gate_values(model) -> dict:
    if not hasattr(model, "mbb"):
        return {
            "sem_gate": None,
            "geo_gate": None,
        }

    return {
        "sem_gate": float(model.mbb.sem_gate.detach().item()),
        "geo_gate": float(model.mbb.geo_gate.detach().item()),
    }


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch: int,
    mode: str,
    cfg: dict,
    best_cd: float,
    best_acc: Optional[float],
    validation: Optional[dict],
) -> dict:
    return {
        "epoch": int(epoch),
        "pcn_mode": mode,
        "optimizer_method": "standard_training",
        "selection_rule": "lowest_validation_cd_l1_x1000",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_cd_l1_x1000": float(best_cd),
        "best_accuracy": (
            None if best_acc is None else float(best_acc)
        ),
        "validation_at_save": validation,
        "config": cfg,
    }


def lightweight_payload(
    model,
    epoch: int,
    mode: str,
    cfg: dict,
    selection_rule: str,
    metrics: dict,
) -> dict:
    return {
        "epoch": int(epoch),
        "pcn_mode": mode,
        "optimizer_method": "standard_training",
        "selection_rule": selection_rule,
        "model_state_dict": model.state_dict(),
        "metrics_at_save": metrics,
        "config": cfg,
    }


@torch.no_grad()
def validate(
    model,
    loader,
    cd_metric,
    device,
    mode: str,
) -> dict:
    model.eval()

    completion_only = mode == "completionOnly"

    cd_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(
        loader,
        desc=f"Validation/{mode}",
        leave=False,
    )

    for partial, gt, label in progress:
        partial = partial.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        _coarse, fine, logits = model(partial)

        cd_value = cd_l1_x1000(
            cd_metric,
            fine,
            gt,
        )

        batch_size = partial.shape[0]
        cd_meter.update(
            float(cd_value.item()),
            batch_size,
        )

        if completion_only:
            if logits is not None:
                raise RuntimeError(
                    "CompletionOnly unexpectedly returned logits."
                )
        else:
            if logits is None:
                raise RuntimeError(
                    f"{mode} returned no classification logits."
                )

            accuracy = calc_acc(
                logits,
                label,
            )
            acc_meter.update(
                float(accuracy),
                batch_size,
            )

    return {
        "cd_l1_x1000": float(cd_meter.avg),
        "accuracy": (
            None
            if completion_only
            else float(acc_meter.avg)
        ),
    }


def build_scheduler(optimizer, cfg):
    if "LR_MILESTONES" in cfg:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(cfg["LR_MILESTONES"]),
            gamma=float(cfg.get("GAMMA", 0.5)),
        )

    if "STEP_SIZE" in cfg:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg["STEP_SIZE"]),
            gamma=float(cfg.get("GAMMA", 0.5)),
        )

    raise KeyError(
        "PCN config must define LR_MILESTONES or STEP_SIZE."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train paper-facing PCN CompletionOnly / Full / MBB-Net modes."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=PAPER_PCN_MODES,
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT / "cfgs" / "PCN_config.yaml"
        ),
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help=(
            "Override PCN/ShapeNetCompletion DATASET_ROOT "
            "from the config."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--train_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--val_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Default: outputs/PCN/training/<mode>/seed_<seed>. "
            "Distributed checkpoints are never overwritten by default."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run one real forward/backward batch and exit.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = normalize_pcn_mode(args.mode)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the SnowflakeNet extensions."
        )

    set_seed(args.seed)
    device = torch.device("cuda")

    config_path = (
        Path(args.config)
        .expanduser()
        .resolve()
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        cfg = yaml.safe_load(handle)

    if args.data_root is not None:
        cfg["DATASET_ROOT"] = str(
            Path(args.data_root)
            .expanduser()
            .resolve()
        )

    cfg["NUM_CLASSES"] = 8

    epochs = int(
        args.epochs
        or cfg["EPOCHS"]
    )
    batch_size = int(
        args.batch_size
        or cfg["BATCH_SIZE"]
    )
    alpha = float(
        args.alpha
        if args.alpha is not None
        else cfg.get("ALPHA", 0.4)
    )

    if args.output_dir:
        output_dir = (
            Path(args.output_dir)
            .expanduser()
            .resolve()
        )
    else:
        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / "PCN"
            / "training"
            / mode
            / f"seed_{args.seed}"
        )

    best_model_path = (
        output_dir / "best_model.pth"
    )
    best_full_state_path = (
        output_dir / "best_model_full_state.pth"
    )
    best_accuracy_path = (
        output_dir / "best_accuracy_audit.pth"
    )
    last_full_path = (
        output_dir / "last_full.pth"
    )
    history_path = (
        output_dir / "history.jsonl"
    )

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
    ):
        if (
            args.resume is None
            and not args.overwrite
            and not args.dry_run
        ):
            raise RuntimeError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --resume or --overwrite."
            )

        if (
            args.overwrite
            and args.resume is None
            and not args.dry_run
        ):
            shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        f"PCN paper-mode training | mode={mode}"
    )
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Config:           {config_path}")
    print(f"Epochs:           {epochs}")
    print(f"Batch size:       {batch_size}")
    print(f"Alpha:            {alpha}")
    print(f"Output:           {output_dir}")
    print(
        "Selection rule:   lowest validation CD-L1 x1000"
    )
    print("=" * 88)

    train_dataset = PCNDataset(
        cfg,
        subset="train",
    )
    val_dataset = PCNDataset(
        cfg,
        subset="val",
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.train_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(
            args.train_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=(
            args.val_workers > 0
        ),
    )

    model = build_pcn_model(
        cfg,
        mode,
    ).to(device)

    completion_only = (
        mode == "completionOnly"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["LR"]),
        weight_decay=float(
            cfg["WEIGHT_DECAY"]
        ),
    )

    scheduler = build_scheduler(
        optimizer,
        cfg,
    )

    criterion = PCNLoss(
        cd_scale=float(
            cfg.get("CD_SCALE", 1000.0)
        ),
        use_mbb=not completion_only,
        alpha=alpha,
    ).to(device)

    cd_metric = (
        ChamferDistanceL1()
        .to(device)
    )

    start_epoch = 1
    best_cd = float("inf")
    best_acc = (
        None
        if completion_only
        else float("-inf")
    )

    if args.resume:
        resume_path = (
            Path(args.resume)
            .expanduser()
            .resolve()
        )
        raw = torch.load(
            resume_path,
            map_location="cpu",
        )

        required = (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
        )

        missing = [
            key
            for key in required
            if key not in raw
        ]

        if missing:
            raise RuntimeError(
                "Resume checkpoint lacks fields: "
                f"{missing}"
            )

        saved_mode = normalize_pcn_mode(
            raw.get("pcn_mode", mode)
        )

        if saved_mode != mode:
            raise RuntimeError(
                f"Resume mode={saved_mode} "
                f"does not match requested mode={mode}."
            )

        model.load_state_dict(
            raw["model_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            raw["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            raw["scheduler_state_dict"]
        )

        start_epoch = int(
            raw["epoch"]
        ) + 1
        best_cd = float(
            raw.get(
                "best_cd_l1_x1000",
                best_cd,
            )
        )

        if not completion_only:
            stored_acc = raw.get(
                "best_accuracy",
                best_acc,
            )
            if stored_acc is not None:
                best_acc = float(stored_acc)

        print(
            f"[Resume] {resume_path} "
            f"-> epoch {start_epoch}"
        )

    if args.dry_run:
        model.train()

        partial, gt, label = next(
            iter(train_loader)
        )

        partial = partial.to(
            device,
            non_blocking=True,
        )
        gt = gt.to(
            device,
            non_blocking=True,
        )
        label = label.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        coarse, fine, logits = model(
            partial
        )

        if completion_only:
            if logits is not None:
                raise RuntimeError(
                    "CompletionOnly returned logits."
                )

            loss, completion_value = (
                criterion(
                    coarse,
                    fine,
                    None,
                    gt,
                    None,
                )
            )
        else:
            if logits is None:
                raise RuntimeError(
                    f"{mode} returned no logits."
                )

            loss, completion_value = (
                criterion(
                    coarse,
                    fine,
                    logits,
                    gt,
                    label,
                )
            )

        loss.backward()

        print("[Dry run OK]")
        print(
            f"  mode:    {mode}"
        )
        print(
            f"  partial: {tuple(partial.shape)}"
        )
        print(
            f"  coarse:  {tuple(coarse.shape)}"
        )
        print(
            f"  fine:    {tuple(fine.shape)}"
        )
        print(
            "  logits:  "
            + (
                "None"
                if logits is None
                else str(tuple(logits.shape))
            )
        )
        print(
            f"  loss:    {float(loss.item()):.6f}"
        )
        print(
            f"  gates:   {gate_values(model)}"
        )
        return

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        epoch_start = time.time()

        model.train()

        loss_meter = AverageMeter()
        completion_meter = AverageMeter()
        acc_meter = AverageMeter()

        progress = tqdm(
            train_loader,
            desc=(
                f"{mode} Epoch "
                f"[{epoch}/{epochs}]"
            ),
            leave=False,
        )

        for partial, gt, label in progress:
            partial = partial.to(
                device,
                non_blocking=True,
            )
            gt = gt.to(
                device,
                non_blocking=True,
            )
            label = label.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            coarse, fine, logits = model(
                partial
            )

            if completion_only:
                if logits is not None:
                    raise RuntimeError(
                        "CompletionOnly returned logits."
                    )

                loss, completion_value = (
                    criterion(
                        coarse,
                        fine,
                        None,
                        gt,
                        None,
                    )
                )
            else:
                if logits is None:
                    raise RuntimeError(
                        f"{mode} returned no logits."
                    )

                loss, completion_value = (
                    criterion(
                        coarse,
                        fine,
                        logits,
                        gt,
                        label,
                    )
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch "
                    f"{epoch}: {loss.item()}"
                )

            loss.backward()
            optimizer.step()

            n = partial.shape[0]

            loss_meter.update(
                float(loss.item()),
                n,
            )

            if isinstance(
                completion_value,
                torch.Tensor,
            ):
                completion_scalar = float(
                    completion_value.mean().item()
                )
            else:
                completion_scalar = float(
                    completion_value
                )

            completion_meter.update(
                completion_scalar,
                n,
            )

            postfix = {
                "loss": f"{loss_meter.avg:.4f}",
                "comp": f"{completion_meter.avg:.4f}",
            }

            if not completion_only:
                accuracy = float(
                    calc_acc(
                        logits,
                        label,
                    )
                )
                acc_meter.update(
                    accuracy,
                    n,
                )
                postfix["acc"] = (
                    f"{100.0 * acc_meter.avg:.2f}%"
                )

            gates = gate_values(model)

            if gates["sem_gate"] is not None:
                postfix["sem_g"] = (
                    f"{gates['sem_gate']:.4f}"
                )
                postfix["geo_g"] = (
                    f"{gates['geo_gate']:.4f}"
                )

            progress.set_postfix(
                postfix
            )

        scheduler.step()

        should_validate = (
            epoch == 1
            or epoch % args.val_interval == 0
            or epoch == epochs
        )

        validation = None

        if should_validate:
            validation = validate(
                model,
                val_loader,
                cd_metric,
                device,
                mode,
            )

            if completion_only:
                print(
                    f"[Val {epoch}] "
                    f"CD-L1 x1000="
                    f"{validation['cd_l1_x1000']:.6f}"
                )
            else:
                print(
                    f"[Val {epoch}] "
                    f"CD-L1 x1000="
                    f"{validation['cd_l1_x1000']:.6f} | "
                    f"Acc="
                    f"{100.0 * validation['accuracy']:.4f}%"
                )

            if (
                validation["cd_l1_x1000"]
                < best_cd
            ):
                best_cd = validation[
                    "cd_l1_x1000"
                ]

                atomic_torch_save(
                    lightweight_payload(
                        model=model,
                        epoch=epoch,
                        mode=mode,
                        cfg=cfg,
                        selection_rule=(
                            "lowest_validation_"
                            "cd_l1_x1000"
                        ),
                        metrics=validation,
                    ),
                    best_model_path,
                )

                atomic_torch_save(
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        mode=mode,
                        cfg=cfg,
                        best_cd=best_cd,
                        best_acc=best_acc,
                        validation=validation,
                    ),
                    best_full_state_path,
                )

                print(
                    "[Saved best CD] "
                    f"{best_model_path}"
                )

            if (
                not completion_only
                and validation["accuracy"]
                > best_acc
            ):
                best_acc = validation[
                    "accuracy"
                ]

                atomic_torch_save(
                    lightweight_payload(
                        model=model,
                        epoch=epoch,
                        mode=mode,
                        cfg=cfg,
                        selection_rule=(
                            "highest_validation_"
                            "accuracy_audit_only"
                        ),
                        metrics=validation,
                    ),
                    best_accuracy_path,
                )

        atomic_torch_save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                mode=mode,
                cfg=cfg,
                best_cd=best_cd,
                best_acc=best_acc,
                validation=validation,
            ),
            last_full_path,
        )

        gates = gate_values(model)

        history_row = {
            "epoch": int(epoch),
            "mode": mode,
            "selection_rule": (
                "lowest_validation_cd_l1_x1000"
            ),
            "lr": float(
                optimizer.param_groups[0]["lr"]
            ),
            "train": {
                "loss": float(
                    loss_meter.avg
                ),
                "completion_objective": float(
                    completion_meter.avg
                ),
                "accuracy": (
                    None
                    if completion_only
                    else float(acc_meter.avg)
                ),
                **gates,
            },
            "validation": validation,
            "best_cd_l1_x1000": float(
                best_cd
            ),
            "best_accuracy": (
                None
                if best_acc is None
                else float(best_acc)
            ),
            "elapsed_seconds": float(
                time.time() - epoch_start
            ),
        }

        append_jsonl(
            history_path,
            history_row,
        )

        if completion_only:
            train_acc_text = "N/A"
        else:
            train_acc_text = (
                f"{100.0 * acc_meter.avg:.3f}%"
            )

        print(
            f"[Epoch {epoch}] "
            f"TrainLoss={loss_meter.avg:.5f} | "
            f"TrainComp={completion_meter.avg:.5f} | "
            f"TrainAcc={train_acc_text} | "
            f"LR={optimizer.param_groups[0]['lr']:.3e} | "
            f"Time={time.time() - epoch_start:.1f}s"
        )

    print("\nTraining finished.")
    print(
        f"Best-CD model:      {best_model_path}"
    )
    print(
        f"Best-CD full state: {best_full_state_path}"
    )

    if not completion_only:
        print(
            f"Best-Acc audit:     {best_accuracy_path}"
        )

    print(
        f"Last resumable:     {last_full_path}"
    )
    print(
        f"History:            {history_path}"
    )
    print(
        "\nFinal paper metrics must be produced by "
        "`python main.py pcn-test ...`, "
        "not copied from validation."
    )


if __name__ == "__main__":
    main()
