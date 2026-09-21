#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train the PCN version of the exact ShapeNet-55 MBB-Net "Ours" bridge.

Ours forward definition
-----------------------
G2S branch:
    sem_feat = sem_attn(sem_token, geo_tokens.detach())

S2G branch:
    geo_feat = geo_attn(geo_tokens, sem_token)

Residual outputs:
    sem_out = sem_norm(sem_token + sem_gate * sem_feat)
    geo_out = geo_norm(geo_tokens + geo_gate * geo_feat)

Only the geometry tensor used by the G2S branch is detached. This matches the
ShapeNet-55 Ours implementation; it is NOT the ordinary PCN Full bridge.

Checkpoint selection
--------------------
To remain directly comparable with the existing PCN ablation experiments, the
canonical checkpoint is selected by the lowest validation CD-L1 x1000.
A separate best-accuracy checkpoint is also saved for auditing, but it must not
replace the canonical result after observing the test set.

Recommended command
-------------------
CUDA_VISIBLE_DEVICES=0 python -u \
  scripts/Ablation/PCN_ablation/train_pcn_ours_clean.py \
  2>&1 | tee paper/PCN_ours_training.log
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_project_root(start: Path) -> Path:
    """Find MBB-Net from this script location."""
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        if (
            (candidate / "cfgs" / "PCN_config.yaml").is_file()
            and (candidate / "models" / "pcn_model.py").is_file()
            and (candidate / "datasets" / "PCN_dataset.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        "Cannot locate the MBB-Net project root. Place this script anywhere "
        "inside the project, or pass --project_root explicitly."
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SNOW_ROOT = PROJECT_ROOT / "backbones" / "snowflakenet"
if str(SNOW_ROOT) not in sys.path:
    # Keep the project datasets package ahead of any backbone datasets package.
    sys.path.append(str(SNOW_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets.PCN_dataset import PCNDataset  # noqa: E402
from extensions.chamfer_dist import ChamferDistanceL1  # noqa: E402
from models.pcn_model import MBB_Model_PCN  # noqa: E402
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
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def apply_ours_patch(model: MBB_Model_PCN) -> None:
    """
    Apply the exact ShapeNet Ours bridge to one PCN model instance.

    Important:
    - Forward interaction remains bidirectional.
    - Only geo_tokens on the G2S branch are detached.
    - The S2G branch receives raw sem_token, matching ShapeNet Ours.
    - This is deliberately instance-local; it does not mutate the bridge class.
    """
    if not hasattr(model, "mbb"):
        raise AttributeError("MBB_Model_PCN has no 'mbb' module.")

    def ours_forward(
        bridge_self,
        sem_token: torch.Tensor,
        geo_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        geo_for_sem = geo_tokens.detach()
        sem_feat = bridge_self.sem_attn(sem_token, geo_for_sem)
        geo_feat = bridge_self.geo_attn(geo_tokens, sem_token)

        sem_token_out = bridge_self.sem_norm(
            sem_token + bridge_self.sem_gate * sem_feat
        )
        geo_tokens_out = bridge_self.geo_norm(
            geo_tokens + bridge_self.geo_gate * geo_feat
        )
        return sem_token_out, geo_tokens_out

    model.mbb.forward = types.MethodType(ours_forward, model.mbb)
    print(
        "[MBB mode] Ours applied: bidirectional forward interaction; "
        "geo_tokens detached only on the G2S branch."
    )


def atomic_torch_save(payload: Any, save_path: Path) -> None:
    """Safely write a checkpoint and report free space on write failure."""
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
        free_gib = shutil.disk_usage(save_path.parent).free / (1024**3)
        raise RuntimeError(
            f"Checkpoint write failed: {save_path}\n"
            f"Free space in target filesystem: {free_gib:.2f} GiB"
        ) from exc


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cd_l1_x1000(
    metric: torch.nn.Module,
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """
    Return a scalar validation CD-L1 x1000.

    This follows the existing PCN ablation validation behavior while handling
    either a scalar Chamfer module output or raw per-point distance tensors.
    """
    result = metric(pred.float(), gt.float())

    if isinstance(result, (tuple, list)) and len(result) >= 2:
        dist_pred_to_gt, dist_gt_to_pred = result[0], result[1]
        if (
            isinstance(dist_pred_to_gt, torch.Tensor)
            and isinstance(dist_gt_to_pred, torch.Tensor)
            and dist_pred_to_gt.ndim == 2
            and dist_gt_to_pred.ndim == 2
        ):
            per_sample = 0.5 * (
                torch.sqrt(dist_pred_to_gt.clamp_min(0.0) + 1e-8).mean(dim=1)
                + torch.sqrt(dist_gt_to_pred.clamp_min(0.0) + 1e-8).mean(dim=1)
            )
            return per_sample.mean() * 1000.0

    if not isinstance(result, torch.Tensor):
        raise TypeError(
            "ChamferDistanceL1 returned an unsupported value: "
            f"{type(result).__name__}"
        )
    return result.mean() * 1000.0


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    cfg: dict,
    best_cd: float,
    best_acc: float,
    validation: Optional[dict],
) -> dict:
    return {
        "epoch": int(epoch),
        "mbb_mode": "ours",
        "optimizer_method": "standard_joint_training",
        "selection_rule": "lowest_validation_cd_l1_x1000",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_cd_l1_x1000": float(best_cd),
        "best_accuracy": float(best_acc),
        "validation_at_save": validation,
        "config": cfg,
    }


def lightweight_eval_payload(
    model: torch.nn.Module,
    epoch: int,
    cfg: dict,
    selection_rule: str,
    validation: dict,
) -> dict:
    return {
        "epoch": int(epoch),
        "mbb_mode": "ours",
        "optimizer_method": "standard_joint_training",
        "selection_rule": selection_rule,
        "model_state_dict": model.state_dict(),
        "metrics_at_save": validation,
        "config": cfg,
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    cd_metric: torch.nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    cd_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(loader, desc="Validation", leave=False)
    for partial, gt, label in progress:
        partial = partial.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        _coarse, fine, logits = model(partial)
        if logits is None:
            raise RuntimeError("Ours returned no classification logits.")

        cd_value = cd_l1_x1000(cd_metric, fine, gt)
        accuracy = calc_acc(logits, label)
        batch_size = partial.shape[0]

        cd_meter.update(float(cd_value.item()), batch_size)
        acc_meter.update(float(accuracy), batch_size)

    return {
        "cd_l1_x1000": float(cd_meter.avg),
        "accuracy": float(acc_meter.avg),
    }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
):
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
        "PCN config must define LR_MILESTONES or STEP_SIZE for the scheduler."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PCN with the exact ShapeNet-55 MBB Ours bridge."
    )
    parser.add_argument(
        "--project_root",
        default=str(PROJECT_ROOT),
        help="Normally auto-detected; retained for experiment logging.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "cfgs" / "PCN_config.yaml"),
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override PCN/ShapeNetCompletion DATA_ROOT from the config.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_workers", type=int, default=8)
    parser.add_argument("--val_workers", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "checkpoints" / "PCN" / "ours_clean"),
    )
    parser.add_argument(
        "--canonical_checkpoint",
        default=str(
            PROJECT_ROOT
            / "checkpoints"
            / "PCN"
            / "pcn_mbb_ablation_ours_best.pth"
        ),
        help="Canonical lowest-validation-CD checkpoint used by the evaluator.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from ours_clean/last_full.pth or another full checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a fresh run to overwrite an existing Ours output directory.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run one forward/backward batch and exit without saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the local SnowflakeNet extensions.")

    set_seed(args.seed)
    device = torch.device("cuda")

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if args.data_root is not None:
        cfg["DATASET_ROOT"] = str(
            Path(args.data_root).expanduser().resolve()
        )

    cfg["NUM_CLASSES"] = 8
    epochs = int(args.epochs or cfg["EPOCHS"])
    batch_size = int(args.batch_size or cfg["BATCH_SIZE"])
    alpha = float(
        args.alpha
        if args.alpha is not None
        else cfg.get("ALPHA", 0.4)
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    canonical_path = Path(args.canonical_checkpoint).expanduser().resolve()
    history_path = output_dir / "history.jsonl"
    best_cd_full_path = output_dir / "best_cd_full.pth"
    best_acc_eval_path = output_dir / "best_accuracy.pth"
    last_full_path = output_dir / "last_full.pth"

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.resume is None and not args.overwrite and not args.dry_run:
            raise RuntimeError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --resume for the same run or --overwrite for an intentional "
                "fresh run. This guard prevents accidental checkpoint replacement."
            )
        if args.overwrite and args.resume is None and not args.dry_run:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("PCN MBB-Net Ours training")
    print(f"Project root:          {PROJECT_ROOT}")
    print(f"Config:                {config_path}")
    print(f"Epochs:                {epochs}")
    print(f"Batch size:            {batch_size}")
    print(f"Alpha:                 {alpha}")
    print(f"Canonical checkpoint:  {canonical_path}")
    print("Selection rule:        lowest validation CD-L1 x1000")
    print("=" * 88)

    train_dataset = PCNDataset(cfg, subset="train")
    val_dataset = PCNDataset(cfg, subset="val")

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
        persistent_workers=(args.train_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=(args.val_workers > 0),
    )

    model = MBB_Model_PCN(cfg, use_mbb=True).to(device)
    apply_ours_patch(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["LR"]),
        weight_decay=float(cfg["WEIGHT_DECAY"]),
    )
    scheduler = build_scheduler(optimizer, cfg)
    criterion = PCNLoss(
        cd_scale=float(cfg.get("CD_SCALE", 1000.0)),
        use_mbb=True,
        alpha=alpha,
    ).to(device)
    cd_metric = ChamferDistanceL1().to(device)

    start_epoch = 1
    best_cd = float("inf")
    best_acc = float("-inf")

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        raw = torch.load(resume_path, map_location="cpu")
        required = (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise RuntimeError(
                f"Resume checkpoint lacks required fields: {missing}"
            )

        model.load_state_dict(raw["model_state_dict"], strict=True)
        optimizer.load_state_dict(raw["optimizer_state_dict"])
        scheduler.load_state_dict(raw["scheduler_state_dict"])
        start_epoch = int(raw["epoch"]) + 1
        best_cd = float(raw.get("best_cd_l1_x1000", best_cd))
        best_acc = float(raw.get("best_accuracy", best_acc))
        print(f"[Resume] {resume_path} -> epoch {start_epoch}")

    if args.dry_run:
        model.train()
        partial, gt, label = next(iter(train_loader))
        partial = partial.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        coarse, fine, logits = model(partial)
        loss, completion_value = criterion(coarse, fine, logits, gt, label)
        loss.backward()

        print("[Dry run OK]")
        print(f"  partial: {tuple(partial.shape)}")
        print(f"  coarse:  {tuple(coarse.shape)}")
        print(f"  fine:    {tuple(fine.shape)}")
        print(f"  logits:  {tuple(logits.shape)}")
        print(f"  loss:    {float(loss.item()):.6f}")
        print(
            "  gates:   "
            f"sem={float(model.mbb.sem_gate.detach().item()):.6f}, "
            f"geo={float(model.mbb.geo_gate.detach().item()):.6f}"
        )
        return

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()

        loss_meter = AverageMeter()
        completion_meter = AverageMeter()
        acc_meter = AverageMeter()

        progress = tqdm(
            train_loader,
            desc=f"Ours Epoch [{epoch}/{epochs}]",
            leave=False,
        )

        for partial, gt, label in progress:
            partial = partial.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            coarse, fine, logits = model(partial)

            if logits is None:
                raise RuntimeError("Ours returned no classification logits.")

            loss, completion_value = criterion(
                coarse,
                fine,
                logits,
                gt,
                label,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}: {loss.item()}"
                )

            loss.backward()
            optimizer.step()

            batch_size_now = partial.shape[0]
            loss_meter.update(float(loss.item()), batch_size_now)

            if isinstance(completion_value, torch.Tensor):
                completion_scalar = float(completion_value.mean().item())
            else:
                completion_scalar = float(completion_value)
            completion_meter.update(completion_scalar, batch_size_now)

            accuracy = float(calc_acc(logits, label))
            acc_meter.update(accuracy, batch_size_now)

            progress.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                comp=f"{completion_meter.avg:.4f}",
                acc=f"{100.0 * acc_meter.avg:.2f}%",
                sem_g=f"{model.mbb.sem_gate.detach().item():.4f}",
                geo_g=f"{model.mbb.geo_gate.detach().item():.4f}",
            )

        scheduler.step()

        should_validate = (
            epoch == 1
            or epoch % args.val_interval == 0
            or epoch == epochs
        )
        validation: Optional[dict] = None

        if should_validate:
            validation = validate(model, val_loader, cd_metric, device)
            print(
                f"[Val {epoch}] "
                f"CD-L1 x1000={validation['cd_l1_x1000']:.6f} | "
                f"Acc={100.0 * validation['accuracy']:.4f}% | "
                f"sem_gate={model.mbb.sem_gate.detach().item():.6f} | "
                f"geo_gate={model.mbb.geo_gate.detach().item():.6f}"
            )

            if validation["cd_l1_x1000"] < best_cd:
                best_cd = validation["cd_l1_x1000"]

                canonical_payload = lightweight_eval_payload(
                    model=model,
                    epoch=epoch,
                    cfg=cfg,
                    selection_rule="lowest_validation_cd_l1_x1000",
                    validation=validation,
                )
                atomic_torch_save(canonical_payload, canonical_path)

                full_payload = checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    cfg=cfg,
                    best_cd=best_cd,
                    best_acc=best_acc,
                    validation=validation,
                )
                atomic_torch_save(full_payload, best_cd_full_path)
                print(f"[Saved canonical best CD] {canonical_path}")

            if validation["accuracy"] > best_acc:
                best_acc = validation["accuracy"]
                best_acc_payload = lightweight_eval_payload(
                    model=model,
                    epoch=epoch,
                    cfg=cfg,
                    selection_rule="highest_validation_accuracy_audit_only",
                    validation=validation,
                )
                atomic_torch_save(best_acc_payload, best_acc_eval_path)
                print(f"[Saved best accuracy audit] {best_acc_eval_path}")

        last_payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            cfg=cfg,
            best_cd=best_cd,
            best_acc=best_acc,
            validation=validation,
        )
        atomic_torch_save(last_payload, last_full_path)

        history_row = {
            "epoch": epoch,
            "mode": "ours",
            "selection_rule": "lowest_validation_cd_l1_x1000",
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": {
                "joint_loss": float(loss_meter.avg),
                "completion_objective": float(completion_meter.avg),
                "accuracy": float(acc_meter.avg),
                "sem_gate": float(model.mbb.sem_gate.detach().item()),
                "geo_gate": float(model.mbb.geo_gate.detach().item()),
            },
            "validation": validation,
            "best_cd_l1_x1000": float(best_cd),
            "best_accuracy": float(best_acc),
            "elapsed_seconds": float(time.time() - epoch_start),
        }
        append_jsonl(history_path, history_row)

        print(
            f"[Epoch {epoch}] "
            f"TrainLoss={loss_meter.avg:.5f} | "
            f"TrainComp={completion_meter.avg:.5f} | "
            f"TrainAcc={100.0 * acc_meter.avg:.3f}% | "
            f"LR={optimizer.param_groups[0]['lr']:.3e} | "
            f"Time={time.time() - epoch_start:.1f}s"
        )

    print("\nTraining finished.")
    print(f"Canonical checkpoint: {canonical_path}")
    print(f"Best-CD full state:   {best_cd_full_path}")
    print(f"Best-Acc audit only:  {best_acc_eval_path}")
    print(f"Last resumable state: {last_full_path}")
    print(f"History:              {history_path}")
    print(
        "\nFinal PCN paper metrics must be produced by "
        "tools/test/eval_pcn.py, not copied from validation."
    )


if __name__ == "__main__":
    main()
