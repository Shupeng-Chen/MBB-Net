#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ShapeNet-55 PC-Grad training aligned with the project's original ShapeNet protocol.

Supported experiments
---------------------
1. PC-Grad + S2G:
       MBB_Model_ShapeNet(..., mbb_mode='s2g')

2. PC-Grad + Ours:
       MBB_Model_ShapeNet(..., mbb_mode='ours')

Key alignment choices
---------------------
- Uses the complete ShapeNet-55 train split (no 95%/5% split).
- Uses the existing ShapeNetDataset and ShapeNetLoss without modifying them.
- Uses ALPHA=0.4 and CD_SCALE=1.0 by default.
- Uses the existing ShapeNet test pipeline during training-time evaluation,
  matching the original S2G/Ours checkpoint-selection convention.
- Selects best_model.pth by the same average CD criterion.
- Uses native bridge modes directly; no monkey patch.
- Saves to new folders and never overwrites the old S2G/Ours checkpoints.
- Final paper metrics must still be recomputed with the separate official
  eight-view + FPS evaluator.

Examples
--------
PC-Grad + S2G:
    CUDA_VISIBLE_DEVICES=0 python -u \
      scripts/Ablation/train_pcgrad_s2g_ours_aligned.py \
      --mbb_mode s2g

PC-Grad + Ours:
    CUDA_VISIBLE_DEVICES=0 python -u \
      scripts/Ablation/train_pcgrad_s2g_ours_aligned.py \
      --mbb_mode ours
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
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------
# Public release paths / imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

from reproduce.ShapeNet.ShapeNet55.pcgrad_support.dataset import ShapeNetDataset
from reproduce.ShapeNet.ShapeNet55.pcgrad_support.loss import ShapeNetLoss
from reproduce.ShapeNet.ShapeNet55.pcgrad_support.metrics import (
    AverageMeter,
    calc_acc,
    calc_shapenet_metrics,
)
from models.shapenet_table2_model import ShapeNet55Table2Model


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[*] Global random seed fixed at {seed}")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ---------------------------------------------------------------------
# Safe checkpoint I/O
# ---------------------------------------------------------------------
def atomic_torch_save(payload, save_path: Path) -> None:
    """
    Write to a temporary file, then replace the destination after a complete
    write. Broken temporary files are removed automatically.
    """
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


def save_model_only(
    model: torch.nn.Module,
    epoch: int,
    mode: str,
    cfg: dict,
    selection_rule: str,
    metrics: Optional[dict],
    save_path: Path,
) -> None:
    """
    Lightweight evaluation checkpoint. The official evaluator can read the
    nested model_state_dict directly.
    """
    payload = {
        "epoch": int(epoch),
        "mbb_mode": mode,
        "optimizer_method": "PC-Grad",
        "selection_rule": selection_rule,
        "model_state_dict": model.state_dict(),
        "metrics_at_save": metrics,
        "config": cfg,
    }
    atomic_torch_save(payload, save_path)


def save_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    mode: str,
    cfg: dict,
    best_cd: float,
    save_path: Path,
) -> None:
    """
    One rotating full checkpoint for interruption recovery.
    """
    payload = {
        "epoch": int(epoch),
        "mbb_mode": mode,
        "optimizer_method": "PC-Grad",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_cd": float(best_cd),
        "config": cfg,
    }
    atomic_torch_save(payload, save_path)


def load_resume_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    expected_mode: str,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    mode_in_checkpoint = str(
        checkpoint.get("mbb_mode", "")
    ).lower()
    if mode_in_checkpoint != expected_mode:
        raise RuntimeError(
            f"Checkpoint mode '{mode_in_checkpoint}' does not match "
            f"requested mode '{expected_mode}'."
        )

    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(
            f"{checkpoint_path} is not a resumable checkpoint. "
            f"Missing keys: {missing}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = int(checkpoint["epoch"]) + 1
    best_cd = float(
        checkpoint.get("best_cd", float("inf"))
    )

    print(
        f"[*] Resumed from: {checkpoint_path}\n"
        f"    Next epoch: {start_epoch}\n"
        f"    Best CD:    {best_cd:.6f}"
    )
    return start_epoch, best_cd


# ---------------------------------------------------------------------
# Standard two-task PC-Grad
# ---------------------------------------------------------------------
def pcgrad_optimizer_step(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    loss_completion: torch.Tensor,
    loss_classification_weighted: torch.Tensor,
) -> Tuple[bool, float]:
    """
    Two-task PC-Grad.

    For conflicting gradients, each task gradient is projected against the
    original gradient of the other task, then both projected gradients are
    summed. Parameters used by only one task retain that task's gradient.
    """
    optimizer.zero_grad(set_to_none=True)

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    completion_grads = torch.autograd.grad(
        loss_completion,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    classification_grads = torch.autograd.grad(
        loss_classification_weighted,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )

    device = loss_completion.device
    dot_product = torch.zeros((), device=device)
    completion_norm_sq = torch.zeros((), device=device)
    classification_norm_sq = torch.zeros((), device=device)

    for grad_completion, grad_classification in zip(
        completion_grads,
        classification_grads,
    ):
        if grad_completion is not None:
            completion_norm_sq += torch.sum(
                grad_completion.detach() ** 2
            )

        if grad_classification is not None:
            classification_norm_sq += torch.sum(
                grad_classification.detach() ** 2
            )

        if (
            grad_completion is not None
            and grad_classification is not None
        ):
            dot_product += torch.sum(
                grad_completion.detach()
                * grad_classification.detach()
            )

    cosine = dot_product / (
        torch.sqrt(
            completion_norm_sq.clamp_min(1e-20)
            * classification_norm_sq.clamp_min(1e-20)
        )
        + 1e-12
    )

    has_conflict = bool((dot_product < 0).item())

    if has_conflict:
        completion_projection_coeff = (
            dot_product / (classification_norm_sq + 1e-12)
        )
        classification_projection_coeff = (
            dot_product / (completion_norm_sq + 1e-12)
        )
    else:
        completion_projection_coeff = torch.zeros(
            (),
            device=device,
        )
        classification_projection_coeff = torch.zeros(
            (),
            device=device,
        )

    for (
        parameter,
        grad_completion,
        grad_classification,
    ) in zip(
        parameters,
        completion_grads,
        classification_grads,
    ):
        if (
            grad_completion is None
            and grad_classification is None
        ):
            parameter.grad = None
            continue

        projected_completion = grad_completion
        projected_classification = grad_classification

        if (
            has_conflict
            and grad_completion is not None
            and grad_classification is not None
        ):
            projected_completion = (
                grad_completion
                - completion_projection_coeff
                * grad_classification
            )
            projected_classification = (
                grad_classification
                - classification_projection_coeff
                * grad_completion
            )

        if projected_completion is None:
            final_gradient = projected_classification
        elif projected_classification is None:
            final_gradient = projected_completion
        else:
            final_gradient = (
                projected_completion
                + projected_classification
            )

        parameter.grad = final_gradient.detach().clone()

    optimizer.step()

    return has_conflict, float(cosine.item())


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class ShapeNet55PCGradTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        cfg: dict,
        mbb_mode: str,
        output_dir: Path,
        resume_save_interval: int,
    ) -> None:
        self.model = model.cuda()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.mbb_mode = mbb_mode
        self.output_dir = output_dir
        self.resume_save_interval = resume_save_interval

        self.alpha = float(cfg.get("ALPHA", 0.4))
        self.cd_scale = 1.0

        # Reuse the project's original loss implementation unchanged.
        self.criterion = ShapeNetLoss(
            alpha=self.alpha,
            cd_scale=self.cd_scale,
        ).cuda()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg["LR"]),
            weight_decay=float(cfg["WEIGHT_DECAY"]),
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(cfg["STEP_SIZE"]),
            gamma=float(cfg["GAMMA"]),
        )

        self.best_cd = float("inf")

        self.best_model_path = (
            self.output_dir / "best_model.pth"
        )
        self.final_model_path = (
            self.output_dir / "final_model.pth"
        )
        self.last_model_path = (
            self.output_dir / "last_model.pth"
        )
        self.history_path = (
            self.output_dir / "history.jsonl"
        )

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()

        loss_meter = AverageMeter()
        cd_meter = AverageMeter()
        acc_meter = AverageMeter()
        conflict_meter = AverageMeter()
        cosine_meter = AverageMeter()

        progress = tqdm(
            self.train_loader,
            desc=(
                f"[PCGrad+{self.mbb_mode.upper()}] "
                f"Epoch {epoch} Train"
            ),
        )

        for partial, gt, label in progress:
            partial = partial.cuda(
                non_blocking=True
            ).float()
            gt = gt.cuda(
                non_blocking=True
            ).float()
            label = label.cuda(
                non_blocking=True
            ).long()

            out_list, logits = self.model(partial)

            if logits is None:
                raise RuntimeError(
                    "PC-Grad training requires classification logits."
                )

            total_loss, loss_comp, loss_ce = self.criterion(
                out_list,
                logits,
                gt,
                label,
            )

            # ShapeNetLoss returns raw CE as loss_ce. Weight it exactly as in:
            # total_loss = loss_comp + ALPHA * loss_ce
            loss_cls_weighted = self.alpha * loss_ce

            has_conflict, cosine = pcgrad_optimizer_step(
                optimizer=self.optimizer,
                model=self.model,
                loss_completion=loss_comp,
                loss_classification_weighted=loss_cls_weighted,
            )

            batch_size = partial.size(0)
            accuracy = calc_acc(logits, label)

            loss_meter.update(
                total_loss.item(),
                batch_size,
            )
            cd_meter.update(
                loss_comp.item(),
                batch_size,
            )
            acc_meter.update(
                accuracy,
                batch_size,
            )
            conflict_meter.update(
                1.0 if has_conflict else 0.0,
                1,
            )
            cosine_meter.update(
                cosine,
                1,
            )

            progress.set_postfix(
                {
                    "Loss": f"{loss_meter.avg:.4f}",
                    "CD": f"{cd_meter.avg:.4f}",
                    "Acc": f"{acc_meter.avg * 100:.2f}%",
                    "Conflict": (
                        f"{conflict_meter.avg * 100:.1f}%"
                    ),
                    "Cos": f"{cosine_meter.avg:.4f}",
                }
            )

        self.scheduler.step()

        return {
            "loss": loss_meter.avg,
            "completion_loss": cd_meter.avg,
            "accuracy": acc_meter.avg,
            "conflict_ratio": conflict_meter.avg,
            "cosine_similarity": cosine_meter.avg,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """
        Uses the same old ShapeNet test/evaluation pipeline used by the original
        S2G/Ours scripts, so checkpoint selection is directly aligned.
        """
        self.model.eval()

        metrics = {
            "simple": AverageMeter(),
            "moderate": AverageMeter(),
            "hard": AverageMeter(),
        }
        f1_metrics = {
            "simple": AverageMeter(),
            "moderate": AverageMeter(),
            "hard": AverageMeter(),
        }
        acc_meter = AverageMeter()

        progress = tqdm(
            self.test_loader,
            desc=(
                f"[PCGrad+{self.mbb_mode.upper()}] "
                f"Epoch {epoch} Eval"
            ),
        )

        for res_dict, gt, label in progress:
            gt = gt.cuda(
                non_blocking=True
            ).float()
            label = label.cuda(
                non_blocking=True
            ).long()

            moderate_partial = res_dict[
                "moderate"
            ].cuda(
                non_blocking=True
            ).float()

            _, logits = self.model(moderate_partial)
            accuracy = calc_acc(logits, label)
            acc_meter.update(
                accuracy,
                label.size(0),
            )

            for difficulty in (
                "simple",
                "moderate",
                "hard",
            ):
                partial = res_dict[
                    difficulty
                ].cuda(
                    non_blocking=True
                ).float()

                out_list, _ = self.model(partial)
                final_prediction = out_list[-1]

                cd_value, f1_value = (
                    calc_shapenet_metrics(
                        final_prediction,
                        gt,
                    )
                )

                metrics[difficulty].update(
                    float(cd_value),
                    partial.size(0),
                )
                f1_metrics[difficulty].update(
                    float(f1_value),
                    partial.size(0),
                )

        average_cd = (
            metrics["simple"].avg
            + metrics["moderate"].avg
            + metrics["hard"].avg
        ) / 3.0

        average_f1 = (
            f1_metrics["simple"].avg
            + f1_metrics["moderate"].avg
            + f1_metrics["hard"].avg
        ) / 3.0

        result = {
            "epoch": epoch,
            "cd": {
                "simple": metrics["simple"].avg,
                "moderate": metrics["moderate"].avg,
                "hard": metrics["hard"].avg,
                "average": average_cd,
            },
            "f1": {
                "simple": f1_metrics["simple"].avg,
                "moderate": f1_metrics["moderate"].avg,
                "hard": f1_metrics["hard"].avg,
                "average": average_f1,
            },
            "accuracy": acc_meter.avg,
        }

        print(f"\n--- Evaluation Results (Epoch {epoch}) ---")
        print(
            "CD(L2*1000) -> "
            f"Simple: {metrics['simple'].avg:.3f} | "
            f"Mod: {metrics['moderate'].avg:.3f} | "
            f"Hard: {metrics['hard'].avg:.3f} | "
            f"Avg: {average_cd:.3f}"
        )
        print(
            "F1-Score    -> "
            f"Simple: {f1_metrics['simple'].avg:.3f} | "
            f"Mod: {f1_metrics['moderate'].avg:.3f} | "
            f"Hard: {f1_metrics['hard'].avg:.3f} | "
            f"Avg: {average_f1:.3f}"
        )
        print(
            f"Accuracy    -> {acc_meter.avg * 100:.2f}%"
        )

        if average_cd < self.best_cd:
            self.best_cd = average_cd

            save_model_only(
                model=self.model,
                epoch=epoch,
                mode=self.mbb_mode,
                cfg=self.cfg,
                selection_rule=(
                    "minimum_average_cd_on_existing_"
                    "ShapeNet_test_evaluation_pipeline"
                ),
                metrics=result,
                save_path=self.best_model_path,
            )

            print(
                f"[*] New Best CD! Saved to "
                f"{self.best_model_path}"
            )

        return result

    def save_resume(self, epoch: int) -> None:
        save_resume_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            mode=self.mbb_mode,
            cfg=self.cfg,
            best_cd=self.best_cd,
            save_path=self.last_model_path,
        )

        print(
            f"[*] Resumable checkpoint saved: "
            f"{self.last_model_path}"
        )

    def save_final(self, epoch: int) -> None:
        save_model_only(
            model=self.model,
            epoch=epoch,
            mode=self.mbb_mode,
            cfg=self.cfg,
            selection_rule="fixed_final_epoch",
            metrics=None,
            save_path=self.final_model_path,
        )

        print(
            f"[*] Final epoch checkpoint saved: "
            f"{self.final_model_path}"
        )

    def append_history(
        self,
        epoch: int,
        train_result: dict,
        eval_result: Optional[dict],
        elapsed_seconds: float,
    ) -> None:
        row = {
            "epoch": epoch,
            "mbb_mode": self.mbb_mode,
            "train": train_result,
            "evaluation": eval_result,
            "best_cd": self.best_cd,
            "elapsed_seconds": elapsed_seconds,
        }

        with open(
            self.history_path,
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------
def train_one_mode(args, mode: str) -> None:
    set_seed(args.seed)

    config_path = (
        Path(args.config)
        if args.config
        else Path(PROJECT_ROOT)
        / "configs"
        / "ShapeNet55_config.yaml"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as handle:
        cfg = yaml.safe_load(handle)

    if args.data_root:
        cfg["DATA_ROOT"] = str(
            Path(args.data_root).expanduser().resolve()
        )
    else:
        configured = Path(cfg["DATA_ROOT"])
        cfg["DATA_ROOT"] = str(
            configured.resolve()
            if configured.is_absolute()
            else (PROJECT_ROOT / configured).resolve()
        )

    required_data = [
        Path(cfg["DATA_ROOT"]) / "train.txt",
        Path(cfg["DATA_ROOT"]) / "test.txt",
        Path(cfg["DATA_ROOT"]) / "shapenet_pc",
    ]
    missing_data = [str(x) for x in required_data if not x.exists()]
    if missing_data:
        raise FileNotFoundError(
            "ShapeNet-55 data root is incomplete:\n  "
            + "\n  ".join(missing_data)
        )

    # Keep original project code unchanged. Override only inside this script.
    cfg["ALPHA"] = float(args.alpha)
    cfg["NUM_CLASSES"] = 55

    train_dataset = ShapeNetDataset(
        cfg,
        subset="train",
    )
    test_dataset = ShapeNetDataset(
        cfg,
        subset="test",
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(cfg["NUM_WORKERS"]),
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(
            int(cfg["NUM_WORKERS"]) > 0
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=(
            args.test_workers > 0
        ),
    )

    model = ShapeNet55Table2Model(
        cfg,
        mode=mode,
    )

    output_dir = (
        Path(args.output_root)
        if args.output_root
        else Path(PROJECT_ROOT)
        / "outputs"
        / "ShapeNet55"
        / f"pcgrad_{mode}_aligned"
    )

    if (
        args.mbb_mode == "both"
        and args.output_root
    ):
        output_dir = output_dir / mode

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer = ShapeNet55PCGradTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=cfg,
        mbb_mode=mode,
        output_dir=output_dir,
        resume_save_interval=args.resume_save_interval,
    )

    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        start_epoch, trainer.best_cd = (
            load_resume_checkpoint(
                checkpoint_path=resume_path,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                expected_mode=mode,
            )
        )
    elif (
        args.auto_resume
        and trainer.last_model_path.is_file()
    ):
        start_epoch, trainer.best_cd = (
            load_resume_checkpoint(
                checkpoint_path=trainer.last_model_path,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                expected_mode=mode,
            )
        )

    maximum_epoch = int(
        args.epochs
        if args.epochs is not None
        else cfg["MAX_EPOCH"]
    )

    print("\n" + "=" * 88)
    print("ShapeNet-55 aligned PC-Grad training")
    print("=" * 88)
    print(f"Mode:                  PC-Grad + {mode.upper()}")
    print(f"Actual bridge mode:    {mode}")
    print("Monkey patch:          NO")
    print("Training split:        complete ShapeNet train split")
    print("Checkpoint selection:  existing ShapeNet test evaluation pipeline")
    print("Final reporting:       separate official 8-view + FPS evaluation")
    print(f"Train samples:         {len(train_dataset)}")
    print(f"Train batches:         {len(train_loader)}")
    print(f"Test samples:          {len(test_dataset)}")
    print(f"Alpha:                 {trainer.alpha}")
    print(f"CD scale:              {trainer.cd_scale}")
    print(f"Batch size:            {cfg['BATCH_SIZE']}")
    print(f"Epochs:                {maximum_epoch}")
    print(f"Evaluation frequency:  every 5 epochs and every epoch from 280")
    print(f"Resume save interval:  {args.resume_save_interval}")
    print(f"Output directory:      {output_dir}")
    print("=" * 88)

    for epoch in range(
        start_epoch,
        maximum_epoch + 1,
    ):
        epoch_start = time.time()

        train_result = trainer.train_epoch(epoch)
        eval_result = None

        should_evaluate = (
            epoch % 5 == 0
            or epoch >= 280
        )

        if should_evaluate:
            eval_result = trainer.validate(epoch)

        should_save_resume = (
            epoch == 1
            or epoch % args.resume_save_interval == 0
            or epoch == maximum_epoch
        )

        if should_save_resume:
            trainer.save_resume(epoch)

        if epoch == maximum_epoch:
            trainer.save_final(epoch)

        trainer.append_history(
            epoch=epoch,
            train_result=train_result,
            eval_result=eval_result,
            elapsed_seconds=(
                time.time() - epoch_start
            ),
        )

    print("\n" + "=" * 88)
    print(f"Training completed: PC-Grad + {mode.upper()}")
    print(f"Best checkpoint:  {trainer.best_model_path}")
    print(f"Final checkpoint: {trainer.final_model_path}")
    print(f"Resume checkpoint:{trainer.last_model_path}")
    print("=" * 88)
    print(
        "Use best_model.pth for the protocol-aligned PC-Grad comparison, "
        "then recompute final paper metrics with the official 8-view + FPS "
        "evaluation script."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train PC-Grad + S2G and PC-Grad + Ours "
            "with the original ShapeNet checkpoint-selection protocol."
        )
    )

    parser.add_argument(
        "--mbb_mode",
        required=True,
        choices=("s2g", "ours", "both"),
    )
    parser.add_argument(
        "--config",
        default=None,
    )

    parser.add_argument(
        "--data_root",
        default=None,
        help="ShapeNet-55 root containing train.txt, test.txt and shapenet_pc/.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--test_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--resume_save_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output_root",
        default=None,
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Explicit last_model.pth path. "
            "Available only for one mode."
        ),
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if (
        args.resume
        and args.mbb_mode == "both"
    ):
        raise ValueError(
            "--resume cannot be combined with --mbb_mode both."
        )

    if args.resume_save_interval <= 0:
        raise ValueError(
            "--resume_save_interval must be positive."
        )

    modes = (
        ("s2g", "ours")
        if args.mbb_mode == "both"
        else (args.mbb_mode,)
    )

    for mode in modes:
        train_one_mode(args, mode)


if __name__ == "__main__":
    main()