#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Canonical historical ShapeNet-55 Table-2 ablation training entry.

This entry reproduces the historical training protocol for the modes
that were present in the original unified ShapeNet55 ablation script:

    baseline
    full
    g2s
    s2g
    ours

It intentionally DOES NOT use:
    - a dedicated torch.Generator for DataLoader
    - worker_init_fn
    - persistent_workers
    - pin_memory

Those additions alter the historical RNG/data trajectory.

The released checkpoint evaluator remains:
    reproduce/ShapeNet/ShapeNet55/eval_table2.py

This script saves retrained checkpoints to a separate output directory
and never overwrites the released paper checkpoints by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


CANONICAL_MODES = (
    "baseline",
    "full",
    "g2s",
    "s2g",
    "ours",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a canonical historical "
            "ShapeNet-55 Table-2 ablation mode."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=CANONICAL_MODES,
    )

    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "ShapeNet55_config.yaml"
        ),
    )

    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "ShapeNet55 root containing "
            "train.txt, test.txt and shapenet_pc/. "
            "If omitted, DATA_ROOT from the config "
            "is resolved relative to the repository."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Retraining output directory. "
            "Default: outputs/ShapeNet55/"
            "Table2_historical/<mode>/seed_42"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing best_model.pth "
            "inside --output-dir."
        ),
    )

    return parser.parse_args()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(
        self,
        val,
        n=1,
    ):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += int(n)

        self.avg = (
            self.sum / self.count
            if self.count
            else 0.0
        )


def calc_acc(
    logits,
    labels,
):
    if logits is None:
        return 0.0

    preds = torch.argmax(
        logits,
        dim=1,
    )

    correct = (
        preds == labels
    ).sum().item()

    return (
        correct
        / labels.size(0)
    )


def compute_squared_dists_chunked(
    xyz1,
    xyz2,
    chunk_size=256,
):
    """
    Historical validation metric implementation.
    Returns squared nearest-neighbor distances.
    """

    batch_size, n, _ = xyz1.shape
    _, m, _ = xyz2.shape

    dist1 = torch.zeros(
        batch_size,
        n,
        device=xyz1.device,
    )

    dist2 = torch.zeros(
        batch_size,
        m,
        device=xyz2.device,
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n,
        )

        diff = (
            xyz1[:, start:end]
            .unsqueeze(2)
            - xyz2.unsqueeze(1)
        )

        d = torch.sum(
            diff ** 2,
            dim=-1,
        )

        dist1[:, start:end] = (
            torch.min(
                d,
                dim=2,
            )[0]
        )

        del diff, d

    for start in range(
        0,
        m,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            m,
        )

        diff = (
            xyz2[:, start:end]
            .unsqueeze(2)
            - xyz1.unsqueeze(1)
        )

        d = torch.sum(
            diff ** 2,
            dim=-1,
        )

        dist2[:, start:end] = (
            torch.min(
                d,
                dim=2,
            )[0]
        )

        del diff, d

    return dist1, dist2


def calc_shapenet_metrics(
    pred,
    gt,
):
    pred = (
        pred
        .contiguous()
        .float()
    )

    gt = (
        gt
        .contiguous()
        .float()
    )

    with torch.no_grad():

        dist1, dist2 = (
            compute_squared_dists_chunked(
                pred,
                gt,
                chunk_size=256,
            )
        )

        cd_val = (
            torch.mean(dist1)
            + torch.mean(dist2)
        ) * 1000.0

        d1 = torch.sqrt(
            dist1 + 1e-12
        )

        d2 = torch.sqrt(
            dist2 + 1e-12
        )

        precision = torch.mean(
            (d1 < 0.01).float(),
            dim=1,
        )

        recall = torch.mean(
            (d2 < 0.01).float(),
            dim=1,
        )

        fscore = (
            2
            * precision
            * recall
            / (
                precision
                + recall
                + 1e-8
            )
        )

        return (
            cd_val.item(),
            fscore.mean().item(),
        )


class ShapeNetLoss(nn.Module):
    """
    Historical ShapeNet training objective:

        sum(CD-L2 over every decoder output)
        + alpha * CrossEntropy
    """

    def __init__(
        self,
        chamfer_cls,
        alpha=0.4,
        cd_scale=1.0,
    ):
        super().__init__()

        self.alpha = float(alpha)
        self.cd_scale = float(
            cd_scale
        )

        self.cd_l2 = chamfer_cls()
        self.ce = nn.CrossEntropyLoss()

    def _safe_cd_l2(
        self,
        pc1,
        pc2,
    ):
        result = self.cd_l2(
            pc1.contiguous().float(),
            pc2.contiguous().float(),
        )

        if isinstance(
            result,
            (list, tuple),
        ):
            d1, d2 = result

            cd = (
                torch.mean(
                    d1,
                    dim=1,
                )
                + torch.mean(
                    d2,
                    dim=1,
                )
            )

            return cd.mean()

        return result.mean()

    def forward(
        self,
        out_list,
        logits,
        gt,
        label,
    ):
        # Historical implementation intentionally preserved.
        #
        # Do not rewrite the in-place += accumulation as
        # ``loss_comp = loss_comp + ...``. Although mathematically
        # equivalent in the forward pass, that changes the autograd
        # graph used by the historical training implementation.
        loss_comp = torch.tensor(0.0).cuda()

        for pc in out_list:
            loss_comp += self._safe_cd_l2(pc, gt)

        loss_comp = loss_comp * self.cd_scale

        loss_ce = torch.tensor(0.0).cuda()

        if logits is not None:
            loss_ce = self.ce(logits, label)

        total_loss = loss_comp + self.alpha * loss_ce

        return total_loss, loss_comp, loss_ce


def set_seed(
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    epoch,
):
    model.train()

    loss_meter = AverageMeter()
    cd_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} Train",
    )

    for (
        partial,
        gt,
        label,
    ) in progress:

        partial = partial.cuda()
        gt = gt.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        out_list, logits = model(
            partial
        )

        (
            loss,
            loss_comp,
            loss_ce,
        ) = criterion(
            out_list,
            logits,
            gt,
            label,
        )

        loss.backward()
        optimizer.step()

        acc = calc_acc(
            logits,
            label,
        )

        batch_size = partial.size(0)

        loss_meter.update(
            loss.item(),
            batch_size,
        )

        cd_meter.update(
            loss_comp.item(),
            batch_size,
        )

        acc_meter.update(
            acc,
            batch_size,
        )

        progress.set_postfix(
            {
                "Loss":
                    f"{loss_meter.avg:.4f}",
                "CD":
                    f"{cd_meter.avg:.4f}",
                "Acc":
                    f"{acc_meter.avg * 100:.2f}%",
            }
        )

    scheduler.step()

    return {
        "loss":
            loss_meter.avg,
        "completion_loss":
            cd_meter.avg,
        "accuracy":
            acc_meter.avg,
    }


def validate(
    model,
    loader,
    epoch,
):
    """
    Historical checkpoint-selection validation.

    IMPORTANT:
    This is intentionally NOT the newer official
    8-view evaluator. The historical checkpoint was
    selected using this original validation path.
    """

    model.eval()

    cd_metrics = {
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

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch} Eval",
        )

        for (
            res_dict,
            gt,
            label,
        ) in progress:

            gt = gt.cuda()
            label = label.cuda()

            # Historical trainer performed a separate
            # moderate forward for classification.
            moderate_partial = (
                res_dict["moderate"]
                .cuda()
            )

            _, logits = model(
                moderate_partial
            )

            acc = calc_acc(
                logits,
                label,
            )

            acc_meter.update(
                acc,
                label.size(0),
            )

            # Then it performed the three completion
            # forwards, including moderate again.
            for difficulty in (
                "simple",
                "moderate",
                "hard",
            ):
                partial = (
                    res_dict[difficulty]
                    .cuda()
                )

                out_list, _ = model(
                    partial
                )

                if not out_list:
                    raise RuntimeError(
                        f"Mode returned no completion "
                        f"output during historical "
                        f"validation: {difficulty}"
                    )

                final_pred = out_list[-1]

                (
                    cd_val,
                    f1_val,
                ) = calc_shapenet_metrics(
                    final_pred,
                    gt,
                )

                batch_size = (
                    partial.size(0)
                )

                cd_metrics[
                    difficulty
                ].update(
                    cd_val,
                    batch_size,
                )

                f1_metrics[
                    difficulty
                ].update(
                    f1_val,
                    batch_size,
                )

    avg_cd = sum(
        cd_metrics[d].avg
        for d in (
            "simple",
            "moderate",
            "hard",
        )
    ) / 3.0

    avg_f1 = sum(
        f1_metrics[d].avg
        for d in (
            "simple",
            "moderate",
            "hard",
        )
    ) / 3.0

    result = {
        "epoch": epoch,
        "cd_l2_x1000": {
            d: cd_metrics[d].avg
            for d in (
                "simple",
                "moderate",
                "hard",
            )
        },
        "cd_l2_x1000_avg":
            avg_cd,
        "f1_at_1pct": {
            d: f1_metrics[d].avg
            for d in (
                "simple",
                "moderate",
                "hard",
            )
        },
        "f1_at_1pct_avg":
            avg_f1,
        "accuracy":
            acc_meter.avg,
    }

    print()
    print(
        f"--- Evaluation Results "
        f"(Epoch {epoch}) ---"
    )

    print(
        "CD(L2*1000) -> "
        f"Simple: {cd_metrics['simple'].avg:.3f} | "
        f"Mod: {cd_metrics['moderate'].avg:.3f} | "
        f"Hard: {cd_metrics['hard'].avg:.3f} | "
        f"Avg: {avg_cd:.3f}"
    )

    print(
        "F1-Score    -> "
        f"Simple: {f1_metrics['simple'].avg:.3f} | "
        f"Mod: {f1_metrics['moderate'].avg:.3f} | "
        f"Hard: {f1_metrics['hard'].avg:.3f} | "
        f"Avg: {avg_f1:.3f}"
    )

    print(
        "Accuracy    -> "
        f"{acc_meter.avg * 100:.2f}%"
    )

    return result


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for ShapeNet55 training."
        )

    # Lazy project imports: --help never imports model/CUDA extensions.
    from data.shapenet55_train import (
        ShapeNet55HistoricalTrainDataset,
    )

    from models.shapenet_table2_model import (
        ShapeNet55Table2Model,
    )

    try:
        from extensions.chamfer_dist import (
            ChamferDistanceL2,
        )
    except Exception as exc:
        raise RuntimeError(
            "Cannot import the released Chamfer CUDA "
            "extension. Build/install "
            "extensions/chamfer_dist first."
        ) from exc

    config_path = (
        Path(args.config)
        .expanduser()
        .resolve()
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        data_root = (
            Path(args.data_root)
            .expanduser()
            .resolve()
        )
    else:
        configured = Path(
            cfg["DATA_ROOT"]
        )

        data_root = (
            configured
            if configured.is_absolute()
            else PROJECT_ROOT / configured
        ).resolve()

    required = (
        data_root / "train.txt",
        data_root / "test.txt",
        data_root / "shapenet_pc",
    )

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "ShapeNet55 data root is incomplete:\n  "
            + "\n  ".join(missing)
            + "\nUse --data-root /path/to/ShapeNet55"
        )

    cfg["DATA_ROOT"] = str(data_root)

    mode = args.mode

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
            / "ShapeNet55"
            / "Table2_historical"
            / mode
            / "seed_42"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        output_dir
        / "best_model.pth"
    )

    if (
        best_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{best_path} already exists. "
            "Use another --output-dir or "
            "--overwrite."
        )

    # --------------------------------------------------------
    # Historical RNG protocol begins here.
    # No torch.Generator.
    # No worker_init_fn.
    # --------------------------------------------------------
    set_seed(42)

    train_dataset = (
        ShapeNet55HistoricalTrainDataset(
            data_root=data_root,
            subset="train",
            num_classes=int(
                cfg["NUM_CLASSES"]
            ),
            num_partial_points=int(
                cfg["NUM_PARTIAL_POINTS"]
            ),
            num_gt_points=int(
                cfg["NUM_GT_POINTS"]
            ),
        )
    )

    test_dataset = (
        ShapeNet55HistoricalTrainDataset(
            data_root=data_root,
            subset="test",
            num_classes=int(
                cfg["NUM_CLASSES"]
            ),
            num_partial_points=int(
                cfg["NUM_PARTIAL_POINTS"]
            ),
            num_gt_points=int(
                cfg["NUM_GT_POINTS"]
            ),
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            cfg["BATCH_SIZE"]
        ),
        shuffle=True,
        num_workers=int(
            cfg["NUM_WORKERS"]
        ),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=24,
        shuffle=False,
        num_workers=8,
    )

    # Historical order:
    # seed -> datasets -> loaders -> model.
    model = ShapeNet55Table2Model(
        cfg,
        mode=mode,
    ).cuda()

    criterion = ShapeNetLoss(
        chamfer_cls=ChamferDistanceL2,
        alpha=float(
            cfg.get(
                "ALPHA",
                0.4,
            )
        ),
        cd_scale=1.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["LR"]),
        weight_decay=float(
            cfg["WEIGHT_DECAY"]
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(
                cfg["STEP_SIZE"]
            ),
            gamma=float(
                cfg["GAMMA"]
            ),
        )
    )

    protocol = {
        "dataset": "ShapeNet55",
        "mode": mode,
        "seed": 42,
        "epochs": int(
            cfg["MAX_EPOCH"]
        ),
        "batch_size": int(
            cfg["BATCH_SIZE"]
        ),
        "train_workers": int(
            cfg["NUM_WORKERS"]
        ),
        "test_batch_size": 24,
        "test_workers": 8,
        "optimizer": "AdamW",
        "lr": float(
            cfg["LR"]
        ),
        "weight_decay": float(
            cfg["WEIGHT_DECAY"]
        ),
        "scheduler": "StepLR",
        "step_size": int(
            cfg["STEP_SIZE"]
        ),
        "gamma": float(
            cfg["GAMMA"]
        ),
        "alpha": float(
            cfg.get(
                "ALPHA",
                0.4,
            )
        ),
        "eval_schedule": (
            "epoch % 5 == 0 or epoch >= 280"
        ),
        "selection": (
            "minimum historical validation "
            "average CD-L2 over "
            "simple/moderate/hard"
        ),
        "dataloader_generator": None,
        "worker_init_fn": None,
        "shuffle": True,
        "drop_last": True,
        "config": str(
            config_path
        ),
        "data_root": str(
            data_root
        ),
    }

    (
        output_dir
        / "protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("ShapeNet55 historical canonical training")
    print("=" * 88)

    print("Mode       :", mode)
    print("Seed       : 42")
    print("Data root  :", data_root)
    print("Train size :", len(train_dataset))
    print("Test size  :", len(test_dataset))
    print("Output     :", output_dir)

    print(
        "RNG        : global RNG only; "
        "generator=None; worker_init_fn=None"
    )

    print(
        "Selection  : minimum historical "
        "validation average CD-L2"
    )

    best_cd = float("inf")
    best_epoch = None
    best_metrics = None

    history_path = (
        output_dir
        / "history.jsonl"
    )

    # Fresh retraining record.
    history_path.write_text(
        "",
        encoding="utf-8",
    )

    max_epoch = int(
        cfg["MAX_EPOCH"]
    )

    for epoch in range(
        1,
        max_epoch + 1,
    ):
        train_result = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )

        record = {
            "epoch": epoch,
            "train": train_result,
            "evaluation": None,
        }

        if (
            epoch % 5 == 0
            or epoch >= 280
        ):
            evaluation = validate(
                model=model,
                loader=test_loader,
                epoch=epoch,
            )

            record[
                "evaluation"
            ] = evaluation

            current_cd = float(
                evaluation[
                    "cd_l2_x1000_avg"
                ]
            )

            if current_cd < best_cd:

                best_cd = current_cd
                best_epoch = epoch
                best_metrics = evaluation

                # Preserve historical raw-state-dict
                # checkpoint format.
                torch.save(
                    model.state_dict(),
                    best_path,
                )

                print(
                    "[*] New Best CD! "
                    f"Saved to {best_path}"
                )

                metadata = {
                    "mode": mode,
                    "seed": 42,
                    "best_epoch":
                        best_epoch,
                    "best_cd_l2_x1000":
                        best_cd,
                    "metrics_at_save":
                        best_metrics,
                    "selection_rule":
                        "min historical validation avg CD",
                    "checkpoint_format":
                        "raw state_dict",
                }

                (
                    output_dir
                    / "best_model.meta.json"
                ).write_text(
                    json.dumps(
                        metadata,
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        with history_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    if not best_path.is_file():
        raise RuntimeError(
            "Training completed but no "
            "best checkpoint was saved."
        )

    print()
    print("=" * 88)
    print("FINAL HISTORICAL VALIDATION")
    print("=" * 88)

    state = torch.load(
        best_path,
        map_location="cpu",
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.cuda()

    final_result = validate(
        model=model,
        loader=test_loader,
        epoch="Final Test",
    )

    final_record = {
        "mode": mode,
        "seed": 42,
        "best_epoch": best_epoch,
        "best_cd_l2_x1000":
            best_cd,
        "best_checkpoint":
            str(best_path),
        "best_checkpoint_sha256":
            sha256_file(
                best_path
            ),
        "historical_final_validation":
            final_result,
    }

    (
        output_dir
        / "training_result.json"
    ).write_text(
        json.dumps(
            final_record,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Best epoch :",
        best_epoch,
    )

    print(
        "Best CD    :",
        best_cd,
    )

    print(
        "Checkpoint :",
        best_path,
    )

    print(
        "SHA256     :",
        sha256_file(
            best_path
        ),
    )

    print()
    print(
        "IMPORTANT: run eval_table2.py "
        "separately for the released official "
        "8-view paper evaluation protocol."
    )


if __name__ == "__main__":
    main()
