#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.pcn_official_eval import (
    PCN_TAXONOMY,
    PCNSnowflakeOfficialEvalDataset,
    resolve_pcn_data_root,
)

from metrics.pcn_metrics import (
    compute_squared_dists,
    calc_pcn_cd,
    calc_fscore,
)

from models.pcn_model import (
    MBB_Model_PCN,
)


EXPECTED_CKPT_SHA256 = (
    "b32d7a03dd759741ded8ca22b4fb48e"
    "60d7f5039cee120050229875058f36940"
)


CFG = {
    "NUM_CLASSES": 8,
    "NUM_PC": 256,
    "NUM_P0": 512,
    "UP_FACTORS": [1, 4, 8],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def safe_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def parse_state_dict(path):
    raw = safe_load(path)

    state = raw

    if isinstance(raw, dict):
        for candidate in (
            "model_state_dict",
            "state_dict",
            "model",
            "base_model",
            "net",
        ):
            value = raw.get(candidate)

            if (
                isinstance(value, dict)
                and value
            ):
                state = value
                break

    cleaned = {}

    for key, value in state.items():
        if not isinstance(
            value,
            torch.Tensor,
        ):
            continue

        new_key = key

        while new_key.startswith(
            "module."
        ):
            new_key = new_key[
                len("module.") :
            ]

        if new_key in cleaned:
            raise RuntimeError(
                f"Duplicate key: {new_key}"
            )

        cleaned[new_key] = value

    if not cleaned:
        raise RuntimeError(
            "No tensor state_dict found."
        )

    return cleaned


def make_audit_indices(
    dataset,
    per_class,
):
    if per_class <= 0:
        return list(
            range(len(dataset))
        )

    counts = defaultdict(int)
    selected = []

    for index, sample in enumerate(
        dataset.samples
    ):
        taxonomy_id = (
            sample["taxonomy_id"]
        )

        if counts[taxonomy_id] < per_class:
            selected.append(index)
            counts[taxonomy_id] += 1

    return selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--f1_threshold",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--audit_per_class",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        default="expected_results/"
                "pcn_release_eval.json",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    set_seed(
        args.seed
    )

    checkpoint = Path(
        args.checkpoint
    ).resolve()

    actual_sha = sha256(
        checkpoint
    )

    print(
        "checkpoint SHA256 =",
        actual_sha,
    )

    if (
        actual_sha
        != EXPECTED_CKPT_SHA256
    ):
        raise RuntimeError(
            "Canonical PCN checkpoint "
            "SHA256 mismatch."
        )

    print(
        "[PASS] canonical checkpoint SHA256"
    )

    data_root = (
        resolve_pcn_data_root(
            explicit_root=args.data_root
        )
    )

    dataset = (
        PCNSnowflakeOfficialEvalDataset(
            data_root=data_root,
            num_partial=2048,
            num_complete=16384,
            seed=args.seed,
            cache=True,
        )
    )

    indices = make_audit_indices(
        dataset,
        args.audit_per_class,
    )

    eval_dataset = (
        dataset
        if len(indices) == len(dataset)
        else Subset(
            dataset,
            indices,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = MBB_Model_PCN(
        CFG,
        use_mbb=True,
    ).cuda()

    state = parse_state_dict(
        checkpoint
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
            "Strict load failed."
        )

    print(
        "[PASS] checkpoint strict-load"
    )

    model.eval()

    per_class = {
        taxonomy_id: {
            "name": class_name,
            "cd": [],
            "f1": [],
            "correct": 0,
            "count": 0,
        }
        for taxonomy_id, class_name
        in PCN_TAXONOMY
    }

    with torch.inference_mode():

        progress = tqdm(
            loader,
            desc="PCN-MBB",
        )

        for (
            partial,
            gt,
            labels,
            taxonomy_ids,
            _model_ids,
        ) in progress:

            partial = partial.cuda(
                non_blocking=True
            )

            gt = gt.cuda(
                non_blocking=True
            )

            labels = labels.cuda(
                non_blocking=True
            )

            coarse, fine, logits = model(
                partial
            )

            if (
                fine.ndim != 3
                or fine.shape[-1] != 3
            ):
                raise RuntimeError(
                    f"Invalid prediction "
                    f"shape={tuple(fine.shape)}"
                )

            d1, d2 = (
                compute_squared_dists(
                    fine,
                    gt,
                )
            )

            cd_batch = calc_pcn_cd(
                d1,
                d2,
            )

            f1_batch = calc_fscore(
                d1,
                d2,
                threshold=args.f1_threshold,
            )

            predictions = (
                logits.argmax(dim=1)
            )

            for i, taxonomy_id in enumerate(
                taxonomy_ids
            ):
                entry = per_class[
                    str(taxonomy_id)
                ]

                entry["count"] += 1

                entry["cd"].append(
                    float(
                        cd_batch[i].item()
                    )
                )

                entry["f1"].append(
                    float(
                        f1_batch[i].item()
                    )
                )

                entry["correct"] += int(
                    predictions[i].item()
                    == labels[i].item()
                )

    class_rows = []

    for taxonomy_id, class_name in PCN_TAXONOMY:

        entry = per_class[
            taxonomy_id
        ]

        row = {
            "taxonomy_id":
                taxonomy_id,

            "class_name":
                class_name,

            "count":
                entry["count"],

            "cd_l1_x1000":
                float(
                    np.mean(
                        entry["cd"]
                    )
                ),

            "f1_at_1pct":
                float(
                    np.mean(
                        entry["f1"]
                    )
                ),

            "accuracy":
                float(
                    entry["correct"]
                    / entry["count"]
                ),
        }

        class_rows.append(
            row
        )

    all_cd = [
        value
        for entry in per_class.values()
        for value in entry["cd"]
    ]

    all_f1 = [
        value
        for entry in per_class.values()
        for value in entry["f1"]
    ]

    total_correct = sum(
        entry["correct"]
        for entry in per_class.values()
    )

    total_count = sum(
        entry["count"]
        for entry in per_class.values()
    )

    output = {
        "protocol":
            "PCN SnowflakeNet official "
            "deterministic evaluation",

        "checkpoint_sha256":
            actual_sha,

        "seed":
            args.seed,

        "objects":
            total_count,

        "cd_l1_x1000_macro":
            float(
                np.mean(
                    [
                        r["cd_l1_x1000"]
                        for r in class_rows
                    ]
                )
            ),

        "cd_l1_x1000_micro":
            float(
                np.mean(all_cd)
            ),

        "f1_at_1pct_macro":
            float(
                np.mean(
                    [
                        r["f1_at_1pct"]
                        for r in class_rows
                    ]
                )
            ),

        "f1_at_1pct_micro":
            float(
                np.mean(all_f1)
            ),

        "accuracy_micro":
            float(
                total_correct
                / total_count
            ),

        "accuracy_macro":
            float(
                np.mean(
                    [
                        r["accuracy"]
                        for r in class_rows
                    ]
                )
            ),

        "per_class":
            class_rows,
    }

    print()
    print("=" * 78)
    print("PCN RELEASE EVALUATION")
    print("=" * 78)

    print(
        "Objects          =",
        output["objects"],
    )

    print(
        "CD-L1 x1000     =",
        repr(
            output[
                "cd_l1_x1000_micro"
            ]
        ),
    )

    print(
        "F1@1%           =",
        repr(
            output[
                "f1_at_1pct_micro"
            ]
        ),
    )

    print(
        "Accuracy        =",
        repr(
            output[
                "accuracy_micro"
            ]
        ),
    )

    print(
        "Paper CD        =",
        f"{output['cd_l1_x1000_micro']:.2f}",
    )

    print(
        "Paper Accuracy  =",
        f"{100.0 * output['accuracy_micro']:.2f}%",
    )

    print("=" * 78)

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
