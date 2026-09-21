#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portable verifier for Figure 5.

This script does NOT rerun model backpropagation.

It independently recomputes the paper-level Figure 5 statistics
from the archived final sample-level gradient records, and checks:

1. protocol metadata;
2. 5 modes x 220 objects;
3. identical paired sample set across modes;
4. valid-record counts;
5. mean adverse alignment;
6. mean active-support overlap;
7. rounded paper values;
8. archived summary consistency;
9. checkpoint SHA256 provenance.

No GPU is required.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    ROOT
    / "expected_results"
    / "Analysis"
    / "Fig5"
)

SAMPLES = (
    DATA_DIR
    / "gradient_interaction_shapenet55_all5_final_samples.csv"
)

SUMMARY = (
    DATA_DIR
    / "gradient_interaction_shapenet55_all5_final_summary.json"
)

PROTOCOL = (
    DATA_DIR
    / "gradient_interaction_shapenet55_all5_final_protocol.json"
)

CKPT_META = (
    DATA_DIR
    / "gradient_interaction_shapenet55_all5_final_checkpoints.json"
)

PAPER_REF = (
    DATA_DIR
    / "fig5_reference.json"
)

AUDIT_OUT = (
    ROOT
    / "expected_results"
    / "audit"
    / "fig5_verification.json"
)


MODE_TO_PAPER = {
    "NO_BRIDGE": "No-Bridge",
    "FULL": "Full",
    "G2S": "G2S",
    "S2G": "S2G",
    "OURS": "MBB",
}


CHECKPOINT_PATHS = {
    "NO_BRIDGE":
        ROOT
        / "checkpoints/ShapeNet/ShapeNet55/"
          "no_bridge/best_model.pth",

    "FULL":
        ROOT
        / "checkpoints/ShapeNet/ShapeNet55/"
          "full/best_model.pth",

    "G2S":
        ROOT
        / "checkpoints/ShapeNet/ShapeNet55/"
          "g2s/best_model.pth",

    "S2G":
        ROOT
        / "checkpoints/ShapeNet/ShapeNet55/"
          "s2g/best_model.pth",

    "OURS":
        ROOT
        / "checkpoints/ShapeNet/ShapeNet55/"
          "ours/best_model.pth",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def as_bool(value: str) -> bool:
    x = value.strip().lower()

    if x == "true":
        return True

    if x == "false":
        return False

    raise ValueError(
        f"Invalid boolean string: {value!r}"
    )


# ============================================================
# Load archived artifacts
# ============================================================

for path in (
    SAMPLES,
    SUMMARY,
    PROTOCOL,
    CKPT_META,
    PAPER_REF,
):
    if not path.is_file():
        raise FileNotFoundError(path)


summary_json = json.loads(
    SUMMARY.read_text(encoding="utf-8")
)

protocol_json = json.loads(
    PROTOCOL.read_text(encoding="utf-8")
)

ckpt_json = json.loads(
    CKPT_META.read_text(encoding="utf-8")
)

paper_ref = json.loads(
    PAPER_REF.read_text(encoding="utf-8")
)


rows_by_mode = defaultdict(list)

with SAMPLES.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        mode = row["mode"]

        rows_by_mode[mode].append(
            {
                "dataset":
                    row["dataset"],

                "sample_order":
                    int(row["sample_order"]),

                "dataset_index":
                    int(row["dataset_index"]),

                "label":
                    int(row["label"]),

                "valid":
                    as_bool(row["valid"]),

                "conflict":
                    as_bool(row["conflict"]),

                "cosine":
                    float(row["cosine"]),

                "adverse_alignment":
                    float(
                        row["adverse_alignment"]
                    ),

                "support_overlap":
                    float(
                        row["support_overlap"]
                    ),
            }
        )


failed = []
report = {
    "protocol": {},
    "modes": {},
    "checkpoint_provenance": {},
}


# ============================================================
# Protocol checks
# ============================================================

expected_protocol = {
    "dataset": "ShapeNet55",
    "subset": "test",
    "partial": "moderate",
    "seed": 42,
    "per_class": 4,
    "selected_object_count": 220,
    "alpha": 0.4,
    "batch_size": 1,
    "num_workers": 0,
}


for key, expected in (
    expected_protocol.items()
):
    got = protocol_json.get(key)

    same = got == expected

    report["protocol"][key] = {
        "actual": got,
        "expected": expected,
        "exact": same,
    }

    if not same:
        failed.append(
            f"protocol:{key}"
        )


expected_modes = list(
    MODE_TO_PAPER.keys()
)

if protocol_json.get("modes") != expected_modes:
    failed.append(
        "protocol:modes"
    )


# ============================================================
# Raw row / paired-sample checks
# ============================================================

if set(rows_by_mode) != set(expected_modes):
    failed.append(
        "samples:mode_set"
    )


reference_pairs = None


for mode in expected_modes:

    rows = rows_by_mode[mode]

    if len(rows) != 220:
        failed.append(
            f"{mode}:N_total"
        )

    orders = [
        x["sample_order"]
        for x in rows
    ]

    if orders != list(range(220)):
        failed.append(
            f"{mode}:sample_order"
        )

    pairs = [
        (
            x["dataset_index"],
            x["label"],
        )
        for x in rows
    ]

    if reference_pairs is None:
        reference_pairs = pairs

    elif pairs != reference_pairs:
        failed.append(
            f"{mode}:paired_sample_set"
        )


# ============================================================
# Independently recompute Fig.5 values
# ============================================================

for mode in expected_modes:

    rows = rows_by_mode[mode]

    valid_rows = [
        x
        for x in rows
        if x["valid"]
    ]

    adverse = np.asarray(
        [
            x["adverse_alignment"]
            for x in valid_rows
        ],
        dtype=np.float64,
    )

    overlap = np.asarray(
        [
            x["support_overlap"]
            for x in valid_rows
        ],
        dtype=np.float64,
    )

    conflict = np.asarray(
        [
            float(x["conflict"])
            for x in valid_rows
        ],
        dtype=np.float64,
    )

    mean_adverse = float(
        adverse.mean()
    )

    mean_overlap = float(
        overlap.mean()
    )

    conflict_ratio_percent = float(
        conflict.mean() * 100.0
    )

    archived = (
        summary_json["summary"][mode]
    )

    # Raw -> archived final summary
    numeric_checks = {
        "mean_adverse_alignment":
            (
                mean_adverse,
                float(
                    archived[
                        "mean_adverse_alignment"
                    ]
                ),
            ),

        "support_overlap_mean":
            (
                mean_overlap,
                float(
                    archived[
                        "support_overlap_mean"
                    ]
                ),
            ),

        "conflict_ratio_percent":
            (
                conflict_ratio_percent,
                float(
                    archived[
                        "conflict_ratio_percent"
                    ]
                ),
            ),
    }


    archived_exact = {}

    for name, (got, expected) in (
        numeric_checks.items()
    ):

        # CSV decimal parsing + NumPy aggregation can differ
        # by tiny last-bit floating arithmetic only.
        same = bool(
            np.isclose(
                got,
                expected,
                rtol=0.0,
                atol=1e-14,
            )
        )

        archived_exact[name] = {
            "recomputed": got,
            "archived": expected,
            "within_1e-14": same,
        }

        if not same:
            failed.append(
                f"{mode}:summary:{name}"
            )


    paper_name = MODE_TO_PAPER[mode]

    expected_paper = (
        paper_ref["methods"][paper_name]
    )

    paper_adverse = round(
        mean_adverse,
        3,
    )

    paper_overlap = round(
        mean_overlap,
        3,
    )

    adverse_ok = (
        paper_adverse
        == expected_paper[
            "adverse_ratio"
        ]
    )

    overlap_ok = (
        paper_overlap
        == expected_paper[
            "overlap_ratio"
        ]
    )

    if not adverse_ok:
        failed.append(
            f"{mode}:paper:adverse"
        )

    if not overlap_ok:
        failed.append(
            f"{mode}:paper:overlap"
        )


    report["modes"][mode] = {
        "paper_name":
            paper_name,

        "N_total":
            len(rows),

        "N_valid":
            len(valid_rows),

        "conflict_ratio_percent":
            conflict_ratio_percent,

        "mean_adverse_alignment":
            mean_adverse,

        "support_overlap_mean":
            mean_overlap,

        "paper_rounded": {
            "adverse_ratio":
                paper_adverse,

            "overlap_ratio":
                paper_overlap,
        },

        "paper_expected": {
            "adverse_ratio":
                expected_paper[
                    "adverse_ratio"
                ],

            "overlap_ratio":
                expected_paper[
                    "overlap_ratio"
                ],
        },

        "archived_summary_checks":
            archived_exact,
    }


# ============================================================
# Checkpoint provenance
# ============================================================

for mode in expected_modes:

    path = CHECKPOINT_PATHS[mode]

    expected_sha = (
        ckpt_json[mode]["sha256"]
    )

    if not path.is_file():

        failed.append(
            f"{mode}:checkpoint_missing"
        )

        report[
            "checkpoint_provenance"
        ][mode] = {
            "path":
                str(
                    path.relative_to(ROOT)
                ),

            "expected_sha256":
                expected_sha,

            "actual_sha256":
                None,

            "exact":
                False,
        }

        continue


    actual_sha = sha256_file(path)

    exact = (
        actual_sha == expected_sha
    )

    if not exact:
        failed.append(
            f"{mode}:checkpoint_sha256"
        )


    report[
        "checkpoint_provenance"
    ][mode] = {
        "path":
            str(
                path.relative_to(ROOT)
            ),

        "expected_sha256":
            expected_sha,

        "actual_sha256":
            actual_sha,

        "exact":
            exact,
    }


# ============================================================
# Final result
# ============================================================

report["failed"] = failed

report["status"] = (
    "PASS"
    if not failed
    else "FAIL"
)


AUDIT_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_OUT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)


print(
    "=" * 88
)

print(
    "FIGURE 5 PORTABLE VERIFICATION"
)

print(
    "=" * 88
)


for mode in expected_modes:

    r = report["modes"][mode]

    print()
    print(
        f"{mode:10s}"
        f" N={r['N_valid']:3d}"
        f" conflict={r['conflict_ratio_percent']:.6f}%"
    )

    print(
        "  adverse:"
        f" raw={r['mean_adverse_alignment']:.15f}"
        f" -> paper={r['paper_rounded']['adverse_ratio']:.3f}"
    )

    print(
        "  overlap:"
        f" raw={r['support_overlap_mean']:.15f}"
        f" -> paper={r['paper_rounded']['overlap_ratio']:.3f}"
    )


print()
print(
    "checkpoint provenance:"
)

for mode in expected_modes:

    item = (
        report[
            "checkpoint_provenance"
        ][mode]
    )

    print(
        f"  {mode:10s}"
        f" exact={item['exact']}"
    )


print()
print(
    "failed =",
    failed,
)

print(
    "status =",
    report["status"],
)

print(
    "audit  =",
    AUDIT_OUT,
)


if failed:
    raise SystemExit(
        "[FAIL] Figure 5 verification failed."
    )


print()
print(
    "[PASS] FIGURE 5 SAMPLE-LEVEL ARTIFACT "
    "REPRODUCES ALL PAPER VALUES."
)

print(
    "[PASS] FIGURE 5 CHECKPOINT "
    "PROVENANCE IS EXACT."
)
