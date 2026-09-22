#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FINAL Figure 5: ShapeNet-55 Optimization and Training Analysis
===============================================================

Paper-final data policy
-----------------------
1. Gradient panels (a)(b) read ONE AND ONLY ONE source file:
   paper/figures/gradient_analysis_final/
   gradient_interaction_shapenet55_all5_final_samples.csv
2. That CSV is the unified 220-object recomputation for:
   No-Bridge / Full / G2S / S2G / MBB-Net.
3. No v2/v3/v4 gradient CSV is read by this script.
4. Training panels (c)(d) continue to read the original raw training logs.
5. No smoothing, interpolation, or fabricated points are used.
6. The script audits N=220 and exact object/label alignment across all five modes
   before plotting.

This file is intended to be the single paper-final Figure 5 plotting entry point.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 0. Locate project root
# ============================================================

def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (
            (candidate / "expected_results").exists()
            and (candidate / "checkpoints").exists()
            and (candidate / "models").exists()
        ):
            return candidate
    raise RuntimeError(
        "Cannot locate MBB-Net project root.\n"
        "Expected expected_results/, checkpoints/, and models/."
    )


PROJECT_ROOT = find_project_root()


# ============================================================
# 1. Input / output paths
# ============================================================

GRADIENT_DIR = PROJECT_ROOT / "expected_results" / "Analysis" / "Fig5"
FINAL_GRADIENT_CSV = (
    GRADIENT_DIR / "gradient_interaction_shapenet55_all5_final_samples.csv"
)

DYNAMICS_JSON = (
    GRADIENT_DIR / "training_dynamics.json"
)

OUTPUT_DIR = PROJECT_ROOT / "assets" / "fig5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUT_DIR / "Fig5_ShapeNet55_Optimization_Training_FINAL.pdf"
PNG_PATH = OUTPUT_DIR / "Fig5_ShapeNet55_Optimization_Training_FINAL.png"
STATS_PATH = OUTPUT_DIR / "Fig5_ShapeNet55_Gradient_Statistics_FINAL.csv"


# ============================================================
# 2. Check files
# ============================================================

REQUIRED_FILES = {
    "FINAL all-five gradient CSV": FINAL_GRADIENT_CSV,
    "ShapeNet-55 training dynamics": DYNAMICS_JSON,
}

print("=" * 90)
print("Checking required Figure 5 source files")
print("=" * 90)

missing = []
for name, path in REQUIRED_FILES.items():
    if path.is_file():
        print(f"[OK] {name}\n     {path}")
    else:
        print(f"[MISSING] {name}\n          {path}")
        missing.append(path)

if missing:
    raise FileNotFoundError(
        "\nMissing required source files:\n" + "\n".join(str(p) for p in missing)
    )


# ============================================================
# 3. Publication style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.2,
    "axes.labelsize": 8.8,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.7,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.82,
    "xtick.major.width": 0.72,
    "ytick.major.width": 0.72,
    "xtick.major.size": 3.1,
    "ytick.major.size": 3.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 900,
})


# ============================================================
# 4. Visual encoding
# ============================================================

COLORS = {
    "single": "#4D4D4D",
    "NO_BRIDGE": "#7A7A7A",
    "FULL": "#D55E00",
    "full": "#D55E00",
    "G2S": "#56B4E9",
    "S2G": "#009E73",
    "OURS": "#0072B2",
    "mbb": "#0072B2",
}

DISPLAY_NAMES = {
    "NO_BRIDGE": "No-Bridge",
    "FULL": "Full",
    "G2S": "G2S",
    "S2G": "S2G",
    "OURS": "MBB-Net",
}

GRADIENT_MODES = ["NO_BRIDGE", "FULL", "G2S", "S2G", "OURS"]

LINE_STYLES = {
    "single": (0, (4.0, 2.2)),
    "full": (0, (5.0, 2.0, 1.2, 2.0)),
    "mbb": "-",
}

MARKERS = {
    "single": "o",
    "full": "s",
    "mbb": "^",
}


# ============================================================
# 5. Common axis style
# ============================================================

def style_axis(ax):
    ax.grid(axis="y", linestyle="--", linewidth=0.42, alpha=0.18, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.82)
    ax.spines["bottom"].set_linewidth(0.82)
    ax.tick_params(axis="both", direction="out", width=0.72, length=3.1, pad=2.0)


def add_panel_subcaption(ax, text):
    ax.text(
        0.5, -0.245, text,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8.8, fontweight="bold",
        clip_on=False,
    )


# ============================================================
# 6. Gradient CSV parser
# ============================================================

def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def read_gradient_csv(path: Path):
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "dataset", "mode", "sample_order", "dataset_index",
            "label", "valid", "cosine", "support_overlap",
        }
        actual_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - actual_columns
        if missing_columns:
            raise RuntimeError(f"{path.name} missing columns: {sorted(missing_columns)}")

        for row in reader:
            try:
                cosine = float(row["cosine"])
            except (TypeError, ValueError):
                cosine = np.nan
            try:
                support = float(row["support_overlap"])
            except (TypeError, ValueError):
                support = np.nan

            records.append({
                "dataset": row["dataset"],
                "mode": row["mode"],
                "sample_order": int(row["sample_order"]),
                "dataset_index": int(row["dataset_index"]),
                "label": int(row["label"]),
                "valid": parse_bool(row["valid"]),
                "cosine": cosine,
                "support_overlap": support,
            })
    return records


# ============================================================
# 7. Load ShapeNet-55 gradient measurements
# ============================================================

gradient_records = read_gradient_csv(FINAL_GRADIENT_CSV)

gradient_grouped = defaultdict(list)
unexpected_records = []
for record in gradient_records:
    if record["dataset"] == "ShapeNet55" and record["mode"] in GRADIENT_MODES:
        gradient_grouped[record["mode"]].append(record)
    else:
        unexpected_records.append((record["dataset"], record["mode"]))

if unexpected_records:
    raise RuntimeError(
        "FINAL gradient CSV contains unexpected dataset/mode records: "
        f"{sorted(set(unexpected_records))}"
    )


def valid_gradient_records(mode):
    return [
        record for record in gradient_grouped[mode]
        if record["valid"] is not False and np.isfinite(record["cosine"])
    ]


# ============================================================
# 8. Gradient audit
# ============================================================

EXPECTED_N = 220
reference_indices = None
reference_labels = None

print("\n" + "=" * 90)
print("ShapeNet-55 FINAL single-source gradient data audit")
print("=" * 90)

for mode in GRADIENT_MODES:
    subset = valid_gradient_records(mode)
    if len(subset) != EXPECTED_N:
        raise RuntimeError(f"ShapeNet55/{mode}: expected {EXPECTED_N} valid samples, found {len(subset)}.")

    sample_orders = [record["sample_order"] for record in subset]
    indices = [record["dataset_index"] for record in subset]
    labels = {record["dataset_index"]: record["label"] for record in subset}

    if sorted(sample_orders) != list(range(EXPECTED_N)):
        raise RuntimeError(
            f"ShapeNet55/{mode}: sample_order must be exactly 0..{EXPECTED_N - 1}."
        )

    if len(set(indices)) != len(indices):
        raise RuntimeError(f"Duplicate indices in ShapeNet55/{mode}.")

    if reference_indices is None:
        reference_indices = set(indices)
        reference_labels = labels
    else:
        if set(indices) != reference_indices:
            raise RuntimeError(f"ShapeNet-55 object set differs across gradient modes: {mode}")
        for index, label in labels.items():
            if reference_labels[index] != label:
                raise RuntimeError(f"ShapeNet-55 label mismatch at dataset index {index}")

    print(f"{mode:<5s} | N={len(subset):>3d} | same object set: OK")


# ============================================================
# 9. Gradient statistics
# ============================================================

def summarize_gradient(mode):
    subset = valid_gradient_records(mode)
    cosines = np.asarray([record["cosine"] for record in subset], dtype=float)
    supports = np.asarray([record["support_overlap"] for record in subset], dtype=float)
    adverse = np.maximum(0.0, -cosines)
    return {
        "N": len(cosines),
        "mean": float(np.mean(cosines)),
        "median": float(np.median(cosines)),
        "q1": float(np.percentile(cosines, 25)),
        "q3": float(np.percentile(cosines, 75)),
        "mean_adverse": float(np.mean(adverse)),
        "support_mean": float(np.nanmean(supports)),
    }


gradient_stats = {mode: summarize_gradient(mode) for mode in GRADIENT_MODES}

with STATS_PATH.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "mode", "N", "cosine_mean", "cosine_median", "q1", "q3",
        "mean_adverse_alignment", "support_overlap_mean",
    ])
    for mode in GRADIENT_MODES:
        s = gradient_stats[mode]
        writer.writerow([
            mode, s["N"], s["mean"], s["median"], s["q1"], s["q3"],
            s["mean_adverse"], s["support_mean"],
        ])

print("\nGradient statistics used by new Figure 5:")
for mode in GRADIENT_MODES:
    s = gradient_stats[mode]
    print(f"{mode:<5s} | A-={s['mean_adverse']:.5f} | Omega={s['support_mean']:.5f}")


# ============================================================
# 10. Training log parsers
# ============================================================

def parse_shapenet_joint_log(log_path: Path):
    epochs, cds, accs = [], [], []
    current_epoch = None
    current_cd = None

    epoch_pattern = re.compile(r"--- Evaluation Results \(Epoch\s+(\d+)\) ---")
    cd_pattern = re.compile(r"CD\(L2\*1000\)\s*->.*Avg:\s*([0-9.]+)")
    acc_pattern = re.compile(r"Accuracy\s*->\s*([0-9.]+)%")

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                current_cd = None
                continue
            if current_epoch is None:
                continue
            cd_match = cd_pattern.search(line)
            if cd_match:
                current_cd = float(cd_match.group(1))
                continue
            acc_match = acc_pattern.search(line)
            if acc_match:
                epochs.append(current_epoch)
                cds.append(np.nan if current_cd is None else current_cd)
                accs.append(float(acc_match.group(1)))
                current_epoch = None
                current_cd = None

    if len(epochs) == 0:
        raise RuntimeError(f"No ShapeNet-55 joint validation records found:\n{log_path}")

    return {
        "epoch": np.asarray(epochs, dtype=float),
        "cd": np.asarray(cds, dtype=float),
        "acc": np.asarray(accs, dtype=float),
    }


def parse_shapenet_cls_only_log(log_path: Path):
    epochs, accs = [], []
    current_epoch = None
    epoch_pattern = re.compile(r"--- Evaluation Results \(Epoch\s+(\d+)\) ---")
    acc_pattern = re.compile(r"Accuracy\s*->\s*([0-9.]+)%")

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            if current_epoch is None:
                continue
            acc_match = acc_pattern.search(line)
            if acc_match:
                epochs.append(current_epoch)
                accs.append(float(acc_match.group(1)))
                current_epoch = None

    if len(epochs) == 0:
        raise RuntimeError(f"No ShapeNet-55 Classification-only records found:\n{log_path}")

    return {
        "epoch": np.asarray(epochs, dtype=float),
        "acc": np.asarray(accs, dtype=float),
    }


# ============================================================
# 11. Load training data
# ============================================================

with DYNAMICS_JSON.open("r", encoding="utf-8") as handle:
    _dynamics = json.load(handle)

def _restore_record(record):
    return {
        key: np.asarray(value)
        if isinstance(value, list)
        else value
        for key, value in record.items()
    }

sn_comp = _restore_record(_dynamics["completion_only"])
sn_full = _restore_record(_dynamics["full"])
sn_mbb = _restore_record(_dynamics["mbb"])
sn_cls = _restore_record(_dynamics["classification_only"])

print("\n" + "=" * 90)
print("ShapeNet-55 training-log audit")
print("=" * 90)
print(f"Completion-only:      {len(sn_comp['epoch'])} points")
print(f"Full:                 {len(sn_full['epoch'])} points")
print(f"MBB-Net:              {len(sn_mbb['epoch'])} points")
print(f"Classification-only:  {len(sn_cls['epoch'])} points")


# ============================================================
# 12. Marker thinning
# ============================================================

SN_MARKER_EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]


def get_marker_indices(epoch_array, selected_epochs):
    epoch_to_index = {int(epoch): index for index, epoch in enumerate(epoch_array)}
    return [epoch_to_index[epoch] for epoch in selected_epochs if epoch in epoch_to_index]


sn_comp_markers = get_marker_indices(sn_comp["epoch"], SN_MARKER_EPOCHS)
sn_full_markers = get_marker_indices(sn_full["epoch"], SN_MARKER_EPOCHS)
sn_mbb_markers = get_marker_indices(sn_mbb["epoch"], SN_MARKER_EPOCHS)
sn_cls_markers = get_marker_indices(sn_cls["epoch"], SN_MARKER_EPOCHS)


# ============================================================
# 13. Panel (a): gradient alignment
# ============================================================

def plot_gradient_alignment(ax, jitter_rng):
    positions = np.arange(1, len(GRADIENT_MODES) + 1, dtype=float)
    arrays = [
        np.asarray([record["cosine"] for record in valid_gradient_records(mode)], dtype=float)
        for mode in GRADIENT_MODES
    ]

    boxplot = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops=dict(color="black", linewidth=1.20),
        whiskerprops=dict(color="#555555", linewidth=0.85),
        capprops=dict(color="#555555", linewidth=0.85),
        boxprops=dict(edgecolor="#333333", linewidth=0.90),
    )

    for patch, mode in zip(boxplot["boxes"], GRADIENT_MODES):
        patch.set_facecolor(COLORS[mode])
        patch.set_alpha(0.43)

    for position, mode, values in zip(positions, GRADIENT_MODES, arrays):
        jitter = jitter_rng.normal(loc=0.0, scale=0.055, size=len(values))
        jitter = np.clip(jitter, -0.14, +0.14)
        # IMPORTANT: jitter is horizontal only.
        # No-Bridge rho values remain exactly at y=0; no vertical noise is added.
        point_size = 6.2 if mode == "NO_BRIDGE" else 5.3
        point_alpha = 0.22 if mode == "NO_BRIDGE" else 0.16

        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=point_size,
            color=COLORS[mode],
            alpha=point_alpha,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    ax.axhline(0.0, color="#666666", linestyle=(0, (3.2, 2.0)), linewidth=0.90, zorder=1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlim(0.45, len(GRADIENT_MODES) + 0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_NAMES[mode] for mode in GRADIENT_MODES])
    ax.set_ylabel(r"Task-gradient cosine $\rho$")
    style_axis(ax)
    add_panel_subcaption(ax, "(a) ShapeNet-55: Gradient Alignment")


# ============================================================
# 14. Panel (b): backward interaction
# ============================================================

def plot_backward_interaction(ax):
    positions = np.arange(len(GRADIENT_MODES), dtype=float)

    # Wider separation between the two bars in each group.
    width = 0.26
    center_offset = 0.18

    adverse_values = np.asarray([gradient_stats[mode]["mean_adverse"] for mode in GRADIENT_MODES])
    support_values = np.asarray([gradient_stats[mode]["support_mean"] for mode in GRADIENT_MODES])

    adverse_bars = []
    for x, mode, value in zip(positions, GRADIENT_MODES, adverse_values):
        bar = ax.bar(
            x - center_offset,
            value,
            width=width,
            align="center",
            facecolor=COLORS[mode],
            edgecolor=COLORS[mode],
            alpha=0.72,
            linewidth=0.80,
            zorder=3,
        )
        adverse_bars.append(bar[0])

    support_bars = []
    for x, mode, value in zip(positions, GRADIENT_MODES, support_values):
        bar = ax.bar(
            x + center_offset,
            value,
            width=width,
            align="center",
            facecolor="white",
            edgecolor=COLORS[mode],
            hatch="////",
            linewidth=1.00,
            zorder=3,
        )
        support_bars.append(bar[0])

    YMAX = 0.72
    label_offset = YMAX * 0.018

    # Shift label anchors horizontally to prevent text from touching the opposite bar edge.
    left_dx = 0.018
    right_dx = 0.018

    for bar, value in zip(adverse_bars, adverse_values):
        # Lift tiny/zero values above the x-axis so the text is readable.
        y = max(value + label_offset, YMAX * 0.045)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0 - left_dx,
            y,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#333333",
        )

    for bar, value in zip(support_bars, support_values):
        y = max(value + label_offset, YMAX * 0.045)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0 + right_dx,
            y,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#333333",
        )

    ax.set_xlim(-0.55, len(GRADIENT_MODES) - 0.45)
    ax.set_ylim(0.0, YMAX)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_NAMES[mode] for mode in GRADIENT_MODES])
    ax.set_ylabel("Metric value")
    style_axis(ax)

    legend_handles = [
        Patch(facecolor="#777777", edgecolor="#777777", alpha=0.72,
              label=r"Mean adverse alignment $\overline{A^-}$"),
        Patch(facecolor="white", edgecolor="#555555", hatch="////",
              label=r"Parameter-support overlap $\Omega$"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.00, 1.00),
        frameon=False,
        handlelength=1.45,
        handleheight=0.90,
        borderaxespad=0.10,
        labelspacing=0.25,
    )

    add_panel_subcaption(ax, "(b) ShapeNet-55: Backward Interaction")


# ============================================================
# 15. Panel (c): completion dynamics
# ============================================================

def plot_completion_dynamics(ax):
    ax.plot(sn_comp["epoch"], sn_comp["cd"],
            color=COLORS["single"], linestyle=LINE_STYLES["single"],
            marker=MARKERS["single"], markevery=sn_comp_markers,
            linewidth=1.60, markersize=3.5, markeredgewidth=0.50,
            label="Completion-only", zorder=2)

    ax.plot(sn_full["epoch"], sn_full["cd"],
            color=COLORS["full"], linestyle=LINE_STYLES["full"],
            marker=MARKERS["full"], markevery=sn_full_markers,
            linewidth=1.60, markersize=3.5, markeredgewidth=0.50,
            label="Full", zorder=2)

    ax.plot(sn_mbb["epoch"], sn_mbb["cd"],
            color=COLORS["mbb"], linestyle=LINE_STYLES["mbb"],
            marker=MARKERS["mbb"], markevery=sn_mbb_markers,
            linewidth=1.85, markersize=3.8, markeredgewidth=0.50,
            label="MBB-Net", zorder=3)

    ax.set_xlim(0, 300)
    ax.set_ylim(1.10, 3.20)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(r"CD-L2 ($\times 10^3$) $\downarrow$")
    style_axis(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=2.05,
              handletextpad=0.42, borderaxespad=0.18, labelspacing=0.25)
    add_panel_subcaption(ax, "(c) ShapeNet-55: Completion Dynamics")


# ============================================================
# 16. Panel (d): classification dynamics
# ============================================================

def plot_classification_dynamics(ax):
    ax.plot(sn_cls["epoch"], sn_cls["acc"],
            color=COLORS["single"], linestyle=LINE_STYLES["single"],
            marker=MARKERS["single"], markevery=sn_cls_markers,
            linewidth=1.60, markersize=3.5, markeredgewidth=0.50,
            label="Classification-only", zorder=2)

    ax.plot(sn_full["epoch"], sn_full["acc"],
            color=COLORS["full"], linestyle=LINE_STYLES["full"],
            marker=MARKERS["full"], markevery=sn_full_markers,
            linewidth=1.60, markersize=3.5, markeredgewidth=0.50,
            label="Full", zorder=2)

    ax.plot(sn_mbb["epoch"], sn_mbb["acc"],
            color=COLORS["mbb"], linestyle=LINE_STYLES["mbb"],
            marker=MARKERS["mbb"], markevery=sn_mbb_markers,
            linewidth=1.85, markersize=3.8, markeredgewidth=0.50,
            label="MBB-Net", zorder=3)

    ax.set_xlim(0, 300)
    ax.set_ylim(64.0, 90.5)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(r"Accuracy (\%) $\uparrow$")
    style_axis(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.05,
              handletextpad=0.42, borderaxespad=0.18, labelspacing=0.25)
    add_panel_subcaption(ax, "(d) ShapeNet-55: Classification Dynamics")


# ============================================================
# 17. Build figure
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(7.80, 5.55))
ax_a, ax_b, ax_c, ax_d = axes.ravel()

rng = np.random.default_rng(20260807)
plot_gradient_alignment(ax_a, rng)
plot_backward_interaction(ax_b)
plot_completion_dynamics(ax_c)
plot_classification_dynamics(ax_d)

plt.subplots_adjust(
    left=0.088,
    right=0.987,
    bottom=0.123,
    top=0.988,
    wspace=0.31,
    hspace=0.64,
)

fig.canvas.draw()
fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.025)
fig.savefig(PNG_PATH, bbox_inches="tight", pad_inches=0.025, dpi=1000)
plt.close(fig)


# ============================================================
# 18. Report
# ============================================================

print("\n" + "=" * 90)
print("NEW SHAPENET-55 FIGURE 5 FINAL SINGLE-SOURCE GENERATED (No-Bridge added)")
print("=" * 90)
print(f"\nPDF:\n  {PDF_PATH}")
print(f"\nPNG:\n  {PNG_PATH}")
print(f"\nGradient statistics:\n  {STATS_PATH}")
print("\nPanels:")
print("  (a) ShapeNet-55: Gradient Alignment")
print("  (b) ShapeNet-55: Backward Interaction")
print("  (c) ShapeNet-55: Completion Dynamics")
print("  (d) ShapeNet-55: Classification Dynamics")
print("\nFixes in v2:")
print("  - No-Bridge added to panels (a) and (b)")
print("  - No-Bridge y-values are kept exactly at rho=0; only x-jitter is used")
print("  - panel (b) label/bar overlap fixed via larger bar separation")
print("  - panel (b) labels shifted left/right")
print("  - higher export quality (PDF + 1000 dpi PNG)")
print("  - panel (b) y-axis fixed to [0, 0.72]")
print("\nDone.")

# Public-release CSV newline normalization
# Python's csv module defaults to CRLF. Normalize the generated public
# statistics artifact to LF so repeated regeneration is Git-clean.
from pathlib import Path as _ReleasePath

_release_csv = (
    _ReleasePath(__file__).resolve().parents[3]
    / "assets"
    / "fig5"
    / "Fig5_ShapeNet55_Gradient_Statistics_FINAL.csv"
)

if _release_csv.is_file():
    _release_bytes = _release_csv.read_bytes()
    if b"\r\n" in _release_bytes:
        _release_csv.write_bytes(
            _release_bytes.replace(b"\r\n", b"\n")
        )
