#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reproduce the paper-facing PCN qualitative Figure 2.

Layout
------
Input | SnowflakeNet | PoinTr | SeedFormer |
SymmCompletion | CompletionOnly | Full | MBB-Net | GT

The script composes the exact eight frozen paper-facing
three-panel renders released under:

    assets/qualitative/PCN/Figure2/

External baseline renders are frozen experiment artifacts.
CompletionOnly, Full, and MBB-Net additionally have released
model/checkpoint reproduction paths.

The script performs visualization-only cropping/padding.
It does not modify point-cloud coordinates or metrics.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 0. Locate project root
# ============================================================

def find_project_root() -> Path:
    current = Path(__file__).resolve().parent

    for candidate in (current, *current.parents):
        if (
            (candidate / "main.py").is_file()
            and (candidate / "assets").is_dir()
            and (candidate / "tools").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Cannot locate the MBB-Net public repository root."
    )


PROJECT_ROOT = find_project_root()


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = (
    PROJECT_ROOT
    / "assets"
    / "qualitative"
    / "PCN"
    / "Figure2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "PCN_Figure2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PDF_PATH = (
    OUTPUT_DIR
    / "Fig2_PCN_Qualitative.pdf"
)

PNG_PATH = (
    OUTPUT_DIR
    / "Fig2_PCN_Qualitative_600dpi.png"
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "Fig2_PCN_Qualitative_Audit.txt"
)


# ============================================================
# 2. Typography
# ============================================================

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


# ============================================================
# 3. Final rows
#
# Exactly the same representative PCN samples used by
# the previous manuscript figure.
# ============================================================

ROWS = [
    ("Plane",      "plane_idx3.png"),
    ("Cabinet",    "cabinet_idx160.png"),
    ("Car",        "car_idx309.png"),
    ("Chair",      "chair_idx452.png"),
    ("Lamp",       "lamp_idx602.png"),
    ("Couch",      "couch_idx753.png"),
    ("Table",      "table_idx909.png"),
    ("Watercraft", "watercraft_idx1061.png"),
]


# ============================================================
# 4. Final columns
#
# format:
#   display title
#   source folder
#   source section inside the original 3-panel image
#
# section:
#   left   = Input
#   middle = Prediction
#   right  = GT
# ============================================================

COLUMNS = [
    ("Input",          "MBB-Net",         "left"),
    ("SnowflakeNet",   "SnowflakeNet",    "middle"),
    ("PoinTr",         "PoinTr",          "middle"),
    ("SeedFormer",     "SeedFormer",      "middle"),
    ("SymmCompletion", "SymmCompletion",  "middle"),
    ("CompletionOnly", "CompletionOnly",  "middle"),
    ("Full",           "Full",            "middle"),
    ("MBB-Net",        "MBB-Net",         "middle"),
    ("GT",             "MBB-Net",         "right"),
]


# ============================================================
# 5. Compact-layout parameters
#
# These are intentionally close to the old "Perfect" composer.
# ============================================================

# Original individual figure:
# figure(figsize=(15, 5))
#
# Titles occupy approximately the top 12%.
# Bottom blank region is removed at ~82%.
TOP_CUT_RATIO = 0.12
BOTTOM_KEEP_RATIO = 0.82


# Non-white detection.
#
# Since the title region is already physically removed,
# remaining non-white pixels should primarily be point clouds.
WHITE_THRESHOLD = 0.985


# Small margin around detected point cloud.
CONTENT_MARGIN_X = 5
CONTENT_MARGIN_Y = 4


# Crucial: very small inter-cell padding.
#
# Old code used 15. The new final figure uses smaller padding,
# because the user wants the same compact visual style as the
# previous PDF.
PADDING_X = 7
PADDING_Y = 3


# Prevent tiny columns when an Input is extremely sparse.
#
# IMPORTANT:
# this pads; it does NOT resize the point cloud.
MIN_COLUMN_WIDTH = 180


# Header
HEADER_HEIGHT = 125

HEADER_FONT_SIZE = 31
HEADER_FONT_SIZE_LONG = 27


# Export
PREVIEW_DPI = 300
PNG_DPI = 600


# ============================================================
# 6. Read image
# ============================================================

def load_rgb(path: Path) -> np.ndarray:

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing visualization image:\n{path}"
        )

    image = (
        np.asarray(
            Image.open(path).convert("RGB"),
            dtype=np.float32,
        )
        / 255.0
    )

    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(
            f"Invalid RGB image: {path}\n"
            f"shape={image.shape}"
        )

    return image


# ============================================================
# 7. Extract Input / Prediction / GT from 3-panel image
# ============================================================

def extract_panel(
    image: np.ndarray,
    section: str,
) -> np.ndarray:

    h, w, _ = image.shape

    crop_w = w // 3

    if crop_w <= 0:
        raise RuntimeError(
            f"Image width too small: {w}"
        )


    # --------------------------------------------------------
    # Deliberately use exact equal-width blocks.
    #
    # Any remainder pixels on the far right are ignored,
    # preventing 1-2 px inconsistencies.
    # --------------------------------------------------------

    if section == "left":

        patch = image[
            :,
            0:crop_w,
            :,
        ]

    elif section == "middle":

        patch = image[
            :,
            crop_w:2 * crop_w,
            :,
        ]

    elif section == "right":

        patch = image[
            :,
            2 * crop_w:3 * crop_w,
            :,
        ]

    else:

        raise ValueError(
            f"Unknown section: {section}"
        )


    ph = patch.shape[0]

    top = int(
        ph * TOP_CUT_RATIO
    )

    bottom = int(
        ph * BOTTOM_KEEP_RATIO
    )


    if bottom <= top:
        raise RuntimeError(
            "Invalid vertical crop."
        )


    return patch[
        top:bottom,
        :,
        :
    ]


# ============================================================
# 8. Non-white masks
# ============================================================

def nonwhite_mask(
    patch: np.ndarray,
) -> np.ndarray:

    # Use per-pixel channel minimum instead of grayscale mean.
    #
    # This is more reliable for red / blue / green point-cloud
    # pixels because highly saturated pixels are detected even
    # if their RGB average is relatively bright.

    return np.any(
        patch < WHITE_THRESHOLD,
        axis=2,
    )


# ============================================================
# 9. Determine shared Y crop for one row
#
# Same Y range is used by all methods in a row.
# ============================================================

def get_joint_y_bounds(
    patches,
):

    valid = [
        p
        for p in patches
        if p is not None
    ]


    if not valid:
        return 0, 100


    base_h = min(
        p.shape[0]
        for p in valid
    )


    union_y = np.zeros(
        base_h,
        dtype=bool,
    )


    for patch in valid:

        mask = nonwhite_mask(
            patch[:base_h]
        )


        row_mask = np.any(
            mask,
            axis=1,
        )


        union_y |= row_mask


    if not np.any(
        union_y
    ):

        return 0, base_h


    y_indices = np.where(
        union_y
    )[0]


    ymin = max(
        0,
        int(y_indices[0])
        - CONTENT_MARGIN_Y,
    )


    # +1 because slicing endpoint is exclusive
    ymax = min(
        base_h,
        int(y_indices[-1])
        + CONTENT_MARGIN_Y
        + 1,
    )


    return ymin, ymax


# ============================================================
# 10. Individual X crop
#
# IMPORTANT:
# This removes blank margins only.
# It does NOT resize the point cloud.
# ============================================================

def crop_x_only(
    patch: np.ndarray,
    ymin: int,
    ymax: int,
) -> np.ndarray:

    patch = patch[
        ymin:ymax,
        :,
        :
    ]


    mask = nonwhite_mask(
        patch
    )


    if not np.any(
        mask
    ):

        return patch


    col_mask = np.any(
        mask,
        axis=0,
    )


    x_indices = np.where(
        col_mask
    )[0]


    xmin = max(
        0,
        int(x_indices[0])
        - CONTENT_MARGIN_X,
    )


    xmax = min(
        patch.shape[1],
        int(x_indices[-1])
        + CONTENT_MARGIN_X
        + 1,
    )


    return patch[
        :,
        xmin:xmax,
        :
    ]


# ============================================================
# 11. Pad WITHOUT resizing
# ============================================================

def pad_to_size(
    image: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:

    h, w, c = (
        image.shape
    )


    if h > target_h:
        raise RuntimeError(
            f"Image height {h} > target height {target_h}"
        )


    if w > target_w:
        raise RuntimeError(
            f"Image width {w} > target width {target_w}"
        )


    canvas = np.ones(
        (
            target_h,
            target_w,
            c,
        ),
        dtype=np.float32,
    )


    y0 = (
        target_h - h
    ) // 2


    x0 = (
        target_w - w
    ) // 2


    canvas[
        y0:y0 + h,
        x0:x0 + w,
        :
    ] = image


    return canvas


# ============================================================
# 12. Main
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "PCN QUALITATIVE FINAL"
    )

    print(
        "=" * 100
    )


    print(
        f"\nProject root:\n"
        f"  {PROJECT_ROOT}"
    )


    print(
        f"\nSource root:\n"
        f"  {BASE_DIR}"
    )


    # ========================================================
    # 12.1 Input-file audit
    # ========================================================

    missing_files = []


    print(
        "\n"
        + "-" * 100
    )

    print(
        "SOURCE IMAGE AUDIT"
    )

    print(
        "-" * 100
    )


    for row_name, file_name in ROWS:

        for (
            column_title,
            folder_name,
            section,
        ) in COLUMNS:

            path = (
                BASE_DIR
                / folder_name
                / file_name
            )


            if not path.is_file():

                missing_files.append(
                    str(path)
                )


    if missing_files:

        print(
            "\nMissing files:"
        )

        for path in missing_files:

            print(
                f"  {path}"
            )

        raise FileNotFoundError(
            f"{len(missing_files)} required "
            f"visualization files are missing."
        )


    print(
        f"[OK] All "
        f"{len(ROWS) * len(COLUMNS)} "
        f"required source entries exist."
    )


    # ========================================================
    # 12.2 Load / split / crop
    # ========================================================

    grid_images = []


    audit_lines = []


    for row_index, (
        row_name,
        file_name,
    ) in enumerate(
        ROWS
    ):

        print(
            f"\n[{row_index + 1}/{len(ROWS)}] "
            f"{row_name:<12s} "
            f"{file_name}"
        )


        raw_patches = []


        # ----------------------------------------------------
        # Load every method patch first
        # ----------------------------------------------------

        for (
            column_title,
            folder_name,
            section,
        ) in COLUMNS:

            image_path = (
                BASE_DIR
                / folder_name
                / file_name
            )


            image = load_rgb(
                image_path
            )


            patch = extract_panel(
                image,
                section,
            )


            raw_patches.append(
                patch
            )


            audit_lines.append(
                f"{row_name:12s} | "
                f"{column_title:16s} | "
                f"{folder_name:20s} | "
                f"{section:6s} | "
                f"source={image.shape} | "
                f"panel={patch.shape}"
            )


        # ----------------------------------------------------
        # Shared vertical crop
        # ----------------------------------------------------

        row_ymin, row_ymax = (
            get_joint_y_bounds(
                raw_patches
            )
        )


        print(
            f"  Shared Y crop: "
            f"[{row_ymin}, {row_ymax})"
        )


        # ----------------------------------------------------
        # Independent X white-margin removal
        #
        # NO resizing.
        # ----------------------------------------------------

        aligned_patches = []


        for (
            patch,
            (
                column_title,
                _,
                _,
            ),
        ) in zip(
            raw_patches,
            COLUMNS,
        ):

            clean = crop_x_only(
                patch,
                row_ymin,
                row_ymax,
            )


            aligned_patches.append(
                clean
            )


            print(
                f"    {column_title:<16s} "
                f"{clean.shape[1]:>4d} x "
                f"{clean.shape[0]:>4d}"
            )


        grid_images.append(
            aligned_patches
        )


    # ========================================================
    # 12.3 Column widths
    #
    # A common width is used for each COLUMN over all rows.
    #
    # IMPORTANT:
    # This is only padding, never resizing.
    # ========================================================

    n_rows = len(
        ROWS
    )

    n_cols = len(
        COLUMNS
    )


    col_widths = []


    for c in range(
        n_cols
    ):

        max_content_width = max(

            grid_images[r][c]
            .shape[1]

            for r in range(
                n_rows
            )
        )


        target_width = max(

            MIN_COLUMN_WIDTH,

            max_content_width
            + PADDING_X,
        )


        col_widths.append(
            target_width
        )


    # ========================================================
    # 12.4 Row heights
    # ========================================================

    row_heights = []


    for r in range(
        n_rows
    ):

        max_content_height = max(

            grid_images[r][c]
            .shape[0]

            for c in range(
                n_cols
            )
        )


        target_height = (
            max_content_height
            + PADDING_Y
        )


        row_heights.append(
            target_height
        )


    # ========================================================
    # 12.5 Build the compact image grid
    # ========================================================

    final_rows = []


    for r in range(
        n_rows
    ):

        row_cells = []


        for c in range(
            n_cols
        ):

            padded = pad_to_size(

                grid_images[r][c],

                target_h=
                    row_heights[r],

                target_w=
                    col_widths[c],
            )


            row_cells.append(
                padded
            )


        final_rows.append(
            np.hstack(
                row_cells
            )
        )


    point_cloud_grid = np.vstack(
        final_rows
    )


    # ========================================================
    # 12.6 Header
    # ========================================================

    h, w, channels = (
        point_cloud_grid.shape
    )


    final_canvas = np.ones(
        (
            h + HEADER_HEIGHT,
            w,
            channels,
        ),
        dtype=np.float32,
    )


    final_canvas[
        HEADER_HEIGHT:,
        :,
        :
    ] = point_cloud_grid


    # ========================================================
    # 12.7 Final figure
    #
    # Same philosophy as your old Perfect fusion code:
    #
    #     image pixel dimensions
    #         ->
    #     matplotlib figure dimensions
    #
    # This avoids subplot-generated whitespace.
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(
            w / 100.0,
            (
                h
                + HEADER_HEIGHT
            )
            / 100.0,
        ),

        dpi=PREVIEW_DPI,

        facecolor="white",
    )


    ax.imshow(
        final_canvas,

        interpolation="nearest",
    )


    ax.axis(
        "off"
    )


    # ========================================================
    # 12.8 Column titles
    # ========================================================

    x_offset = 0


    for c, (
        column_title,
        _,
        _,
    ) in enumerate(
        COLUMNS
    ):

        x_center = (
            x_offset
            + col_widths[c]
            / 2.0
        )


        # ----------------------------------------------------
        # Only the final proposed method is emphasized.
        # ----------------------------------------------------

        is_final_method = (
            column_title
            == "MBB-Net"
        )


        # Long names receive slightly smaller text.
        if len(
            column_title
        ) >= 14:

            font_size = (
                HEADER_FONT_SIZE_LONG
            )

        else:

            font_size = (
                HEADER_FONT_SIZE
            )


        ax.text(

            x_center,

            HEADER_HEIGHT
            * 0.61,

            column_title,

            fontsize=
                font_size,

            fontweight=(
                "bold"
                if is_final_method
                else "normal"
            ),

            family="serif",

            ha="center",

            va="center",

            color="black",
        )


        x_offset += (
            col_widths[c]
        )


    # ========================================================
    # 12.9 Zero external margins
    # ========================================================

    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
    )


    # ========================================================
    # 12.10 Export
    # ========================================================

    fig.savefig(

        PDF_PATH,

        format="pdf",

        bbox_inches="tight",

        pad_inches=0,

        facecolor="white",
    )


    fig.savefig(

        PNG_PATH,

        format="png",

        bbox_inches="tight",

        pad_inches=0,

        dpi=PNG_DPI,

        facecolor="white",
    )


    plt.close(
        fig
    )


    # ========================================================
    # 12.11 Audit file
    # ========================================================

    with AUDIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "PCN Qualitative Final Figure Audit\n"
        )

        handle.write(
            "=" * 80
            + "\n\n"
        )


        handle.write(
            "Important:\n"
        )

        handle.write(
            "- Source images are never resized.\n"
        )

        handle.write(
            "- Cropping removes white margins only.\n"
        )

        handle.write(
            "- Shared Y crop is used within each row.\n"
        )

        handle.write(
            "- X crop is used only for blank-margin removal.\n"
        )

        handle.write(
            "- Only MBB-Net is highlighted in the final header.\n\n"
        )


        handle.write(
            "Rows:\n"
        )

        for row_name, file_name in ROWS:

            handle.write(
                f"  {row_name:<12s} "
                f"{file_name}\n"
            )


        handle.write(
            "\nColumns:\n"
        )

        for (
            title,
            folder,
            section,
        ) in COLUMNS:

            handle.write(
                f"  {title:<16s} | "
                f"{folder:<20s} | "
                f"{section}\n"
            )


        handle.write(
            "\nSource details:\n"
        )

        for line in audit_lines:

            handle.write(
                line
                + "\n"
            )


        handle.write(
            "\nFinal column widths:\n"
        )

        for title, width in zip(
            COLUMN_TITLES_FROM_COLUMNS(),
            col_widths,
        ):

            handle.write(
                f"  {title:<16s}: "
                f"{width}px\n"
            )


        handle.write(
            "\nFinal row heights:\n"
        )

        for (
            (
                row_name,
                _,
            ),
            height,
        ) in zip(
            ROWS,
            row_heights,
        ):

            handle.write(
                f"  {row_name:<12s}: "
                f"{height}px\n"
            )


    # ========================================================
    # 12.12 Terminal summary
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PCN QUALITATIVE FINAL GENERATED"
    )

    print(
        "=" * 100
    )


    print(
        "\nFinal layout:"
    )

    print(
        "  Input | SnowflakeNet | PoinTr | SeedFormer | "
        "SymmCompletion | CompletionOnly | Full | MBB-Net | GT"
    )


    print(
        "\nRows:"
    )

    for row_name, file_name in ROWS:

        print(
            f"  {row_name:<12s} "
            f"{file_name}"
        )


    print(
        "\nVisual semantics inherited from source renderings:"
    )

    print(
        "  Input      = #B22222"
    )

    print(
        "  Prediction = #0000CD"
    )

    print(
        "  GT         = #228B22"
    )


    print(
        "\nNo source point-cloud image was resized."
    )


    print(
        "\nOutputs:"
    )

    print(
        f"  PDF:\n"
        f"    {PDF_PATH}"
    )

    print(
        f"  PNG:\n"
        f"    {PNG_PATH}"
    )

    print(
        f"  Audit:\n"
        f"    {AUDIT_PATH}"
    )


    print(
        "\nDone."
    )


# ============================================================
# 13. Convenience helper for audit output
# ============================================================

def COLUMN_TITLES_FROM_COLUMNS():

    return [
        title
        for (
            title,
            _,
            _,
        )
        in COLUMNS
    ]


# ============================================================
# 14. Entry
# ============================================================

if __name__ == "__main__":
    main()
