#!/usr/bin/env python3
"""Regenerate the figures in the README and findings, from the same fixtures.

Every panel here is produced by running the tool, not by drawing what it would
look like, so a figure that stops being true stops being generated.

    pip install matplotlib
    python scripts/make_figures.py --out docs/img
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

INK = "#14171A"
MUTED = "#5B626B"
SIGNAL = "#A81458"
QUIET = "#12626E"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C9CCC7")
    ax.grid(alpha=0.25, linewidth=0.6)


def figure_seam(out: str) -> str:
    """What a sheet switch looks like, and what a clean surface looks like."""
    from labelscope.mesh import _line_scores, displace, edge_dip
    from test_mesh import mesh_on_sheet, wrapped_volume

    volume = wrapped_volume(spacing=24.0)
    clean = mesh_on_sheet(rows=40, cols=40, step=3.0)
    moved = displace(clean, 24.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=170)
    limits = []
    for ax, mesh, label, colour in (
        (axes[0], clean, "as published", QUIET),
        (axes[1], moved, "one winding planted in half the grid", SIGNAL),
    ):
        dip = edge_dip(mesh, volume, steps=17)
        means = np.nanmean(dip[1], axis=0)
        scores = _line_scores(dip[1], along=0)
        ax.plot(means, color=colour, linewidth=1.4)
        peak = int(np.nanargmax(means))
        ax.text(
            0.04,
            0.92,
            f"max z = {scores[peak]:.1f}   peak darkening {means[peak]:.1f}",
            transform=ax.transAxes,
            fontsize=9,
            color=colour,
            va="top",
        )
        style(ax, label, "grid line", "mean darkening along the line")
        limits.append(float(np.nanmax(means)))
    for ax in axes:
        ax.set_ylim(-2, max(limits) * 1.25)
    fig.suptitle(
        "A seam is a whole grid line crossing the gap between two wraps",
        fontsize=11.5,
        color=INK,
        x=0.007,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out, "seam.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def figure_degenerate(out: str) -> str:
    """Why a z-score needs a distribution to be a z-score."""
    from labelscope.mesh import QuadMesh, edge_dip, find_sheet_switches
    from test_mesh import masked_volume, mesh_on_sheet

    volume = masked_volume()
    covered = mesh_on_sheet(rows=40, cols=40, step=3.0)
    absent = QuadMesh(
        points=covered.points + np.array([0.0, 0.0, 170.0], np.float32),
        valid=covered.valid.copy(),
        meta={},
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=170)
    for ax, mesh, label in (
        (axes[0], covered, "where the scan covers the surface"),
        (axes[1], absent, "where the scan is masked out"),
    ):
        dip = edge_dip(mesh, volume, steps=17)
        means = np.nanmean(dip[1], axis=0)
        result = find_sheet_switches(mesh, volume)
        colour = SIGNAL if result["dip_degenerate"] else QUIET
        ax.plot(means, color=colour, linewidth=1.4)
        verdict = (
            "refused: dip_degenerate"
            if result["dip_degenerate"]
            else f"scored, {result['n_seams']} seam(s)"
        )
        ax.text(
            0.03,
            0.88,
            f"median intensity at the surface: {result['surface_intensity_median']:.2f}\n{verdict}",
            transform=ax.transAxes,
            fontsize=8.5,
            color=colour,
            va="top",
        )
        style(ax, label, "grid line", "mean darkening along the line")
    axes[1].set_ylim(axes[0].get_ylim())
    fig.suptitle(
        "A null with no spread in it has no z-scores in it",
        fontsize=11.5,
        color=INK,
        x=0.007,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out, "degenerate.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def figure_radius(out: str) -> str:
    """The failure that motivated the aggregated estimator."""
    from labelscope.alignment import aggregate_alignment, ridge_alignment
    from test_aggregate import scroll_like, sheet_label, upward

    shape = (64, 96, 96)
    volume = scroll_like(shape)
    mask = sheet_label(shape, 32)
    field = upward(shape)
    radii = [2, 3, 4, 5, 6, 7, 8, 9]
    naive = [
        ridge_alignment(volume, mask, radius=r, n_samples=3000, orient_field=field)[
            "median_abs_offset"
        ]
        for r in radii
    ]
    aggregated = [
        abs(
            aggregate_alignment(
                volume,
                mask,
                radius=r,
                orient_field=field,
                orient_by="field",
                n_samples=6000,
                bootstrap=0,
            )["global_peak_offset_raw"]
        )
        for r in radii
    ]

    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=170)
    ax.plot(
        radii,
        naive,
        "o-",
        color=SIGNAL,
        linewidth=1.5,
        markersize=4,
        label="nearest intensity maximum, per voxel",
    )
    ax.plot(
        radii,
        aggregated,
        "o-",
        color=QUIET,
        linewidth=1.5,
        markersize=4,
        label="aggregated over a 64-voxel cell",
    )
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    style(
        ax,
        "A correctly placed label, measured two ways",
        "search radius (voxels)",
        "reported |offset| (voxels)",
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = os.path.join(out, "radius.png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for build in (figure_seam, figure_degenerate, figure_radius):
        path = build(args.out)
        print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
