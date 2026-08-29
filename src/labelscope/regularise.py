"""Removing the part of a surface label's placement that is not the convention.

`aggregate_alignment` measures how far a labelled surface sits from the scan's
own ridge, per cell of the volume.  Two things are mixed into that number.

One is a **convention**: the labels mark the recto face, the side the ink is on,
so they sit a consistent distance off the sheet's density maximum.  That is
intended, it is the same in both public releases to within a third of a voxel,
and moving the labels onto the maximum would fight it.

The other is **wobble**: the same patch's cells disagree with each other, with a
p90 |offset| well above the spread the convention alone would produce.  A model
trained on that is being told the surface is in slightly different places in
different regions of the same patch.

This module removes the second while preserving the first.  The target for every
cell is the patch's *own* global offset, not zero, so a patch labelled
consistently 2.3 voxels off its ridge comes out unchanged.  Only the deviation
from the patch's own convention is taken out.

Nothing is moved where the measurement had no power: a cell whose profile has no
peak in the window, or whose peak is below the SNR floor, is not reported by
`aggregate_alignment` at all, and its region of the field is left at zero.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from labelscope.alignment import (
    aggregate_alignment,
    orient_by_intensity,
    orient_normals,
    point_normals,
    propagate_orientation,
)


def delta_field(
    shape: Tuple[int, int, int],
    cells,
    global_offset: float,
    cell: int,
    smooth: float = 0.5,
) -> np.ndarray:
    """A smooth per-voxel correction, in voxels, from the per-cell offsets.

    A per-cell shift applied as-is would step at every cell boundary, and a step
    in a surface is exactly the seam `sheetswitch` looks for -- the correction
    would manufacture the defect the rest of the tool detects.  So the cell
    deltas are laid on a coarse grid, smoothed there, and interpolated up.

    ``smooth`` is in *cells*, not voxels, and it costs signal: a 300 cubed patch
    at ``cell=64`` is only five cells across, so smoothing of 1.0 spans most of
    the patch and damps a one-cell excursion to a quarter of its size.  Measured
    on the wobble fixture, the residual spread after correction runs 0.64x at
    ``smooth=0``, 0.58x at 0.3 and 0.5, 0.66x at 0.8 and 0.89x at 1.0 -- so the
    default sits at the bottom of that curve rather than at the smoothest end.
    """
    coarse_shape = tuple(int(np.ceil(s / cell)) for s in shape)
    coarse = np.zeros(coarse_shape, dtype=np.float32)
    known = np.zeros(coarse_shape, dtype=bool)
    for c in cells:
        k = tuple(int(v) for v in c["key"])
        if all(0 <= k[i] < coarse_shape[i] for i in range(3)):
            coarse[k] = float(c["offset"]) - global_offset
            known[k] = True
    if not known.any():
        return np.zeros(shape, dtype=np.float32)
    if smooth > 0 and known.sum() > 1:
        # Smooth only over the cells that carry a measurement, then normalise by
        # the smoothed mask.  Blurring the zeros of unmeasured cells into their
        # neighbours would pull real corrections toward zero for no reason.
        num = ndi.gaussian_filter(coarse * known, smooth, mode="nearest")
        den = ndi.gaussian_filter(known.astype(np.float32), smooth, mode="nearest")
        coarse = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        coarse[~ndi.binary_dilation(known, iterations=max(1, int(round(smooth))))] = 0.0

    grid = np.meshgrid(
        *[(np.arange(s, dtype=np.float32) + 0.5) / cell - 0.5 for s in shape],
        indexing="ij",
    )
    return (
        ndi.map_coordinates(
            coarse, np.stack([g.ravel() for g in grid]), order=1, mode="nearest"
        )
        .reshape(shape)
        .astype(np.float32)
    )


def regularise_label(
    image: np.ndarray,
    mask: np.ndarray,
    cell: int = 64,
    smooth: float = 0.5,
    threshold: float = 0.5,
    max_shift: float = 4.0,
    alignment: Optional[Dict] = None,
    orient_field: Optional[np.ndarray] = None,
    **align_kwargs,
) -> Tuple[np.ndarray, Dict]:
    """Warp a surface label so its cells agree with the patch's own convention.

    Returns the new mask and a report of what was done, including the case where
    nothing was: an unreliable global measurement, or no resolved cell, leaves
    the label exactly as it came in.
    """
    if alignment is None:
        alignment = aggregate_alignment(
            image, mask, cell=cell, return_cells=True, **align_kwargs
        )

    report: Dict = {
        "n_cells": int(alignment.get("n_cells", 0)),
        "global_offset": alignment.get("global_peak_offset_raw"),
        "global_offset_reliable": bool(alignment.get("global_offset_reliable", False)),
        "changed": False,
        "voxels_before": int(mask.sum()),
    }
    cells = alignment.get("cells") or []
    if not cells or not report["global_offset_reliable"]:
        report["reason"] = "no resolved cells" if not cells else "global offset unreliable"
        report["voxels_after"] = report["voxels_before"]
        return mask.copy(), report

    field = delta_field(
        mask.shape, cells, float(alignment["global_peak_offset_raw"]), cell, smooth=smooth
    )
    np.clip(field, -max_shift, max_shift, out=field)
    report["max_abs_shift"] = float(np.abs(field).max())
    report["mean_abs_shift_on_label"] = (
        float(np.abs(field)[mask].mean()) if mask.any() else 0.0
    )
    if report["max_abs_shift"] < 1e-3:
        report["reason"] = "every cell already agrees with the patch"
        report["voxels_after"] = report["voxels_before"]
        return mask.copy(), report

    # Everything below is done inside the band the label can actually reach.
    # A 364 cubed patch would otherwise need a (3, 364, 364, 364) index array
    # for the nearest-neighbour normals and two more for the warp, which is
    # several GB per patch and what broke a 32-way parallel run.
    box = _band_box(mask, int(np.ceil(max_shift)) + 2)
    sub_mask = mask[box]
    sub_image = image[box]
    sub_field = field[box]

    normals = _dense_normals(sub_image, sub_mask, orient_field=_crop(orient_field, box))

    coords = np.indices(sub_mask.shape, dtype=np.float32)
    # new(p) = old(p - d(p) n(p)): a cell whose label reads +d relative to the
    # patch is pulled back by d along its own normal.
    source = coords - normals * sub_field[None]
    del coords, normals
    warped = ndi.map_coordinates(
        sub_mask.astype(np.float32),
        source.reshape(3, -1),
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(sub_mask.shape)
    del source
    out = np.zeros_like(mask)
    out[box] = warped >= threshold
    report["changed"] = bool((out != mask).any())
    report["voxels_after"] = int(out.sum())
    report["voxels_moved"] = int((out != mask).sum())
    return out, report


def _band_box(mask: np.ndarray, margin: int):
    """Bounding box of the label, grown by the furthest it may be moved."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return tuple(slice(0, s) for s in mask.shape)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _crop(field, box):
    return None if field is None else field[box]


def _dense_normals(
    image: np.ndarray, mask: np.ndarray, orient_field: Optional[np.ndarray] = None
) -> np.ndarray:
    """A (3, Z, Y, X) normal field, defined everywhere, not only on the label.

    The label has to be able to *move*, and the voxels it moves into are by
    definition not labelled yet.  A field that exists only on the mask leaves
    those voxels with a zero displacement and the label stays exactly where it
    was -- which looks like a working no-op and is really a broken warp.  Each
    voxel therefore borrows the normal of the nearest labelled voxel.

    The orientation has to be *the same* pipeline the measurement used --
    locally estimated, made consistent by MST propagation, then pointed either
    by an external field or by the scan's own intensity.  An offset measured
    along one orientation and applied along another is a correction with the
    wrong sign wherever the two disagree, which makes the label worse in exactly
    the places it was already worst.
    """
    coords = np.argwhere(mask)
    points = coords.astype(np.float32)
    normals = point_normals(coords, points)
    normals, components = propagate_orientation(points, normals)
    if orient_field is not None:
        normals = orient_normals(normals, points.T, orient_field, components=components)
    else:
        normals = orient_by_intensity(image, points.T, normals, components=components)

    nearest = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    lookup = -np.ones(mask.shape, dtype=np.int64)
    lookup[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(len(coords))
    index = lookup[tuple(nearest)]
    return np.asarray(normals, dtype=np.float32)[:, index]
