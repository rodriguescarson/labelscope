"""Rendering the findings as images.

Numbers say a dataset has a problem; a picture says *where*.  Two renderers:

``render_overlay``   CT cross-section with the label outlined on top — the
                     "is this label even on a sheet?" view.
``render_drift_map`` the same section with each labelled voxel coloured by its
                     signed offset from the CT ridge — blue where the label sits
                     inboard of the sheet, red where it sits outboard.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


def _to_gray(slice2d: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Contrast-stretch a CT slice to 0-255 uint8."""
    data = slice2d.astype(np.float32)
    lo, hi = np.percentile(data, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip(255.0 * (data - lo) / (hi - lo), 0, 255).astype(np.uint8)


def _outline(mask2d: np.ndarray) -> np.ndarray:
    """One-voxel boundary of a 2-D mask."""
    return mask2d & ~ndi.binary_erosion(mask2d, np.ones((3, 3), bool))


def _save(rgb: np.ndarray, path: str, scale: int = 1) -> str:
    from PIL import Image

    image = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        image = image.resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    image.save(path)
    return path


def render_overlay(
    image: np.ndarray,
    label: np.ndarray,
    path: str,
    axis: int = 0,
    index: Optional[int] = None,
    surface_class: int = 1,
    crop: Optional[Tuple[int, int, int, int]] = None,
    scale: int = 1,
) -> str:
    """CT cross-section with the surface label outlined in red."""
    index = image.shape[axis] // 2 if index is None else index
    ct = np.take(image, index, axis=axis)
    lab = np.take(label, index, axis=axis) == surface_class
    if crop:
        y0, y1, x0, x1 = crop
        ct, lab = ct[y0:y1, x0:x1], lab[y0:y1, x0:x1]

    gray = _to_gray(ct)
    rgb = np.stack([gray] * 3, axis=-1)
    edge = _outline(lab)
    rgb[edge] = (230, 40, 40)
    return _save(rgb, path, scale)


def render_drift_map(
    image: np.ndarray,
    label: np.ndarray,
    path: str,
    axis: int = 0,
    index: Optional[int] = None,
    surface_class: int = 1,
    orient_field: Optional[np.ndarray] = None,
    clip: float = 2.0,
    crop: Optional[Tuple[int, int, int, int]] = None,
    scale: int = 1,
) -> Optional[str]:
    """CT section with the label coloured by its signed offset from the ridge.

    Blue is a label sitting inboard of the sheet the scan shows, red is outboard,
    grey-green is on it.  ``clip`` sets the colour saturation point, in voxels.
    """
    from labelscope.alignment import _sample_at, orient_normals, point_normals

    index = image.shape[axis] // 2 if index is None else index
    mask3d = label == surface_class
    coords = np.argwhere(mask3d)
    if coords.shape[0] == 0:
        return None

    on_slice = coords[coords[:, axis] == index]
    if on_slice.shape[0] == 0:
        return None

    points = on_slice.astype(np.float32)
    normals = point_normals(coords, points)
    if orient_field is not None:
        normals = orient_normals(normals, points.T, orient_field)

    step, radius = 0.25, 6
    offsets = np.arange(-radius, radius + step / 2, step, dtype=np.float32)
    walk = points.T[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    profile = _sample_at(image.astype(np.float32), walk.reshape(3, -1)).reshape(
        offsets.size, -1
    )
    smooth = ndi.gaussian_filter1d(profile, 2.0, axis=0)
    signed = (np.argmax(smooth, axis=0) - (offsets.size - 1) / 2.0) * step

    ct = np.take(image, index, axis=axis)
    gray = _to_gray(ct)
    rgb = np.stack([gray] * 3, axis=-1) // 2 + 40  # dim the background

    plane_axes = [a for a in range(3) if a != axis]
    rows = on_slice[:, plane_axes[0]]
    cols = on_slice[:, plane_axes[1]]
    scaled = np.clip(signed / clip, -1.0, 1.0)
    red = (255 * np.clip(scaled, 0, 1)).astype(np.uint8)
    blue = (255 * np.clip(-scaled, 0, 1)).astype(np.uint8)
    green = (200 * (1.0 - np.abs(scaled))).astype(np.uint8)
    rgb[rows, cols] = np.stack([red, green, blue], axis=-1)

    if crop:
        y0, y1, x0, x1 = crop
        rgb = rgb[y0:y1, x0:x1]
    return _save(rgb, path, scale)
