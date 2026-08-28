"""Reading tifxyz quad meshes, and finding where one jumps to another wrap.

A traced surface is stored as three TIFFs — x, y and z — holding a 2-D grid of
3-D vertex coordinates, with -1 marking a missing vertex.  The grid is the point:
grid-adjacent vertices are meant to be adjacent *on the same papyrus sheet*.

The failure mode this module detects is the one the Open Problems post lists
fourth in its bottleneck table, "Meshes can jump from one wrap to another", and
asks conservative failure detection for.  It is invisible to the spiral
satisfaction metric, which derives its target from the patch's own position and
so scores a patch displaced by a whole winding identically to a correct one
(ScrollPrize/villa#1621).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi


@dataclass
class QuadMesh:
    """A tifxyz surface: ``points`` is (rows, cols, 3) in z, y, x order."""

    points: np.ndarray
    valid: np.ndarray
    meta: dict
    path: str = ""

    @property
    def shape(self) -> Tuple[int, int]:
        return self.points.shape[:2]

    def grid_step(self) -> float:
        """Median 3-D distance between grid-adjacent vertices."""
        steps = []
        for axis in (0, 1):
            a = np.take(self.points, range(self.points.shape[axis] - 1), axis=axis)
            b = np.take(self.points, range(1, self.points.shape[axis]), axis=axis)
            m = np.take(self.valid, range(self.valid.shape[axis] - 1), axis=axis) & np.take(
                self.valid, range(1, self.valid.shape[axis]), axis=axis
            )
            if m.any():
                steps.append(np.linalg.norm(b - a, axis=-1)[m])
        return float(np.median(np.concatenate(steps))) if steps else float("nan")

    def normals(self) -> np.ndarray:
        du = np.gradient(self.points, axis=0)
        dv = np.gradient(self.points, axis=1)
        n = np.cross(du, dv)
        length = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / (length + 1e-9)

    def bounds(self, margin: int = 0):
        pts = self.points[self.valid]
        lo = np.floor(pts.min(0)).astype(int) - margin
        hi = np.ceil(pts.max(0)).astype(int) + margin
        return lo, hi


def read_tifxyz(directory: str) -> QuadMesh:
    """Load a tifxyz surface.  Missing vertices are marked by -1 in any channel."""
    import tifffile

    parts = {}
    for axis in ("x", "y", "z"):
        path = os.path.join(directory, f"{axis}.tif")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{directory} is not a tifxyz surface: no {axis}.tif")
        parts[axis] = tifffile.imread(path).astype(np.float32)

    meta_path = os.path.join(directory, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            meta = json.load(handle)

    points = np.stack([parts["z"], parts["y"], parts["x"]], axis=-1)
    valid = np.all(points >= 0, axis=-1) & np.all(np.isfinite(points), axis=-1)
    return QuadMesh(points=points, valid=valid, meta=meta, path=directory)


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #
def edge_dip(
    mesh: QuadMesh, volume: np.ndarray, origin=None, steps: int = 17
) -> Dict[int, np.ndarray]:
    """How far the scan darkens *between* grid-adjacent vertices.

    Two vertices on the same sheet are joined by a path that stays on papyrus.
    Two vertices on different wraps are joined by a path that must cross the gap
    between them, and the gap is dark.  The statistic is therefore the depth of
    the trough along each grid edge, relative to the edge's own endpoints — which
    keeps it insensitive to how bright this part of the scroll happens to be.

    Returns ``{axis: dip}`` for axis 0 (row edges) and 1 (column edges), each with
    NaN where either endpoint is missing.
    """
    origin = np.zeros(3) if origin is None else np.asarray(origin)
    local = mesh.points - origin
    volume = volume.astype(np.float32, copy=False)
    fractions = np.linspace(0.0, 1.0, steps)[:, None, None, None]

    out = {}
    for axis in (0, 1):
        a = np.take(local, range(local.shape[axis] - 1), axis=axis)
        b = np.take(local, range(1, local.shape[axis]), axis=axis)
        m = np.take(mesh.valid, range(mesh.valid.shape[axis] - 1), axis=axis) & np.take(
            mesh.valid, range(1, mesh.valid.shape[axis]), axis=axis
        )
        walk = a[None] + (b - a)[None] * fractions
        values = ndi.map_coordinates(
            volume, walk.reshape(-1, 3).T, order=1, mode="constant", cval=0.0
        ).reshape(steps, *a.shape[:2])
        dip = np.minimum(values[0], values[-1]) - values[1:-1].min(0)
        dip[~m] = np.nan
        out[axis] = dip
    return out


def _line_scores(dip: np.ndarray, along: int) -> np.ndarray:
    """Robust z-score of each grid line's mean dip, against the other lines.

    A sheet switch is a *seam*: a whole line of edges crosses the gap at once.
    Judging each edge on its own drowns that in the ordinary roughness of the
    scan — the seam is about 1% of the edges in a mesh — so the dip is averaged
    along the seam direction first.
    """
    with np.errstate(invalid="ignore"):
        means = np.nanmean(dip, axis=along)
    finite = np.isfinite(means)
    if finite.sum() < 4:
        return np.zeros_like(means)
    centre = np.median(means[finite])
    spread = 1.4826 * np.median(np.abs(means[finite] - centre))
    if spread < 1e-6:
        spread = float(np.std(means[finite])) or 1.0
    scores = (means - centre) / spread
    scores[~finite] = 0.0
    return scores


def find_sheet_switches(
    mesh: QuadMesh,
    volume: np.ndarray,
    origin=None,
    z_threshold: float = 5.0,
    steps: int = 17,
) -> Dict:
    """Locate seams where the surface appears to jump to a neighbouring wrap.

    Conservative by construction: it reports a seam only where a whole grid line
    darkens together, which is what crossing the gap between two wraps looks
    like, and not where single edges happen to run over damage.
    """
    dip = edge_dip(mesh, volume, origin=origin, steps=steps)
    result: Dict = {
        "grid_shape": list(mesh.shape),
        "grid_step": mesh.grid_step(),
        "z_threshold": z_threshold,
        "seams": [],
    }
    for axis, along in ((0, 1), (1, 0)):
        scores = _line_scores(dip[axis], along=along)
        flagged = np.flatnonzero(scores >= z_threshold)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(dip[axis], axis=along)
        for line in flagged:
            result["seams"].append(
                {
                    "axis": int(axis),
                    "line": int(line),
                    "z": float(scores[line]),
                    "mean_dip": float(means[line]),
                    "edges": int(np.isfinite(np.take(dip[axis], line, axis=1 - along)).sum()),
                }
            )
        result[f"axis{axis}_max_z"] = float(scores.max()) if scores.size else 0.0
        result[f"axis{axis}_median_dip"] = float(np.nanmedian(means)) if means.size else 0.0
    result["seams"].sort(key=lambda s: -s["z"])
    result["n_seams"] = len(result["seams"])
    result["max_z"] = max((result.get("axis0_max_z", 0.0), result.get("axis1_max_z", 0.0)))
    return result


def displace(mesh: QuadMesh, distance: float, region=None) -> QuadMesh:
    """Return a copy with part of the surface pushed along its own normal.

    Used to plant a known sheet switch: displacing by a whole winding is the
    move that villa's satisfaction metric scores as no change at all.
    """
    points = mesh.points.copy()
    normals = mesh.normals()
    if region is None:
        region = np.zeros(mesh.shape, dtype=bool)
        region[:, mesh.shape[1] // 2 :] = True
    points[region] = points[region] + normals[region] * distance
    return QuadMesh(
        points=points, valid=mesh.valid.copy(), meta=dict(mesh.meta), path=mesh.path
    )
