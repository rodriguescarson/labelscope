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

    def window(self, r0: int, r1: int, c0: int, c1: int) -> QuadMesh:
        """The sub-grid ``[r0:r1, c0:c1]`` as its own mesh."""
        return QuadMesh(
            points=self.points[r0:r1, c0:c1],
            valid=self.valid[r0:r1, c0:c1],
            meta=self.meta,
            path=self.path,
        )


class LazyQuadMesh:
    """A tifxyz surface left on disk; only the windows asked for are read.

    A published surface can run to 250 MB per axis.  Materialising all three
    axes, the stacked points, the validity mask and then the normals is several
    gigabytes, which is more than the small machine this check is meant to run
    on.  The on-sheet check only ever looks at a few dozen small blocks, so it
    can memory-map the TIFFs and page in just those.
    """

    def __init__(self, parts, meta: dict, path: str = ""):
        self._parts = parts  # (z, y, x), each a memory-mapped 2-D array
        self.meta = meta
        self.path = path

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self._parts[0].shape)

    def window(self, r0: int, r1: int, c0: int, c1: int) -> QuadMesh:
        """Read ``[r0:r1, c0:c1]`` off disk and return it as an in-memory mesh."""
        parts = [np.asarray(p[r0:r1, c0:c1], dtype=np.float32) for p in self._parts]
        points = np.stack(parts, axis=-1)
        valid = np.all(points >= 0, axis=-1) & np.all(np.isfinite(points), axis=-1)
        return QuadMesh(points=points, valid=valid, meta=self.meta, path=self.path)


def _tifxyz_paths(directory: str):
    paths = {}
    for axis in ("x", "y", "z"):
        path = os.path.join(directory, f"{axis}.tif")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{directory} is not a tifxyz surface: no {axis}.tif")
        paths[axis] = path
    return paths


def _read_meta(directory: str) -> dict:
    meta_path = os.path.join(directory, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            return json.load(handle)
    return {}


def read_tifxyz(directory: str, lazy=False):
    """Load a tifxyz surface.  Missing vertices are marked by -1 in any channel.

    ``lazy=True`` memory-maps the TIFFs instead of reading them, returning a
    :class:`LazyQuadMesh` that only supports windowed access; ``lazy="auto"``
    does so when the three files together exceed 256 MB.  Mapping needs the
    TIFFs to be uncompressed and contiguous, which is how the published corpus
    is written; a file that cannot be mapped is read whole instead.
    """
    import tifffile

    paths = _tifxyz_paths(directory)
    meta = _read_meta(directory)

    if lazy == "auto":
        lazy = sum(os.path.getsize(p) for p in paths.values()) > 256 * 1024 * 1024

    if lazy:
        try:
            parts = tuple(tifffile.memmap(paths[axis], mode="r") for axis in ("z", "y", "x"))
            if all(p.ndim == 2 for p in parts):
                return LazyQuadMesh(parts, meta=meta, path=directory)
        except (ValueError, OSError):
            pass  # not mappable: fall through and read it whole

    parts = {axis: tifffile.imread(path).astype(np.float32) for axis, path in paths.items()}
    points = np.stack([parts["z"], parts["y"], parts["x"]], axis=-1)
    valid = np.all(points >= 0, axis=-1) & np.all(np.isfinite(points), axis=-1)
    return QuadMesh(points=points, valid=valid, meta=meta, path=directory)


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #
def _sampler(volume, origin):
    """Accept either an in-memory array or a ChunkedVolume."""
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float32)
    if hasattr(volume, "sample"):
        return lambda pts: volume.sample(np.asarray(pts, np.float32) - origin), True
    array = volume.astype(np.float32, copy=False)
    return (
        lambda pts: ndi.map_coordinates(
            array,
            (np.asarray(pts, np.float32) - origin).reshape(-1, 3).T,
            order=1,
            mode="constant",
            cval=0.0,
        ),
        False,
    )


def edge_dip(mesh: QuadMesh, volume, origin=None, steps: int = 17) -> Dict[int, np.ndarray]:
    """How far the scan darkens *between* grid-adjacent vertices.

    Two vertices on the same sheet are joined by a path that stays on papyrus.
    Two vertices on different wraps are joined by a path that must cross the gap
    between them, and the gap is dark.  The statistic is therefore the depth of
    the trough along each grid edge, relative to the edge's own endpoints — which
    keeps it insensitive to how bright this part of the scroll happens to be.

    Returns ``{axis: dip}`` for axis 0 (row edges) and 1 (column edges), each with
    NaN where either endpoint is missing.
    """
    sample, remote = _sampler(volume, origin)
    fractions = np.linspace(0.0, 1.0, steps)[:, None, None, None]

    out = {}
    for axis in (0, 1):
        a = np.take(mesh.points, range(mesh.points.shape[axis] - 1), axis=axis)
        b = np.take(mesh.points, range(1, mesh.points.shape[axis]), axis=axis)
        m = np.take(mesh.valid, range(mesh.valid.shape[axis] - 1), axis=axis) & np.take(
            mesh.valid, range(1, mesh.valid.shape[axis]), axis=axis
        )
        walk = a[None] + (b - a)[None] * fractions
        flat = walk.reshape(-1, 3)
        if remote and hasattr(volume, "prefetch"):
            volume.prefetch(flat - (np.zeros(3) if origin is None else np.asarray(origin)))
        values = sample(flat).reshape(steps, *a.shape[:2])
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
        # No usable spread.  This happens on a masked volume, where the window
        # sits in the air around the scroll and every edge reads zero: the line
        # means are all 0.00 and a fallback to the standard deviation turns a
        # 0.25 grey-level wobble into z = 12.6.  That was the highest score in a
        # 56-surface sweep and it was made of nothing.  A degenerate null has no
        # z-scores in it, so say so instead of inventing them.
        return np.zeros_like(means)
    scores = (means - centre) / spread
    scores[~finite] = 0.0
    return scores


def winding_spacing(
    mesh: QuadMesh,
    volume,
    origin=None,
    reach: float = 6.0,
    n_samples: int = 2000,
    seed: int = 0,
) -> float:
    """Distance to the next wrap, measured from the scan along the surface normal.

    This is what decides whether the detector can work at all here.  A seam is
    visible because the edge crossing it dips into the gap between two wraps; if
    the mesh's own grid step is already comparable to the distance between wraps,
    every edge crosses gaps and there is no seam to find.
    """
    sample, remote = _sampler(volume, origin)
    idx = np.argwhere(mesh.valid)
    if len(idx) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(idx) > n_samples:
        idx = idx[rng.choice(len(idx), n_samples, replace=False)]
    normals = mesh.normals()
    base = mesh.points[idx[:, 0], idx[:, 1]]
    nrm = normals[idx[:, 0], idx[:, 1]]

    step = mesh.grid_step()
    span = max(reach * step, 40.0)
    offsets = np.arange(-span, span + 1.0, max(span / 120.0, 0.5), dtype=np.float32)
    walk = base[None] + nrm[None] * offsets[:, None, None]
    flat = walk.reshape(-1, 3)
    # Without this the gate fetches its chunks one at a time.  Over a remote
    # store that is latency, not bandwidth: on a 2.4 um scroll most of the walk
    # lands in masked-out air whose chunks the store simply omits, so the cost
    # is thousands of serial round trips for chunks that turn out to be empty.
    if remote and hasattr(volume, "prefetch"):
        volume.prefetch(flat - (np.zeros(3) if origin is None else np.asarray(origin)))
    values = sample(flat).reshape(offsets.size, -1)
    profile = ndi.gaussian_filter1d(values.mean(1), 3.0)
    peaks = [
        i
        for i in range(2, profile.size - 2)
        if profile[i] > profile[i - 1] and profile[i] > profile[i + 1]
    ]
    if not peaks:
        return float("nan")
    dominant = min(peaks, key=lambda i: abs(float(offsets[i])))
    centre = float(offsets[dominant])
    others = [
        abs(float(offsets[i]) - centre) for i in peaks if abs(float(offsets[i]) - centre) > 2
    ]
    return min(others) if others else float("nan")


def surface_intensity(mesh, volume, origin=None, n_samples: int = 4000, seed: int = 0):
    """Median and robust spread of the scan *at* the surface.

    The check this exists for: a masked volume reads zero in the air around the
    scroll, and a mesh can be perfectly well-formed over a region the scan does
    not cover.  Every edge then dips by nothing, the line means are all 0.00,
    and a robust z-score computed on that null reports a seam made of nothing.
    """
    sample, _ = _sampler(volume, origin)
    points = mesh.points[mesh.valid] if hasattr(mesh, "valid") else mesh.points
    if len(points) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    if len(points) > n_samples:
        points = points[rng.choice(len(points), n_samples, replace=False)]
    values = sample(np.asarray(points, np.float32))
    centre = float(np.median(values))
    return centre, float(1.4826 * np.median(np.abs(values - centre)))


def find_sheet_switches(
    mesh: QuadMesh,
    volume,
    origin=None,
    z_threshold: float = 5.0,
    steps: int = 17,
    min_dip: float = 1.0,
    check_resolution: bool = True,
) -> Dict:
    """Locate seams where the surface appears to jump to a neighbouring wrap.

    Conservative by construction: it reports a seam only where a whole grid line
    darkens together, which is what crossing the gap between two wraps looks
    like, and not where single edges happen to run over damage.
    """
    step = mesh.grid_step()
    spacing = (
        winding_spacing(mesh, volume, origin=origin) if check_resolution else float("nan")
    )
    # The seam is only visible if a grid edge normally stays on one wrap.  Once
    # the grid step approaches the distance between wraps, every edge crosses a
    # gap and the statistic saturates -- at 45.5 um on PHercParis4 the step is
    # about 18 voxels against a 12.5 voxel spacing, and the detector there is
    # measuring the scan's roughness, not sheet switches.
    adequate = bool(np.isfinite(spacing) and step <= 0.5 * spacing)
    dip = edge_dip(mesh, volume, origin=origin, steps=steps)
    result: Dict = {
        "grid_shape": list(mesh.shape),
        "grid_step": step,
        "winding_spacing": float(spacing),
        "steps_per_winding": float(spacing / step) if step else float("nan"),
        "resolution_adequate": adequate,
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
    # A z-score is a statement about a distribution, so the distribution has to
    # exist.  Where the scan is masked out the whole surface reads flat and the
    # dip is a fraction of a grey level; 11 of 56 published surfaces sat in that
    # regime, and the loudest false positive in the sweep came from it.
    level, spread = surface_intensity(mesh, volume, origin=origin)
    result["surface_intensity_median"] = level
    result["surface_intensity_mad"] = spread
    result["min_dip"] = min_dip
    result["dip_degenerate"] = bool(level < min_dip)
    if result["dip_degenerate"]:
        result["seams_degenerate"] = result.pop("seams")
        result["seams"] = []
    if not adequate:
        # report the measurement, refuse the conclusion
        result["seams_unreliable"] = result.pop("seams")
        result["seams"] = []
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
