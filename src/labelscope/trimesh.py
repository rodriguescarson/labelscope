"""Reading triangular surface meshes, and finding sheet switches on them.

`mesh.py` detects a sheet switch on a tifxyz quad mesh, where the seam is a
whole *grid line* of edges.  Triangular meshes -- OBJ and PLY, the other formats
the community exchanges surfaces in -- have no grid, so the seam has to be
described without one.

The observation the quad detector rests on does not need a grid.  Two vertices
on the same sheet are joined by an edge that stays on papyrus; two vertices on
different wraps are joined by an edge that crosses the dark gap between them.
What the grid provided was only a way to require that the darkening be
*collective*: a seam is many edges failing together, not one edge running over
damage.  On a triangular mesh the same requirement is a connected component of
the flagged-edge subgraph -- a cut through the surface rather than a line of the
grid.  A single unlucky edge is a component of size one and is discarded by the
same rule that makes the quad detector conservative.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from labelscope.mesh import _sampler


@dataclass
class TriMesh:
    """A triangular surface.  ``points`` is (V, 3) in z, y, x order."""

    points: np.ndarray
    faces: np.ndarray
    path: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def edges(self) -> np.ndarray:
        """Unique undirected edges, (E, 2), each pair sorted ascending."""
        f = self.faces
        pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        pairs = np.sort(pairs, axis=1)
        return np.unique(pairs, axis=0)

    def edge_length(self) -> float:
        """Median edge length -- the triangular counterpart of ``grid_step``."""
        e = self.edges()
        if len(e) == 0:
            return float("nan")
        d = np.linalg.norm(self.points[e[:, 0]] - self.points[e[:, 1]], axis=1)
        d = d[np.isfinite(d)]
        return float(np.median(d)) if d.size else float("nan")

    def normals(self) -> np.ndarray:
        """Area-weighted vertex normals, unit length, arbitrary sign."""
        p = self.points
        f = self.faces
        fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
        out = np.zeros_like(p)
        for k in range(3):
            np.add.at(out, f[:, k], fn)
        length = np.linalg.norm(out, axis=1, keepdims=True)
        return out / (length + 1e-9)

    def bounds(self, margin: int = 0):
        lo = np.floor(self.points.min(0)).astype(int) - margin
        hi = np.ceil(self.points.max(0)).astype(int) + margin
        return lo, hi


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def _to_zyx(xyz: np.ndarray) -> np.ndarray:
    """OBJ and PLY store x, y, z; everything downstream indexes z, y, x."""
    return np.ascontiguousarray(xyz[:, ::-1])


def _fan(indices: List[int]) -> List[Tuple[int, int, int]]:
    """Triangulate one polygon face by a fan.  Quads are the common case."""
    return [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]


_OBJ_INDEX = re.compile(r"^(-?\d+)")


def read_obj(path: str) -> TriMesh:
    """Read a Wavefront OBJ.  Polygon faces are fanned; ``v`` lines only."""
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    with open(path, errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                bits = line.split()
                verts.append((float(bits[1]), float(bits[2]), float(bits[3])))
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    match = _OBJ_INDEX.match(token)
                    if match is None:
                        continue
                    value = int(match.group(1))
                    # OBJ is 1-based, and negative indices count back from the
                    # end of the vertex list read *so far*.
                    idx.append(value - 1 if value > 0 else len(verts) + value)
                if len(idx) >= 3:
                    faces.extend(_fan(idx))
    if not verts:
        raise ValueError(f"{path} contains no vertices")
    if not faces:
        raise ValueError(f"{path} contains no faces")
    return TriMesh(
        points=_to_zyx(np.asarray(verts, dtype=np.float32)),
        faces=np.asarray(faces, dtype=np.int64),
        path=path,
    )


_PLY_NUMPY = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
_PLY_STRUCT = {
    "i1": "b",
    "u1": "B",
    "i2": "h",
    "u2": "H",
    "i4": "i",
    "u4": "I",
    "f4": "f",
    "f8": "d",
}


def _ply_header(handle):
    """Parse a PLY header, returning (format, elements) with the stream past it."""
    magic = handle.readline().strip()
    if magic != b"ply":
        raise ValueError("not a PLY file: missing magic")
    fmt = None
    elements: List[dict] = []
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("truncated PLY header")
        bits = line.split()
        if not bits:
            continue
        key = bits[0]
        if key == b"format":
            fmt = bits[1].decode()
        elif key == b"element":
            elements.append({"name": bits[1].decode(), "count": int(bits[2]), "props": []})
        elif key == b"property":
            if not elements:
                raise ValueError("PLY property outside any element")
            if bits[1] == b"list":
                elements[-1]["props"].append(
                    {
                        "list": True,
                        "count_type": _PLY_NUMPY[bits[2].decode()],
                        "type": _PLY_NUMPY[bits[3].decode()],
                        "name": bits[4].decode(),
                    }
                )
            else:
                elements[-1]["props"].append(
                    {
                        "list": False,
                        "type": _PLY_NUMPY[bits[1].decode()],
                        "name": bits[2].decode(),
                    }
                )
        elif key == b"end_header":
            break
    if fmt is None:
        raise ValueError("PLY header declares no format")
    return fmt, elements


def read_ply(path: str) -> TriMesh:
    """Read a PLY.  ``ascii`` and ``binary_little_endian`` are supported."""
    with open(path, "rb") as handle:
        fmt, elements = _ply_header(handle)
        if fmt not in ("ascii", "binary_little_endian"):
            raise ValueError(f"unsupported PLY format: {fmt}")
        verts = None
        faces: List[Tuple[int, int, int]] = []
        for element in elements:
            if fmt == "ascii":
                rows = [handle.readline().split() for _ in range(element["count"])]
                if element["name"] == "vertex":
                    names = [p["name"] for p in element["props"]]
                    cols = [names.index(a) for a in ("x", "y", "z")]
                    verts = np.array(
                        [[float(r[c]) for c in cols] for r in rows], dtype=np.float32
                    )
                elif element["name"] == "face":
                    for row in rows:
                        n = int(row[0])
                        faces.extend(_fan([int(v) for v in row[1 : 1 + n]]))
            else:
                simple = all(not p["list"] for p in element["props"])
                if simple:
                    dtype = np.dtype([(p["name"], "<" + p["type"]) for p in element["props"]])
                    block = np.frombuffer(
                        handle.read(dtype.itemsize * element["count"]),
                        dtype=dtype,
                        count=element["count"],
                    )
                    if element["name"] == "vertex":
                        verts = np.stack([block["x"], block["y"], block["z"]], axis=1).astype(
                            np.float32
                        )
                else:
                    for _ in range(element["count"]):
                        values = {}
                        for prop in element["props"]:
                            if prop["list"]:
                                ct = _PLY_STRUCT[prop["count_type"]]
                                n = struct.unpack("<" + ct, handle.read(struct.calcsize(ct)))[
                                    0
                                ]
                                st = _PLY_STRUCT[prop["type"]]
                                raw = handle.read(struct.calcsize(st) * n)
                                values[prop["name"]] = list(struct.unpack("<" + st * n, raw))
                            else:
                                st = _PLY_STRUCT[prop["type"]]
                                values[prop["name"]] = struct.unpack(
                                    "<" + st, handle.read(struct.calcsize(st))
                                )[0]
                        if element["name"] == "face":
                            idx = values.get("vertex_indices") or values.get("vertex_index")
                            if idx and len(idx) >= 3:
                                faces.extend(_fan([int(v) for v in idx]))
    if verts is None or len(verts) == 0:
        raise ValueError(f"{path} contains no vertices")
    if not faces:
        raise ValueError(f"{path} contains no faces")
    return TriMesh(points=_to_zyx(verts), faces=np.asarray(faces, dtype=np.int64), path=path)


def read_trimesh(path: str) -> TriMesh:
    """Read a triangular mesh, dispatching on the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return read_obj(path)
    if ext == ".ply":
        return read_ply(path)
    raise ValueError(f"unsupported mesh format: {ext or path!r} (expected .obj or .ply)")


def from_quad(mesh) -> TriMesh:
    """Triangulate a :class:`~labelscope.mesh.QuadMesh`, dropping missing vertices.

    Used to check the two detectors against each other on the same surface.
    """
    rows, cols = mesh.shape
    index = -np.ones((rows, cols), dtype=np.int64)
    flat = mesh.points[mesh.valid]
    index[mesh.valid] = np.arange(len(flat))
    quad = (
        mesh.valid[:-1, :-1] & mesh.valid[1:, :-1] & mesh.valid[:-1, 1:] & mesh.valid[1:, 1:]
    )
    r, c = np.nonzero(quad)
    a, b = index[r, c], index[r + 1, c]
    d, e = index[r, c + 1], index[r + 1, c + 1]
    faces = np.concatenate([np.stack([a, b, d], axis=1), np.stack([b, e, d], axis=1)], axis=0)
    return TriMesh(points=flat, faces=faces, path=getattr(mesh, "path", ""))


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #
def edge_dip(mesh: TriMesh, volume, origin=None, steps: int = 17) -> np.ndarray:
    """How far the scan darkens along each edge, relative to its own endpoints.

    Same statistic as :func:`labelscope.mesh.edge_dip`, over an unstructured
    edge list instead of the two grid axes.
    """
    sample, remote = _sampler(volume, origin)
    e = mesh.edges()
    if len(e) == 0:
        return np.zeros(0, dtype=np.float32)
    a = mesh.points[e[:, 0]]
    b = mesh.points[e[:, 1]]
    fractions = np.linspace(0.0, 1.0, steps)[:, None, None]
    walk = a[None] + (b - a)[None] * fractions
    flat = walk.reshape(-1, 3)
    if remote and hasattr(volume, "prefetch"):
        volume.prefetch(flat - (np.zeros(3) if origin is None else np.asarray(origin)))
    values = sample(flat).reshape(steps, len(e))
    return np.minimum(values[0], values[-1]) - values[1:-1].min(0)


def _components(edges: np.ndarray, flagged: np.ndarray) -> List[np.ndarray]:
    """Connected components of the subgraph made of the flagged edges."""
    picked = np.flatnonzero(flagged)
    if picked.size == 0:
        return []
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in picked:
        union(int(edges[i, 0]), int(edges[i, 1]))
    groups: Dict[int, List[int]] = {}
    for i in picked:
        groups.setdefault(find(int(edges[i, 0])), []).append(int(i))
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def winding_spacing(
    mesh: TriMesh,
    volume,
    origin=None,
    reach: float = 6.0,
    n_samples: int = 2000,
    seed: int = 0,
) -> float:
    """Distance to the next wrap, measured from the scan along the surface normal.

    The triangular counterpart of :func:`labelscope.mesh.winding_spacing`, and it
    plays the same gatekeeping role: below two edges per winding there is no seam
    to find, because every edge already crosses a gap.
    """
    from scipy import ndimage as ndi

    sample, remote = _sampler(volume, origin)
    rng = np.random.default_rng(seed)
    idx = np.arange(mesh.n_vertices)
    if idx.size == 0:
        return float("nan")
    if idx.size > n_samples:
        idx = rng.choice(idx, n_samples, replace=False)
    base = mesh.points[idx]
    nrm = mesh.normals()[idx]

    step = mesh.edge_length()
    span = max(reach * step, 40.0)
    offsets = np.arange(-span, span + 1.0, max(span / 120.0, 0.5), dtype=np.float32)
    walk = base[None] + nrm[None] * offsets[:, None, None]
    flat = walk.reshape(-1, 3)
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


def _span_fraction(points: np.ndarray, extent: float) -> float:
    """How far a set of vertices reaches, along its own principal direction.

    Measured as a fraction of the mesh's largest extent, so it answers "does
    this run across the surface?" rather than "is it big in voxels?".
    """
    if len(points) < 2 or not np.isfinite(extent) or extent <= 0:
        return 0.0
    centred = points - points.mean(0)
    _, _, basis = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ basis[0]
    return float(projected.max() - projected.min()) / extent


def find_sheet_switches(
    mesh: TriMesh,
    volume,
    origin=None,
    z_threshold: float = 5.0,
    steps: int = 17,
    min_edges: int = 4,
    min_span: float = 0.4,
    min_dip: float = 1.0,
    check_resolution: bool = True,
) -> Dict:
    """Locate seams on a triangular surface.

    On a quad mesh the conservatism comes from the grid: a seam is reported only
    when a whole grid *line* darkens, which is exactly a cut running the width of
    the surface.  Without a grid the same requirement has to be stated
    geometrically.  A component of flagged edges counts as a seam only if it
    reaches across at least ``min_span`` of the mesh's own largest extent -- a
    sheet switch cuts across the surface, while damage, a hole edge or one
    unlucky edge does not.

    On the planted-displacement fixture the separation is not marginal: the real
    seam spans 1.00 of the surface and the largest noise component spans 0.15.
    """
    step = mesh.edge_length()
    spacing = (
        winding_spacing(mesh, volume, origin=origin) if check_resolution else float("nan")
    )
    adequate = bool(np.isfinite(spacing) and step <= 0.5 * spacing)

    from labelscope.mesh import surface_intensity

    level, intensity_spread = surface_intensity(mesh, volume, origin=origin)
    dip = edge_dip(mesh, volume, origin=origin, steps=steps)
    edges = mesh.edges()
    finite = np.isfinite(dip)
    result: Dict = {
        "n_vertices": mesh.n_vertices,
        "n_faces": mesh.n_faces,
        "n_edges": int(len(edges)),
        "edge_length": step,
        "winding_spacing": float(spacing),
        "steps_per_winding": float(spacing / step) if step else float("nan"),
        "resolution_adequate": adequate,
        "z_threshold": z_threshold,
        "min_edges": min_edges,
        "min_span": min_span,
        "min_dip": min_dip,
        "surface_intensity_median": level,
        "surface_intensity_mad": intensity_spread,
        "dip_degenerate": bool(level < min_dip),
        "seams": [],
        "max_z": 0.0,
        "median_dip": 0.0,
    }
    if finite.sum() < 4 or result["dip_degenerate"]:
        # Same degenerate null as the quad detector: a surface over masked-out
        # air has no dip distribution to score against.
        result["n_seams"] = 0
        return result

    centre = float(np.median(dip[finite]))
    spread = 1.4826 * float(np.median(np.abs(dip[finite] - centre)))
    if spread < 1e-6:
        result["dip_degenerate"] = True
        result["n_seams"] = 0
        return result
    scores = np.zeros_like(dip)
    scores[finite] = (dip[finite] - centre) / spread
    result["median_dip"] = centre
    result["max_z"] = float(scores.max())

    extent = float((mesh.points.max(0) - mesh.points.min(0)).max())
    result["mesh_extent"] = extent
    for component in _components(edges, scores >= z_threshold):
        if component.size < min_edges:
            continue
        verts = np.unique(edges[component].ravel())
        span = _span_fraction(mesh.points[verts], extent)
        if span < min_span:
            continue
        result["seams"].append(
            {
                "edges": int(component.size),
                "vertices": int(verts.size),
                "span_fraction": span,
                "z_max": float(scores[component].max()),
                "z_mean": float(scores[component].mean()),
                "mean_dip": float(dip[component].mean()),
                "centroid_zyx": [float(v) for v in mesh.points[verts].mean(0)],
            }
        )
    result["seams"].sort(key=lambda s: -s["span_fraction"])
    if not adequate:
        result["seams_unreliable"] = result.pop("seams")
        result["seams"] = []
    result["n_seams"] = len(result["seams"])
    return result


def displace(mesh: TriMesh, distance: float, region=None) -> TriMesh:
    """Return a copy with part of the surface pushed along its own normal."""
    points = mesh.points.copy()
    normals = mesh.normals()
    if region is None:
        # split on the median of the widest extent, so the cut runs across the
        # surface rather than clipping a corner
        extent = points.max(0) - points.min(0)
        axis = int(np.argmax(extent))
        region = points[:, axis] > np.median(points[:, axis])
    points[region] = points[region] + normals[region] * distance
    return TriMesh(
        points=points, faces=mesh.faces.copy(), path=mesh.path, meta=dict(mesh.meta)
    )
