"""Seam detection on Will Stevens' components, in their native level-2 space.

The meshes are stored in level-2 coordinates (scale 0.25). Scaling them to
level 0 does not improve the detector: steps_per_winding is a *ratio* of winding
spacing to grid step, and scaling both by 4 leaves it unchanged, while costing
64x the chunk traffic. So the surfaces are read against level 2 as authored.

Grid step is 4.0 voxels and the winding spacing at 9.6 um/voxel should be
16-36 voxels, so steps_per_winding lands well above the >=2 gate -- unlike the
PHerc0172 case in the August corpus, where the gate failed because the grid step
was large relative to the winding, not because the scan was coarse.

Each surface is measured as delivered and again with a whole winding planted in
half of it, because the August pass showed a fixed z threshold does not transfer
between surfaces; the planted run is the per-surface reference.
"""
import json
import sys
import time

import numpy as np

from labelscope.mesh import (
    QuadMesh, displace, find_sheet_switches, read_tifxyz, surface_intensity, winding_spacing,
)
from labelscope.remote_zarr import ChunkedVolume

B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
V = f"{B}/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr"
LEVEL = "2"


def best_window(mesh, w):
    """The densest w x w window: components are sparse and mostly hole."""
    rows, cols = mesh.shape
    if rows <= w and cols <= w:
        return mesh.window(0, rows, 0, cols)
    best, bestv = None, -1.0
    for r0 in range(0, max(rows - w, 1), max((rows - w) // 6, 1)):
        for c0 in range(0, max(cols - w, 1), max((cols - w) // 6, 1)):
            win = mesh.window(r0, min(r0 + w, rows), c0, min(c0 + w, cols))
            v = float(win.valid.mean())
            if v > bestv:
                best, bestv = win, v
    return best


def run(path, vol, window, plant=0.0):
    name = path.rstrip("/").split("/")[-1]
    t0 = time.time()
    m = read_tifxyz(path, lazy="auto")
    win = best_window(m, window)
    mesh = QuadMesh(points=win.points.copy(), valid=win.valid.copy(), meta={}, path=path)
    row = {"name": name, "planted": plant, "grid": list(mesh.shape),
           "valid_fraction": round(float(mesh.valid.mean()), 3),
           "grid_step": round(float(mesh.grid_step()), 2)}
    if plant:
        sp = winding_spacing(mesh, vol)
        spacing = sp.get("spacing") if isinstance(sp, dict) else sp
        if not spacing or not np.isfinite(spacing):
            row["error"] = "no winding spacing to plant with"
            return row
        region = np.zeros(mesh.shape, bool)
        region[: mesh.shape[0] // 2] = True
        mesh = displace(mesh, float(spacing), region=region)
    si = surface_intensity(mesh, vol)
    row["surface_intensity"] = round(float(si[0]), 1) if si else None
    out = find_sheet_switches(mesh, vol)
    for k in ("winding_spacing", "steps_per_winding", "resolution_adequate",
              "dip_degenerate", "n_seams", "max_z"):
        if k in out:
            v = out[k]
            row[k] = round(float(v), 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
    row["seconds"] = round(time.time() - t0, 1)
    row["chunks"] = getattr(vol, "chunks_fetched", None)
    return row


if __name__ == "__main__":
    window = int(sys.argv[1])
    vol = ChunkedVolume.from_store(V, level=LEVEL)
    results = []
    for path in sys.argv[2:]:
        for plant in (0.0, 1.0):
            try:
                r = run(path, vol, window, plant)
            except Exception as e:  # noqa: BLE001
                r = {"name": path.split("/")[-1], "planted": plant,
                     "error": f"{type(e).__name__}: {str(e)[:150]}"}
            print(json.dumps(r), flush=True)
            results.append(r)
    with open("/root/will_l2.json", "w") as fh:
        json.dump(results, fh, indent=1)
