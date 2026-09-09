"""Cheap first: is the seam detector even applicable to these surfaces?

find_sheet_switches walks every grid edge, which at level 0 over a large window
is tens of millions of samples. The resolution gate decides whether that work is
worth doing at all, and it needs only the winding spacing and the grid step.
"""
import sys
import numpy as np
from labelscope.mesh import QuadMesh, read_tifxyz, surface_intensity, winding_spacing
from labelscope.remote_zarr import ChunkedVolume

B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
V = f"{B}/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr"

vol = ChunkedVolume.from_store(V, level="0")
for path in sys.argv[1:]:
    name = path.rstrip("/").split("/")[-1]
    m = read_tifxyz(path, lazy="auto")
    rows, cols = m.shape
    w = 240
    r0, c0 = max((rows - w) // 2, 0), max((cols - w) // 2, 0)
    win = m.window(r0, min(r0 + w, rows), c0, min(c0 + w, cols))
    scaled = QuadMesh(points=win.points * 4.0, valid=win.valid, meta={}, path=path)
    step = float(scaled.grid_step())
    si = surface_intensity(scaled, vol)
    sp = winding_spacing(scaled, vol)
    spacing = sp.get("spacing") if isinstance(sp, dict) else sp
    spw = (spacing / step) if (spacing and step) else float("nan")
    print(f"{name:9s} grid {rows}x{cols}  window {w}  step {step:6.2f} vx  "
          f"winding {spacing if spacing is None else round(float(spacing),1)}  "
          f"steps/winding {spw:5.2f}  gate {'PASS' if spw >= 2 else 'FAIL'}  "
          f"surface_intensity {round(float(si[0]),1) if si else None}", flush=True)
