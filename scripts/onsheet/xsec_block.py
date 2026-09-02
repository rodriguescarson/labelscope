#!/usr/bin/env python3
"""Cross-sections through chosen blocks of a large surface, without loading it.

`render_crosssections.py` renders whole rows of a grid, which on a published
20-million-cell surface is a thousand images and more memory than a small
machine has.  What the on-sheet question needs is narrower: a cross-section
through a block whose profile range is already known, so a flat block and a
structured block of the same surface can be looked at side by side.

Reads the mesh lazily (memory-mapped) and renders, for each requested block, a
strip `span` grid cells wide through the block's centre row, resampled to one
image column per voxel of arc length and one row per voxel along the normal.

    python scripts/onsheet/xsec_block.py --mesh seg.tifxyz --volume <zarr> --remote \\
        --block 1200,3456 --block 84,7104 --out xsec/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from render_crosssections import strip_for_rows, to_png  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument(
        "--block", action="append", required=True, help="r0,c0 of a block (grid cells)"
    )
    ap.add_argument("--block-size", type=int, default=12)
    ap.add_argument("--span", type=int, default=60, help="grid cells across the strip")
    ap.add_argument("--reach", type=float, default=70.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from labelscope.mesh import read_tifxyz

    if args.remote:
        from labelscope.remote_zarr import ChunkedVolume

        volume = ChunkedVolume.from_store(args.volume, cache_dir=args.cache)
    else:
        import tifffile

        volume = tifffile.imread(args.volume)

    mesh = read_tifxyz(args.mesh, lazy=True)
    rows, cols = mesh.shape
    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(args.mesh.rstrip("/"))[:30]
    for spec in args.block:
        r0, c0 = (int(v) for v in spec.split(","))
        rc = r0 + args.block_size // 2
        c_lo = max(c0 + args.block_size // 2 - args.span // 2, 0)
        c_hi = min(c_lo + args.span, cols)
        win = mesh.window(max(rc - 1, 0), min(rc + 2, rows), c_lo, c_hi)
        local_row = rc - max(rc - 1, 0)
        strips, kept, _ = strip_for_rows(win, volume, None, [local_row], args.reach, 1.0)
        if not strips:
            print(f"  block {spec}: too few valid vertices")
            continue
        path = os.path.join(args.out, f"{name}__r{r0:05d}_c{c0:05d}.png")
        to_png(strips[0], path)
        print(f"  block {spec}: {strips[0].shape[1]} px wide -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
