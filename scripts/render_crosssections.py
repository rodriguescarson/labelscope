#!/usr/bin/env python3
"""Render a traced surface as flattened cross-sections a human can judge.

The question "did this surface jump to a neighbouring wrap?" is hard to see in a
list of numbers and easy to see in one picture. For each row of the mesh grid,
sample the scan along the surface normal from -R to +R voxels and stack those
columns side by side. The traced surface is then the horizontal centre line of
the image.

On a surface that stays on one sheet, the bright papyrus band runs flat along the
centre. Where the surface steps to the next wrap, the band steps with it -- a
visible discontinuity, at the place where it happens.

This exists so the ground truth in the GrowPatch validation comes from a person
looking at the data, not from the detector being asked to grade itself.

    python scripts/render_crosssections.py --mesh seg.tifxyz --volume <zarr-url> \
        --remote --out strips/
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def strip_for_rows(mesh, volume, origin, rows, reach: float, step: float):
    """One image: grid position across, distance along the normal down."""
    from labelscope.mesh import _sampler

    sample, remote = _sampler(volume, origin)
    offsets = np.arange(-reach, reach + step / 2, step, dtype=np.float32)
    normals = mesh.normals()

    columns, keep = [], []
    for r in rows:
        valid = mesh.valid[r]
        if valid.sum() < 8:
            continue
        cols = np.flatnonzero(valid)
        base = mesh.points[r, cols]
        nrm = normals[r, cols]
        walk = base[None] + nrm[None] * offsets[:, None, None]
        flat = walk.reshape(-1, 3)
        if remote and hasattr(volume, "prefetch"):
            volume.prefetch(flat - (np.zeros(3) if origin is None else np.asarray(origin)))
        values = sample(flat).reshape(offsets.size, len(cols))
        columns.append(values)
        keep.append(r)
    return columns, keep, offsets


def to_png(image: np.ndarray, path: str, centre_line: bool = True):
    from PIL import Image

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return False
    lo, hi = np.percentile(finite, [1, 99])
    scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=2)
    if centre_line:
        mid = rgb.shape[0] // 2
        # a thin marker on the traced surface itself, so a step is unmistakable
        rgb[mid, :, 0] = 220
        rgb[mid, :, 1] = 40
        rgb[mid, :, 2] = 90
    Image.fromarray(rgb).save(path)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="tifxyz directory")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--reach", type=float, default=60.0, help="voxels either side of the surface"
    )
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--rows", type=int, default=12, help="how many cross-sections to render")
    args = ap.parse_args(argv)

    from labelscope.mesh import read_tifxyz

    mesh = read_tifxyz(args.mesh)
    if not mesh.valid.any():
        print("no valid vertices", file=sys.stderr)
        return 2

    if args.remote:
        from labelscope.remote_zarr import ChunkedVolume

        volume, origin = ChunkedVolume.from_store(args.volume, cache_dir=args.cache), None
    else:
        import tifffile

        volume, origin = tifffile.imread(args.volume), None

    usable = np.flatnonzero(mesh.valid.sum(axis=1) >= 8)
    if usable.size == 0:
        print("no row has enough valid vertices", file=sys.stderr)
        return 2
    rows = usable[np.linspace(0, usable.size - 1, min(args.rows, usable.size)).astype(int)]

    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(args.mesh.rstrip("/"))
    columns, kept, offsets = strip_for_rows(mesh, volume, origin, rows, args.reach, args.step)

    written = 0
    for image, row in zip(columns, kept):
        path = os.path.join(args.out, f"{name}__row{row:05d}.png")
        if to_png(image, path):
            written += 1
    print(f"{name}: grid {mesh.shape}, {written} cross-sections -> {args.out}")
    if args.remote:
        print(f"  streamed {getattr(volume, 'bytes_fetched', 0) / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
