#!/usr/bin/env python3
"""Is this traced surface actually on papyrus?

A tracer can complete normally, report a plausible area, place every vertex
inside the scan, and still produce a surface that cuts *across* the windings
instead of following a sheet. Nothing in the toolchain catches that: the meta
looks right, renders look like credible fibrous texture at every depth, and any
ink model probing the surface returns structured noise
(ScrollPrize/villa#1675 -- and independently reproduced here on a
full-resolution L0 prediction, not the coarse L2 one that issue describes).

The check: sample the scan along the surface normal, averaged over a *coherent*
neighbourhood of the grid. A surface lying on a sheet sits on a density ridge, so
its profile has real dynamic range. A surface cutting across sheets sees the same
fibrous material at every depth, so the profile is flat.

Averaging has to be local. Over a whole patch the winding phase varies and the
periodicity cancels, which makes a good surface look as flat as a bad one -- that
mistake cost an afternoon before this was written.

    python scripts/onsheet_check.py --mesh seg.tifxyz --baseline published.tifxyz \
        --volume <zarr-url> --remote

Calibration on PHercParis4 at 2.4 um: published surfaces give 51-53 grey levels
of range, off-sheet grown surfaces give 11-12.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def block_profiles(mesh, volume, origin, reach, step, blocks, block_size, seed):
    """Mean intensity along the normal, one profile per coherent grid block."""
    from labelscope.mesh import _sampler

    sample, remote = _sampler(volume, origin)
    offsets = np.arange(-reach, reach + step / 2, step, dtype=np.float32)
    normals = mesh.normals()
    rows, cols = mesh.shape
    rng = np.random.default_rng(seed)

    out = []
    tries = 0
    while len(out) < blocks and tries < blocks * 20:
        tries += 1
        r0 = int(rng.integers(0, max(1, rows - block_size)))
        c0 = int(rng.integers(0, max(1, cols - block_size)))
        sl = (slice(r0, r0 + block_size), slice(c0, c0 + block_size))
        valid = mesh.valid[sl]
        if valid.sum() < (block_size * block_size) // 2:
            continue
        base = mesh.points[sl][valid].astype(np.float32)
        nrm = normals[sl][valid]
        walk = base[None] + nrm[None] * offsets[:, None, None]
        flat = walk.reshape(-1, 3)
        if remote and hasattr(volume, "prefetch"):
            volume.prefetch(flat - (np.zeros(3) if origin is None else np.asarray(origin)))
        profile = sample(flat).reshape(offsets.size, -1).mean(1)
        if not np.isfinite(profile).all() or profile.max() <= 0:
            continue
        out.append(
            {
                "block": [r0, c0],
                "n": int(valid.sum()),
                "range": float(profile.max() - profile.min()),
                "at_zero": float(profile[len(profile) // 2]),
                "peak_offset": float(offsets[int(np.argmax(profile))]),
            }
        )
    return out


def summarise(name, blocks):
    if not blocks:
        return {"mesh": name, "error": "no usable blocks"}
    ranges = np.array([b["range"] for b in blocks])
    peaks = np.array([abs(b["peak_offset"]) for b in blocks])
    return {
        "mesh": name,
        "blocks": len(blocks),
        "range_median": float(np.median(ranges)),
        "range_min": float(ranges.min()),
        "range_max": float(ranges.max()),
        "peak_offset_abs_median": float(np.median(peaks)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", action="append", required=True, help="tifxyz dir; repeatable")
    ap.add_argument("--baseline", help="a known-good published tifxyz, measured the same way")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument("--reach", type=float, default=70.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the full result as JSON")
    args = ap.parse_args(argv)

    from labelscope.mesh import read_tifxyz

    if args.remote:
        from labelscope.remote_zarr import ChunkedVolume

        volume = ChunkedVolume.from_store(args.volume, cache_dir=args.cache)
    else:
        import tifffile

        volume = tifffile.imread(args.volume)

    targets = [(m, False) for m in args.mesh]
    if args.baseline:
        targets.append((args.baseline, True))

    results = []
    for path, is_baseline in targets:
        mesh = read_tifxyz(path)
        blocks = block_profiles(
            mesh, volume, None, args.reach, args.step, args.blocks, args.block_size, args.seed
        )
        row = summarise(os.path.basename(path.rstrip("/")), blocks)
        row["baseline"] = is_baseline
        row["per_block"] = blocks
        results.append(row)

    base = next((r for r in results if r.get("baseline") and "error" not in r), None)
    print(f"{'surface':44s} {'blocks':>6s} {'range':>8s} {'|peak|':>7s}  verdict")
    print("-" * 84)
    for r in results:
        if "error" in r:
            print(f"{r['mesh'][:44]:44s} {'-':>6s} {'-':>8s} {'-':>7s}  {r['error']}")
            continue
        verdict = "baseline" if r.get("baseline") else ""
        if base and not r.get("baseline"):
            frac = r["range_median"] / max(base["range_median"], 1e-6)
            verdict = (
                "ON SHEET" if frac >= 0.5 else ("marginal" if frac >= 0.3 else "OFF SHEET")
            )
            verdict += f" ({frac:.0%} of baseline)"
        print(
            f"{r['mesh'][:44]:44s} {r['blocks']:6d} {r['range_median']:8.1f} "
            f"{r['peak_offset_abs_median']:7.1f}  {verdict}"
        )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"results": results}, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
