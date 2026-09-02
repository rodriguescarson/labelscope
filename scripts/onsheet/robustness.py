#!/usr/bin/env python3
"""Does the w128-129 result depend on how the blocks were drawn?

Re-measures the two suspect surfaces and their adjacent windings across a grid
of block sizes and seeds, and reports the Mann-Whitney comparison for each.  A
claim that only holds at one block size and one seed is not a claim.

    python scripts/onsheet/robustness.py --meshes <dir with the four tifxyz> \\
        --volume <zarr url> --remote --out robustness.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from labelscope.mesh import read_tifxyz
from labelscope.onsheet import block_profiles, compare

PAIRS = [("20260623", "w128-129", "w126-127"), ("20260701", "w128-129", "w126-127")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", required=True, help="directory holding the four tifxyz dirs")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--sizes", default="8,12,16")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.remote:
        from labelscope.remote_zarr import ChunkedVolume

        volume = ChunkedVolume.from_store(args.volume, cache_dir=args.cache)
    else:
        import tifffile

        volume = tifffile.imread(args.volume)

    def find(prefix, tag):
        hits = [
            d for d in glob.glob(os.path.join(args.meshes, "*")) if prefix in d and tag in d
        ]
        if len(hits) != 1:
            raise SystemExit(f"expected one mesh for {prefix} {tag}, found {hits}")
        return hits[0]

    meshes = {(p, t): read_tifxyz(find(p, t)) for p, a, b in PAIRS for t in (a, b)}
    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    print(
        f"{'series':9s} {'size':>4s} {'seed':>4s} {'n':>3s} {'suspect':>8s} {'neighbour':>9s} {'ratio':>6s} {'p':>9s}"
    )
    print("-" * 62)
    for prefix, sus, nb in PAIRS:
        for size in sizes:
            for seed in seeds:
                a = block_profiles(
                    meshes[(prefix, sus)], volume, None, 70.0, 1.0, args.blocks, size, seed
                )
                b = block_profiles(
                    meshes[(prefix, nb)], volume, None, 70.0, 1.0, args.blocks, size, seed
                )
                st = compare(a, b)
                if "error" in st:
                    print(f"{prefix:9s} {size:4d} {seed:4d}   {st['error']}")
                    continue
                ratio = st["median_a"] / max(st["median_b"], 1e-9)
                rows.append(
                    {"series": prefix, "block_size": size, "seed": seed, **st, "ratio": ratio}
                )
                print(
                    f"{prefix:9s} {size:4d} {seed:4d} {st['n_a']:3d} {st['median_a']:8.1f} "
                    f"{st['median_b']:9.1f} {ratio:6.2f} {st['p_less']:9.4g}"
                )

    if rows:
        ps = np.array([r["p_less"] for r in rows])
        rt = np.array([r["ratio"] for r in rows])
        print(
            f"\n{len(rows)} configurations: p max {ps.max():.4g}, ratio range {rt.min():.2f}-{rt.max():.2f}, "
            f"all p < 0.05: {bool((ps < 0.05).all())}"
        )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "blocks": args.blocks}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
