#!/usr/bin/env python3
"""The on-sheet question, asked of the team's own surface volumes.

Every published segment ships a ``surface-volumes/*.zarr``: the scan resampled
in a band around the traced surface by the team's renderer, 109 layers deep
with the surface at the middle layer.  That is the same measurement
``labelscope onsheet`` makes, produced by a different sampler.  If their band
is flat for a surface where ours is flat, the finding no longer depends on our
code at all.

Chunks are 109 x 128 x 128 uint8, uncompressed, C order, one chunk per 128 x 128
column of the surface, so a single 1.8 MB fetch gives full-depth profiles for
16,384 columns.  The store is sparse -- only chunks the surface covers exist --
so chunks are chosen from a listing rather than by coordinate.

    python scripts/onsheet/surface_volume_profile.py \\
        --store https://.../segments/<id>/surface-volumes/<vol>.zarr --chunks 24
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request

import numpy as np

LAYERS, SIDE = 109, 128


def _list_prefixes(bucket: str, prefix: str):
    url = (
        f"{bucket}/?list-type=2&prefix={urllib.parse.quote(prefix)}&delimiter=/&max-keys=1000"
    )
    xml = urllib.request.urlopen(url, timeout=60).read().decode()
    return [p for p in re.findall(r"<Prefix>([^<]+)</Prefix>", xml) if p != prefix]


def _list_keys(bucket: str, prefix: str):
    url = f"{bucket}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
    xml = urllib.request.urlopen(url, timeout=60).read().decode()
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def chunk_profiles(store: str, chunks: int, seed: int, min_coverage: float = 0.5):
    """Per-chunk layer profiles over the surface's own footprint."""
    bucket, _, path = store.partition("/PHerc")
    path = "PHerc" + path
    rng = np.random.default_rng(seed)
    columns = _list_prefixes(bucket, f"{path}/0/0/")
    if not columns:
        raise SystemExit(f"no chunk columns under {store}/0/0/")
    out, tries = [], 0
    while len(out) < chunks and tries < chunks * 6:
        tries += 1
        col = columns[int(rng.integers(len(columns)))]
        keys = _list_keys(bucket, col)
        if not keys:
            continue
        key = keys[int(rng.integers(len(keys)))]
        raw = urllib.request.urlopen(f"{bucket}/{key}", timeout=120).read()
        if len(raw) != LAYERS * SIDE * SIDE:
            continue
        cube = np.frombuffer(raw, dtype=np.uint8).reshape(LAYERS, SIDE, SIDE)
        footprint = cube.max(axis=0) > 0  # the surface exists where any layer is non-zero
        cov = float(footprint.mean())
        if cov < min_coverage:
            continue
        profile = cube[:, footprint].astype(np.float32).mean(axis=1)
        mid = LAYERS // 2
        out.append(
            {
                "chunk": key.rsplit("/", 2)[-2:],
                "coverage": cov,
                "range": float(profile.max() - profile.min()),
                "at_middle": float(profile[mid]),
                "peak_offset": int(np.argmax(profile) - mid),
                "profile": [round(float(v), 2) for v in profile],
            }
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--store", action="append", required=True, help="…/surface-volumes/<vol>.zarr"
    )
    ap.add_argument("--chunks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    results = []
    print(f"{'segment':32s} {'chunks':>6s} {'range':>7s} {'at_mid':>7s} {'|peak|':>7s}")
    print("-" * 66)
    for store in args.store:
        seg = store.split("/segments/", 1)[1].split("/", 1)[0]
        found = chunk_profiles(store, args.chunks, args.seed)
        if not found:
            print(f"{seg[:32]:32s}   none")
            results.append({"segment": seg, "store": store, "error": "no usable chunks"})
            continue
        rng_ = np.array([c["range"] for c in found])
        pk = np.array([abs(c["peak_offset"]) for c in found])
        mid = np.array([c["at_middle"] for c in found])
        row = {
            "segment": seg,
            "store": store,
            "chunks": len(found),
            "range_median": float(np.median(rng_)),
            "at_middle_median": float(np.median(mid)),
            "peak_offset_abs_median": float(np.median(pk)),
            "per_chunk": found,
        }
        results.append(row)
        print(
            f"{seg[:32]:32s} {len(found):6d} {row['range_median']:7.1f} "
            f"{row['at_middle_median']:7.1f} {row['peak_offset_abs_median']:7.1f}"
        )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"results": results}, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
