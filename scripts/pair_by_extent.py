#!/usr/bin/env python3
"""Pair a surface with the scan it was traced on by reading both, not the name.

Most published surfaces encode their volume in the directory name
(``<segment>-on-<volume-id>-<voxel>um.tifxyz``).  A large minority do not: they
sit under ``mesh/intermediate/tifxyz_original/`` with no volume id anywhere in
the path, and on some scrolls that is the *only* surface published -- PHerc 1447,
PHerc 0800 and PHerc 1203 among them, all three eligible for the Grand Prize.

Those surfaces are still pairable, because a mesh in voxel coordinates has to fit
inside the volume it came from.  This reads each surface's own extent, reads each
candidate volume's shape from its ``.zarray``, and keeps the volumes that contain
the surface.  Where several fit, the finest wins, and the margin is reported so a
suspiciously loose fit is visible rather than silent.

    python scripts/pair_by_extent.py --meshes DIR --scroll PHerc1447 --out pairs.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import requests

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOXEL = re.compile(r"-(\d+\.?\d*)um-")


def volumes_for(scroll: str, session) -> list:
    """Every published volume of a scroll, with its shape and voxel size."""
    url = f"{BUCKET}/?list-type=2&prefix={scroll}/volumes/&delimiter=/"
    body = session.get(url, timeout=60).text
    out = []
    for key in re.findall(rf"<Prefix>({scroll}/volumes/[^<]+?)/</Prefix>", body):
        meta = session.get(f"{BUCKET}/{key}/0/.zarray", timeout=60)
        if meta.status_code != 200:
            continue
        shape = json.loads(meta.text)["shape"]
        match = VOXEL.search(key)
        out.append(
            {
                "key": key,
                "shape": shape,
                "um": float(match.group(1)) if match else float("nan"),
            }
        )
    return out


def extent(mesh_dir: str):
    """The surface's own bounding box in voxels, ignoring missing vertices."""
    import tifffile

    parts = {}
    for axis in ("x", "y", "z"):
        parts[axis] = tifffile.imread(os.path.join(mesh_dir, f"{axis}.tif")).astype(np.float32)
    points = np.stack([parts["z"], parts["y"], parts["x"]], axis=-1)
    valid = np.all(points >= 0, axis=-1) & np.all(np.isfinite(points), axis=-1)
    if not valid.any():
        return None, None
    inside = points[valid]
    return inside.min(0), inside.max(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", required=True, help="directory of local tifxyz directories")
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", help="write the fit margins here as JSON")
    args = ap.parse_args(argv)

    session = requests.Session()
    candidates = volumes_for(args.scroll, session)
    if not candidates:
        print(f"{args.scroll}: no volumes published", file=sys.stderr)
        return 2
    print(f"{args.scroll}: {len(candidates)} candidate volumes")

    rows, report = [], []
    for name in sorted(os.listdir(args.meshes)):
        directory = os.path.join(args.meshes, name)
        if not os.path.isdir(directory) or not name.startswith(args.scroll):
            continue
        lo, hi = extent(directory)
        if lo is None:
            report.append({"mesh": name, "error": "no valid vertices"})
            continue
        fits = [
            c
            for c in candidates
            if all(hi[i] < c["shape"][i] for i in range(3))
            and all(lo[i] >= 0 for i in range(3))
        ]
        if not fits:
            report.append(
                {
                    "mesh": name,
                    "error": "no published volume contains this surface",
                    "extent_hi": [float(v) for v in hi],
                    "shapes": {c["key"]: c["shape"] for c in candidates},
                }
            )
            continue
        best = min(fits, key=lambda c: (c["um"], c["key"]))
        # how much of the volume the surface leaves unused on each axis; a fit
        # that only just squeaks in is the interesting case to eyeball
        margin = [float(best["shape"][i] - hi[i]) / best["shape"][i] for i in range(3)]
        rows.append((directory, f"{BUCKET}/{best['key']}"))
        report.append(
            {
                "mesh": name,
                "volume": best["key"],
                "um": best["um"],
                "n_fitting": len(fits),
                "headroom_fraction": [round(m, 4) for m in margin],
            }
        )

    with open(args.out, "w") as handle:
        for directory, volume in rows:
            handle.write(f"{directory}\t{volume}\n")
    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
    unpaired = len([r for r in report if "error" in r])
    print(f"paired {len(rows)}, unpaired {unpaired} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
