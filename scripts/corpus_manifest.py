#!/usr/bin/env python3
"""Pick one surface per segment across the whole published corpus.

Input is the raw enumeration of every published tifxyz paired with the volume it
was traced on, one ``scroll<TAB>voxel_um<TAB>mesh_key<TAB>volume_key`` per line.
Most segments are traced several times over -- Scroll 1 alone ships meshes on
five different scans of itself -- so a corpus pass has to choose, and the choice
has to be stated rather than left to whichever key sorted first.

The rule: prefer the scan closest to 2.4 um among those between 1 and 5 um, and
fall back to the finest available otherwise.

Why not simply take the finest scan every time?  The detector's own resolution
gate is scale-invariant for a *given* mesh -- both the grid step and the winding
spacing are in voxels of the same volume, so their ratio does not move when the
scan gets finer -- while the streaming cost rises with the cube of it.  2.4 um is
also the tier the detector was validated at, so the fleet numbers stay comparable
with the Scroll 1 pass.

    python scripts/corpus_manifest.py corpus_raw.tsv --out manifest.tsv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
PREFERRED_UM = 2.4
BAND = (1.0, 5.0)


def choose(rows):
    """The one row to measure for this segment."""
    in_band = [r for r in rows if BAND[0] <= r["um"] <= BAND[1]]
    if in_band:
        return min(in_band, key=lambda r: (abs(r["um"] - PREFERRED_UM), r["mesh"]))
    return min(rows, key=lambda r: (r["um"], r["mesh"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="scroll<TAB>um<TAB>mesh_key<TAB>volume_key per line")
    ap.add_argument("--out", required=True, help="mesh_key<TAB>volume_url per line")
    ap.add_argument("--local-root", default="", help="prefix local mesh paths with this")
    ap.add_argument("--skip-scroll", action="append", default=[])
    args = ap.parse_args(argv)

    by_segment = defaultdict(list)
    with open(args.raw) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            scroll, um, mesh, volume = parts
            if scroll in args.skip_scroll:
                continue
            try:
                value = float(um)
            except ValueError:
                continue
            segment = mesh.split("/segments/")[1].split("/")[0]
            by_segment[(scroll, segment)].append(
                {"scroll": scroll, "um": value, "mesh": mesh, "volume": volume}
            )

    picked = [choose(rows) for rows in by_segment.values()]
    picked.sort(key=lambda r: (r["scroll"], r["mesh"]))

    with open(args.out, "w") as handle:
        for row in picked:
            name = row["mesh"].split("/segments/")[1].replace("/mesh/", "__")
            local = (
                f"{args.local_root.rstrip('/')}/{row['scroll']}__{name}"
                if args.local_root
                else row["mesh"]
            )
            handle.write(f"{local}\t{BUCKET}/{row['volume']}\n")

    per_scroll = defaultdict(lambda: defaultdict(int))
    for row in picked:
        per_scroll[row["scroll"]][row["um"]] += 1
    print(f"{len(picked)} surfaces, one per segment, across {len(per_scroll)} scrolls")
    for scroll in sorted(per_scroll):
        detail = ", ".join(f"{n} at {um} um" for um, n in sorted(per_scroll[scroll].items()))
        print(f"  {scroll:14s} {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
