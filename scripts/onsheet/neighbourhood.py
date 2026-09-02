"""Are the w128-129 tracings different from their own neighbours?

The remaining alternative explanation is that this region of the scroll is
simply featureless, in which case a low profile range says nothing about the
tracing. Published surfaces covering nearby windings settle it: if they score
normally, the region carries structure.
"""

import glob
import json
import os
import re

import numpy as np

rows = []
for f in sorted(glob.glob("/workspace/onsheet_corpus/*.json")):
    with open(f) as fh:
        d = json.load(fh)
    res = d["results"]
    base = next((r for r in res if r.get("baseline") and "error" not in r), None)
    t = next((r for r in res if not r.get("baseline")), None)
    if not t or "error" in t or not base:
        continue
    n = os.path.basename(f)[:-5]
    m = re.search(r"w(\d+)-(\d+)", n)
    w = (int(m.group(1)) + int(m.group(2))) / 2 if m else None
    rows.append(
        (
            n,
            w,
            t["range_median"] / max(base["range_median"], 1e-9),
            t["peak_offset_abs_median"],
        )
    )

win = [r for r in rows if r[1] is not None]
print(f"{len(win)} of {len(rows)} surfaces name a winding range")
print()
print("neighbourhood of w128-129 (winding midpoint 105-155):")
print(f"  {'wind':>6s} {'frac':>6s} {'|peak|':>7s}  surface")
band = sorted([r for r in win if 105 <= r[1] <= 155], key=lambda r: r[1])
for n, w, fr, pk in band:
    flag = "   <-- SUSPECT" if fr < 0.4 and pk > 40 else ""
    print(f"  {w:6.1f} {fr:6.2f} {pk:7.1f}  {n[:42]}{flag}")

nb = [r for r in band if not (r[2] < 0.4 and r[3] > 40)]
print()
if nb:
    fr = [r[2] for r in nb]
    print(
        f"neighbours excluding the suspects: n={len(nb)}  "
        f"median frac {np.median(fr):.2f}  min {min(fr):.2f}"
    )
    print(
        "VERDICT: region carries structure -> the two suspects differ from their neighbours"
        if min(fr) > 0.5
        else "VERDICT: neighbours also weak -> region may be featureless, claim NOT supported"
    )
else:
    print(
        "VERDICT: no published neighbours in this band -- cannot rule out a featureless region"
    )
