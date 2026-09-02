"""Is the last patch of a tracing run systematically off-sheet?

Both w128-129 tracings are terminal patches of their series. If terminal patches
score worse than interior ones across the whole corpus, that is a population
finding rather than two anomalies -- and a cheap rule for anyone releasing data.
"""

import glob
import json
import os
import re
from collections import defaultdict

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
    m = re.search(r"^(\d{8})\d*-w(\d+)-(\d+)", n)
    if not m:
        continue
    rows.append(
        {
            "name": n,
            "day": m.group(1),
            "w": (int(m.group(2)) + int(m.group(3))) / 2,
            "frac": t["range_median"] / max(base["range_median"], 1e-9),
        }
    )

series = defaultdict(list)
for r in rows:
    series[r["day"]].append(r)

term, interior = [], []
print(
    f"{'series':10s} {'n':>3s} {'windings':>12s} {'last w':>7s} {'last frac':>10s} {'interior median':>16s}"
)
for day, rs in sorted(series.items()):
    rs.sort(key=lambda r: r["w"])
    if len(rs) < 4:
        continue
    last, rest = rs[-1], rs[:-1]
    term.append(last["frac"])
    interior += [r["frac"] for r in rest]
    print(
        f"{day:10s} {len(rs):3d} {f'{rs[0][chr(119)]:.0f}-{rs[-1][chr(119)]:.0f}':>12s} "
        f"{last['w']:7.0f} {last['frac']:10.2f} {np.median([r['frac'] for r in rest]):16.2f}"
    )

print()
print(f"terminal patches  n={len(term):3d}  median {np.median(term):.2f}  min {min(term):.2f}")
print(
    f"interior patches  n={len(interior):3d}  median {np.median(interior):.2f}  min {min(interior):.2f}"
)
below = sum(1 for t in term if t < min(interior))
print(f"terminal patches below the worst interior patch: {below} of {len(term)}")
