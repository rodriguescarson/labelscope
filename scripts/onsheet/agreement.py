"""Does labelscope's sheet-switch detector flag the surfaces an independent
measurement says are off-sheet?

This is the structure of the test Paul asked for -- find real errors, then check
whether the detector finds those and only those -- run against published data
instead of grown segments, so it needs no tracing run at all.
"""
import csv
import glob
import json
import os
import re

import numpy as np

# sheetswitch results from the August corpus pass
sw = {}
for f in glob.glob("findings/corpus/per_surface/real/PHercParis4__*.csv"):
    with open(f) as fh:
        row = next(csv.DictReader(fh))
    key = os.path.basename(f)[:-4].split("__", 1)[1]
    sw[key] = row

# on-sheet results from this session
on = {}
for f in glob.glob("findings/onsheet/onsheet_corpus/*.json"):
    d = json.load(open(f))
    res = d["results"]
    base = next((r for r in res if r.get("baseline") and "error" not in r), None)
    t = next((r for r in res if not r.get("baseline")), None)
    if not t or "error" in t or not base:
        continue
    on[os.path.basename(f)[:-5]] = t["range_median"] / max(base["range_median"], 1e-9)

joined = []
for k, frac in on.items():
    row = sw.get(k)
    if row is None:
        cands = [kk for kk in sw if kk.split("__")[0] == k.split("__")[0]]
        row = sw[cands[0]] if cands else None
    if row is None:
        continue
    joined.append({
        "name": k, "frac": frac,
        "usable": row["resolution_adequate"] == "True" and row["dip_degenerate"] != "True",
        "n_seams": int(float(row["n_seams"])),
        "max_z": float(row["max_z"]) if row["max_z"] else 0.0,
    })

print(f"joined {len(joined)} of {len(on)} on-sheet results to sheetswitch output")
use = [r for r in joined if r["usable"]]
print(f"{len(use)} clear the sheetswitch gates (resolution + non-degenerate)\n")

offsheet = [r for r in use if r["frac"] < 0.30]
onsheet = [r for r in use if r["frac"] >= 0.50]
print(f"{'surface':46s} {'frac':>5s} {'seams':>6s} {'max_z':>6s}")
print("-" * 68)
for r in sorted(use, key=lambda r: r["frac"])[:6]:
    print(f"{r['name'][:46]:46s} {r['frac']:5.2f} {r['n_seams']:6d} {r['max_z']:6.2f}")

print()
if offsheet:
    fl = sum(1 for r in offsheet if r["n_seams"] > 0)
    print(f"OFF SHEET surfaces that sheetswitch flags: {fl} of {len(offsheet)}")
else:
    print("no OFF SHEET surface clears the sheetswitch gates -- "
          "the detector cannot see the one error we independently found")
if onsheet:
    fp = sum(1 for r in onsheet if r["n_seams"] > 0)
    print(f"ON SHEET surfaces sheetswitch also flags:  {fp} of {len(onsheet)} "
          f"({100*fp/len(onsheet):.0f}%)")
    z = [r["max_z"] for r in onsheet]
    print(f"max_z on ON SHEET surfaces: median {np.median(z):.2f}, max {max(z):.2f}")
if offsheet:
    print(f"max_z on OFF SHEET surfaces: {[round(r['max_z'],2) for r in offsheet]}")
