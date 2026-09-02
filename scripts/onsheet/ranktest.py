"""The terminal-patch rank test, exactly as pre-registered in 8d6faad.

Primary: series of >=5 surfaces on scrolls clearing the resolution gate.
Secondary, excluded from the combined p-value: PHerc0172 (0.90 steps/winding).
"""

import glob
import json
import os
import re
from collections import defaultdict


def winding(name):
    m = re.search(r"w(\d+)-(\d+)", name)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    m = re.search(r"w(\d+)", name)
    return float(m.group(1)) if m else None


def load(pattern, scroll_from_name):
    out = []
    for f in glob.glob(pattern):
        with open(f) as fh:
            d = json.load(fh)
        t = next((r for r in d["results"] if not r.get("baseline") and "error" not in r), None)
        if not t:
            continue
        n = os.path.basename(f)[:-5]
        scroll = scroll_from_name(n)
        w = winding(n)
        m = re.search(r"(\d{8})\d*-w", n)
        if w is None or not m:
            continue
        out.append(
            {
                "scroll": scroll,
                "day": m.group(1),
                "w": w,
                "range": t["range_median"],
                "name": n,
            }
        )
    return out


rows = load("/workspace/onsheet_corpus/*.json", lambda n: "PHercParis4")
rows += load("/workspace/onsheet_series/*.json", lambda n: n.split("__")[0])

series = defaultdict(list)
for r in rows:
    series[(r["scroll"], r["day"])].append(r)

PRIMARY = {("PHercParis4", "20260623"), ("PHercParis4", "20260701"), ("PHerc0139", "20250108")}
SECONDARY = {("PHerc0172", "20250926")}

print(
    f"{'scroll':13s} {'series':10s} {'n':>3s} {'last w':>7s} {'last range':>11s} "
    f"{'rank of last':>13s} {'worst?':>7s}"
)
print("-" * 74)
combined = 1.0
tested = 0
for key in sorted(PRIMARY | SECONDARY):
    rs = series.get(key)
    if not rs:
        print(f"{key[0]:13s} {key[1]:10s}  -- not measured --")
        continue
    rs.sort(key=lambda r: r["w"])
    n = len(rs)
    last = rs[-1]
    order = sorted(rs, key=lambda r: r["range"])
    rank = order.index(last) + 1
    worst = rank == 1
    tag = "" if key in PRIMARY else "   (secondary)"
    if key in PRIMARY:
        combined *= (1.0 / n) if worst else 1.0
        tested += 1
    print(
        f"{key[0]:13s} {key[1]:10s} {n:3d} {last['w']:7.0f} {last['range']:11.1f} "
        f"{f'{rank} of {n}':>13s} {'YES' if worst else 'no':>7s}{tag}"
    )

prim = [series[k] for k in PRIMARY if k in series]
hits = sum(
    1
    for rs in prim
    if sorted(rs, key=lambda r: r["range"])[0] is max(rs, key=lambda r: r["w"])
)
print()
print(f"PRIMARY: {hits} of {len(prim)} terminal patches are the worst in their series")
if hits == len(prim) and prim:
    print(f"combined p = {combined:.2e}  (1 in {1 / combined:.0f})")
else:
    print(
        "not all terminal patches are worst -> reported as pre-registered, "
        "no combined p-value claimed"
    )
