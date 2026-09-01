"""Do the other six scrolls have tracing series with an orderable winding index?

The terminal-patch rank test needs, per scroll, at least one series of >=5
surfaces whose winding order can be read from the name. Check that from names
alone before spending pod time on measurement.
"""
import re
from collections import defaultdict

rows = []
with open("/workspace/labelscope/findings/corpus/inputs/corpus_manifest.tsv") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or not parts[0].startswith("/"):
            continue
        base = parts[0].rsplit("/", 1)[-1]
        scroll, rest = base.split("__", 1)
        rows.append((scroll, rest))


def winding(name):
    m = re.search(r"w(\d+)-(\d+)", name)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    m = re.search(r"w(\d+)", name)
    return float(m.group(1)) if m else None


by_scroll = defaultdict(lambda: defaultdict(list))
nowind = defaultdict(int)
for scroll, rest in rows:
    w = winding(rest)
    if w is None:
        nowind[scroll] += 1
        continue
    m = re.match(r"(\d{8})", rest)
    by_scroll[scroll][m.group(1) if m else "?"].append(w)

print(f"{'scroll':14s} {'total':>5s} {'no-w':>5s}  usable series (date: n, winding span)")
total_series = total_surf = 0
for scroll in sorted(set(r[0] for r in rows)):
    ser = by_scroll[scroll]
    good = {d: sorted(ws) for d, ws in ser.items() if len(ws) >= 5}
    total_series += len(good)
    total_surf += sum(len(v) for v in good.values())
    desc = "  ".join(f"{d}: n={len(ws)}, w{ws[0]:.0f}-{ws[-1]:.0f}" for d, ws in sorted(good.items())) or "-- none --"
    n = sum(1 for r in rows if r[0] == scroll)
    print(f"{scroll:14s} {n:5d} {nowind[scroll]:5d}  {desc}")

print()
print(f"testable: {total_series} series covering {total_surf} surfaces")
