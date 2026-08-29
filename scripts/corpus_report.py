#!/usr/bin/env python3
"""Turn a corpus sweep and its planted control into one report.

The unplanted pass is the finding.  The planted pass is what says whether the
finding means anything: the same surfaces, each with a whole winding displaced
into half of it, which is the case the spiral satisfaction metric scores as no
change at all.  A detector that cannot see the plant on a given surface cannot
be trusted to have seen its absence either, and the pairing is what makes that
checkable per surface instead of assumed once.

    python scripts/corpus_report.py --real corpus_real --plant corpus_plant \
        --out findings/corpus
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load(directory: str) -> dict:
    rows = {}
    for path in sorted(glob.glob(os.path.join(directory, "*", "sheetswitch.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                rows[row["name"]] = row
    return rows


def num(row, key, default=float("nan")):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truthy(row, key):
    return str(row.get(key, "")).lower() == "true"


def bootstrap_ci(values, n_boot=4000, seed=0):
    if len(values) < 3:
        return None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean": float(arr.mean()),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--plant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--z-threshold", type=float, default=5.0)
    args = ap.parse_args(argv)

    real, plant = load(args.real), load(args.plant)
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(real)} measured, {len(plant)} with the control planted")

    per_scroll = defaultdict(list)
    for name, row in real.items():
        per_scroll[name.split("__")[0]].append(row)

    table = []
    for scroll in sorted(per_scroll):
        rows = per_scroll[scroll]
        ok = [r for r in rows if not r.get("error")]
        gate = [r for r in ok if truthy(r, "resolution_adequate")]
        degen = [r for r in ok if truthy(r, "dip_degenerate")]
        usable = [r for r in gate if not truthy(r, "dip_degenerate")]
        flagged = [r for r in usable if int(r.get("n_seams") or 0) > 0]
        spw = [num(r, "steps_per_winding") for r in ok]
        spw = [v for v in spw if np.isfinite(v)]
        gb = [num(r, "mb_fetched", 0.0) / 1000 for r in ok]
        table.append(
            {
                "scroll": scroll,
                "surfaces": len(rows),
                "errors": len(rows) - len(ok),
                "resolution_adequate": len(gate),
                "dip_degenerate": len(degen),
                "usable": len(usable),
                "flagged": len(flagged),
                "steps_per_winding_median": float(np.median(spw)) if spw else None,
                "gb_per_surface_median": float(np.median(gb)) if gb else None,
            }
        )

    # the paired control, on surfaces where both passes produced a usable answer
    paired, ratios = [], []
    for name, row in real.items():
        other = plant.get(name)
        if other is None or row.get("error") or other.get("error"):
            continue
        if not truthy(row, "resolution_adequate") or truthy(row, "dip_degenerate"):
            continue
        if truthy(other, "dip_degenerate"):
            continue
        a, b = num(row, "max_z"), num(other, "max_z")
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
            continue
        paired.append({"name": name, "real_max_z": a, "plant_max_z": b, "ratio": b / a})
        ratios.append(b / a)

    summary = {
        "z_threshold": args.z_threshold,
        "n_measured": len(real),
        "per_scroll": table,
        "paired": {
            "n": len(paired),
            "plant_exceeds_real": int(
                sum(1 for p in paired if p["plant_max_z"] > p["real_max_z"])
            ),
            "ratio_median": float(np.median(ratios)) if ratios else None,
            "ratio_min": float(np.min(ratios)) if ratios else None,
            "ratio_max": float(np.max(ratios)) if ratios else None,
            "real_max_z": {
                "min": float(np.min([p["real_max_z"] for p in paired])) if paired else None,
                "median": float(np.median([p["real_max_z"] for p in paired]))
                if paired
                else None,
                "max": float(np.max([p["real_max_z"] for p in paired])) if paired else None,
            },
            "plant_max_z": {
                "min": float(np.min([p["plant_max_z"] for p in paired])) if paired else None,
                "median": float(np.median([p["plant_max_z"] for p in paired]))
                if paired
                else None,
                "max": float(np.max([p["plant_max_z"] for p in paired])) if paired else None,
            },
            "separable_by_a_fixed_threshold": bool(
                paired
                and min(p["plant_max_z"] for p in paired)
                > max(p["real_max_z"] for p in paired)
            ),
            "log_ratio_ci": bootstrap_ci([np.log(r) for r in ratios]) if ratios else None,
        },
    }

    with open(os.path.join(args.out, "corpus_summary.json"), "w") as handle:
        json.dump({"summary": summary, "paired": paired}, handle, indent=2)
    with open(os.path.join(args.out, "corpus_by_scroll.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(table[0].keys()) if table else ["scroll"]
        )
        writer.writeheader()
        writer.writerows(table)

    head = (
        f"{'scroll':13s} {'n':>4s} {'err':>4s} {'gate':>5s} {'degen':>6s} "
        f"{'usable':>7s} {'flag':>5s} {'spw':>6s} {'GB':>5s}"
    )
    print("\n" + head)
    print("-" * len(head))
    for row in table:
        print(
            f"{row['scroll']:13s} {row['surfaces']:4d} {row['errors']:4d} "
            f"{row['resolution_adequate']:5d} {row['dip_degenerate']:6d} {row['usable']:7d} "
            f"{row['flagged']:5d} {row['steps_per_winding_median'] or float('nan'):6.2f} "
            f"{row['gb_per_surface_median'] or float('nan'):5.1f}"
        )
    p = summary["paired"]
    print(
        f"\npaired control on {p['n']} surfaces: plant beats real on "
        f"{p['plant_exceeds_real']}, ratio median {p['ratio_median']:.2f}x "
        f"(range {p['ratio_min']:.2f}-{p['ratio_max']:.2f})"
    )
    print(
        f"  real  max z: {p['real_max_z']['min']:.2f} / {p['real_max_z']['median']:.2f} / {p['real_max_z']['max']:.2f}"
    )
    print(
        f"  plant max z: {p['plant_max_z']['min']:.2f} / {p['plant_max_z']['median']:.2f} / {p['plant_max_z']['max']:.2f}"
    )
    print(f"  a fixed threshold separates them: {p['separable_by_a_fixed_threshold']}")
    print(f"\nwrote {args.out}/corpus_summary.json, corpus_by_scroll.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
