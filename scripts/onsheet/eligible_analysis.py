#!/usr/bin/env python3
"""The pre-registered test against pscamillo's eye screening.

Everything here is fixed in findings/eligible-meshes-preregistration.md, written
before the predictor was computed. The labels are third-party and were published
on 1 September 2026, before the test was designed.

    python scripts/onsheet/eligible_analysis.py --scores em/out \\
        --index findings/eligible_meshes_index.csv --out findings/eligible-meshes-result.json
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

PASS_AUC = 0.70  # pre-registered
BOTTOM_FRACTION = 1 / 3


def auc(scores, positives):
    """P(a random positive scores below a random negative). 0.5 is chance."""
    s = np.asarray(scores, float)
    y = np.asarray(positives, bool)
    if y.all() or (~y).all():
        return float("nan")
    pos, neg = s[y], s[~y]
    wins = (pos[:, None] < neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="dir of labelscope onsheet JSONs")
    ap.add_argument("--index", required=True, help="pscamillo data/index.csv")
    ap.add_argument("--out")
    ap.add_argument("--csv", help="write the per-mesh score table for the PR")
    args = ap.parse_args(argv)

    with open(args.index) as fh:
        index = {f"{r['scroll']}__z{r['window_z']}_{r['wrap']}": r for r in csv.DictReader(fh)}

    rows = []
    failed = []
    for f in glob.glob(os.path.join(args.scores, "*.json")):
        key = os.path.basename(f)[:-5]
        with open(f) as fh:
            res = json.load(fh)["results"][0]
        meta = index.get(key)
        if meta is None:
            continue
        if "error" in res:
            failed.append((key, res["error"]))
            continue
        ranges = np.array([b["range"] for b in res["per_block"]])
        rows.append(
            {
                "key": key,
                "scroll": meta["scroll"],
                "wrap": meta["wrap"],
                "area_cm2": float(meta["area_cm2"]),
                "verdict": meta["gate_verdict"],
                "mean_range": float(ranges.mean()),
                "median_range": float(np.median(ranges)),
                "flat_frac": float((ranges < 5).mean()),
                "tiles": int(ranges.size),
            }
        )

    if not rows:
        print("no scored meshes matched the index")
        return 1

    by_scroll = defaultdict(list)
    for r in rows:
        by_scroll[r["scroll"]].append(r)
    for rs in by_scroll.values():
        means = np.array([r["mean_range"] for r in rs])
        for r in rs:
            r["pct"] = float((means < r["mean_range"]).mean() + 0.5 * (means == r["mean_range"]).mean())

    labelled = [r for r in rows if r["verdict"] in ("aprova", "reprova")]
    partial = [r for r in rows if r["verdict"] == "parcial"]
    unlabelled = [r for r in rows if not r["verdict"]]

    print(f"scored {len(rows)} meshes ({len(failed)} failed), of which "
          f"{len(labelled)} carry an aprova/reprova verdict, {len(partial)} parcial, "
          f"{len(unlabelled)} unlabelled")
    print()
    print(f"{'verdict':10s} {'n':>4s} {'mean range':>11s} {'median':>8s} {'flat frac':>10s}")
    print("-" * 48)
    for v in ("aprova", "parcial", "reprova", ""):
        g = [r for r in rows if r["verdict"] == v]
        if not g:
            continue
        mr = np.array([r["mean_range"] for r in g])
        ff = np.array([r["flat_frac"] for r in g])
        print(f"{v or '(none)':10s} {len(g):4d} {mr.mean():11.1f} {np.median(mr):8.1f} {ff.mean():10.2f}")

    a = auc([r["pct"] for r in labelled], [r["verdict"] == "reprova" for r in labelled])
    base = float(np.mean([r["verdict"] == "reprova" for r in labelled]))
    bottom = [r for r in labelled if r["pct"] <= BOTTOM_FRACTION]
    bottom_rate = float(np.mean([r["verdict"] == "reprova" for r in bottom])) if bottom else float("nan")
    passed = bool(a >= PASS_AUC and len(bottom) >= 10 and bottom_rate >= 2 * base)

    print()
    print(f"AUC of within-scroll percentile vs 'reprova': {a:.2f}   (pre-registered pass >= {PASS_AUC})")
    print(f"base 'reprova' rate: {base:.2f}   bottom third: n={len(bottom)} rate={bottom_rate:.2f} "
          f"(pass needs >= {2 * base:.2f})")
    print(f"PRE-REGISTERED RESULT: {'PASS' if passed else 'FAIL'}")

    print()
    print(f"{'scroll':12s} {'labelled':>8s} {'AUC':>6s}")
    per_scroll = {}
    for scroll in sorted(by_scroll):
        g = [r for r in by_scroll[scroll] if r["verdict"] in ("aprova", "reprova")]
        sa = auc([r["pct"] for r in g], [r["verdict"] == "reprova" for r in g]) if len(g) >= 4 else float("nan")
        per_scroll[scroll] = {"labelled": len(g), "auc": sa, "total": len(by_scroll[scroll])}
        print(f"{scroll:12s} {len(g):8d} {sa:6.2f}")

    miss = sorted([r for r in labelled if r["pct"] <= BOTTOM_FRACTION and r["verdict"] == "aprova"], key=lambda r: r["pct"])
    late = sorted([r for r in labelled if r["pct"] > 0.5 and r["verdict"] == "reprova"], key=lambda r: -r["pct"])
    print(f"\nthe eye passed but the score ranks low: {len(miss)}")
    for r in miss[:10]:
        print(f"   pct {r['pct']:.2f}  mean {r['mean_range']:6.1f}  {r['key']}")
    print(f"the eye failed but the score ranks high: {len(late)}")
    for r in late[:10]:
        print(f"   pct {r['pct']:.2f}  mean {r['mean_range']:6.1f}  {r['key']}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["key", "scroll", "wrap", "area_cm2", "verdict",
                                               "mean_range", "median_range", "flat_frac", "tiles", "pct"])
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["scroll"], -r["pct"])):
                w.writerow(r)
        print(f"\nwrote {args.csv}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "per_scroll": per_scroll, "failed": failed,
                       "test": {"auc": a, "base_reprova": base, "bottom_n": len(bottom),
                                "bottom_rate": bottom_rate, "passed": passed}}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
