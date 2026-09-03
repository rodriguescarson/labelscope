#!/usr/bin/env python3
"""The pre-registered analysis for the blind ink-render test.

Written and committed before any label existed, so the statistics cannot be
chosen after seeing the outcome.  Everything computed here is fixed in
findings/ink-blind-preregistration.md.

    python scripts/onsheet/blind_analysis.py --labels drafts/ink-labels-a.csv drafts/ink-labels-b.csv \\
        --key drafts/ink-labeler-key.json \\
        --predictors findings/onsheet/onsheet_sv/ --out findings/ink-blind-result.json
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

TEXT_FLOOR = 0.20  # scrolls with fewer text labels carry no rank information
FLAT = 5.0  # secondary statistic: fraction of chunks with range below this


def auc(scores, positives):
    """Probability a random positive scores below a random negative (rank AUC)."""
    s = np.asarray(scores, float)
    y = np.asarray(positives, bool)
    if y.all() or (~y).all():
        return float("nan")
    pos, neg = s[y], s[~y]
    wins = (pos[:, None] < neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="code,label CSV(s) from the labeller; sets A and B",
    )
    ap.add_argument("--key", required=True)
    ap.add_argument(
        "--predictors", required=True, help="dir of labelscope onsheet --surface-volume JSONs"
    )
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    with open(args.key) as fh:
        key = json.load(fh)
    labels = {}
    for path in args.labels:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row.get("label"):
                    labels[row["code"].strip()] = row["label"].strip()

    pred = {}
    for f in glob.glob(os.path.join(args.predictors, "*.json")):
        with open(f) as fh:
            d = json.load(fh)
        r = d["results"][0]
        if "error" in r:
            continue
        rs = np.array([b["range"] for b in r["per_block"]])
        pred[os.path.basename(f)[:-5]] = {
            "mean": float(rs.mean()),
            "median": float(np.median(rs)),
            "flat_frac": float((rs < FLAT).mean()),
            "n": int(rs.size),
        }

    rows = []
    for code, lab in labels.items():
        meta = key.get(code)
        if not meta or meta["base"] not in pred:
            continue
        rows.append(
            {
                "code": code,
                "label": lab,
                "scroll": meta["scroll"],
                "segment": meta["segment"],
                **pred[meta["base"]],
            }
        )

    # within-scroll percentile of the pre-registered predictor (mean chunk range)
    by = defaultdict(list)
    for r in rows:
        by[r["scroll"]].append(r)
    for rs in by.values():
        means = np.array([r["mean"] for r in rs])
        for r in rs:
            r["pct"] = float((means < r["mean"]).mean() + 0.5 * (means == r["mean"]).mean())
        n_text = sum(1 for r in rs if r["label"] == "text")
        n_lab = sum(1 for r in rs if r["label"] in ("text", "no text"))
        frac = n_text / n_lab if n_lab else 0.0
        for r in rs:
            r["scroll_text_frac"] = frac
            r["qualifying"] = frac >= TEXT_FLOOR

    print(
        f"{'scroll':13s} {'labelled':>8s} {'text':>5s} {'no text':>8s} {'unsure':>7s} {'qualifies':>9s} {'AUC(no text)':>13s}"
    )
    print("-" * 70)
    per_scroll = {}
    for scroll in sorted(by):
        rs = by[scroll]
        lab = [r for r in rs if r["label"] in ("text", "no text")]
        a = auc([r["pct"] for r in lab], [r["label"] == "no text" for r in lab])
        per_scroll[scroll] = {
            "labelled": len(rs),
            "text": sum(r["label"] == "text" for r in rs),
            "no_text": sum(r["label"] == "no text" for r in rs),
            "unsure": sum(r["label"] == "unsure" for r in rs),
            "qualifies": bool(rs[0]["qualifying"]),
            "auc_no_text": a,
        }
        p = per_scroll[scroll]
        print(
            f"{scroll:13s} {p['labelled']:8d} {p['text']:5d} {p['no_text']:8d} {p['unsure']:7d} {str(p['qualifies']):>9s} {a:13.2f}"
        )

    # the pre-registered pass/fail
    q = [r for r in rows if r["qualifying"] and r["label"] in ("text", "no text")]
    base_rate = float(np.mean([r["label"] == "no text" for r in q])) if q else float("nan")
    bottom = [r for r in q if r["pct"] <= 0.10]
    precision = (
        float(np.mean([r["label"] == "no text" for r in bottom])) if bottom else float("nan")
    )
    passed = bool(len(bottom) >= 10 and precision >= 0.80 and precision >= 2 * base_rate)
    overall_auc = auc([r["pct"] for r in q], [r["label"] == "no text" for r in q])

    print()
    print(f"qualifying scrolls: {sorted({r['scroll'] for r in q})}")
    print(f"segments in test: {len(q)}   base rate of 'no text': {base_rate:.2f}")
    print(
        f"bottom decile of within-scroll percentile: n={len(bottom)}   precision(no text) = {precision:.2f}"
    )
    print(
        f"pre-registered pass (n>=10, precision>=0.80, >=2x base rate): {'PASS' if passed else 'FAIL'}"
    )
    print(f"descriptive AUC of within-scroll percentile vs 'no text': {overall_auc:.2f}")

    misses = sorted(
        [r for r in q if r["pct"] <= 0.10 and r["label"] == "text"], key=lambda r: r["pct"]
    )
    blanks = sorted(
        [r for r in q if r["pct"] >= 0.50 and r["label"] == "no text"], key=lambda r: -r["pct"]
    )
    print(f"\nlow-score segments that show text (the tool's misses): {len(misses)}")
    for r in misses[:15]:
        print(f"   pct {r['pct']:.2f} mean {r['mean']:5.1f}  {r['scroll']} {r['segment']}")
    print(
        f"high-score segments with no text (blank sheet or model miss, not the tool's): {len(blanks)}"
    )
    for r in blanks[:15]:
        print(f"   pct {r['pct']:.2f} mean {r['mean']:5.1f}  {r['scroll']} {r['segment']}")

    if args.out:
        payload = {
            "rows": rows,
            "per_scroll": per_scroll,
            "test": {
                "n": len(q),
                "base_rate_no_text": base_rate,
                "bottom_decile_n": len(bottom),
                "bottom_decile_precision": precision,
                "passed": passed,
                "auc": overall_auc,
            },
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
