#!/usr/bin/env python3
"""The pre-registered test of the seam detector against pscamillo's 12 rejects.

Rules are fixed in findings/reject-seam-preregistration.md, committed before
the detector ran. This script only applies them.

    python scripts/onsheet/reject_seam_analysis.py --out-dir findings/reject_seam/out \
        --md findings/reject-seam-result.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

STEP = [
    "PHerc0813__z12496_w100", "PHerc0813__z11296_w100", "PHerc0813__z4704_w100",
    "PHerc0813__z13088_w100", "PHerc0813__z11904_w100", "PHerc0813__z11904_w080",
    "PHerc0211__z4912_w080", "PHerc0211__z13920_w080", "PHerc0800__z17072_w100",
    "PHerc0125__z6944_w080",
]
DRIFT = ["PHerc0211__z6720_w100", "PHerc0211__z9120_w100"]


def read_one(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    r = rows[0]
    return {
        "steps_per_winding": float(r["steps_per_winding"]),
        "gate": r["resolution_adequate"] == "True",
        "n_seams": int(r["n_seams"]),
        "max_z": float(r["max_z"]),
        "winding_spacing": float(r["winding_spacing"]),
        "grid_step": float(r["grid_step"]),
        "mb": float(r["mb_fetched"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--md")
    a = ap.parse_args(argv)

    table = []
    for tag in STEP + DRIFT:
        d0 = read_one(os.path.join(a.out_dir, f"{tag}__plant0.json", "sheetswitch.csv")) if os.path.exists(os.path.join(a.out_dir, f"{tag}__plant0.json", "sheetswitch.csv")) else None
        d1 = read_one(os.path.join(a.out_dir, f"{tag}__plant1.json", "sheetswitch.csv")) if os.path.exists(os.path.join(a.out_dir, f"{tag}__plant1.json", "sheetswitch.csv")) else None
        table.append({"tag": tag, "set": "step" if tag in STEP else "drift", "d": d0, "p": d1})

    missing = [t["tag"] for t in table if t["d"] is None or t["p"] is None]
    G = [t for t in table if t["d"] and t["p"] and t["d"]["gate"]]
    g_step = [t for t in G if t["set"] == "step"]
    g_drift = [t for t in G if t["set"] == "drift"]

    # pre-registered rules, in order
    if len(g_step) < 5:
        verdict = "INCONCLUSIVE BY GATE"
        why = f"only {len(g_step)} of 10 step meshes pass the resolution gate (need 5)"
    else:
        ctrl_ok = sum(1 for t in G if t["p"]["max_z"] > t["d"]["max_z"])
        if ctrl_ok / len(G) < 0.70:
            verdict = "CONTROL VOID"
            why = f"planted copy scored higher on only {ctrl_ok}/{len(G)} gate-passing meshes (need 70%)"
        else:
            hit = sum(1 for t in g_step if t["d"]["n_seams"] >= 1)
            fp = sum(1 for t in g_drift if t["d"]["n_seams"] >= 1)
            ok = hit / len(g_step) >= 0.70 and fp == 0
            verdict = "PASS" if ok else "FAIL"
            why = f"seams on {hit}/{len(g_step)} step meshes, {fp}/{len(g_drift)} drift meshes"

    lines = []
    lines.append(f"PRE-REGISTERED RESULT: {verdict}  ({why})")
    if missing:
        lines.append(f"missing outputs: {', '.join(missing)}")
    lines.append("")
    hdr = f"{'mesh':26s} {'set':5s} {'step/wind':>9s} {'gate':>5s} {'seams':>5s} {'max_z':>6s} {'max_z+1w':>8s} {'MB':>7s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for t in table:
        d, p = t["d"], t["p"]
        if not d:
            lines.append(f"{t['tag']:26s} {t['set']:5s} {'(no output)':>9s}")
            continue
        lines.append(
            f"{t['tag']:26s} {t['set']:5s} {d['steps_per_winding']:9.2f} {str(d['gate']):>5s} "
            f"{d['n_seams']:5d} {d['max_z']:6.2f} {(p['max_z'] if p else float('nan')):8.2f} {d['mb']:7.0f}"
        )
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write("```\n" + text + "\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
