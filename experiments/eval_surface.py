"""Compare two surface models without ever looking at a label.

Dice against either label set is circular: each arm is favoured by the labels it
was trained on, so the arm trained on the corrected labels wins on the corrected
labels and loses on the originals, and neither result says anything about the
surface.

So the primary metric never touches a label.  For every held-out patch the model
predicts, the prediction is binarised, and the *prediction's own* placement is
measured against the CT with the same estimator the audit uses.  A model that has
learned a cleaner surface should put its prediction somewhere the scan agrees
with, and its cells should agree with each other.

Reported as paired per-patch differences with a bootstrap interval, because the
patches vary far more than the arms do and an unpaired comparison would drown in
that.  Dice against both label sets is reported too, and labelled circular.

    python experiments/eval_surface.py --runs runs/orig_seed0 runs/reg_seed0
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from labelscope.alignment import aggregate_alignment

METRICS = (
    "cell_abs_offset_p90",
    "cell_frac_ge_1vx",
    "global_profile_snr",
    "cell_offset_spread",
)


def load_model(run_dir: str, device: str):
    from train_surface import UNet3D

    state = torch.load(
        os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False
    )
    model = UNet3D().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["config"]


@torch.no_grad()
def predict(model, image: np.ndarray, device: str, crop: int = 128, stride: int = 96):
    """Sliding-window logits over a whole patch, averaged on the overlaps."""
    x = image.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    pad = [max(0, crop - s) for s in x.shape]
    if any(pad):
        x = np.pad(x, [(0, p) for p in pad])
    acc = np.zeros(x.shape, np.float32)
    hits = np.zeros(x.shape, np.float32)
    starts = [
        sorted({*range(0, s - crop + 1, stride), s - crop}) if s > crop else [0]
        for s in x.shape
    ]
    for z in starts[0]:
        for y in starts[1]:
            for xx in starts[2]:
                sl = (slice(z, z + crop), slice(y, y + crop), slice(xx, xx + crop))
                block = torch.from_numpy(x[sl])[None, None].to(device)
                with torch.amp.autocast(device):
                    out = model(block)
                acc[sl] += torch.sigmoid(out.float())[0, 0].cpu().numpy()
                hits[sl] += 1
    probs = acc / np.maximum(hits, 1)
    return probs[tuple(slice(0, s) for s in image.shape)]


def measure(image: np.ndarray, mask: np.ndarray, seed: int = 0) -> dict:
    """The prediction's own placement, measured against the scan."""
    if mask.sum() < 2000:
        return {"error": "prediction too small to measure", "voxels": int(mask.sum())}
    result = aggregate_alignment(
        image, mask, cell=64, min_per_cell=200, n_samples=20_000, bootstrap=0, seed=seed
    )
    if result.get("n_cells", 0) < 2:
        return {"error": "too few resolved cells", "n_cells": result.get("n_cells", 0)}
    offsets = [c["offset"] for c in result.get("cells", [])]
    return {
        "voxels": int(mask.sum()),
        "n_cells": result["n_cells"],
        "cell_abs_offset_p90": result.get("cell_abs_offset_p90"),
        "cell_frac_ge_1vx": result.get("cell_frac_ge_1vx"),
        "global_profile_snr": result.get("global_profile_snr"),
        "cell_offset_spread": float(np.std(offsets)) if len(offsets) > 1 else None,
    }


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = float((pred & truth).sum())
    return 2 * inter / max(1.0, float(pred.sum() + truth.sum()))


def paired_ci(a, b, n_boot=2000, seed=0):
    """Bootstrap interval on the mean paired difference b - a."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    diff = np.array([y - x for x, y in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(diff),
        "mean_difference": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, required=True, metavar=("BASELINE", "TREATMENT"))
    ap.add_argument("--images", default="/workspace/d059/imagesTr")
    ap.add_argument("--labels-orig", default="/workspace/d059/labelsTr")
    ap.add_argument("--labels-reg", default="/workspace/d059/labelsTr_reg")
    ap.add_argument("--splits", default="findings/full/d059_leakage/splits_final.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import tifffile

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.splits) as handle:
        names = json.load(handle)[args.fold]["val"]
    have = {f[: -len(".tif")] for f in os.listdir(args.labels_orig) if f.endswith(".tif")}
    names = [n for n in names if n in have]
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names)} held-out patches, device {device}", flush=True)

    models = {tag: load_model(tag, device) for tag in args.runs}
    rows = []
    for n, name in enumerate(names, 1):
        image = tifffile.imread(os.path.join(args.images, f"{name}_0000.tif"))
        row = {"name": name}
        for tag, (model, _) in models.items():
            probs = predict(model, image, device)
            mask = probs > 0.5
            row[tag] = measure(image, mask, seed=0)
            for which, folder in (("orig", args.labels_orig), ("reg", args.labels_reg)):
                path = os.path.join(folder, f"{name}.tif")
                if os.path.exists(path):
                    truth = tifffile.imread(path) == 1
                    row[tag][f"dice_vs_{which}_circular"] = dice(mask, truth)
        rows.append(row)
        if n % 10 == 0 or n == len(names):
            print(f"  {n}/{len(names)}", flush=True)
            with open(args.out, "w") as handle:
                json.dump({"per_patch": rows}, handle, indent=2)

    base, treat = args.runs
    summary = {}
    for metric in METRICS:
        a = [r[base].get(metric) for r in rows]
        b = [r[treat].get(metric) for r in rows]
        summary[metric] = paired_ci(a, b)
    for which in ("orig", "reg"):
        key = f"dice_vs_{which}_circular"
        summary[key] = paired_ci(
            [r[base].get(key) for r in rows], [r[treat].get(key) for r in rows]
        )

    payload = {
        "baseline": base,
        "treatment": treat,
        "n_patches": len(rows),
        "primary_metrics_are_label_free": list(METRICS),
        "summary": summary,
        "per_patch": rows,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
