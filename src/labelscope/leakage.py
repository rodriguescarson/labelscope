"""Measuring the leak in the units that matter: how much of a validation
patch's answer the model was already shown.

The overlap statistics in :mod:`labelscope.geometry` establish that patches share
voxels.  This module goes one step further and asks, for a concrete split, what
fraction of each validation patch's *labelled surface* also sits inside a
training patch — voxel for voxel, in scroll coordinates.  That number is an
upper bound on how much of the validation target is reproducible by memory
alone, and it needs no GPU and no training run to compute.

It also checks the assumption underneath: that two patches covering the same
scroll voxel agree about what is there.  Where they do not, the disagreement is
a labelling inconsistency worth reporting in its own right.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from labelscope.geometry import Patch, overlap_graph
from labelscope.io import read_volume


def _intersection(a: Patch, b: Patch):
    """Overlap box in ``a``'s local coordinates, or None."""
    lo, hi = [], []
    for axis in range(3):
        start = max(a.origin[axis], b.origin[axis])
        stop = min(a.origin[axis] + a.size[axis], b.origin[axis] + b.size[axis])
        if stop <= start:
            return None
        lo.append(start - a.origin[axis])
        hi.append(stop - a.origin[axis])
    return tuple(lo), tuple(hi)


def _fold_of(splits: Sequence[dict]) -> Dict[str, int]:
    fold = {}
    for index, split in enumerate(splits):
        for name in split["val"]:
            fold[name] = index
    return fold


def _train_sets(splits: Sequence[dict]) -> List[set]:
    """The training names of each fold, taken from the split itself.

    Not "everything outside this fold": a buffered split deliberately drops the
    patches that touch validation, and counting those as training data would
    make the split look like it still leaks.
    """
    return [set(split["train"]) for split in splits]


def measure_seen_fraction(
    patches: Sequence[Patch],
    splits: Sequence[dict],
    labels_dir: str,
    surface_class: int = 1,
    graph: Optional[Dict] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """Fraction of each validation patch's labelled voxels that a training patch
    also covers, under the given split.

    One label volume is read per patch: a patch sits in exactly one validation
    fold, so its training neighbours are simply the neighbours outside that fold.
    """
    graph = graph if graph is not None else overlap_graph(patches)
    fold = _fold_of(splits)
    train_sets = _train_sets(splits)
    by_name = {p.name: i for i, p in enumerate(patches)}

    per_patch: List[Dict] = []
    for count, patch in enumerate(patches, 1):
        if patch.name not in fold:
            continue
        path = os.path.join(labels_dir, patch.name + ".tif")
        if not os.path.exists(path):
            continue
        try:
            label = read_volume(path)
        except Exception as exc:
            per_patch.append({"name": patch.name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        mask = label == surface_class
        total = int(mask.sum())
        if total == 0:
            continue

        my_fold = fold[patch.name]
        training = train_sets[my_fold]
        seen = np.zeros(mask.shape, dtype=bool)
        n_neighbours = 0
        for other in graph.get(by_name[patch.name], ()):
            neighbour = patches[other]
            if neighbour.name not in training:
                continue  # validation, or dropped into the buffer zone
            box = _intersection(patch, neighbour)
            if box is None:
                continue
            (z0, y0, x0), (z1, y1, x1) = box
            seen[z0:z1, y0:y1, x0:x1] = True
            n_neighbours += 1

        per_patch.append(
            {
                "name": patch.name,
                "fold": my_fold,
                "labelled_voxels": total,
                "seen_voxels": int((mask & seen).sum()),
                "seen_fraction": float((mask & seen).sum() / total),
                "volume_seen_fraction": float(seen.mean()),
                "training_neighbours": n_neighbours,
            }
        )
        if progress and count % 25 == 0:
            progress(count, len(patches))

    usable = [r for r in per_patch if "seen_fraction" in r]
    if not usable:
        return {"n_patches": 0, "per_patch": per_patch}

    fractions = np.array([r["seen_fraction"] for r in usable])
    weights = np.array([r["labelled_voxels"] for r in usable], dtype=float)
    return {
        "n_patches": len(usable),
        "seen_fraction_mean": float(fractions.mean()),
        "seen_fraction_median": float(np.median(fractions)),
        "seen_fraction_p90": float(np.percentile(fractions, 90)),
        "seen_fraction_max": float(fractions.max()),
        "seen_fraction_voxel_weighted": float((fractions * weights).sum() / weights.sum()),
        "patches_with_any_seen": int((fractions > 0).sum()),
        "patches_with_any_seen_pct": float(100.0 * (fractions > 0).mean()),
        "patches_over_25pct_seen": int((fractions > 0.25).sum()),
        "per_patch": per_patch,
    }


def check_overlap_consistency(
    patches: Sequence[Patch],
    labels_dir: str,
    graph: Optional[Dict] = None,
    surface_class: int = 1,
    max_pairs: int = 300,
    seed: int = 0,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """Do two patches covering the same scroll voxel agree about what is there?

    They should: the overlap is the same physical papyrus, labelled once.  A
    disagreement means the release carries two different answers for one voxel,
    which is a problem for anything trained on both.
    """
    graph = graph if graph is not None else overlap_graph(patches)
    pairs = sorted({(a, b) for a, neighbours in graph.items() for b in neighbours if a < b})
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        chosen = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in chosen]

    cache: Dict[str, np.ndarray] = {}

    def load(patch: Patch):
        if patch.name not in cache:
            path = os.path.join(labels_dir, patch.name + ".tif")
            if not os.path.exists(path):
                return None
            if len(cache) > 24:
                cache.pop(next(iter(cache)))
            cache[patch.name] = read_volume(path) == surface_class
        return cache[patch.name]

    results = []
    for n, (i, j) in enumerate(pairs, 1):
        a, b = patches[i], patches[j]
        box_a, box_b = _intersection(a, b), _intersection(b, a)
        if box_a is None or box_b is None:
            continue
        la, lb = load(a), load(b)
        if la is None or lb is None:
            continue
        (az0, ay0, ax0), (az1, ay1, ax1) = box_a
        (bz0, by0, bx0), (bz1, by1, bx1) = box_b
        va = la[az0:az1, ay0:ay1, ax0:ax1]
        vb = lb[bz0:bz1, by0:by1, bx0:bx1]
        if va.shape != vb.shape or va.size == 0:
            continue
        both = int((va & vb).sum())
        either = int((va | vb).sum())
        results.append(
            {
                "a": a.name,
                "b": b.name,
                "voxels": int(va.size),
                "iou": float(both / either) if either else 1.0,
                "dice": float(2 * both / (va.sum() + vb.sum()))
                if (va.sum() + vb.sum())
                else 1.0,
                "disagreeing_voxels": int((va ^ vb).sum()),
                "disagreement_rate": float((va ^ vb).sum() / va.size),
            }
        )
        if progress and n % 25 == 0:
            progress(n, len(pairs))

    if not results:
        return {"n_pairs": 0}
    ious = np.array([r["iou"] for r in results])
    return {
        "n_pairs": len(results),
        "iou_mean": float(ious.mean()),
        "iou_median": float(np.median(ious)),
        "iou_min": float(ious.min()),
        "pairs_below_0_9_iou": int((ious < 0.9).sum()),
        "pairs_below_0_5_iou": int((ious < 0.5).sum()),
        "worst": sorted(results, key=lambda r: r["iou"])[:10],
    }


def nnunet_reference_splits(names, k: int = 5, seed: int = 12345):
    """The split nnU-Net writes when none is supplied — re-exported here so the
    leak can be measured against it directly."""
    from labelscope.geometry import nnunet_default_split

    return nnunet_default_split(names, k=k, seed=seed)
