"""Patch geometry: where each training patch came from, and what that implies
for cross-validation.

Many Vesuvius surface-label datasets are cut from a handful of scroll volumes on
a sliding window whose stride is smaller than the patch, so neighbouring patches
share voxels.  A random k-fold split over such patches puts overlapping voxels on
both sides of the split, and the validation score that follows is optimistic —
which matters, because that score is what checkpoint selection and loss-variant
comparisons are decided on.

This module measures that overlap and emits a spatially blocked split that
removes it.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: ``s1_z10240_y2560_x2560`` — scroll id plus the patch's origin in scroll voxels
PATCH_NAME_RE = re.compile(r"^(?P<vol>[A-Za-z0-9.]+)_z(?P<z>\d+)_y(?P<y>\d+)_x(?P<x>\d+)$")


@dataclass(frozen=True)
class Patch:
    name: str
    volume: str
    origin: Tuple[int, int, int]  # z, y, x
    size: Tuple[int, int, int]  # z, y, x

    def overlap_fraction(self, other: Patch) -> float:
        """Shared volume as a fraction of this patch's volume (0 if disjoint)."""
        if self.volume != other.volume:
            return 0.0
        shared = 1
        for axis in range(3):
            lo = max(self.origin[axis], other.origin[axis])
            hi = min(
                self.origin[axis] + self.size[axis], other.origin[axis] + other.size[axis]
            )
            if hi <= lo:
                return 0.0
            shared *= hi - lo
        return shared / float(self.size[0] * self.size[1] * self.size[2])

    def gap(self, other: Patch) -> Optional[int]:
        """Smallest per-axis gap in voxels; 0 when the patches touch or overlap.

        ``None`` when the patches are from different volumes and so incomparable.
        """
        if self.volume != other.volume:
            return None
        gaps = []
        for axis in range(3):
            lo_a, hi_a = self.origin[axis], self.origin[axis] + self.size[axis]
            lo_b, hi_b = other.origin[axis], other.origin[axis] + other.size[axis]
            gaps.append(max(0, max(lo_a, lo_b) - min(hi_a, hi_b)))
        return max(gaps)


def parse_patch_names(
    names: Iterable[str],
    size: Tuple[int, int, int],
    sizes: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> Tuple[List[Patch], List[str]]:
    """Parse coordinate-encoded patch names.  Returns (parsed, unparsed).

    ``sizes`` supplies each patch's real shape, read from the volume rather than
    assumed.  That matters: ``Dataset059`` names look uniform but ships 170, 172,
    236 and 300 voxel cubes in one directory, and computing overlaps at a single
    assumed size gets every one of them wrong.
    """
    parsed, unparsed = [], []
    for name in names:
        match = PATCH_NAME_RE.match(name)
        if match is None:
            unparsed.append(name)
            continue
        parsed.append(
            Patch(
                name=name,
                volume=match.group("vol"),
                origin=(int(match.group("z")), int(match.group("y")), int(match.group("x"))),
                size=(sizes or {}).get(name, size),
            )
        )
    return parsed, unparsed


def sizes_from_volumes(names: Iterable[str], directory: str) -> Tuple[Dict, List[str]]:
    """Read each patch's true shape from its label volume header.

    Returns ``(sizes, unreadable)``.  Header-only, so it costs a file open per
    volume and no decoding.
    """
    import os

    from labelscope.io import probe_volume

    sizes, unreadable = {}, []
    for name in names:
        for extension in (".tif", ".tiff"):
            path = os.path.join(directory, name + extension)
            if os.path.exists(path):
                info = probe_volume(path)
                if info.shape and len(info.shape) == 3:
                    sizes[name] = tuple(int(v) for v in info.shape)
                else:
                    unreadable.append(name)
                break
        else:
            unreadable.append(name)
    return sizes, unreadable


# --------------------------------------------------------------------------- #
# overlap graph
# --------------------------------------------------------------------------- #
def overlap_graph(patches: Sequence[Patch], buffer: int = 0) -> Dict[int, Dict[int, float]]:
    """Adjacency between patches that share voxels (or come within ``buffer``).

    ``buffer`` widens the notion of contamination: with ``buffer=32`` two patches
    that stop 32 voxels short of each other still count as neighbours, because
    the same papyrus sheet almost certainly runs through both.
    """
    by_volume: Dict[str, List[int]] = defaultdict(list)
    for idx, patch in enumerate(patches):
        by_volume[patch.volume].append(idx)

    graph: Dict[int, Dict[int, float]] = defaultdict(dict)
    for indices in by_volume.values():
        # sort by z so the inner loop can stop early
        indices.sort(key=lambda i: patches[i].origin[0])
        for a_pos, a in enumerate(indices):
            pa = patches[a]
            z_limit = pa.origin[0] + pa.size[0] + buffer
            for b in indices[a_pos + 1 :]:
                pb = patches[b]
                if pb.origin[0] >= z_limit:
                    break
                fraction = pa.overlap_fraction(pb)
                gap = pa.gap(pb)
                if fraction > 0 or (buffer > 0 and gap is not None and gap <= buffer):
                    graph[a][b] = fraction
                    graph[b][a] = fraction
    return graph


def simulate_random_kfold(
    n: int, graph: Dict[int, Dict[int, float]], k: int = 5, trials: int = 200, seed: int = 0
) -> dict:
    """How much of a random k-fold validation set is contaminated by its own
    training set?  Returns mean/sd over ``trials`` shuffles."""
    rng = random.Random(seed)
    contaminated_pct, shared_pct = [], []
    all_indices = list(range(n))
    for _ in range(trials):
        rng.shuffle(all_indices)
        folds = [all_indices[i::k] for i in range(k)]
        n_contaminated = n_val = 0
        shared_total = 0.0
        for fold in folds:
            val = set(fold)
            for v in val:
                n_val += 1
                fractions = [f for nb, f in graph.get(v, {}).items() if nb not in val]
                if fractions:
                    n_contaminated += 1
                    shared_total += max(fractions)
        contaminated_pct.append(100.0 * n_contaminated / n_val)
        shared_pct.append(100.0 * shared_total / n_val)

    def _stats(values):
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        return mean, var**0.5

    contaminated_mean, contaminated_sd = _stats(contaminated_pct)
    shared_mean, _ = _stats(shared_pct)
    return {
        "k": k,
        "trials": trials,
        "val_patches_contaminated_pct_mean": contaminated_mean,
        "val_patches_contaminated_pct_sd": contaminated_sd,
        "mean_max_shared_volume_pct": shared_mean,
    }


def nnunet_default_split(names: Sequence[str], k: int = 5, seed: int = 12345) -> List[dict]:
    """Reproduce exactly the split nnU-Net v2 writes when none is supplied.

    ``nnUNetTrainer.do_split`` calls ``generate_crossval_split(sorted_keys,
    seed=12345, n_splits=5)``, which is ``sklearn.model_selection.KFold(
    n_splits, shuffle=True, random_state=seed)`` over the sorted case names.
    Reproducing it means the leak can be reported for *the* split that is
    actually trained on, not an average over hypothetical ones.
    """
    ordered = sorted(names)
    n = len(ordered)
    try:
        from sklearn.model_selection import KFold

        folds = [
            list(test) for _, test in KFold(k, shuffle=True, random_state=seed).split(ordered)
        ]
    except ImportError:  # sklearn's own algorithm
        import numpy as np

        indices = np.arange(n)
        np.random.RandomState(seed).shuffle(indices)
        sizes = np.full(k, n // k, dtype=int)
        sizes[: n % k] += 1
        folds, cursor = [], 0
        for size in sizes:
            folds.append(list(indices[cursor : cursor + size]))
            cursor += size
    out = []
    for fold in folds:
        val = {ordered[i] for i in fold}
        out.append(
            {
                "train": [x for x in ordered if x not in val],
                "val": [x for x in ordered if x in val],
            }
        )
    return out


def measure_split_leakage(
    patches: Sequence[Patch], splits: Sequence[dict], graph: Dict[int, Dict[int, float]]
) -> dict:
    """How much of a given split's validation data is contaminated by its own
    training data?"""
    index = {patch.name: i for i, patch in enumerate(patches)}
    contaminated = n_val = 0
    shared_total = 0.0
    worst = 0.0
    for split in splits:
        train = {index[name] for name in split["train"] if name in index}
        for name in split["val"]:
            v = index.get(name)
            if v is None:
                continue
            n_val += 1
            fractions = [f for nb, f in graph.get(v, {}).items() if nb in train]
            if fractions:
                contaminated += 1
                shared_total += max(fractions)
                worst = max(worst, max(fractions))
    return {
        "n_val_evaluations": n_val,
        "val_patches_contaminated": contaminated,
        "val_patches_contaminated_pct": 100.0 * contaminated / n_val if n_val else 0.0,
        "mean_max_shared_volume_pct": 100.0 * shared_total / n_val if n_val else 0.0,
        "worst_shared_volume_pct": 100.0 * worst,
    }


# --------------------------------------------------------------------------- #
# a split that does not leak
# --------------------------------------------------------------------------- #
def connected_components(n: int, graph: Dict[int, Dict[int, float]]) -> List[List[int]]:
    seen = [False] * n
    components = []
    for start in range(n):
        if seen[start]:
            continue
        stack, component = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in graph.get(node, ()):
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _block_id(patch: Patch, block_size: Tuple[int, int, int]) -> Tuple:
    centre = tuple(patch.origin[a] + patch.size[a] // 2 for a in range(3))
    return (patch.volume,) + tuple(centre[a] // block_size[a] for a in range(3))


def blocked_kfold(
    patches: Sequence[Patch],
    k: int = 5,
    buffer: int = 0,
    seed: int = 0,
    mode: str = "block",
    block_factor: int = 4,
) -> Tuple[List[dict], dict]:
    """A k-fold split in which no validation patch touches a training patch.

    Two strategies, both leak-free:

    ``mode="block"`` (default, recommended)
        Spatial block cross-validation.  Each source volume is tiled into blocks
        ``block_factor`` times the patch size; whole blocks are dealt to folds so
        the validation folds stay near-equal in size.  Any training patch that
        would still touch a validation patch is *dropped* into a buffer zone
        rather than silently leaking.  This is the standard remedy for spatially
        autocorrelated data, and it costs a few percent of the training set.

    ``mode="component"``
        Whole connected components of the overlap graph are dealt to folds.
        Nothing is discarded, but fold sizes follow component sizes and can be
        very uneven when one component spans most of a scroll.

    Returns ``(splits, stats)`` where ``splits`` is nnU-Net's
    ``splits_final.json`` structure: a list of ``{"train": [...], "val": [...]}``.
    """
    graph = overlap_graph(patches, buffer=buffer)
    rng = random.Random(seed)

    if mode == "component":
        groups = connected_components(len(patches), graph)
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
    elif mode == "block":
        # blocks are sized off the largest patch, so a block is never smaller
        # than the patches it is meant to separate
        largest = tuple(max((p.size[a] for p in patches), default=1) for a in range(3))
        block_size = tuple(max(1, s * block_factor) for s in largest)
        buckets: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, patch in enumerate(patches):
            buckets[_block_id(patch, block_size)].append(idx)
        groups = list(buckets.values())
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'block' or 'component'")

    fold_members: List[List[int]] = [[] for _ in range(k)]
    for group in groups:
        target = min(range(k), key=lambda i: len(fold_members[i]))
        fold_members[target].extend(group)

    splits, dropped_per_fold, residual = [], [], 0
    for i in range(k):
        val_set = set(fold_members[i])
        train, dropped = [], 0
        for j in range(len(patches)):
            if j in val_set:
                continue
            if any(nb in val_set for nb in graph.get(j, ())):
                dropped += 1  # buffer zone: touches validation, so excluded
                continue
            train.append(j)
        dropped_per_fold.append(dropped)
        train_set = set(train)
        residual += sum(1 for v in val_set if any(nb in train_set for nb in graph.get(v, ())))
        splits.append(
            {
                "train": [patches[j].name for j in sorted(train)],
                "val": [patches[j].name for j in sorted(val_set)],
            }
        )

    stats = {
        "mode": mode,
        "k": k,
        "buffer": buffer,
        "n_patches": len(patches),
        "n_groups": len(groups),
        "largest_group": max((len(g) for g in groups), default=0),
        "val_fold_sizes": [len(f) for f in fold_members],
        "train_fold_sizes": [len(s["train"]) for s in splits],
        "buffer_dropped_per_fold": dropped_per_fold,
        "buffer_dropped_pct_mean": (
            100.0 * sum(dropped_per_fold) / (k * len(patches)) if patches else 0.0
        ),
        "residual_leaking_val_patches": residual,
    }
    return splits, stats


def write_splits(splits: List[dict], path: str) -> None:
    with open(path, "w") as handle:
        json.dump(splits, handle, indent=1)
