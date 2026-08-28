"""Label-only quality metrics.

Everything here is computed from the label volume alone, so it runs over a whole
dataset without downloading the (far larger) CT volumes.  The metrics are chosen
to catch the failure modes the Vesuvius Challenge Open Problems post describes:
labels that merge two windings, labels that are blobs rather than sheets, and
labels whose class scheme silently disagrees with the rest of the dataset.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi

#: 26-connectivity — two label voxels touching at a corner are the same sheet
_CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)

#: 6-connectivity ball, for peeling a mask one Euclidean-ish layer at a time
_BALL = ndi.generate_binary_structure(3, 1)


def label_scheme(label: np.ndarray) -> Dict:
    """Which class values the volume actually uses, and how much of each."""
    values, counts = np.unique(label, return_counts=True)
    total = float(label.size)
    return {
        "values": [int(v) for v in values],
        "value_fractions": {int(v): float(c / total) for v, c in zip(values, counts)},
        "n_values": int(values.size),
        "foreground_fraction": float((label > 0).sum() / total),
        "border_touch_fraction": _border_touch_fraction(label > 0),
    }


def _border_touch_fraction(mask: np.ndarray) -> float:
    """Fraction of foreground voxels sitting on the volume's outer faces.

    A thin sheet crossing the patch touches the border; a blob that fills the
    patch touches it far more.  Useful as a cheap sanity signal.
    """
    if not mask.any():
        return 0.0
    faces = (
        mask[0].sum()
        + mask[-1].sum()
        + mask[:, 0].sum()
        + mask[:, -1].sum()
        + mask[:, :, 0].sum()
        + mask[:, :, -1].sum()
    )
    return float(faces / mask.sum())


def thickness_stats(
    mask: np.ndarray,
    sample: int = 200_000,
    max_thickness: Optional[int] = 24,
    seed: int = 0,
) -> Dict:
    """Local sheet thickness, as twice the distance to the nearest background voxel.

    A traced writing surface should be a few voxels thick and tightly
    distributed; a fat upper tail is where a label has swallowed the gap between
    two windings.

    ``max_thickness`` bounds the answer and buys a great deal of speed.  An exact
    Euclidean distance transform over a 320³ volume costs about 90 seconds, which
    would be the entire cost of auditing a release; peeling the mask with
    successive erosions instead costs a couple of seconds and saturates cleanly
    beyond the cap.  Since the question being asked is "is this label a sheet", a
    measurement that stops caring above 24 voxels loses nothing.  Pass
    ``max_thickness=None`` for the exact transform.

    The erosion ladder measures city-block depth rather than Euclidean distance.
    Across a thin sheet the two agree, because the nearest background voxel lies
    along the surface normal; they diverge only for bulky regions, where the
    number is saturated and uninformative anyway.  ``saturated`` reports the
    share of voxels that hit the cap.
    """
    if not mask.any():
        return {"median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0, "saturated": 0.0}

    if max_thickness is None:
        distance = ndi.distance_transform_edt(mask)
        values = 2.0 * distance[mask]
        saturated = 0.0
    else:
        # depth[v] = how many erosions v survives; 1 for a surface voxel
        depth = np.zeros(mask.shape, dtype=np.uint8)
        current = mask
        for step in range(1, int(max_thickness) + 1):
            depth[current] = step
            # border_value=1 keeps the volume's own faces from acting as
            # background, which is what distance_transform_edt does too — a sheet
            # running out of the patch is not thinner for it
            current = ndi.binary_erosion(current, _BALL, border_value=1)
            if not current.any():
                break
        values = 2.0 * depth[mask].astype(np.float32)
        saturated = float((depth[mask] >= max_thickness).mean())

    if values.size > sample:
        rng = np.random.default_rng(seed)
        values = rng.choice(values, size=sample, replace=False)
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "saturated": saturated,
    }


def component_stats(
    mask: np.ndarray,
    min_size: int = 64,
    max_components: int = 40,
    pca_sample: int = 200_000,
    seed: int = 0,
) -> Dict:
    """Connected-component census, plus how sheet-like the biggest pieces are.

    ``planarity`` is the smallest PCA eigenvalue of a component's voxel cloud as
    a share of the total — 0 for a perfect plane, 1/3 for an isotropic blob.
    A traced papyrus sheet should sit near 0.
    """
    if not mask.any():
        return {
            "n_components": 0,
            "n_components_ge_min_size": 0,
            "largest_component_fraction": 0.0,
            "fragment_fraction": 0.0,
            "median_planarity": None,
            "worst_planarity": None,
        }

    labelled, n = ndi.label(mask, structure=_CONNECTIVITY)
    sizes = np.bincount(labelled.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    big = [int(i) for i in order if sizes[i] >= min_size]

    rng = np.random.default_rng(seed)
    planarities = []
    for comp_index in big[:max_components]:
        component = labelled == comp_index + 1
        n_voxels = int(sizes[comp_index])
        if n_voxels > pca_sample:
            # planarity is a shape statistic; a uniform subsample estimates it
            # just as well and keeps a 24-million-voxel component off the heap
            flat = np.flatnonzero(component.ravel())
            flat = rng.choice(flat, pca_sample, replace=False)
            coords = np.stack(np.unravel_index(flat, component.shape), axis=1).astype(
                np.float32
            )
        else:
            coords = np.argwhere(component).astype(np.float32)
        if coords.shape[0] < 8:
            continue
        coords -= coords.mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(np.cov(coords, rowvar=False))
        total = float(eigenvalues.sum())
        if total > 0:
            planarities.append(float(max(eigenvalues.min(), 0.0) / total))

    return {
        "n_components": int(n),
        "n_components_ge_min_size": len(big),
        "largest_component_fraction": float(sizes.max() / sizes.sum()),
        "fragment_fraction": float(sizes[sizes < min_size].sum() / sizes.sum()),
        "median_planarity": float(np.median(planarities)) if planarities else None,
        "worst_planarity": float(max(planarities)) if planarities else None,
    }


def junction_density(
    mask: np.ndarray, radius: int = 5, n_samples: int = 4000, seed: int = 0
) -> Dict:
    """Fraction of the label that sits at a junction — where the sheet forks.

    A correct writing-surface label is locally a single sheet: inside a small
    ball around any of its voxels, the label is one disc, and it meets the ball's
    shell in exactly one connected band.  Where a label has bridged two windings,
    a third arm leaves the ball and the shell intersection breaks into two or
    more pieces.

    Only the piece of the label *connected to the centre voxel inside the ball*
    is considered, so a neighbouring winding that merely passes nearby is not
    mistaken for a fork — that distinction is the whole point of the measure.

    This is the error the Open Problems post singles out as unrecoverable:
    "a small local error can send a traced mesh onto the wrong wrap entirely,
    with no easy way to recover".
    """
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return {"n_samples": 0, "junction_fraction": 0.0, "radius": radius}

    rng = np.random.default_rng(seed)
    if coords.shape[0] > n_samples:
        coords = coords[rng.choice(coords.shape[0], n_samples, replace=False)]

    r = int(radius)
    grid = np.arange(-r, r + 1)
    dz, dy, dx = np.meshgrid(grid, grid, grid, indexing="ij")
    distance = np.sqrt(dz**2 + dy**2 + dx**2)
    ball = distance <= r + 0.5
    shell = (distance >= r - 0.5) & ball
    centre = (r, r, r)

    padded = np.pad(mask, r, mode="constant")
    junctions = 0
    counted = 0
    for z, y, x in coords:
        window = padded[z : z + 2 * r + 1, y : y + 2 * r + 1, x : x + 2 * r + 1] & ball
        blobs, n = ndi.label(window, structure=_CONNECTIVITY)
        home = blobs[centre]
        if home == 0:
            continue
        arms, n_arms = ndi.label((blobs == home) & shell, structure=_CONNECTIVITY)
        counted += 1
        if n_arms >= 2:
            junctions += 1
    return {
        "n_samples": counted,
        "junction_fraction": float(junctions / counted) if counted else 0.0,
        "junctions": junctions,
        "radius": r,
    }


def audit_label(label: np.ndarray, deep: bool = False, max_classes: int = 4) -> Dict:
    """Run every label-only metric on one volume, per class.

    Vesuvius surface labels are not binary: the releases seen so far carry a thin
    writing-surface class alongside a bulky region class (air, or the scroll
    body).  Measuring sheet thickness over ``label > 0`` would average those two
    together into a number that means nothing, so every metric is computed per
    class and the sheet-like class is identified rather than assumed.
    """
    result: Dict = {"shape": list(label.shape), "dtype": str(label.dtype)}
    result.update(label_scheme(label))

    classes = [v for v in result["values"] if v != 0][:max_classes]
    per_class: Dict = {}
    for value in classes:
        mask = label == value
        entry = {
            "fraction": float(mask.mean()),
            "thickness": thickness_stats(mask),
            "components": component_stats(mask),
            "border_touch_fraction": _border_touch_fraction(mask),
        }
        if deep:
            entry["junctions"] = junction_density(mask)
        per_class[value] = entry
    result["per_class"] = per_class

    # the sheet-like class is the thin one: least median thickness, and not
    # occupying most of the volume
    candidates = [
        (v, e) for v, e in per_class.items() if e["fraction"] > 1e-5 and e["fraction"] < 0.5
    ]
    if candidates:
        surface_value, surface = min(candidates, key=lambda kv: kv[1]["thickness"]["median"])
        result["surface_class"] = surface_value
        result["surface_fraction"] = surface["fraction"]
        result["surface_thickness_median"] = surface["thickness"]["median"]
        result["surface_thickness_p95"] = surface["thickness"]["p95"]
        result["surface_components"] = surface["components"]["n_components"]
        result["surface_fragment_fraction"] = surface["components"]["fragment_fraction"]
        result["surface_worst_planarity"] = surface["components"]["worst_planarity"]
        if deep and surface.get("junctions"):
            result["surface_junction_fraction"] = surface["junctions"]["junction_fraction"]
    else:
        result["surface_class"] = None
    return result
