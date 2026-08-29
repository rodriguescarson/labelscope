"""Does the label sit where the CT says the surface is?

The Vesuvius Challenge Open Problems post describes hand-made surface labels as
"approximate — they may wiggle, they may drift slightly off the true surface,
they may avoid the most ambiguous regions", and names label quality as one of
the main unwrapping bottlenecks.  Those are three measurable claims:

``ridge_offset``      how far the label sits from the CT's own sheet ridge, along
                      the surface normal, in voxels.  The *signed mean* is the
                      interesting one: a non-zero mean is systematic bias, not
                      annotator noise.
``ridge_prominence``  how strong a ridge the label is sitting on at all.  Label
                      riding flat CT is label with nothing underneath it.
``layer_separability`` how strongly the papyrus layers stand out from the voxel
                      noise here — the measurable form of the "compressed
                      region" problem the post describes.

Put together they answer a question nobody currently measures: *are the labels
worse exactly where the scroll is hardest?*
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------------- #
# surface normals
# --------------------------------------------------------------------------- #
def point_normals(coords: np.ndarray, neighbours: np.ndarray, k: int = 40) -> np.ndarray:
    """Surface normals at ``neighbours`` by local PCA over the label point cloud.

    A thin open surface has no inside, so the gradient of a distance field is
    degenerate on it — the field has a *minimum* at the sheet, not a zero
    crossing, and its gradient there is zero.  Local PCA has no such problem:
    fit a plane to the k nearest labelled voxels and take the direction of least
    extent.  The result is an unsigned axis; ``orient_normals`` fixes the sign.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    k = int(min(k, coords.shape[0]))
    _, idx = tree.query(neighbours, k=k, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    patch = coords[idx].astype(np.float32)  # (N, k, 3)
    patch -= patch.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", patch, patch) / max(1, k - 1)
    _, vectors = np.linalg.eigh(cov)  # ascending eigenvalues
    normals = vectors[:, :, 0]  # least-extent direction
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    return (normals / norm).T.astype(np.float32)  # (3, N)


def orient_normals(
    normals: np.ndarray,
    points: np.ndarray,
    reference: np.ndarray,
    sigma: float = 3.0,
    components: Optional[np.ndarray] = None,
    pointwise: bool = False,
) -> np.ndarray:
    """Fix the *global* sign of an already-consistent normal field.

    ``reference`` is any field that says which way is outward — the air/void
    class beside the papyrus does the job.  The decision is taken by majority
    over each already-consistent ``components`` piece, or once over everything
    when no components are given: the field is reliable in aggregate even where
    it is useless voxel by voxel.  ``pointwise=True`` restores per-voxel
    flipping, which is only appropriate for a surface whose normals are not
    already consistent and whose reference is close by.
    """
    field = ndi.gaussian_filter(reference.astype(np.float32), sigma)
    grad = np.stack(np.gradient(field), axis=0)
    sampled = np.stack(
        [ndi.map_coordinates(grad[a], points, order=1, mode="nearest") for a in range(3)]
    )
    dots = (normals * sampled).sum(axis=0)
    if pointwise:
        sign = np.sign(dots)
        sign[sign == 0] = 1.0
        return normals * sign
    if components is None:
        return normals if float(dots.sum()) >= 0 else -normals

    oriented = normals.copy()
    for piece in np.unique(components):
        member = components == piece
        if float(dots[member].sum()) < 0:
            oriented[:, member] *= -1.0
    return oriented


def _knn_edges(points: np.ndarray, normals: np.ndarray, k: int):
    from scipy.spatial import cKDTree

    n = points.shape[0]
    tree = cKDTree(points)
    k = int(min(k + 1, n))
    _, idx = tree.query(points, k=k, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    cols = idx[:, 1:].ravel()
    return tree, rows, cols


def _components_of(n: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def propagate_orientation(
    points: np.ndarray, normals: np.ndarray, k: int = 10, max_bridges: int = 200
):
    """Give an unsigned normal field a consistent sign across the surface.

    Local PCA returns an *axis*, not a direction, so half the normals on a sheet
    can come back pointing the wrong way.  Orienting each one independently
    against some distant reference field does not fix it: deep inside a scroll
    the reference is far away, its smoothed gradient is numerically zero, and the
    sign it hands back is noise.  Measured on the Kaggle release, normals inside
    a 64-voxel cell agreed with their own cell mean only 54-64% of the time —
    barely better than a coin.  Profiles averaged over such a cell are half
    mirrored, the average symmetrises, and every offset is pulled toward zero.

    The fix is the standard one for unoriented point clouds (Hoppe et al. 1992):
    build a neighbourhood graph weighted so that strongly-aligned neighbours are
    cheap to traverse, take its minimum spanning tree, and walk the tree flipping
    each normal to agree with its parent.

    Concentric wraps sit far enough apart that a k-nearest-neighbour graph does
    not connect them, so each wrap would otherwise keep its own arbitrary sign.
    They are bridged here by one edge per component to its nearest point
    elsewhere: adjacent wraps are near-parallel, so that edge is cheap and
    carries the orientation across correctly.  Only the single remaining global
    sign is left for :func:`orient_normals` to settle.

    Returns ``(oriented_normals, component_labels)``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

    n = points.shape[0]
    if n < 3:
        return normals, np.zeros(n, dtype=np.int32)

    tree, rows, cols = _knn_edges(points, normals, k)
    labels = _components_of(n, rows, cols)

    # Bridge the wraps.  A k-nearest-neighbour probe cannot find them: inside a
    # sheet the neighbours are one voxel away and the next wrap is a dozen or
    # more, so every candidate in the probe is a same-sheet point.  Query a tree
    # built on everything *outside* the component instead.
    from scipy.spatial import cKDTree

    bridges_r, bridges_c = [], []
    unique = np.unique(labels)
    rng = np.random.default_rng(0)
    if unique.size > 1:
        for piece in unique[:max_bridges]:
            member = np.flatnonzero(labels == piece)
            outside = np.flatnonzero(labels != piece)
            if outside.size == 0:
                continue
            probe_points = member
            if probe_points.size > 2000:
                probe_points = rng.choice(member, 2000, replace=False)
            distances, nearest = cKDTree(points[outside]).query(
                points[probe_points], k=1, workers=-1
            )
            best = int(np.argmin(distances))
            bridges_r.append(int(probe_points[best]))
            bridges_c.append(int(outside[nearest[best]]))
    if bridges_r:
        rows = np.concatenate([rows, np.asarray(bridges_r)])
        cols = np.concatenate([cols, np.asarray(bridges_c)])

    alignment = np.abs((normals[:, rows] * normals[:, cols]).sum(axis=0))
    weights = 1.0 - alignment + 1e-6
    graph = coo_matrix((weights, (rows, cols)), shape=(n, n))
    mst = minimum_spanning_tree(graph)
    mst = mst + mst.T

    oriented = normals.copy()
    components = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    label = 0
    for seed in range(n):
        if visited[seed]:
            continue
        order, predecessors = breadth_first_order(
            mst, seed, directed=False, return_predecessors=True
        )
        for node in order:
            visited[node] = True
            components[node] = label
            parent = predecessors[node]
            if parent < 0:
                continue
            if float(oriented[:, node] @ oriented[:, parent]) < 0:
                oriented[:, node] *= -1.0
        label += 1
        if visited.all():
            break
    return oriented, components


def surface_normals(mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Dense normal field, as a (3, Z, Y, X) array.

    Provided for convenience and for tests; ``ridge_alignment`` uses the far
    cheaper point-wise estimator, since only the sampled voxels need normals.
    """
    coords = np.argwhere(mask)
    out = np.zeros((3,) + mask.shape, dtype=np.float32)
    if coords.shape[0] == 0:
        return out
    normals = point_normals(coords, coords.astype(np.float32))
    out[:, coords[:, 0], coords[:, 1], coords[:, 2]] = normals
    return out


def noise_sigma(
    volume: np.ndarray, sample: int = 500_000, seed: int = 0, void_fraction: float = 0.2
) -> float:
    """Robust estimate of the volume's own voxel noise.

    Prominence has to be judged against this, not against the patch's dynamic
    range: a hazy, compressed patch has a *small* dynamic range, so normalising
    by it would hide exactly the degradation we are trying to detect.

    Voxels that are identically zero are excluded when there are enough of them
    to matter.  51 patches of the Kaggle surface release are masked crops, 53% to
    67% exact zeros, and a median absolute deviation taken over those is simply
    zero — which sends the reported signal-to-noise to 4e7 and floats a patch
    straight through the reliability gate.  A robust estimator is only robust
    while the thing it is estimating is the majority of its input.
    """
    volume = volume.astype(np.float32)
    residual = volume - ndi.gaussian_filter(volume, 1.0)
    void = volume == 0
    if void.mean() > void_fraction:
        residual = residual[~void]
    flat = residual.ravel()
    if flat.size == 0:
        return 1e-6
    if flat.size > sample:
        flat = flat[np.random.default_rng(seed).choice(flat.size, sample, replace=False)]
    return float(max(1.4826 * np.median(np.abs(flat - np.median(flat))), 1e-6))


def _sample_at(volume: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Trilinear sample of ``volume`` at float coordinates (3, N)."""
    return ndi.map_coordinates(volume, points, order=1, mode="nearest")


def _subvoxel_peak(profile: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Parabolic refinement of an integer argmax along axis 0."""
    n = profile.shape[0]
    idx = np.clip(index, 1, n - 2)
    cols = np.arange(profile.shape[1])
    y0 = profile[idx - 1, cols]
    y1 = profile[idx, cols]
    y2 = profile[idx + 1, cols]
    denom = y0 - 2.0 * y1 + y2
    shift = np.where(
        np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / np.where(denom == 0, 1, denom), 0.0
    )
    return idx + np.clip(shift, -1.0, 1.0)


def ridge_alignment(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 6,
    step: float = 0.25,
    n_samples: int = 20_000,
    polarity: str = "bright",
    orient_field: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict:
    """Offset from each labelled surface voxel to the nearest CT ridge.

    For every sampled label voxel we walk along the surface normal, read the CT
    with trilinear interpolation, and locate the intensity extremum.  The signed
    distance from the label to that extremum is the offset, in voxels, positive
    along the outward normal.

    ``polarity`` is ``"bright"`` when papyrus reads brighter than the gaps
    (the usual case for these scans) and ``"dark"`` otherwise.  Left as ``None``
    it is detected from the data.
    """
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return {"n_samples": 0}
    rng = np.random.default_rng(seed)
    if coords.shape[0] > n_samples:
        coords = coords[rng.choice(coords.shape[0], n_samples, replace=False)]

    image = image.astype(np.float32)
    all_coords = np.argwhere(mask)
    sample_points = coords.astype(np.float32)
    normals = point_normals(all_coords, sample_points)  # (3, N)
    normals, components = propagate_orientation(sample_points, normals)
    if orient_field is not None:
        normals = orient_normals(normals, sample_points.T, orient_field, components=components)

    offsets = np.arange(-radius, radius + step / 2, step, dtype=np.float32)
    base = sample_points.T[:, None, :]  # (3, 1, N)
    walk = base + normals[:, None, :] * offsets[None, :, None]  # (3, T, N)
    profile = _sample_at(image, walk.reshape(3, -1)).reshape(offsets.size, -1)

    if polarity in (None, "auto"):
        centre = profile[offsets.size // 2]
        edges = np.concatenate([profile[:2], profile[-2:]]).mean(axis=0)
        polarity = "bright" if float(centre.mean()) >= float(edges.mean()) else "dark"
    if polarity == "dark":
        profile = -profile

    smooth = ndi.gaussian_filter1d(profile, 1.0 / step * 0.5, axis=0)
    peak_index = np.argmax(smooth, axis=0)
    refined = _subvoxel_peak(smooth, peak_index)
    signed_offset = (refined - (offsets.size - 1) / 2.0) * step

    peak_value = smooth.max(axis=0)
    flanks = np.minimum(smooth[0], smooth[-1])
    prominence = peak_value - flanks
    spread = smooth.max(axis=0) - smooth.min(axis=0)
    dynamic_range = float(np.percentile(image, 99) - np.percentile(image, 1)) or 1.0
    sigma_noise = noise_sigma(image)
    snr = prominence / sigma_noise

    interior = np.abs(signed_offset) < (radius - step)  # peak not pinned to the window edge
    valid = signed_offset[interior]
    if valid.size == 0:
        valid = signed_offset

    return {
        "n_samples": int(coords.shape[0]),
        "polarity": polarity,
        "oriented": orient_field is not None,
        "mean_signed_offset": float(valid.mean()),
        "median_abs_offset": float(np.median(np.abs(valid))),
        "p90_abs_offset": float(np.percentile(np.abs(valid), 90)),
        "frac_offset_ge_1vx": float((np.abs(valid) >= 1.0).mean()),
        "frac_offset_ge_2vx": float((np.abs(valid) >= 2.0).mean()),
        "frac_peak_unresolved": float(1.0 - interior.mean()),
        "median_prominence": float(np.median(prominence)),
        "median_prominence_norm": float(np.median(prominence) / dynamic_range),
        "noise_sigma": sigma_noise,
        "median_prominence_snr": float(np.median(snr)),
        "frac_flat_support": float((snr < 3.0).mean()),
        "median_profile_spread": float(np.median(spread)),
    }


# --------------------------------------------------------------------------- #
# label-free difficulty
# --------------------------------------------------------------------------- #
def local_contrast(image: np.ndarray, sigma: float = 2.0, sample: int = 2_000_000) -> Dict:
    """Raw high-frequency energy in the volume.

    .. warning::
       This is **not** a difficulty proxy, though it is tempting to use as one.
       High-frequency energy counts voxel noise as readily as it counts sheet
       structure, so a noisy patch scores *high* while being harder, not easier.
       Measured across 51 patches of the Kaggle surface release, this quantity
       correlates **negatively** with actual layer separability
       (Spearman -0.42).  Use ``global_profile_snr`` from
       :func:`aggregate_alignment` for separability, and the winding spacing for
       compression.  This is kept because it is cheap, label-free, and useful as
       a raw descriptor — not as a verdict.
    """
    image = image.astype(np.float32)
    blurred = ndi.gaussian_filter(image, sigma)
    residual = np.abs(image - blurred)
    if residual.size > sample:
        rng = np.random.default_rng(0)
        flat = residual.ravel()
        residual = flat[rng.choice(flat.size, sample, replace=False)]
    dynamic_range = float(np.percentile(image, 99) - np.percentile(image, 1)) or 1.0
    return {
        "hf_energy": float(residual.mean()),
        "hf_energy_norm": float(residual.mean() / dynamic_range),
        "dynamic_range": dynamic_range,
        "intensity_p50": float(np.percentile(image, 50)),
    }


def audit_alignment(
    image: np.ndarray,
    label: np.ndarray,
    surface_class: Optional[int] = None,
    orient_class: Optional[int] = None,
    naive: bool = True,
    **kwargs,
) -> Dict:
    """Alignment metrics for one image/label pair.

    ``surface_class`` defaults to whichever class is thin, planar and does not
    fill the volume — see :func:`labelscope.quality.audit_label`.  ``orient_class``
    is the bulky region class used to decide which way is outward, so that the
    *sign* of the offset means something; it defaults to the largest remaining
    class, and when there is none only ``|offset|`` is reported.
    """
    from labelscope.quality import audit_label

    if surface_class is None or orient_class is None:
        scheme = audit_label(label)
        if surface_class is None:
            surface_class = scheme.get("surface_class")
        if orient_class is None:
            others = {
                v: e["fraction"]
                for v, e in scheme.get("per_class", {}).items()
                if v != surface_class
            }
            orient_class = max(others, key=others.get) if others else None

    if surface_class is None:
        return {
            "surface_class": None,
            "surface_voxels": 0,
            "error": "no sheet-like class found in this label",
        }

    mask = label == surface_class
    orient_field = (
        (label == orient_class).astype(np.float32) if orient_class is not None else None
    )
    result = {
        "surface_class": surface_class,
        "orient_class": orient_class,
        "surface_voxels": int(mask.sum()),
        "surface_fraction": float(mask.mean()),
    }
    result.update(local_contrast(image))
    result.update(aggregate_alignment(image, mask, orient_field=orient_field, **kwargs))
    if naive:
        # kept for comparison only: on scroll CT this is radius-dependent and
        # does not measure displacement.  See aggregate_alignment's docstring.
        raw = ridge_alignment(
            image, mask, orient_field=orient_field, n_samples=kwargs.get("n_samples", 20_000)
        )
        result.update(
            {
                f"naive_{k}": v
                for k, v in raw.items()
                if k
                in (
                    "median_abs_offset",
                    "mean_signed_offset",
                    "frac_offset_ge_1vx",
                    "frac_flat_support",
                    "median_prominence_snr",
                )
            }
        )
    return result


# --------------------------------------------------------------------------- #
# aggregated alignment — the estimator that actually works on scroll CT
# --------------------------------------------------------------------------- #
def orientation_reference_quality(profiles: np.ndarray, offsets: np.ndarray) -> float:
    """How much a candidate "outward" field actually distinguishes the two sides.

    Returned on 0-1 as the normalised difference between the field's mean on the
    positive side of the surface and on the negative side.  Zero means the field
    is symmetric about the surface and cannot tell inside from outside; one means
    it is entirely on one side.

    This matters because the void class shipped with the Kaggle surface release
    scores near zero — it wraps the writing surface on *both* sides — so offsets
    oriented against it carry a sign that flips arbitrarily from patch to patch.
    """
    if profiles.size == 0:
        return 0.0
    mean = profiles.mean(axis=1)
    baseline = float(mean.min())
    positive = float(mean[offsets > 0].mean() - baseline) if (offsets > 0).any() else 0.0
    negative = float(mean[offsets < 0].mean() - baseline) if (offsets < 0).any() else 0.0
    denominator = abs(positive) + abs(negative)
    if denominator < 1e-9:
        return 0.0
    return float(abs(positive - negative) / denominator)


def orient_by_intensity(
    image: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    components: Optional[np.ndarray] = None,
    distance: float = 3.0,
) -> np.ndarray:
    """Point every normal toward the denser side of the sheet.

    The releases ship no field that says which way is out — the void class is
    symmetric about the writing surface, so orienting against it gives a sign
    that flips arbitrarily from patch to patch.  The scan itself does not have
    that problem: a writing surface has papyrus on one side and a gap on the
    other, so "toward the brighter side" is a convention the data defines and
    every patch can agree on.  Offsets then read consistently as displacement
    toward, or away from, the body of the sheet.
    """
    image = image.astype(np.float32)
    forward = _sample_at(image, points + normals * distance)
    backward = _sample_at(image, points - normals * distance)
    delta = forward - backward
    oriented = normals.copy()
    if components is None:
        return oriented if float(delta.sum()) >= 0 else -oriented
    for piece in np.unique(components):
        member = components == piece
        if float(delta[member].sum()) < 0:
            oriented[:, member] *= -1.0
    return oriented


def sample_profiles(
    image: np.ndarray,
    mask: np.ndarray,
    radius: float = 8.0,
    step: float = 0.25,
    n_samples: int = 20_000,
    orient_field: Optional[np.ndarray] = None,
    polarity: str = "bright",
    seed: int = 0,
):
    """Intensity profiles along the surface normal at sampled label voxels.

    Returns ``(coords, offsets, profiles, polarity)`` where ``profiles`` has shape
    ``(len(offsets), len(coords))``.
    """
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return coords, np.zeros(0), np.zeros((0, 0)), polarity or "bright"

    rng = np.random.default_rng(seed)
    if coords.shape[0] > n_samples:
        coords = coords[rng.choice(coords.shape[0], n_samples, replace=False)]

    image = image.astype(np.float32)
    all_coords = np.argwhere(mask)
    points = coords.astype(np.float32)
    normals = point_normals(all_coords, points)
    normals, components = propagate_orientation(points, normals)
    if isinstance(orient_field, str):
        pass  # explicitly unoriented
    elif orient_field is not None:
        normals = orient_normals(normals, points.T, orient_field, components=components)
    else:
        # nothing external to anchor to: let the scan decide, so that positive
        # always means "toward the body of the sheet"
        normals = orient_by_intensity(image, points.T, normals, components=components)

    offsets = np.arange(-radius, radius + step / 2, step, dtype=np.float32)
    walk = points.T[:, None, :] + normals[:, None, :] * offsets[None, :, None]

    # Sampling outside the volume clamps to the border value, which invents a
    # flat plateau — and a plateau near the window edge can outrank the real
    # ridge.  Keep only the samples whose whole walk stays inside.
    inside = np.ones(points.shape[0], dtype=bool)
    for axis in range(3):
        inside &= (walk[axis] >= 0).all(axis=0)
        inside &= (walk[axis] <= image.shape[axis] - 1).all(axis=0)
    if not inside.any():
        return (
            coords[:0],
            offsets,
            np.zeros((offsets.size, 0), np.float32),
            polarity or "bright",
        )
    coords = coords[inside]
    walk = walk[:, :, inside]

    profiles = _sample_at(image, walk.reshape(3, -1)).reshape(offsets.size, -1)

    if polarity in (None, "auto"):
        # Inferring polarity from the profile assumes the label already sits on
        # the feature — which is exactly what is being measured.  A label a few
        # voxels off sits in the gap between wraps, reads as darker than the
        # window edges, and flips the sense of the whole measurement.  So this
        # is available but never the default: polarity is a property of the
        # modality, and in scroll micro-CT papyrus is denser, hence brighter,
        # than the air between wraps.
        centre = profiles[offsets.size // 2]
        edges = np.concatenate([profiles[:2], profiles[-2:]]).mean(axis=0)
        polarity = "bright" if float(centre.mean()) >= float(edges.mean()) else "dark"
    if polarity == "dark":
        profiles = -profiles
    return coords, offsets, profiles, polarity


def _peak_of(
    profile: np.ndarray, offsets: np.ndarray, smooth: float = 0.5, floor: float = 0.6
) -> float:
    """Sub-voxel location of the profile ridge **nearest the label**.

    Not the tallest ridge: once the search window is wide enough to hold more
    than one wrap, the tallest one may belong to the neighbouring sheet, and
    attributing the label to that sheet turns a 5-voxel displacement into a
    9-voxel one in the opposite direction.  The question being asked is how far
    the label sits from the sheet it is on, so among ridges rising at least
    ``floor`` of the way from the profile's minimum to its maximum, the nearest
    one wins.
    """
    step = float(offsets[1] - offsets[0])
    curve = ndi.gaussian_filter1d(profile, max(smooth / step, 0.5))
    low, high = float(curve.min()), float(curve.max())
    threshold = low + floor * (high - low)
    peaks = [
        i
        for i in range(1, curve.size - 1)
        if curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1] and curve[i] >= threshold
    ]
    if not peaks:
        peaks = [int(np.argmax(curve))]
    i = min(peaks, key=lambda k: abs(float(offsets[k])))
    if 0 < i < curve.size - 1:
        y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
        denom = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
        i = i + float(np.clip(shift, -1.0, 1.0))
    return float(offsets[0] + i * step)


def neighbour_ridge_distance(
    image: np.ndarray,
    mask: np.ndarray,
    orient_field: Optional[np.ndarray] = None,
    max_radius: float = 45.0,
    step: float = 0.5,
    n_samples: int = 12_000,
    polarity: str = "bright",
    seed: int = 0,
) -> Dict:
    """How far the next papyrus wrap sits from the labelled one, in voxels.

    Averaging the normal-direction profiles over the whole labelled surface and
    looking for the nearest secondary maximum measures the local winding spacing
    directly from the scan.  It matters because it bounds what any alignment
    search can honestly do: hunt for a ridge further than half that distance and
    the estimator can lock onto the neighbouring wrap, which is not a
    displacement of the label but a different sheet entirely.

    Measured on the Kaggle surface release, the nearest neighbouring ridge sits
    12.5-31 voxels away and varies from patch to patch, so this is worth
    measuring per patch rather than assuming.
    """
    # a window wider than the volume can support leaves nothing to average
    max_radius = float(min(max_radius, 0.4 * min(image.shape)))
    _, offsets, profiles, _ = sample_profiles(
        image,
        mask,
        radius=max_radius,
        step=step,
        n_samples=n_samples,
        orient_field=orient_field,
        polarity=polarity,
        seed=seed,
    )
    if profiles.size == 0:
        return {
            "neighbour_ridge_pos": None,
            "neighbour_ridge_neg": None,
            "winding_spacing": None,
            "recommended_radius": 6.0,
        }

    curve = ndi.gaussian_filter1d(profiles.mean(axis=1), max(2.0 / step, 1.0))
    peaks = [
        i
        for i in range(2, curve.size - 2)
        if curve[i] > curve[i - 1] and curve[i] > curve[i + 1]
    ]
    if not peaks:
        return {
            "neighbour_ridge_pos": None,
            "neighbour_ridge_neg": None,
            "winding_spacing": None,
            "recommended_radius": 6.0,
            "dominant_ridge_at": None,
        }

    # spacing is measured from the sheet the label is actually on, which is the
    # dominant ridge — not from the label's own position.  A displaced label
    # would otherwise make its own sheet look like the neighbouring wrap.
    # the sheet the label belongs to is the strongest ridge; among near-equal
    # ridges prefer the one closest to the label, since a wrap it does not
    # belong to is not "its" sheet
    strongest = max(curve[i] for i in peaks)
    floor = curve.min() + 0.75 * (strongest - curve.min())
    dominant = min(
        (i for i in peaks if curve[i] >= floor), key=lambda i: abs(float(offsets[i]))
    )
    centre = float(offsets[dominant])
    guard = 2.0
    positions = [float(offsets[i]) - centre for i in peaks]
    forward = [x for x in positions if x > guard]
    backward = [x for x in positions if x < -guard]
    nearest_pos = min(forward) if forward else None
    nearest_neg = max(backward) if backward else None

    candidates = [abs(x) for x in (nearest_pos, nearest_neg) if x is not None]
    spacing = min(candidates) if candidates else None
    # the window must reach the dominant ridge wherever it sits, and stop short
    # of the next wrap beyond it
    recommended = float(np.clip(abs(centre) + 0.45 * spacing, 3.0, 16.0)) if spacing else 6.0
    return {
        "neighbour_ridge_pos": nearest_pos,
        "neighbour_ridge_neg": nearest_neg,
        "winding_spacing": spacing,
        "dominant_ridge_at": centre,
        "recommended_radius": recommended,
    }


def aggregate_alignment(
    image: np.ndarray,
    mask: np.ndarray,
    cell: int = 64,
    min_per_cell: int = 200,
    radius="auto",
    step: float = 0.25,
    n_samples: int = 20_000,
    orient_field: Optional[np.ndarray] = None,
    orient_by: str = "intensity",
    bootstrap: int = 200,
    min_snr: float = 3.0,
    min_global_snr: float = 2.0,
    polarity: str = "bright",
    seed: int = 0,
    return_cells: bool = False,
) -> Dict:
    """Label-to-ridge offset, measured where the measurement has power.

    **Why not simply take each voxel's nearest intensity maximum?**  Because on
    carbonised papyrus that number is meaningless, and measurably so: sweeping
    the search radius over real scroll CT, the median per-voxel ``|offset|``
    tracks the radius almost linearly (0.70 vx at R=2, 1.15 at R=3, 1.53 at R=4,
    2.18 at R=6, 3.09 at R=9).  A genuine displacement would plateau once the
    window exceeded it.  It does not, because along any one normal there are
    fibre maxima, both faces of the sheet, and — at a winding period of roughly
    14 voxels here — the neighbouring wrap.  A per-voxel argmax picks among them
    essentially at random.

    Averaging the profiles over a neighbourhood first fixes this: the incoherent
    maxima cancel, the sheet reinforces.  On the same data the mean profile over
    6,000 labelled voxels is clean, single-peaked, and centred at +0.00 voxels.

    So offsets are reported per ``cell``-sized cube of surface, each backed by at
    least ``min_per_cell`` voxels, plus one global figure with a bootstrap
    confidence interval.  Cells whose averaged profile has no peak inside the
    search window, or whose peak is weaker than ``min_snr``, are counted and
    excluded rather than assigned the window edge as a number — quoting the edge
    would invent a displacement the data does not contain.

    The same applies to the patch as a whole.  Where the sheet contrast at the
    labelled surface is below ``min_global_snr`` times the voxel noise, there is
    nothing to align *to*, and ``global_peak_offset`` is ``None``: on the Kaggle
    surface release, ungated, the worst patch reads +9.5 voxels, and every such
    outlier turns out to have a separability near 1 — one of them with a
    bootstrap interval of [-8.5, +8.4], the estimator flipping between two wraps.
    Gated at 2.0 the largest offset among the patches that can be measured is
    1.53 voxels.  The raw figure stays available as
    ``global_peak_offset_raw``.
    """
    if orient_by == "intensity":
        orient_field = None  # the scan orients itself
    elif orient_by == "none":
        orient_field = "unoriented"
    elif orient_by != "field":
        raise ValueError(
            f"orient_by must be 'intensity', 'field' or 'none', got {orient_by!r}"
        )
    spacing = {}
    if radius == "auto":
        spacing = neighbour_ridge_distance(
            image,
            mask,
            orient_field=orient_field,
            n_samples=min(n_samples, 12_000),
            polarity=polarity,
            seed=seed,
        )
        radius = spacing["recommended_radius"]
    radius = float(radius)

    coords, offsets, profiles, polarity = sample_profiles(
        image,
        mask,
        radius=radius,
        step=step,
        n_samples=n_samples,
        orient_field=orient_field,
        polarity=polarity,
        seed=seed,
    )
    if coords.shape[0] == 0:
        return {"n_samples": 0, "n_cells": 0}

    noise = noise_sigma(image)
    global_profile = profiles.mean(axis=1)
    global_peak = _peak_of(global_profile, offsets)
    contrast = float(global_profile.max() - global_profile.min())

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(bootstrap):
        pick = rng.integers(0, profiles.shape[1], profiles.shape[1])
        boot.append(_peak_of(profiles[:, pick].mean(axis=1), offsets))
    # bootstrap=0 is a legitimate request -- it is what a caller measuring
    # thousands of predictions asks for, since the interval costs more than the
    # estimate does -- and it used to reach np.percentile with an empty list and
    # raise IndexError from inside numpy.
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))

    keys = coords // cell
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)) + 1
    groups = np.split(order, boundaries)

    edge = radius - 2 * step
    cell_offsets, cell_snr, cell_sizes = [], [], []
    # The cell key is carried alongside the offset rather than discarded: an
    # aggregate that cannot say *where* the surface wanders can describe a
    # dataset but cannot correct one.
    cell_keys = []
    n_cells_total = n_unresolved = n_low_snr = 0
    for group in groups:
        if group.size < min_per_cell:
            continue
        key = tuple(int(v) for v in keys[group[0]])
        n_cells_total += 1
        mean_profile = profiles[:, group].mean(axis=1)
        spread = float(mean_profile.max() - mean_profile.min())
        snr = spread / (noise / np.sqrt(group.size))
        peak = _peak_of(mean_profile, offsets)
        if abs(peak) >= edge:
            # the profile keeps rising to the edge of the window: there is no
            # peak to report, and quoting the window edge would invent one
            n_unresolved += 1
            continue
        if snr < min_snr:
            n_low_snr += 1
            continue
        cell_offsets.append(peak)
        cell_snr.append(snr)
        cell_sizes.append(int(group.size))
        cell_keys.append(key)

    separability = contrast / noise if noise else 0.0
    # A noise estimate pinned at the floor means the estimate failed, not that
    # the scan is noiseless; the separability computed from it is meaningless.
    noise_failed = noise <= 1e-5
    reliable = bool(separability >= min_global_snr and not noise_failed)
    result = {
        "n_samples": int(coords.shape[0]),
        "polarity": polarity,
        "oriented": orient_field is not None,
        "search_radius": radius,
        "noise_sigma": noise,
        "global_peak_offset": global_peak if reliable else None,
        "global_peak_offset_raw": global_peak,
        "global_peak_ci95": [float(lo), float(hi)],
        "global_peak_ci95_width": float(hi - lo),
        "global_offset_reliable": reliable,
        "noise_estimate_failed": noise_failed,
        "global_profile_contrast": contrast,
        "global_profile_snr": separability,
        "n_cells": len(cell_offsets),
        "n_cells_considered": n_cells_total,
        "cell_frac_unresolved": (n_unresolved / n_cells_total) if n_cells_total else 0.0,
        "cell_frac_low_snr": (n_low_snr / n_cells_total) if n_cells_total else 0.0,
        "cell_size": cell,
    }
    result.update(spacing)
    if cell_offsets:
        values = np.asarray(cell_offsets)
        result.update(
            {
                "cell_offset_median": float(np.median(values)),
                "cell_offset_mean": float(values.mean()),
                "cell_abs_offset_median": float(np.median(np.abs(values))),
                "cell_abs_offset_p90": float(np.percentile(np.abs(values), 90)),
                "cell_frac_ge_1vx": float((np.abs(values) >= 1.0).mean()),
                "cell_frac_ge_2vx": float((np.abs(values) >= 2.0).mean()),
                "cell_offset_worst": float(values[np.argmax(np.abs(values))]),
                "cell_snr_median": float(np.median(cell_snr)),
                "cell_voxels_median": float(np.median(cell_sizes)),
            }
        )
    if return_cells:
        result["cells"] = [
            {
                "key": list(key),
                "centre_zyx": [float((k + 0.5) * cell) for k in key],
                "offset": float(offset),
                "snr": float(snr),
                "n": int(size),
            }
            for key, offset, snr, size in zip(cell_keys, cell_offsets, cell_snr, cell_sizes)
        ]
    return result
