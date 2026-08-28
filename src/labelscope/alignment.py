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
``local_contrast``    a label-free difficulty proxy for the patch, standing in
                      for the "compressed region" haze the post describes.

Put together they answer a question nobody currently measures: *are the labels
worse exactly where the scroll is hardest?*
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

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
    patch = coords[idx].astype(np.float32)               # (N, k, 3)
    patch -= patch.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", patch, patch) / max(1, k - 1)
    _, vectors = np.linalg.eigh(cov)                     # ascending eigenvalues
    normals = vectors[:, :, 0]                           # least-extent direction
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    return (normals / norm).T.astype(np.float32)         # (3, N)


def orient_normals(
    normals: np.ndarray, points: np.ndarray, reference: np.ndarray, sigma: float = 3.0
) -> np.ndarray:
    """Flip normals so they point up the gradient of ``reference``.

    ``reference`` is any field that says which way is "outward" — in these
    datasets the air/void class next to the papyrus does the job.  Without it
    the offset sign is arbitrary per voxel and only ``|offset|`` is meaningful.
    """
    field = ndi.gaussian_filter(reference.astype(np.float32), sigma)
    grad = np.stack(np.gradient(field), axis=0)
    sampled = np.stack([
        ndi.map_coordinates(grad[a], points, order=1, mode="nearest") for a in range(3)
    ])
    sign = np.sign((normals * sampled).sum(axis=0))
    sign[sign == 0] = 1.0
    return normals * sign


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


def noise_sigma(volume: np.ndarray, sample: int = 500_000, seed: int = 0) -> float:
    """Robust estimate of the volume's own voxel noise.

    Prominence has to be judged against this, not against the patch's dynamic
    range: a hazy, compressed patch has a *small* dynamic range, so normalising
    by it would hide exactly the degradation we are trying to detect.
    """
    residual = volume.astype(np.float32) - ndi.gaussian_filter(volume.astype(np.float32), 1.0)
    flat = residual.ravel()
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
    shift = np.where(np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / np.where(denom == 0, 1, denom), 0.0)
    return idx + np.clip(shift, -1.0, 1.0)


def ridge_alignment(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 6,
    step: float = 0.25,
    n_samples: int = 20_000,
    polarity: Optional[str] = None,
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
    normals = point_normals(all_coords, sample_points)             # (3, N)
    if orient_field is not None:
        normals = orient_normals(normals, sample_points.T, orient_field)

    offsets = np.arange(-radius, radius + step / 2, step, dtype=np.float32)
    base = sample_points.T[:, None, :]                             # (3, 1, N)
    walk = base + normals[:, None, :] * offsets[None, :, None]     # (3, T, N)
    profile = _sample_at(image, walk.reshape(3, -1)).reshape(offsets.size, -1)

    if polarity is None:
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

    interior = np.abs(signed_offset) < (radius - step)   # peak not pinned to the window edge
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
    """A label-free proxy for how separable the papyrus layers are here.

    Compressed regions are described in the Open Problems post as blurred and
    foggy — layer boundaries smear into haze.  High-frequency energy relative to
    the volume's own dynamic range captures that: sharp, well-separated windings
    score high, haze scores low.
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
        return {"surface_class": None, "surface_voxels": 0,
                "error": "no sheet-like class found in this label"}

    mask = label == surface_class
    orient_field = (label == orient_class).astype(np.float32) if orient_class is not None else None
    result = {
        "surface_class": surface_class,
        "orient_class": orient_class,
        "surface_voxels": int(mask.sum()),
        "surface_fraction": float(mask.mean()),
    }
    result.update(local_contrast(image))
    result.update(ridge_alignment(image, mask, orient_field=orient_field, **kwargs))
    return result
