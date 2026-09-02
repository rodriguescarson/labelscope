"""Is this traced surface actually on papyrus?

A tracer can complete normally, report a plausible area, place every vertex
inside the scan, and still produce a surface that cuts *across* the windings
instead of following a sheet. Nothing else in the toolchain catches that: the
meta looks right, renders show credible fibrous texture at every depth, and an
ink model probing the surface returns structured noise.

The check samples the scan along the surface normal over a *coherent*
neighbourhood of the grid. A surface lying on a sheet sits on a density ridge,
so its profile has real dynamic range; one cutting across sheets sees similar
material at every depth, so the profile is flat.

Two things this module is deliberate about, both learned the hard way:

* **Averaging has to be local.** Over a whole patch the winding phase varies and
  the periodicity cancels, which makes a good surface look as flat as a bad one.
* **Absolute range is not comparable across scans.** It tracks scan resolution
  and contrast as much as surface quality, so a threshold fitted on one scroll
  does not transfer. Prefer :func:`compare`, which contrasts two surfaces
  measured in the same volume -- ideally adjacent windings of one tracing run,
  which holds scan, region and provenance fixed.
"""

from __future__ import annotations

import os

import numpy as np


def block_profiles(
    mesh,
    volume,
    origin=None,
    reach: float = 70.0,
    step: float = 1.0,
    blocks: int = 6,
    block_size: int = 12,
    seed: int = 0,
):
    """Mean intensity along the normal, one profile per coherent grid block.

    Blocks are sampled at random and rejected unless at least half their
    vertices are valid, so a patch with sparse coverage yields fewer blocks
    rather than profiles built from scattered points.
    """
    from labelscope.mesh import _sampler

    sample, remote = _sampler(volume, origin)
    offsets = np.arange(-reach, reach + step / 2, step, dtype=np.float32)
    normals = mesh.normals()
    rows, cols = mesh.shape
    rng = np.random.default_rng(seed)

    out = []
    tries = 0
    while len(out) < blocks and tries < blocks * 20:
        tries += 1
        r0 = int(rng.integers(0, max(1, rows - block_size)))
        c0 = int(rng.integers(0, max(1, cols - block_size)))
        sl = (slice(r0, r0 + block_size), slice(c0, c0 + block_size))
        valid = mesh.valid[sl]
        if valid.sum() < (block_size * block_size) // 2:
            continue
        base = mesh.points[sl][valid].astype(np.float32)
        nrm = normals[sl][valid]
        walk = base[None] + nrm[None] * offsets[:, None, None]
        flat = walk.reshape(-1, 3)
        if remote and hasattr(volume, "prefetch"):
            volume.prefetch(flat - (np.zeros(3) if origin is None else np.asarray(origin)))
        profile = sample(flat).reshape(offsets.size, -1).mean(1)
        if not np.isfinite(profile).all() or profile.max() <= 0:
            continue
        out.append(
            {
                "block": [r0, c0],
                "n": int(valid.sum()),
                "range": float(profile.max() - profile.min()),
                "at_zero": float(profile[len(profile) // 2]),
                "peak_offset": float(offsets[int(np.argmax(profile))]),
            }
        )
    return out


def summarise(name: str, blocks) -> dict:
    """Reduce a surface's blocks to the numbers worth reporting."""
    if not blocks:
        return {"mesh": name, "error": "no usable blocks"}
    ranges = np.array([b["range"] for b in blocks])
    peaks = np.array([abs(b["peak_offset"]) for b in blocks])
    return {
        "mesh": name,
        "blocks": len(blocks),
        "range_median": float(np.median(ranges)),
        "range_min": float(ranges.min()),
        "range_max": float(ranges.max()),
        "peak_offset_abs_median": float(np.median(peaks)),
    }


def compare(blocks_a, blocks_b) -> dict:
    """Is surface A drawn from a lower profile-range distribution than B?

    The right test for "this surface is worse than that one", and the one the
    published w128-129 result rests on. Comparing bootstrap intervals of the two
    medians instead is conservative and can hide a real difference: it discards
    the block-level data and asks a weaker question.

    Use B as a surface measured in the *same* volume, ideally the adjacent
    winding of the same tracing run.
    """
    from scipy.stats import mannwhitneyu

    ra = np.array([b["range"] for b in blocks_a])
    rb = np.array([b["range"] for b in blocks_b])
    if ra.size == 0 or rb.size == 0:
        return {"error": "a surface has no usable blocks"}
    stat, p = mannwhitneyu(ra, rb, alternative="less")
    return {
        "n_a": int(ra.size),
        "n_b": int(rb.size),
        "median_a": float(np.median(ra)),
        "median_b": float(np.median(rb)),
        "u": float(stat),
        "p_less": float(p),
    }


def verdict(range_median: float, baseline_median: float) -> tuple[str, float]:
    """Label a surface against a baseline measured in the same volume.

    The 0.5 / 0.3 cuts are calibrated on PHercParis4 at 2.4 um, where published
    surfaces give 51-53 grey levels of range and off-sheet grown surfaces give
    11-12. They are a reading aid for one scan, not a transferable threshold --
    see the module docstring.
    """
    frac = range_median / max(baseline_median, 1e-6)
    if frac >= 0.5:
        return "ON SHEET", frac
    if frac >= 0.3:
        return "marginal", frac
    return "OFF SHEET", frac


def measure(paths, volume, *, reach=70.0, step=1.0, blocks=6, block_size=12, seed=0):
    """Score each tifxyz in ``paths`` against one already-opened volume."""
    from labelscope.mesh import read_tifxyz

    results = []
    for path in paths:
        mesh = read_tifxyz(path)
        found = block_profiles(mesh, volume, None, reach, step, blocks, block_size, seed)
        row = summarise(os.path.basename(str(path).rstrip("/")), found)
        row["per_block"] = found
        results.append(row)
    return results
