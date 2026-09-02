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
* **Surfaces are heterogeneous, so report the distribution.** On a published
  20-million-cell surface the per-block range runs from ~1 to ~80 on healthy
  and defective surfaces alike; they differ in what *fraction* is flat.  A
  median of a bimodal sample at n=24 is unstable by construction -- it moved
  the w128-129 p-value from 0.003 to 0.5 between two random draws.  Sample
  enough, and read the quantiles, not one number.

Two ways to sample.  :func:`block_profiles` walks the raw scan along the
surface normal and works for any tifxyz, including one just traced.  For a
*published* segment, :func:`surface_volume_profiles` reads the team's own
``surface-volumes/*.zarr`` -- the scan already resampled into a 109-layer band
around the surface, dense at one column per voxel -- which is ~300x cheaper per
column and takes our sampler out of the argument entirely.
"""

from __future__ import annotations

import os

import numpy as np


def _tiles(rows: int, cols: int, block_size: int, rng) -> list:
    """Every non-overlapping ``block_size`` tile of the grid, in random order.

    Tiles rather than random placement: two overlapping blocks share vertices,
    and treating them as separate observations in :func:`compare` would count
    the same material twice.  A grid smaller than one tile yields nothing.
    """
    tiles = [
        (r, c)
        for r in range(0, rows - block_size + 1, block_size)
        for c in range(0, cols - block_size + 1, block_size)
    ]
    order = rng.permutation(len(tiles))
    return [tiles[i] for i in order]


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

    Blocks are non-overlapping tiles visited in random order and rejected unless
    at least half their vertices are valid, so a patch with sparse coverage
    yields fewer blocks rather than profiles built from scattered points.

    Normals come from a one-cell-padded window around each block, which gives
    the same central differences as computing them over the whole grid while
    letting a :class:`~labelscope.mesh.LazyQuadMesh` page in only that window.
    """
    from labelscope.mesh import _sampler

    sample, remote = _sampler(volume, origin)
    offsets = np.arange(-reach, reach + step / 2, step, dtype=np.float32)
    rows, cols = mesh.shape
    rng = np.random.default_rng(seed)

    out = []
    for r0, c0 in _tiles(rows, cols, block_size, rng):
        if len(out) >= blocks:
            break
        pr0, pc0 = max(r0 - 1, 0), max(c0 - 1, 0)
        pr1, pc1 = min(r0 + block_size + 1, rows), min(c0 + block_size + 1, cols)
        win = mesh.window(pr0, pr1, pc0, pc1)
        sl = (slice(r0 - pr0, r0 - pr0 + block_size), slice(c0 - pc0, c0 - pc0 + block_size))
        valid = win.valid[sl]
        if valid.sum() < (block_size * block_size) // 2:
            continue
        base = win.points[sl][valid].astype(np.float32)
        nrm = win.normals()[sl][valid]
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


SV_LAYERS, SV_SIDE = 109, 128


def _s3_list(url: str, delimiter: bool):
    import re
    import urllib.parse
    import urllib.request

    bucket, _, prefix = url.partition("amazonaws.com/")
    bucket += "amazonaws.com"
    q = f"{bucket}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
    if delimiter:
        q += "&delimiter=/"
    xml = urllib.request.urlopen(q, timeout=60).read().decode()
    tag = "Prefix" if delimiter else "Key"
    found = re.findall(rf"<{tag}>([^<]+)</{tag}>", xml)
    return [f"{bucket}/{k}" for k in found if k != prefix]


def surface_volume_profiles(store: str, chunks: int = 100, seed: int = 0, min_coverage=0.5):
    """Layer profiles from a published segment's own ``surface-volumes`` zarr.

    Each chunk is 109 layers x 128 x 128 columns, uint8, uncompressed, with the
    traced surface at the middle layer, so one 1.8 MB fetch yields a profile
    averaged over up to 16,384 columns.  The store is sparse -- only chunks the
    surface covers exist -- so chunks are drawn from a listing.  Returns the
    same record shape as :func:`block_profiles` so :func:`summarise` and
    :func:`compare` apply unchanged.
    """
    import urllib.request

    rng = np.random.default_rng(seed)
    columns = _s3_list(f"{store.rstrip('/')}/0/0/", delimiter=True)
    if not columns:
        return []
    out, tries = [], 0
    while len(out) < chunks and tries < chunks * 6:
        tries += 1
        col = columns[int(rng.integers(len(columns)))]
        keys = _s3_list(col, delimiter=False)
        if not keys:
            continue
        key = keys[int(rng.integers(len(keys)))]
        raw = urllib.request.urlopen(key, timeout=120).read()
        if len(raw) != SV_LAYERS * SV_SIDE * SV_SIDE:
            continue
        cube = np.frombuffer(raw, dtype=np.uint8).reshape(SV_LAYERS, SV_SIDE, SV_SIDE)
        footprint = cube.max(axis=0) > 0  # the surface exists where any layer is non-zero
        if footprint.mean() < min_coverage:
            continue
        profile = cube[:, footprint].astype(np.float32).mean(axis=1)
        mid = SV_LAYERS // 2
        out.append(
            {
                "block": key.rsplit("/", 2)[-2:],
                "n": int(footprint.sum()),
                "range": float(profile.max() - profile.min()),
                "at_zero": float(profile[mid]),
                "peak_offset": float(np.argmax(profile) - mid),
            }
        )
    return out


def summarise(name: str, blocks) -> dict:
    """Reduce a surface's blocks to the numbers worth reporting."""
    if not blocks:
        return {"mesh": name, "error": "no usable blocks"}
    ranges = np.array([b["range"] for b in blocks])
    peaks = np.array([abs(b["peak_offset"]) for b in blocks])
    p10, p25, p75, p90 = np.percentile(ranges, [10, 25, 75, 90])
    return {
        "mesh": name,
        "blocks": len(blocks),
        "columns": int(sum(b["n"] for b in blocks)),
        "range_median": float(np.median(ranges)),
        "range_p10": float(p10),
        "range_p90": float(p90),
        "range_iqr_over_median": float((p75 - p25) / max(np.median(ranges), 1e-9)),
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
    """Score each tifxyz in ``paths`` against one already-opened volume.

    Surfaces over 256 MB on disk are memory-mapped rather than loaded, since
    the check only touches a few dozen blocks of them.
    """
    from labelscope.mesh import read_tifxyz

    results = []
    for path in paths:
        mesh = read_tifxyz(path, lazy="auto")
        found = block_profiles(mesh, volume, None, reach, step, blocks, block_size, seed)
        row = summarise(os.path.basename(str(path).rstrip("/")), found)
        row["per_block"] = found
        results.append(row)
    return results
