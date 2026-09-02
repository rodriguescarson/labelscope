"""Compare block distributions directly instead of eyeballing CI overlap.

Overlapping bootstrap CIs of two medians is a conservative test and not the
right one. Mann-Whitney on the per-block ranges asks the actual question: are
the w128-129 blocks drawn from a lower distribution than their neighbours'?
"""

import glob
import sys

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, "/workspace")
from onsheet_check import block_profiles  # noqa: E402

from labelscope.mesh import read_tifxyz  # noqa: E402
from labelscope.remote_zarr import ChunkedVolume  # noqa: E402

VOL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
)
vol = ChunkedVolume.from_store(VOL)


def ranges(d, n=24):
    profs = block_profiles(read_tifxyz(d), vol, None, 70, 1.0, n, 12, 0)
    if profs and isinstance(profs[0], dict):
        return np.array([p["range"] for p in profs])
    return np.array([np.ptp(p) for p in profs])


def find(tag):
    return sorted(d for d in glob.glob("/workspace/corpus_meshes/*/") if tag in d)


pairs = [("20260623", "Jun tracing"), ("20260701", "Jul tracing")]
for pref, label in pairs:
    a = next(d for d in find("w128-129") if pref in d)
    b = next(d for d in find("w126-127") if pref in d)
    ra, rb = ranges(a), ranges(b)
    u, p = mannwhitneyu(ra, rb, alternative="less")
    print(
        f"{label}: w128-129 (n={ra.size}, med {np.median(ra):.1f}) vs "
        f"w126-127 (n={rb.size}, med {np.median(rb):.1f})"
    )
    print(
        f"    Mann-Whitney one-sided p = {p:.4f}   "
        f"{'SEPARATED' if p < 0.05 else 'NOT separated at 0.05'}"
    )

# pooled: both w128-129 tracings against both w126-127 tracings
A = np.concatenate([ranges(next(d for d in find("w128-129") if p in d)) for p, _ in pairs])
B = np.concatenate([ranges(next(d for d in find("w126-127") if p in d)) for p, _ in pairs])
u, p = mannwhitneyu(A, B, alternative="less")
print(
    f"\npooled: w128-129 (n={A.size}, med {np.median(A):.1f}) vs "
    f"w126-127 (n={B.size}, med {np.median(B):.1f})"
)
print(f"    Mann-Whitney one-sided p = {p:.5f}")
