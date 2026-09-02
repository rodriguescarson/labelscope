"""Re-measure the w128-129 pair and their neighbours at 3x the block count.

The headline claim rests on these numbers, so they should not rest on 8 blocks.
Reports a bootstrap interval over blocks rather than a bare median.
"""

import glob
import sys

import numpy as np

sys.path.insert(0, "/workspace")
from onsheet_check import block_profiles  # noqa: E402

from labelscope.mesh import read_tifxyz  # noqa: E402
from labelscope.remote_zarr import ChunkedVolume  # noqa: E402

VOL = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
)
vol = ChunkedVolume.from_store(VOL)

want = ["w126-127", "w128-129"]
dirs = sorted(d for d in glob.glob("/workspace/corpus_meshes/*/") if any(w in d for w in want))
dirs.append("/workspace/pubmesh")

rng = np.random.default_rng(0)
print(f"{'surface':44s} {'blocks':>6s} {'range':>8s} {'95% CI':>16s} {'|peak|':>7s}")
print("-" * 88)
for d in dirs:
    m = read_tifxyz(d)
    profs = block_profiles(m, vol, None, 70, 1.0, 24, 12, 0)
    rs = (
        np.array([p["range"] for p in profs])
        if profs and isinstance(profs[0], dict)
        else np.array([np.ptp(p) for p in profs])
    )
    pk = (
        np.array([abs(np.argmax(p) - len(p) // 2) for p in profs])
        if not isinstance(profs[0], dict)
        else np.array([abs(p["peak_offset"]) for p in profs])
    )
    if rs.size == 0:
        print(f"{d.rsplit('/', 2)[-2][:44]:44s}  no blocks")
        continue
    boot = [np.median(rng.choice(rs, rs.size)) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    name = d.rstrip("/").rsplit("/", 1)[-1]
    print(
        f"{name[:44]:44s} {rs.size:6d} {np.median(rs):8.1f} "
        f"{f'[{lo:.1f}, {hi:.1f}]':>16s} {np.median(pk):7.1f}"
    )
