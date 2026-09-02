#!/usr/bin/env python3
"""Is this traced surface actually on papyrus?

A tracer can complete normally, report a plausible area, place every vertex
inside the scan, and still produce a surface that cuts *across* the windings
instead of following a sheet. Nothing in the toolchain catches that: the meta
looks right, renders look like credible fibrous texture at every depth, and any
ink model probing the surface returns structured noise
(ScrollPrize/villa#1675 -- and independently reproduced here on a
full-resolution L0 prediction, not the coarse L2 one that issue describes).

The check: sample the scan along the surface normal, averaged over a *coherent*
neighbourhood of the grid. A surface lying on a sheet sits on a density ridge, so
its profile has real dynamic range. A surface cutting across sheets sees the same
fibrous material at every depth, so the profile is flat.

Averaging has to be local. Over a whole patch the winding phase varies and the
periodicity cancels, which makes a good surface look as flat as a bad one -- that
mistake cost an afternoon before this was written.

    python scripts/onsheet_check.py --mesh seg.tifxyz --baseline published.tifxyz \
        --volume <zarr-url> --remote

Calibration on PHercParis4 at 2.4 um: published surfaces give 51-53 grey levels
of range, off-sheet grown surfaces give 11-12.
"""

# The implementation now lives in labelscope.onsheet and is exposed as
# `labelscope onsheet`.  This wrapper is kept because the published corpus runs
# and findings/ cite it by path.

import sys

from labelscope.cli import main as _cli


def main(argv=None) -> int:
    return _cli(["onsheet", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
