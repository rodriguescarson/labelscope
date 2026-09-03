# The full-population on-sheet pass, and a correction to the August count

## Correction: the corpus is 258 surfaces, not 253

The August entry reported **"253 published surfaces across 7 scrolls, 0 errors"**.
That denominator is wrong. The input manifest the run was driven from lists
**258** surfaces, and only 253 produced a result row:

| scroll | in manifest | measured in August |
|---|---|---|
| PHerc0814 | 19 | **14** |
| all others | 239 | 239 |

Five PHerc0814 surfaces were never measured and were not counted as failures, so
the published "0 errors" was true of what ran but silent about what did not. All
five are `auto_grown_*` segments whose names carry extra underscores and
`_copy`/`_abf` suffixes, which is the most likely reason the August sharding
missed them.

They have now been measured. **None is defective** — median profile range 53.5
across PHerc0814's full 19, minimum 24.2. Nothing about the August conclusions
changes; the count does.

This is the fourth correction published against this work, and it was found by
cross-checking the manifest against the result rows rather than by anything
external.

## The pass

Every published surface not already measured was scored: 163 surfaces across six
scrolls, on a 2-vCPU CPU pod at $0.06/h (the check never touches a GPU).

| scroll | scored | median range | min |
|---|---|---|---|
| PHerc0139 | 38 | 60.4 | 45.9 |
| PHerc0172 | 53 | 19.7 | 14.8 |
| PHerc0343P | 8 | 34.9 | 23.2 |
| PHerc0500P2 | 39 | 44.1 | 30.0 |
| PHerc0814 | 19 | 53.5 | 24.2 |
| PHerc1667 | 19 | 57.9 | 45.3 |
| PHercParis4 | 81 | 48.0 | 12.2 |
| **total** | **257 of 258** | | |

**Coverage: 258 of 258 against the sheet-switch detector's 114.** The last
surface, `PHerc1667/20260612121456-w011…merged_v4_flatboi_straightened_v4`,
has `x/y/z` grids of 248 MB each and OOM-killed the original reader (exit 137)
in a 4 GB container. `read_tifxyz(lazy=...)` now memory-maps the TIFFs and pages
in only the blocks touched; that surface scores 63.7 (healthy) in the same
container (`onsheet/evidence/big_248mb_surface.json`).

**A second, denser measurement covers 256 of 258:** 100 chunks of each
segment's own `surface-volumes` zarr (`onsheet/onsheet_sv/`). The two not
covered have no surface volume published for the scan the manifest names — a
coarser one exists and is not a substitute. Three scrolls store the band at a
different depth (33 or 118 layers, not 109) and one PHerc1667 store is
blosc-compressed with dot-separated chunk keys; the reader takes all of that
from each store's `.zarray` now, after first silently rejecting 102 segments.

## A limit on how these numbers may be read

Absolute profile range is **not comparable across scrolls**. PHerc0172's median
is 19.7 where other scrolls run 34-60, which reflects that scroll's scan
resolution (median 0.90 steps per winding) rather than 53 defective surfaces. A
low score is only evidence about a surface when compared against surfaces
measured in the *same* scan — ideally the adjacent winding in the same tracing
run, which is how the w128-129 result was established
(`terminal-patch-result.md`).

For the same reason no corpus-wide "off-sheet rate" is quoted here. Setting a
single threshold across scrolls would repeat exactly the mistake the August
planted control already exposed, where a fixed z threshold failed to transfer
between surfaces.

## Reproduce

```
scripts/onsheet/run_full_pass.sh      # drives scripts/onsheet_check.py per surface
python scripts/onsheet/agreement.py   # joins to the sheet-switch results
```
