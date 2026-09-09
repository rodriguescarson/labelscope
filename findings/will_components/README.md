# The seam detector on Will Stevens' pipeline components

Will Stevens shared ten components from his PHerc1667 unwrapping pipeline as
tifxyz on 9 September, pointing at `patch_1` and `patch_9` as the two that run
past his ground truth. His own length-mismatch heatmap needs GT, so those
regions currently have no check; the seam detector needs none, which is why the
comparison was worth running.

**The answer is that this detector cannot give him a trustworthy number, and
the control is what says so.**

## What was measured

Each component was measured as delivered and again with a whole winding planted
in half of it. The planted run is the reference: the August corpus pass showed a
fixed z threshold does not transfer between surfaces, so a score only means
something next to the same surface's planted control.

The meshes are in level-2 coordinates (`scale: 0.25`, confirmed by sampling the
same vertices at levels 0/1/2 with matching factors and getting consistent
tissue values). They are read against level 2 as authored: `steps_per_winding`
is a ratio of winding spacing to grid step, so scaling both by 4 leaves it
unchanged while costing 64x the chunk traffic.

## patch_9, four window sizes on the same surface

| window | steps/winding | gate | seams | max_z delivered | max_z planted |
|---|---|---|---|---|---|
| 120 | 1.89 | fail | 0 | 4.50 | 4.32 |
| 160 | 2.00 | pass | 1 | 6.50 | 6.60 |
| 200 | 2.13 | pass | 1 | 6.32 | 5.61 |
| 280 | 2.02 | pass | 4 | 8.49 | 7.15 |

## patch_1

| window | steps/winding | gate | seams | max_z delivered | max_z planted |
|---|---|---|---|---|---|
| 160 | 1.38 / 2.89 | fail / pass | 0 / 4 | 7.76 | 8.72 |
| 280 | 2.89 / 2.26 | pass / pass | 2 / 3 | 5.56 | 6.20 |

## Why it fails, precisely

**The planted control does not behave.** On patch_9 the surface carrying a
deliberately inserted whole-winding switch scores *lower* than the untouched
surface in three of four windows. A score that does not rise when a real switch
is added is not measuring switches on this data, so the seam counts above are
noise, not findings.

**The winding-spacing estimate is unstable.** On one surface it reads between
5.5 and 11.5 voxels depending only on which window is measured. The resolution
gate is computed from it, so the gate flips pass/fail on the same surface, and
the z-scores inherit the same instability.

**The root cause is sampling, not the scan.** Will's grid step is 4.0 voxels
against a winding spacing of roughly 5.5-11.5, so the grid carries about two
samples per winding -- right at the Nyquist limit for a check that has to see
the dark gap *between* wraps. This is a different failure from PHerc0172 in the
August pass, where the gate failed because the grid step was large relative to
the winding; here the scan resolves the wraps and the mesh does not sample them
finely enough.

For this class of check to work on his output the patches would need roughly
twice the grid resolution -- a grid step near 2 level-2 voxels rather than 4,
giving 4-5 samples per winding rather than 2.

## Reproduce

`will_l2.py <window> <component-dir>` prints one JSON row per run, delivered and
planted. `gate.py` is the cheap resolution check on its own.
