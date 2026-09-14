# Pre-registration: the seam detector against pscamillo's eye-labelled rejects

Committed **before** the detector is run on these meshes. The labels are a
third party's, given in public on 9 September 2026 in the Vesuvius Discord
thread "Surface meshes for eight prize-eligible scrolls", before this test was
designed.

## Why this test

Sean (bruniss) said on 9 September that an intensity-based seam check cannot be
a switch detector because sheets legitimately touch, and that grid step will
not fix it. The planted-winding control on Will Stevens' components agreed
with him: a planted switch did not raise the score. What has been missing is a
set of *real* switches labelled by someone else. pscamillo supplied one.

## Population (12 meshes, fixed)

From `pscamillo/vesuvius-eligible-meshes`, `gate_verdict = reprova`, all 8 to
10 cm^2, all in the outer windings. He described ten as "large ones that step
to a neighbouring winding" and two as having "drifted off the sheet instead of
jumping to a neighbouring one", with very few components, "useful as negative
controls".

Step set (a seam is expected):
`PHerc0813/z12496_w100`, `PHerc0813/z11296_w100`, `PHerc0813/z4704_w100`,
`PHerc0813/z13088_w100`, `PHerc0813/z11904_w100`, `PHerc0813/z11904_w080`,
`PHerc0211/z4912_w080`, `PHerc0211/z13920_w080`, `PHerc0800/z17072_w100`,
`PHerc0125/z6944_w080`.

Drift set (no seam expected): `PHerc0211/z6720_w100`, `PHerc0211/z9120_w100`.

Volumes: the scroll's published masked 9 um zarr (8.64 um for PHerc0800),
streamed over HTTP.

## Predictor, fixed in advance

`labelscope sheetswitch --mesh <tifxyz> --volume <zarr> --remote` with defaults
(`--z-threshold 5.0`, `--steps 17`, whole surface, no window), once as
delivered and once with `--plant 1`, exactly as in the August corpus pass.
Reported per mesh: `steps_per_winding` and the resolution gate, seam count,
`max_z` delivered and planted.

The meshes carry `scale: [0.05, 0.05]`, a 20-voxel grid step, against a
winding spacing of roughly 16 to 32 voxels at 9.36 um. That is about one
sample per winding, so **the resolution gate is expected to refuse most or all
of them.** A refusal is the pre-registered behaviour of the tool and is
reported as such, not worked around.

## Pass / fail, fixed in advance

Let G be the meshes that pass the resolution gate.

* **Inconclusive by gate:** fewer than 5 step meshes in G. Reported as the
  result, with the per-mesh `steps_per_winding`.
* **Control void:** among G, the planted copy scores higher `max_z` than the
  delivered copy on fewer than 70% of meshes. Then the detector is not
  measuring switches on this data and the primary criterion is not evaluated
  (this is what happened on Will Stevens' components).
* **Pass:** control not void, and the detector reports at least one seam on at
  least 70% of the step meshes in G, and zero seams on every drift mesh in G.
* **Fail:** anything else.

Whichever way it comes out is published with the per-mesh table.

## What this cannot show

Ten positives and two negatives is a small set; a pass is evidence the
detector sees these particular switches, not a rate. pscamillo's "step" and
"drift" are eye descriptions, not measurements, so a disagreement is listed,
not resolved.
