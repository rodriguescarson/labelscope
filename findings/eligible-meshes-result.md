# Result: the on-sheet score against a third party's eye screening

Pre-registered in `eligible-meshes-preregistration.md`, committed as `15e781a`
**before** any of these numbers existed. The meshes and the labels are
`pscamillo/vesuvius-eligible-meshes`, published 1 September 2026, a week before
the test was designed.

## Outcome: FAIL on the pre-registered rule, with weak signal underneath

All 340 meshes scored, 0 failures. 72 carry an `aprova`/`reprova` verdict.

| verdict | n | mean tile range |
|---|---|---|
| `aprova` | 41 | 30.7 |
| `parcial` | 12 | 26.5 |
| `reprova` | 31 | 27.5 |
| not inspected | 256 | 27.4 |

| pre-registered criterion | required | achieved | |
|---|---|---|---|
| AUC of within-scroll rank vs `reprova` | >= 0.70 | **0.70** | met |
| bottom third `reprova` rate >= 2x base | >= 0.86 | **0.67** | **not met** |

**The declared result is FAIL**, because both criteria had to hold. The
direction is right and the effect is real but small: `aprova` scores 12% above
`reprova`, and the bottom third of the ranking is 1.55x enriched for `reprova`.

## A criterion I set badly

The base `reprova` rate among labelled meshes is 0.43. Requiring "2x the base
rate" therefore demanded **86% precision**, which leaves almost no headroom —
a nearly perfect classifier would be needed to clear a bar I wrote without
checking whether it was reachable. On a dataset with a 10% base rate the same
words would have demanded 20%.

This does not change the verdict. The rule was fixed in advance and it is
reported as fixed. But the AUC criterion was the informative one and the
precision criterion was close to unsatisfiable by construction, and a reader
comparing 0.67 against 0.86 deserves to know that.

## Where the signal is, and where it is not

| scroll | labelled | AUC |
|---|---|---|
| PHerc0800 | 19 | 0.83 |
| PHerc0813 | 23 | 0.76 |
| PHerc0211 | 20 | 0.71 |
| PHerc0125 | 10 | 0.62 |

Two scrolls separate usefully, two do not. With 10-23 labelled meshes each,
these differences are not individually significant, and no claim is made that
the method works on PHerc0800 and fails on PHerc0125.

**It does not track the author's own quality gradient.** He reports usability
falling from 83% at w020 to 33% at w100. Our score does not follow it:

| wrap | n | our mean score | his aprova/reprova |
|---|---|---|---|
| w020 | 71 | 29.6 | 35/14 |
| w040 | 71 | 26.0 | 2/2 |
| w060 | 70 | 24.8 | 1/2 |
| w080 | 64 | 25.0 | 1/5 |
| w100 | 64 | **33.8** | 2/8 |

w100 scores **highest** while being the least usable. Whatever the score
responds to at w100, it is not sheet quality — most likely the denser, higher
contrast material near the scroll core.

## It is not a sampling-window artifact

The obvious explanation was the normal-walk length. At 2.4 um a winding spans
40-77 voxels, so the default +/-70 stays inside roughly one winding; at 9.36 um
the same physical spacing is 16-32 voxels, so +/-70 sweeps four to nine
windings and would average the structure away.

Tested directly on 5 `aprova` and 5 `reprova` meshes, same tiles, only the
reach changed (`findings/eligible_meshes/reach/`):

| reach | aprova | reprova | ratio | Mann-Whitney p |
|---|---|---|---|---|
| 8 | 11.3 | 11.0 | 1.02 | 0.35 |
| 15 | 14.8 | 14.1 | 1.05 | 0.50 |
| 25 | 17.4 | 16.0 | 1.09 | 0.35 |
| 40 | 19.5 | 18.9 | 1.03 | 0.50 |
| 70 | 21.5 | 21.8 | 0.99 | 0.58 |

No reach separates them. The hypothesis was mine and it is wrong.

A second observation from the same sweep: on known-good meshes the density peak
sits at roughly 40% of the sampling window at *every* reach (4.5 voxels at
reach 8, 28.5 at reach 70). A surface resting on a resolved ridge would peak
near zero regardless of window. These do not, which says the ridge is not being
resolved at all.

## What this says, stated narrowly

At 9.36 um (8.64 um for PHerc0800 and PHerc0268) the on-sheet measurement does
not reliably reproduce a human's usable/not-usable judgement of a surface mesh.
The absolute ranges here (25-34) sit far below the 51-67 measured on 2.4 um
scans, and the August corpus pass saw the same thing on PHerc0172 at 7.91 um
(median 19.7 against 34-60 elsewhere), where 1 of 53 surfaces cleared the
sheet-switch resolution gate.

**Eight of the thirteen prize-eligible scrolls are published only at ~9 um.**
On this evidence neither of the two checks in this tool works there: the
sheet-switch detector refuses such scans by its own gate, and the on-sheet
score does not track human judgement. That is a measured limit on automated
quality control for exactly the scrolls the prize is meant to open, and it is
a checkable argument for finer scans of them rather than an opinion.

## What is published anyway

`findings/eligible_meshes_scores.csv` carries the score for all 340 meshes,
keyed to `data/index.csv`, including the 256 never inspected. Given the result
above it is **not** offered as a screening verdict, and no pull request is sent
proposing it as one. It is published so the negative can be checked and so
anyone testing a different statistic on these meshes has the measurements.

## Reproduce

```
labelscope onsheet --mesh <mesh.tifxyz> --volume <scroll 9um zarr> --remote \
  --blocks 24 --block-size 12 --seed 0
python scripts/onsheet/eligible_analysis.py --scores findings/eligible_meshes \
  --index findings/eligible_meshes_index.csv
```
