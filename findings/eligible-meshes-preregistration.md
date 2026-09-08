# Pre-registration: does the on-sheet score agree with an independent eye screening?

Committed **before** the predictor is computed on these meshes. The labels
already exist and were published by someone else, on 1 September 2026, before
this test was designed — which is the point.

## The dataset, and why it is the right test

Paul's objection to labelscope in August was that its signals are weak, and the
test he named was: find real errors by eye, then check whether the tool flags
those and only those. Every version of that test we could run ourselves has the
same weakness — we generate both the errors and the labels.

`pscamillo/vesuvius-eligible-meshes` removes that. It publishes **340 tifxyz
surface meshes** from eight prize-eligible scrolls (PHerc 0125, 0211, 0257,
0268, 0358, 0800, 0813, 0826), fitted on the 9 µm scans, with `data/index.csv`
carrying a `gate_verdict` column filled in by eye against a fiber-weave
reference:

| verdict | meshes |
|---|---|
| `aprova` (usable weave over most of the mesh) | 41 |
| `reprova` (not usable) | 31 |
| `parcial` | 12 |
| *(blank — not inspected)* | 256 |

Third-party meshes, a third party's eye, published a week before this document,
and 256 meshes the author states are unscreened. If the score separates
`aprova` from `reprova`, the same score screens the other 256 — which is the
contribution, not the validation.

## Predictor, fixed in advance

`labelscope onsheet --mesh <tifxyz> --volume <that scroll's 9 µm zarr> --remote
--blocks 24 --block-size 12 --seed 0`, i.e. 24 non-overlapping grid tiles, the
scan sampled ±70 voxels along the surface normal, one profile per tile.

The per-mesh statistic is the **mean tile range** in grey levels. Mean, not
median: the per-tile range is two-humped on real surfaces (structured tiles
high, fused or empty tiles near zero) and healthy and defective surfaces differ
in the *fraction* that is flat, which the mean is linear in. This is the same
statistic pre-registered for the ink test on 2 September, unchanged.

Scores are ranked **within scroll**, because absolute range tracks scan
resolution and contrast and does not transfer between scans. These are 9 µm
scans (8.64 µm for PHerc0800 and PHerc0268), coarser than anything measured so
far, so the absolute numbers are expected to be low; only the ordering is used.

The sheet-switch detector is **not** used here and is expected to refuse almost
every mesh: at 9 µm the winding spacing is not resolved, and its own gate said
so on PHerc0172 (1 of 53 surfaces clear it). That refusal is the correct
behaviour and is reported, not hidden.

## Pass / fail, fixed in advance

Computed over the 72 meshes labelled `aprova` or `reprova`. `parcial` is
excluded from the primary test and reported separately.

* **Pass:** AUC of within-scroll percentile against the `reprova` label is
  **>= 0.70**, and the `reprova` rate in the bottom third of within-scroll rank
  is at least **twice** the base rate, with at least 10 meshes in that third.
* **Fail:** anything else, published as such with the confusion table.

Descriptive, not gating: per-scroll AUC; the mean-range distributions for each
verdict; and the disagreements both ways, listed by mesh, since a mesh the eye
passed and the score fails is as interesting as the reverse.

**If it passes**, scores for all 340 are published as a CSV keyed to
`data/index.csv`, offered to the author as a pull request, with the 256
previously unscreened meshes ranked. **If it fails**, that is published too,
and it is a stronger negative than our own blind test could produce, because
the labels are not ours.

## What this cannot show

The eye verdict is "does this mesh show usable fiber weave", which is not
identical to "is this surface on a sheet". A mesh can sit correctly on papyrus
and still show poor weave at 9 µm, and a mesh can drift off-sheet in a region
the screener did not look at. Agreement is evidence the score tracks something
real; disagreement is not automatically the score being wrong. Both directions
are listed rather than summarised.
