# Pre-registration: does the on-sheet measurement predict where their ink model finds no text?

Committed **before** any label exists and before the predictor has been computed
over the corpus, so the timestamp is checkable.

## Why this test

Paul's objection to labelscope was that its signals are weak, and the test he
named was: find real errors by eye, then check whether the tool flags those and
only those. Every published segment ships the team's own ink-detection render.
On both published tracings of PHercParis4 windings 128-129 that render is
speckle with no letterforms, while the adjacent winding 126-127 shows columns
of legible Greek from the same model on the same scan. That is a real error
found by eye, on their product, with no segment growing required. This document
fixes how the same comparison is run over the whole corpus.

## Population

Every segment in `findings/corpus/inputs/corpus_manifest.tsv` (258) that has
**both** a `surface-volumes/*.zarr` matching the manifest's volume and an
`ink-detection/downsampled/*-ds8.jpg`. Segments lacking either are counted and
excluded. If a segment has several ink renders, the lexically last is used and
recorded.

## Predictor (computed by the tool, never seen by the labeller)

`labelscope onsheet --surface-volume <zarr> --chunks 100 --seed 0`, i.e. 100
random 128x128 chunks of the team's own 109-layer band around the surface,
each chunk's profile averaged over its surface footprint. The per-segment
statistic is the **mean chunk range** (grey levels, max minus min across the
109 layers).

Mean rather than median: on real surfaces the per-chunk range is two-humped --
structured chunks near 40-80, flat chunks near 1-3 -- on healthy and defective
surfaces alike, and they differ in the fraction that is flat. The mean is
linear in that fraction and needs no threshold; the median of a two-humped
sample is unstable (it moved the w128-129 comparison from p=0.003 to p=0.5
between two draws of 24). Median and the fraction of chunks with range < 5 are
reported as secondary statistics.

The predictor is then converted to a **percentile within its scroll**, because
absolute range tracks scan resolution and contrast (PHerc0172's median is 19.7
where other scrolls run 34-60).

## Ground truth (the labeller never sees names, scrolls, or scores)

Carson labels each segment's ink render, shown as a ~2000 px thumbnail under a
random three-digit code in random order, as one of `text` / `no text` /
`unsure`. The code-to-segment key is sealed in `drafts/` (gitignored); its
SHA-256 is recorded here before labelling begins so the key cannot be changed
afterwards.

## What is being claimed, and its direction

Off-sheet implies no text. **No text does not imply off-sheet** -- blank sheets,
margins and regions the ink model simply misses exist. So the pre-registered
metric is the **precision** of a low on-sheet score as a predictor of `no text`,
and recall is expected to be low and is not a pass criterion.

## Pass / fail, fixed in advance

Computed over scrolls in which at least 20% of labelled segments are `text`
(a scroll where nothing reads carries no information about rank). `unsure`
labels are excluded from the precision and reported.

* **Pass:** among segments in the **bottom decile** of within-scroll
  percentile, the fraction labelled `no text` is >= 0.80, with at least 10
  such segments, and that fraction is at least 2x the pooled base rate of
  `no text` in the same scrolls.
* **Fail:** anything else. Published as such, with the confusion table.

Descriptive, not gating: AUC of within-scroll percentile against the `no text`
label; per-scroll tables; and two lists -- low-score segments that show text
(the tool's misses) and high-score segments with no text (blank sheets or
model misses, not the tool's fault, and labelled as such).

## Separately: the four PHercParis4 surfaces by eye

Carson also judges isotropic cross-sections of the two w128-129 tracings and
their w126-127 neighbours (`findings/onsheet/evidence/xsec/`), marking whether
each surface follows a sheet, sits between sheets, or cuts across them. This is
recorded before the mechanism is written into any text.

## Key hash

`SHA-256(drafts/ink-labeler-key.json)` = `a126d16f00e9953cbcf4ba5be1a46bf19d6178311dc412384724437a4594a86f`

Sealed 2 Sep 2026 before any label was made. The page shows 255 renders; the
three segments with no published ink render and the one with no surface
volume are excluded and counted, as above.
