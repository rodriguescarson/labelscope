# Vesuvius Challenge — August 2026 Progress Prize submission

Form: https://docs.google.com/forms/d/e/1FAIpQLSev2vJobu521iB6OuyehDktzYTEo131F4iUGwt3Qxa9a1fk6A/viewform

The Progress Prize submission is the form and nothing else — five fields, no
attachments, no upload, no email (the `grandprize@scrollprize.org` address is for
the Grand Prize, First Letters and Title prizes only). Verified against the live
form on 2026-08-29.

| field | answer |
|---|---|
| Email | rodriguescarson@gmail.com (pre-filled; tick "record this email") |
| Your full name | Carson Conception Rodrigues |
| Team description | Individual — not a team. |
| URL | the four below |
| Short description | field 5, below |
| Terms and Conditions | Yes, I agree |

**URL:**
- https://github.com/rodriguescarson/labelscope — the tool, MIT, CI on 3.9/3.11/3.12
- Issues filed on ScrollPrize/villa: #1640, #1641, #1642, #1643, #1649
- PRs opened on ScrollPrize/villa: #1644, #1645, #1646
- Fix for #1482 posted with its branch (PR creation is blocked for my account)

---

## Field 5 — how this increases the probability of reading complete scrolls

`labelscope` is a CPU tool that measures things the unwrapping pipeline currently
assumes. It runs on a laptop, needs no GPU and no training, and reads the
standard formats: Zarr/OME-Zarr, tifxyz quad meshes, and `.obj`/`.ply`
triangular meshes.

**1. A detector for the failure the spiral satisfaction metric cannot see, run
over the whole published corpus — and the control that says whether it works.**

The Open Problems bottleneck table lists "meshes can jump from one wrap to
another" and asks for conservative failure detection. villa#1621 shows why the
existing metric cannot give it: it derives its target from the patch's own
position, so a surface displaced by any whole number of windings scores
identically to a correct one — a delta of exactly zero. A displaced surface still
lies on papyrus, so the surface itself gives nothing away. What does is the seam:
the one line of grid edges that must cross the gap between two wraps, and the gap
is dark.

CORPUS_TABLE_HERE

Then the same surfaces again, each with a whole winding planted in half of it —
the case the satisfaction metric scores as no change at all. That pairing is the
part that matters, and it is what a fleet pass without a control cannot tell you:

CORPUS_PAIRED_HERE

**2. Running the full population found a defect in my own detector, and it was
the loudest result in the sweep.**

On the first pass the highest score anywhere was z = 12.60, on a surface that
passed the resolution gate and was flagged on two axes. Its median line dip was
**0.000 grey levels on both axes**. The published volumes are masked — the air
around the scroll is absent from the store and reads as zero — and a mesh can be
perfectly well-formed over a region the scan does not cover. Every edge dipped by
nothing, the robust spread collapsed, and a fallback to the standard deviation
turned a quarter of a grey level into the highest z in the run. Eleven of 56
surfaces sat in that regime.

The detector now samples the scan at the surface before scoring, reports
`dip_degenerate`, and refuses. `_line_scores` returns zeros on a degenerate
spread instead of manufacturing one. This is the second time in this project that
masked-out regions have broken an estimator built on robust statistics, so the
rule is written down rather than just the fix: **a scale estimate of zero is a
refusal, not a small number.**

Both refusals — the resolution gate and this one — are reported as data, so the
count of surfaces the tool *declines* to judge is itself a result about where
this class of check can run at all.

**3. The surface labels mark a face, not a centre-line, in both public
releases.** Measuring the offset from the labelled surface to the local CT
density maximum along the surface normal, over 2,568 pairs in two independently
produced releases:

| | Kaggle surface release | Dataset059 |
|---|---|---|
| pairs | 892 | 1,754 |
| measurable | 836 (93.7%) | 1,732 (98.7%) |
| median \|offset\| | **2.285 vx** | **2.576 vx** |

The same answer to within a third of a voxel, against sheets 7.5–9.3 voxels
thick. This is not an error — a writing surface is a face — and a member of the
team has said as much on villa#1640, along with the sharper point that intensity
peaks will not reliably localise it. I agree with both, and the measurement is
reported here as a quantification of an intended convention rather than a bug:
anything treating these labels as a sheet centre-line inherits a systematic ~2.3
voxel bias, and ink sampling symmetric in ±t about the label is not symmetric
about the sheet.

What the estimator is good for is *consistency*: cells of the same patch
disagreeing with each other is not a convention. `labelscope regularise` removes
that disagreement while preserving the convention — every cell's target is the
patch's own global offset, never zero, so a patch labelled consistently 2.3
voxels off comes back byte-identical. On all 1,754 `Dataset059` patches it
changed 1,735, left 19 alone where the measurement was unreliable, and errored on
none; median shift 0.84 voxels over 36% of surface voxels.

TRACK_A_RESULT_HERE

**4. Data facts worth one line of a release script.** 487 of 892 labels in the
Kaggle release ship uncompressed — 15.52 GB of a 45 GB release, established from
about 1.8 MB of header reads. `Dataset059` ships five patch sizes (170³, 172³,
236³, 300³, 364³) with nothing in a filename saying which. And 55 of the 81
published `PHercParis4` meshes are traced on volume `20230205180739`, which is
not in the S3 open-data bucket in any form (villa#1649) — it exists only as a
TIFF stack on the legacy server, while every other Scroll 1 mesh tier has its
zarr published.

**5. Fixes upstream, each with a regression that fails on `main`.**

| PR | fixes | what `main` does |
|---|---|---|
| #1644 | #1488 | the standalone dice loss consumes raw logits; at mu=-1.0 it returns **1.2e+11**, and at mu=-0.10 it flips sign — a gradient pointing the wrong way |
| #1645 | #1507 | native inference feeds the network depth **3** where training feeds 5 |
| #1646 | #1481 | robust flat normalization runs over the reader's zero padding; padded planes read **-2.2659** instead of 0 |
| branch | #1482 | the native patch retry walks a deterministic cycle and **never terminates** |

Each pairs its regression with a control that passes on both branches. The fourth
is written and tested but GitHub refuses `CreatePullRequest` from my account into
this repository — for nine hours across eight attempts, while issue comments and
issue creation work normally and PR creation into my own fork succeeds — so it is
posted on #1482 with its public branch instead of being sat on.

**6. Three retractions, kept in the findings.** A 92% train/validation leak that
was really 1.5% (I assumed a uniform patch size); eight "malformed" volumes that
my own downloader had truncated; and the z = 12.60 above. All three are now
checks in the tool: it reads every volume's real shape, the fetcher verifies what
it wrote rather than what was promised, and the detector refuses a null with no
spread in it.

Everything is MIT, CPU-only, containerised, and validated against planted ground
truth rather than eyeballed. TEST_COUNT_HERE tests, CI on Python 3.9, 3.11 and
3.12.

---

## Checklist before submitting

- [x] repo public at github.com/rodriguescarson/labelscope, MIT
- [x] CI green on 3.9 / 3.11 / 3.12 plus lint
- [x] `pytest -q` green on a clean clone, no data download required
- [x] Dockerfile built and run end-to-end on the committed sample
- [x] figures generated by running the tool, not drawn
- [x] findings/ committed, including all three retractions
- [x] triangular mesh input (.obj, .ply), and compressed zarr stores
- [x] full-population results for both label releases (2,568 pairs)
- [ ] corpus sheet-switch sweep + planted control across all scrolls
- [ ] label-free evaluation of the regularised-label retrain
- [x] upstream issues filed: #1640, #1641, #1642, #1643, #1649
- [x] upstream PRs opened: #1644, #1645, #1646 (+ #1482 fix posted with branch)
- [ ] posted in the Vesuvius Discord
- [ ] form submitted before 2026-08-31 23:59 PT (2026-09-01 12:29 IST)
