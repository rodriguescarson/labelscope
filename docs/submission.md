# Vesuvius Challenge — August 2026 Progress Prize submission

Form: https://docs.google.com/forms/d/e/1FAIpQLSev2vJobu521iB6OuyehDktzYTEo131F4iUGwt3Qxa9a1fk6A/viewform

**Full name:** Carson Conception Rodrigues
**Team:** individual
**URL:**
- https://github.com/rodriguescarson/labelscope (the tool, MIT)
- Issues filed on ScrollPrize/villa: #1640, #1641, #1642, #1643
- PRs opened on ScrollPrize/villa: #1644, #1645, #1646 (+ a fourth pending)

---

## Field 5 — how this increases the probability of reading complete scrolls

`labelscope` is a CPU tool that measures things the pipeline currently assumes.
It answers four questions, on data you already have, without a GPU or a training
run.

**1. A detector for the failure the spiral satisfaction metric cannot see.**
The Open Problems bottleneck table lists "Meshes can jump from one wrap to
another" and asks for conservative failure detection. villa#1621 shows why the
existing metric cannot give it: it derives its target from the patch's own
position, so a patch displaced by any whole number of windings scores identically
to a correct one — a delta of exactly zero. A displaced surface still lies on
papyrus, so the surface itself gives nothing away. What does is the seam: the one
line of grid edges that must cross the gap between two wraps, and the gap is
dark. On a published PHercParis4 surface against its own 2.4 µm scan, with one
winding planted over half the grid, the seam line's mean darkening goes from 9.1
to 36.3 — a z-score of 0.4 against 11.5 — and it fires at one, two and three
windings alike.

It also refuses to answer when it cannot. The seam is only visible if a grid edge
normally stays on one wrap, so the tool measures the winding spacing from the
scan and declines below two grid steps per winding. That requirement is per
scroll: PHercParis4 needs 2.4 µm, and PHerc0500P2 fails it even at 9.362 µm
because its wraps are physically about 3.5× tighter. My own first fleet run was
at 45.5 µm and was flagging seams that were the scan's ordinary roughness.

It reads a surface out of an 81 TB volume by fetching only the chunks the surface
passes through — about 4 GB per segment, against roughly 50 GB for its bounding
box.

**2. The surface labels mark a face, not a centre-line, in both public
releases.** Measuring the offset from the labelled surface to the local CT
density maximum along the surface normal:

| | Kaggle surface release | Dataset059 |
|---|---|---|
| pairs | 892 | 1,754 |
| measurable | 836 (93.7%) | 1,732 (98.7%) |
| median \|offset\| | **2.285 vx** | **2.576 vx** |
| ≥ 1 voxel | 89% | 98% |

2,568 pairs, two independently produced releases, the same answer to within a
third of a voxel, against sheets 7.5–9.3 voxels thick. This is not an error — a
writing surface is a face — but anything treating these labels as a sheet
centre-line inherits a systematic ~2.3 voxel bias over a winding period whose
median is 19 voxels, and ink sampling symmetric in ±t about the label is not
symmetric about the sheet.

The estimator behind it is the substance. The obvious version — walk the normal,
take the nearest intensity maximum — fails silently on carbonised papyrus: its
answer tracks the search radius rather than any displacement (median |offset|
0.70 vx at R=2, 3.09 at R=9 on the same fixed label). Four separate traps had to
be fixed before the number meant anything, all four now regression tests.

**3. Data facts worth one line of a release script.** 487 of 892 labels in the
Kaggle release ship uncompressed — 15.52 GB of a 45 GB release, where the
compressed half averages 0.87 MB against 32.82 MB. Established from about 1.8 MB
of header reads, because `scan --headers-only` inventories a remote release from
roughly a kilobyte per volume. `Dataset059` ships five patch sizes (170³, 172³,
236³, 300³, 364³) with nothing in a filename saying which; one Kaggle volume,
`sample_00833`, has no class 2 at all.

**4. Four fixes upstream, each with a regression that fails on `main`.**

| PR | fixes | what `main` does |
|---|---|---|
| #1644 | #1488 | the standalone dice loss consumes raw logits; at mu=-1.0 it returns **1.2e+11**, and at mu=-0.10 it flips sign to +0.54 — a gradient pointing the wrong way |
| #1645 | #1507 | native inference feeds the network depth **3** where training feeds 5, dropping the checkpoint's `input_pad_depth_to` |
| #1646 | #1481 | robust flat normalization runs over the reader's zero padding; the padded planes come out at **-2.2659** instead of 0, shifting every real CT voxel |
| pending | #1482 | the native patch retry walks a deterministic cycle and **never terminates** — the test suite had to be killed after 45 s |

Each pairs its regression with a control that passes on both branches, so the
pair distinguishes the fix rather than asserting current behaviour.

**5. Two retractions, kept in the findings.** I reported a 92% train/validation
leak in `Dataset059` that was really 1.5% — I had assumed a uniform 300 cubed
patch size — and eight "malformed" volumes that my own downloader had truncated.
Both mistakes are now checks in the tool: it reads every volume's real shape, and
the fetcher verifies what it wrote rather than what was promised.

Everything is MIT, CPU-only, and validated against planted ground truth rather
than eyeballed. 110 tests, CI on Python 3.9, 3.11 and 3.12.

---

## Checklist before submitting

- [x] repo public at github.com/rodriguescarson/labelscope, MIT
- [x] CI green on 3.9 / 3.11 / 3.12 plus lint
- [x] `pytest -q` green on a clean clone, no data download required
- [x] findings/ committed, including both retractions
- [x] full-population results for both releases (2,568 pairs)
- [ ] fleet-wide sheet-switch sweep over published PHercParis4 surfaces
- [x] upstream issues filed: #1640, #1641, #1642, #1643
- [x] upstream PRs opened: #1644, #1645, #1646 (fourth pending a GitHub rate limit)
- [ ] posted in the Vesuvius Discord
- [ ] form submitted before 2026-08-31 23:59 PT (2026-09-01 12:29 IST)
