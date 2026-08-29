# Findings

Everything here was produced by `labelscope` on public Vesuvius Challenge data.
Every number comes from a full population — all 892 Kaggle surface pairs and all
1,754 `Dataset059` pairs — not a sample, and each section gives the command, so
any of it can be re-run and disagreed with.

| Dataset | Where | What it is |
|---|---|---|
| Kaggle surface-detection release | `hf://buckets/scrollprize/datasets/surfaces/kaggle` | 892 CT/label pairs |
| `Dataset059_s1_s4_s5_patches_frangiedt` | [dl.ash2txt.org](https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/Dataset059_s1_s4_s5_patches_frangiedt/) | 1,754 pairs; the surface-model training set cited in [villa#191](https://github.com/ScrollPrize/villa/issues/191) |

---

## 0. Two retractions, first

Earlier versions of this report carried two claims that were wrong. They are left
here rather than deleted, because both are mistakes anyone auditing this data can
make, and one of them is the reason a check in this tool exists at all.

### "7,527 overlapping patch pairs; nnU-Net's split leaks 91.9% of validation patches"

Wrong by a factor of fifty. The real figures are **28 pairs** and **1.54%**.

Patch names encode only an origin — `s1_z10240_y2560_x2560` — so the size has to
come from somewhere else, and I took it from the nnU-Net convention and from the
first volumes I opened, which are 300³. Reading every volume's header instead
shows five patch sizes in the one directory:

| shape | count |
|---|---|
| 172³ | 840 |
| 236³ | 753 |
| 300³ | 121 |
| 170³ | 39 |
| 364³ | 1 |

At their real sizes the patches barely touch: `s4` is 236³ on a 256-voxel stride,
`s5` is 170³ on strides of 190 and up — patch smaller than stride, so no overlap
by construction.

Two independent checks say the corrected geometry is the right one. Where overlap
*is* predicted, the two patches' labels agree at a **median IoU of 0.9988**
(minimum 0.9925 over 28 pairs), which they could not do if origins or sizes were
being mismodelled. And the whole analysis was re-run on a separate machine with
its own copy of the data, reaching the same numbers.

`labelscope leakage` now reads every volume's real shape by default;
`--assume-patch-size` reproduces the mistake, and two tests pin the difference.

### "Eight malformed volumes in Dataset059"

Also wrong. Those eight files are fine. They were **truncated by my own
downloader**: the server serves them without a `Content-Length`, so a completion
check of "bytes received ≥ declared size" compared against zero and passed
trivially, renaming a partial file as complete. Re-fetched over a fast link, all
eight read cleanly as 300³ binary volumes.

Compounding it, an old TIFF reader turned the truncation into a *plausible shape*
rather than an error — `tifffile` 2024.8.30 reported `(98, 300, 300)` with an
`invalid page offset` warning where 2026.8.23 reads the intact file silently.
Version skew in a reader is not a detail when the reader is the measuring
instrument.

`scripts/fetch_dataset.py` now verifies what it wrote, by HEAD, and treats an
unverifiable file as an error.

---

## 1. The labels mark a face, not a centre-line

**Command**

```bash
labelscope align --images kaggle/images --labels kaggle/labels --jobs 46 --overlays 12 --out audit/
```

**What it found, over all 892 pairs**

836 of 892 (93.7%) have enough sheet contrast at the labelled surface to measure
at all. In those, the sheet's local CT density maximum does not sit under the
label — it sits a little over two voxels away:

| | |
|---|---|
| median \|offset\| | **2.285 voxels** |
| interquartile range | 1.69 to 2.83 |
| patches with \|offset\| ≥ 1 voxel | **745 / 836 (89%)** |
| ≥ 2 voxels | 528 / 836 (63%) |
| ≥ 3 voxels | 151 / 836 (18%) |
| per-patch 95% bootstrap interval | typically ±0.05 |
| measured winding spacing | 9.5 – 46.0 voxels (median 19.0) |
| sheet thickness, FWHM of the mean profile | 7.5 – 9.3 voxels |

The offset is about **half a sheet thickness**, which is what a label placed on
the recto *face* rather than through the sheet's centre should look like. A
51-patch pilot gave 2.34 voxels against the full population's 2.285, so this is
not a sampling artefact either.

**On the sign.** `labelscope` orients normals toward the denser side of the
sheet, because the release ships no field that can say which way is out (§5). The
offset is therefore positive in almost every patch *by that convention*, which is
close to tautological and is not the evidence. The evidence is the magnitude and
its consistency, at a signal-to-noise of 2.0 to 20.9. Two tests check that the
convention cannot manufacture it — a label centred on a symmetric synthetic sheet
still reads under 0.5 voxels across four seeds, and a genuinely asymmetric sheet
measures the same when the volume is mirrored.

**Why it matters**

This is not an error in the labels — a writing surface is a face, and that is what
was annotated. It matters because of what sits downstream:

* anything that treats these labels as a **sheet centre-line** — meshing, surface
  fitting, anything estimating a normal field from them — carries a systematic
  ~2.3 voxel bias, over a winding period whose median is 19 voxels;
* **ink sampling that is symmetric in ±t about the label** is not symmetric about
  the sheet. It reaches roughly 2.3 voxels further into the void on one side and
  2.3 voxels less far into the papyrus on the other.

The figure is reported per patch and per 64³ region, so it can be corrected for
rather than argued about.

**Locally, the labels wander further than that.** Per 64³ cube of surface, the
median absolute offset is 2.30 voxels and the 90th percentile 4.61 — against a
sheet 7.5–9.3 voxels thick. 83% of cells are at least a voxel off, 58% at least
two.

**It replicates on an independent release.** The same command over all 1,754
`Dataset059` pairs — a different scroll set, different patch sizes, a different
production pipeline — gives:

| | Kaggle surface release | `Dataset059` |
|---|---|---|
| pairs | 892 | 1,754 |
| measurable | 836 (93.7%) | 1,732 (98.7%) |
| median \|offset\| | **2.285 vx** | **2.576 vx** |
| interquartile range | 1.69 – 2.83 | 1.88 – 3.79 |
| ≥ 1 voxel | 89% | 98% |
| ≥ 2 voxels | 63% | 70% |
| per-cell \|offset\|, median | 2.30 vx | 2.62 vx |
| cells ≥ 1 voxel off | 83.3% | 85.2% |
| median winding spacing | 19.0 vx | 21.0 vx |

2,568 pairs between them and the same answer to within a third of a voxel. This
is a property of how these surfaces are annotated, not of one release.

**What it does not show.** Binned by layer separability, the four quartiles read
2.21, 2.19, 2.44 and 2.42 voxels of per-cell offset from least to most separable.
There is no gradient: no evidence that the labels are worse where the layers are
harder to see. The measurement is also noisier in the least separable patches, so
this is not evidence of absence either — but the flatness is worth having, since
"labels avoid the ambiguous regions" is a claim that had not been measured.

Overlay and drift-map PNGs for the twelve worst-aligned patches are in
`full/kaggle_align/overlays/`.

---

## 2. `Dataset059` mixes five patch sizes, and nothing says which is which

Full population, all 1,754 volumes:

* five shapes (§0), all LZW-compressed, uniform `[0, 1]` class scheme
* surface class detected in **all 1,754**
* no unpaired volumes, none unreadable

Nothing in a filename or in `dataset.json` distinguishes a 172³ patch from a 300³
one. nnU-Net itself copes with variable sizes; anything that crops, tiles, or
computes patch geometry from names does not — as §0 demonstrates at my expense.

---

## 3. Fifteen gigabytes of the Kaggle release is uncompressed padding

**Command**

```bash
labelscope scan --labels <bucket>/labels --images <bucket>/images \
                --names-file names.txt --headers-only --out audit/
```

`--headers-only` over HTTP reads about a kilobyte per volume — the byte-order
mark, the first image directory, and the content length. That inventoried all
**1,784 volumes** for roughly **1.8 MB of transfer**, against 45 GB to download
them. The figures below were later confirmed by a full local scan of the same
892 label volumes, which reproduced them exactly.

| | files | on disk |
|---|---|---|
| labels, `COMPRESSION.NONE` | **487** | **15.52 GB** |
| labels, `COMPRESSION.LZW` | 405 | 0.36 GB (median 0.87 MB) |
| | | 45 GB total release |

The uncompressed labels are not different in content — class 1 is a ~2-voxel
thick, highly planar writing surface in both groups, class 2 a single bulky
region in both. Only the writer setting differs, and the compressed half of the
same release shows what the other half would cost: a median of 0.87 MB against
32.82 MB.

**Also true, and fine:** the release mixes three patch sizes — 320³ (840 pairs),
256³ (51), 384³ (1) — and images and labels agree on shape in **every** pair, zero
mismatches.

**Not a finding:** 24 sample indices between 1 and 916 have neither an image nor
a label. The numbering is simply not contiguous.

---

## 4. Two anomalies only the full population shows

Both are single-digit counts in 892, invisible to the 227-volume sample this
report previously ran on.

**`sample_00833` has no class 2 at all.** Its class scheme is `[0, 1]` where every
other volume is `[0, 1, 2]`. The sheet label itself is unremarkable — 3.7% of the
volume, 2.0 voxels thick, 6 components. Only the region class is missing. Nothing
using class 2 as a mask or a validity region will fail loudly on this one; it will
just behave differently.

**Fifty-one patches are masked crops**, 53% to 67% exact zeros against 3.7% for a
normal volume — the 256³ subset. That is not a defect, but it broke the tool: a
median-absolute-deviation noise estimate taken over mostly-zeros is zero, so the
reported signal-to-noise came out at **4.4 × 10⁷** and five patches passed the
reliability gate that exists to catch exactly this. A robust estimator is only
robust while the thing it estimates is the majority of its input. Exact zeros are
now excluded once they pass a fifth of the volume, and an estimate still pinned at
the floor marks the patch unmeasurable.

---

## 5. Labels are not binary, and the surface class is not always index 1

`labelscope` detects the sheet-like class rather than assuming it. In the Kaggle
release that is class 1 (median thickness 2.0 voxels, worst-component planarity
0.007), with class 2 a single bulky region occupying 19–85% of the patch.

Any metric computed over `label > 0` here — foreground fraction, sheet thickness,
class balance — averages a 2-voxel sheet together with a region hundreds of voxels
thick, and means nothing.

**Class 2 is also useless as an orientation reference**, which is less obvious. It
wraps the writing surface on *both* sides, so it cannot say which way is out;
`orientation_reference_quality` scores it at 0.009–0.281 out of 1 on real patches.
Offsets oriented against it carry a sign that flips arbitrarily from patch to
patch, which is why `labelscope align` orients by the scan instead.

**How the sheet class is decided also matters, and the obvious way is wrong.**
Gating on foreground fraction — "a class taking a quarter of the patch is a
region" — reported **63 perfectly ordinary Dataset059 volumes as having no surface
class at all**. Those patches are smaller and their labels thicker, so labels run
from 2% to 30% with no gap for a threshold to sit in. An erosion-survival probe
was tried next and rejected: over three erosions a 10-voxel slab and a 22-voxel
block have nearly identical survival curves. The tool now decides from the
thickness measurement's own saturation, which is a measurement rather than a
guess, and finds a surface class in all 1,754.

---

## 6. Label topology is clean

Over a 200-volume seeded sample of the Kaggle release, with `scan --deep`:

| metric | median | p90 | max |
|---|---|---|---|
| surface fraction | 0.049 | 0.084 | 0.154 |
| local thickness (median) | 2.00 | 2.00 | 2.00 |
| local thickness (p95) | 4.00 | 4.00 | 4.00 |
| connected components | 7 | 13 | 22 |
| fragment fraction | 0.000 | 0.000 | 0.000 |
| worst-component planarity | 0.011 | 0.037 | 0.183 |
| junction fraction | 0.0027 | 0.0047 | 0.0070 |

Uniform `[0, 1, 2]` class scheme in all 200; surface class detected as 1 in all
200. No speckle, no blobs, no fat thickness tail, and junctions — the local
signature of a label bridging two windings — sit at a quarter of one percent of
surface voxels with a maximum of 0.7%.

A negative result, and worth having. These particular failure modes can now be
ruled out by measurement rather than by hope, and the same command will say so
again the next time the release changes.

---

## 7. A detector for the failure the spiral metric cannot see

The Open Problems bottleneck table lists "Meshes can jump from one wrap to
another" and asks for conservative failure detection.
[villa#1621](https://github.com/ScrollPrize/villa/issues/1621) shows why the
spiral satisfaction metric cannot provide it: it derives its target from the
patch's own position, so a patch displaced by any whole number of windings scores
identically to a correct one — a delta of exactly zero.

```bash
labelscope sheetswitch --mesh <tifxyz>... --volume <zarr-url> --remote --window 160 --out audit/
```

A displaced surface still lies on papyrus, so nothing about the surface gives it
away. What does is the seam: the one line of grid edges that has to cross the gap
between two wraps, and the gap is dark. On a published PHercParis4 surface against
its own 2.4 µm scan, with one winding planted over half the grid:

| | real mesh | one winding displaced |
|---|---|---|
| seam line mean darkening | 9.1 | **36.3** |
| ratio to the other lines | 1.13 | **4.72×** |
| **z-score** | **0.4** | **11.5** |

It fires at one, two and three windings alike, which is the periodicity the
satisfaction metric is blind to.

**It also refuses to answer when it cannot.** The seam is only visible if a grid
edge normally stays on one wrap, so the detector measures the winding spacing from
the scan and reports `steps_per_winding`. Below two it moves its findings to
`seams_unreliable` and reports none. That matters more than it sounds: at 45.5 µm
on Scroll 1 the grid step is ~18 voxels against a 12.5 voxel spacing, and a first
fleet run there was flagging "seams" that were just the scan's roughness. The
requirement works out as voxel size < winding spacing / 40, and it is per scroll —
`PHerc0500P2` fails it even at 9.362 µm, because its wraps are physically about
3.5× tighter than Scroll 1's.

**Reading a surface out of an 81 TB volume.** A traced surface is a 2-D sheet
threaded through the scan and touches a small fraction of the chunks in its own
bounding box. `--remote` fetches only those chunks, so a segment costs about 4 GB
of transfer against roughly 50 GB for its bounding box, and the 81 TB array is
never a consideration.

**Two simpler detectors were tried first and both fail**, which is worth recording
because both look reasonable:

* *Large 3-D jumps between grid-adjacent vertices.* Published meshes are smooth —
  a switch slides onto the next wrap rather than tearing. Maximum step on a real
  segment: 27.7 voxels against a 20.0 median. Nothing.
* *An aggregate edge-darkening statistic.* A displaced surface still lies on
  papyrus, and only the seam crosses a gap — about 1% of a mesh's edges. Real
  versus one-winding-displaced: 12.29% against 11.72% of edges dipping past
  threshold. Nothing.

---

## 8. The detector's loudest result was made of nothing

A third retraction, and the most useful one, because it was only visible once the
detector was pointed at every published surface rather than a validated example.

Running `sheetswitch` over the published `PHercParis4` surfaces at 2.4 µm, the
highest score in the whole sweep was **z = 12.60** on
`20260701183139-w098-100` — comfortably past the z ≥ 5 threshold, on a surface
that passes the resolution gate, and flagged in two places at once. It looked
like the finding.

It was an artifact of a null with no spread in it:

| surface | max z | median line dip, axis 0 | axis 1 |
|---|---|---|---|
| `20260701183139-w098-100` | **12.60** | **0.000** | **0.000** |
| `20260623160554-w098-100` | 8.43 | 5.972 | 5.131 |
| `20260623141924-w010-027` | 7.24 | 6.387 | 5.106 |
| `20231221180251` | 6.86 | 5.385 | 5.684 |
| *(median over all 56 measured surfaces)* | | 5.404 | 5.124 |

The published volumes are **masked**: the air around the scroll is absent from
the store and reads as zero. A mesh can be perfectly well-formed over a region
the scan does not cover — this window was 100% valid vertices, 160 edges on every
grid line — and then every edge dips by nothing. The line means were all `0.00`
except one at `0.25`. The median was 0, the median absolute deviation was 0, and
the code fell back to a standard deviation of about 0.02, which turned a quarter
of a grey level into z = 12.6.

**Eleven of the 56 measured surfaces sat in that regime**, median line dip under
one grey level on both axes.

### The fix, in two parts

`_line_scores` now returns zeros when the spread is degenerate rather than
substituting a standard deviation. A null with no spread has no z-scores in it,
and manufacturing one is how a measurement invents a result.

And the detector samples the scan at the surface itself before scoring. Below
`--min-dip` the result is marked `dip_degenerate`, the seams move to
`seams_degenerate`, and none are reported — the same shape as the existing
resolution refusal, for a different reason. The triangular detector carries the
same guard.

```bash
labelscope sheetswitch --mesh surface.tifxyz --volume scan.zarr --remote \
                       --min-dip 1.0 --out audit/
```

### Why this keeps happening

This is the second time in this project that **masked-out regions have broken an
estimator built on robust statistics** — the first cost a noise estimator its
sanity on 51 crops, and produced an SNR of 4.4 × 10⁷. Both times the failure was
invisible on a validated example and obvious across the full population. Both
times the robust statistic did exactly what it was asked: MAD is zero when the
data is constant, and every rule for "what to do when the scale estimate is zero"
is a decision about what to invent.

The general lesson, stated so the next estimator here inherits it: **a scale
estimate of zero is a refusal, not a small number.**
