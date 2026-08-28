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

*Pending the deep scan over a 200-volume subsample.*
