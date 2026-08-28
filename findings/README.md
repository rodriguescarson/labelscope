# Findings

Everything here was produced by `labelscope` on public Vesuvius Challenge data,
on a laptop CPU. Each section gives the exact command, so any of it can be re-run
and disagreed with.

| Dataset | Where | What it is |
|---|---|---|
| `Dataset059_s1_s4_s5_patches_frangiedt` | [dl.ash2txt.org](https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/Dataset059_s1_s4_s5_patches_frangiedt/) | 1,754 volume/label pairs; the surface-model training set cited in [villa#191](https://github.com/ScrollPrize/villa/issues/191) |
| Kaggle surface-detection release | `hf://buckets/scrollprize/datasets/surfaces/kaggle` | 892 CT/label pairs |

---

## 0. A correction, first

An earlier version of this report claimed `Dataset059` contained **7,527
overlapping patch pairs** and that nnU-Net's default split leaked **91.9%** of
validation patches. **That was wrong, and it was wrong by a factor of fifty.**

The patch names encode only an origin — `s1_z10240_y2560_x2560` — so the size has
to come from somewhere else. It was taken from the nnU-Net convention and from
the first volumes opened, which are 300³. Reading every volume's header instead
shows the release ships **four** patch sizes:

| shape | count |
|---|---|
| 172³ | 840 |
| 236³ | 753 |
| 300³ | 113 |
| 170³ | 39 |

At their real sizes the patches barely touch: `s4` is 236³ on a 256-voxel stride,
`s5` is 170³ on strides of 190 and up — patch smaller than stride, so no overlap
by construction. The corrected count is **28 overlapping pairs**, and nnU-Net's
default split leaks **1.5%** of validation patches, with a mean of 0.4% of each
one's labelled surface.

Two independent checks say the corrected geometry is the right one. The
downloaded files are byte-identical to the server (120-file sample, zero
mismatches), so this is not truncation. And where overlap *is* predicted, the two
patches' labels agree at a **median IoU of 0.999** (minimum 0.993 over 28 pairs)
— which they could not do if the origins or sizes were being mismodelled.

`labelscope leakage` now reads every volume's real shape by default;
`--assume-patch-size` is the opt-in flag that reproduces the mistake. Two tests
pin the difference. The correction is left here rather than quietly dropped,
because "the declared patch size is not the patch size" is the most transferable
thing in this report.

---

## 1. The labels mark a face, not a centre-line

**Command**

```bash
labelscope align --images kaggle/images --labels kaggle/labels --overlays 10 --out findings/kaggle_align/
```

**What it found**

Across 51 patches of the Kaggle surface release, 48 have enough sheet contrast to
measure. In those 48 the sheet's local CT density maximum does not sit under the
label — it sits about two and a third voxels away:

| | |
|---|---|
| median \|offset\| | **2.34 voxels** |
| interquartile range | 1.79 to 3.02 |
| patches with \|offset\| ≥ 1 voxel | **45 / 48 (94%)** |
| patches with \|offset\| ≥ 2 voxels | 32 / 48 (67%) |
| per-patch 95% bootstrap interval | typically ±0.05 |
| measured winding spacing | 10.5 – 29.0 voxels (median 19.8) |
| sheet thickness, FWHM of the mean profile | 7.5 – 9.3 voxels |

Individually, the four highest-contrast patches read 2.34, 2.61, 2.48 and 2.14
voxels. The offset is about **half a sheet thickness**, which is what a label
placed on the recto *face* rather than through the sheet's centre should look
like.

**On the sign.** `labelscope` orients normals toward the denser side of the
sheet, because the release ships no field that can say which way is out (§4). The
offset is therefore positive in 47 of 48 patches *by that convention*, which is
close to tautological and is not the evidence. The evidence is the magnitude and
its consistency: the density distribution around the labelled surface is
genuinely asymmetric, at a signal-to-noise of 2.0 to 14.8. Two tests check that
the convention cannot manufacture this — a label centred on a symmetric synthetic
sheet still reads under 0.5 voxels across four seeds, and a genuinely asymmetric
sheet measures the same when the volume is mirrored.

**Why it matters**

This is not an error in the labels — a writing surface is a face, and that is
what was annotated. It matters because of what sits downstream:

* anything that treats these labels as a **sheet centre-line** — meshing,
  surface fitting, anything estimating a normal field from them — carries a
  systematic ~2.3 voxel bias, over a winding period of 10–29 voxels;
* **ink sampling that is symmetric in ±t about the label** is not symmetric about
  the sheet. It reaches roughly 2.3 voxels further into the void on one side and
  2.3 voxels less far into the papyrus on the other.

The number is per-patch and per-region, so it can be corrected for rather than
argued about.

**Locally, the labels wander much more than that.** Per 64³ cube of surface, the
median absolute offset is 2.43 voxels and the 90th percentile is 4.66 — against a
sheet 7.5–9.3 voxels thick. 83% of cells are at least a voxel off, 63% at least
two.

**What it does not show.** Binned by layer separability, the least-separable
quartile reads 2.39 voxels of per-cell offset against 2.76 in the most separable
— i.e. no evidence here that the labels are worse where the scroll is harder. The
measurement is also noisier in exactly those patches, so this is not evidence of
absence either. The per-cell maps are emitted so the Annotation Team can look
rather than infer.

---

## 2. `Dataset059` mixes four patch sizes and ships eight malformed volumes

**Command**

```bash
labelscope leakage --labels Dataset059/labelsTr --measure-seen --consistency 250 --out findings/dataset059/
```

Beyond the four cube sizes in §0, eight volumes carry 300×300 pages but **fewer
than 300 of them**:

| volume | pages | page size |
|---|---|---|
| `s1_z10880_y2880_x3200` | 98 | 300×300 |
| `s1_z10880_y3520_x3520` | 117 | 300×300 |
| `s1_z10880_y3200_x3520` | 119 | 300×300 |
| `s1_z10560_y3840_x3840` | 187 | 300×300 |
| `s1_z10880_y2880_x2560` | 218 | 300×300 |
| `s1_z10880_y2560_x2560` | 230 | 300×300 |
| `s1_z10560_y4480_x2880` | 243 | 300×300 |
| `s1_z10560_y4480_x3520` | 293 | 300×300 |

These raise `invalid page offset` on read. Their sizes match the server exactly,
so the copies here are faithful — the volumes are malformed at the source. One
further volume is 364³, larger than any nominal patch size in the release.

**Good news, reported as such:** where patches do overlap, their labels agree.
Median IoU 0.999 across 28 pairs, minimum 0.993, none below 0.9. The release does
not carry two answers for the same scroll voxel.

---

## 3. Fifteen gigabytes of the Kaggle release is uncompressed padding

**Command**

```bash
labelscope scan --labels <bucket-url>/labels --images <bucket-url>/images \
                --names-file names.txt --headers-only --out findings/kaggle_remote/
```

`--headers-only` over HTTP reads about a kilobyte per volume — the byte-order
mark, the first image directory, and the content length. That inventoried all
**1,784 volumes** of the release for roughly **1.8 MB of transfer**, against 45 GB
to download them.

| | files | on disk |
|---|---|---|
| labels, `COMPRESSION.NONE` | **487** | **15.52 GB** |
| labels, `COMPRESSION.LZW` | 405 | 0.36 GB (median 0.87 MB) |
| | | 45 GB total release |

The uncompressed labels are not different in content — class 1 is a ~2-voxel
thick, highly planar writing surface in both groups (median local thickness 2.0,
worst-component planarity 0.007), class 2 a single bulky region in both. Only the
writer setting differs, and the compressed half shows what the other half would
cost: a median of 0.87 MB against 32.82 MB.

**Also true, and fine:** the release mixes three patch sizes — 320³ (840 pairs),
256³ (51), 384³ (1) — and images and labels agree on shape in **every** pair, zero
mismatches. Worth knowing before writing a fixed-size loader; not a defect.

**Not a finding:** 24 sample indices between 1 and 916 have neither an image nor
a label. The numbering is simply not contiguous.

---

## 4. Labels are not binary, and the surface class is not always index 1

`labelscope` detects the sheet-like class rather than assuming it: thin, planar,
and not filling the volume. In the Kaggle release that is class 1 (median
thickness 2.0 voxels, worst-component planarity 0.007), with class 2 a single
bulky region occupying 19–85% of the patch.

Any metric computed over `label > 0` here — foreground fraction, sheet thickness,
class balance — averages a 2-voxel sheet together with a region hundreds of voxels
thick, and means nothing.

**Class 2 is also useless as an orientation reference**, which is less obvious. It
wraps the writing surface on *both* sides, so it cannot say which way is out.
`orientation_reference_quality` scores it at 0.009–0.281 out of 1 on real
patches. Offsets oriented against it carry a sign that flips arbitrarily from
patch to patch — which is why `labelscope align` orients by the scan instead.

---

## 5. Label topology is clean

Over 227 label volumes, measured with `labelscope scan --deep`:

| metric | median | p90 | max |
|---|---|---|---|
| surface fraction | 0.046 | 0.081 | 0.137 |
| local thickness (median) | 2.00 | 2.00 | 2.00 |
| local thickness (p95) | 3.46 | 4.00 | 4.00 |
| connected components | 7 | 12 | 18 |
| fragment fraction | 0.000 | 0.000 | 0.000 |
| worst-component planarity | 0.011 | 0.029 | 0.083 |
| junction fraction | 0.0025 | 0.0047 | 0.0085 |

Uniform `[0, 1, 2]` class scheme in all 227; surface class detected as 1 in all
227. No speckle, no blobs, no fat thickness tail — nothing here suggests labels
that have bridged two windings. A negative result, and worth having: these
particular failure modes can be ruled out by measurement rather than by hope.
