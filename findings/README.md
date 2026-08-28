# Findings

Everything here was produced by `labelscope` on public Vesuvius Challenge data,
on a laptop CPU. Each section gives the exact command, so any of it can be
re-run and disagreed with.

Data used:

| Dataset | Where | What it is |
|---|---|---|
| `Dataset059_s1_s4_s5_patches_frangiedt` | [dl.ash2txt.org](https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/Dataset059_s1_s4_s5_patches_frangiedt/) | 1,754 volume/label pairs; the surface-model training set cited in [villa#191](https://github.com/ScrollPrize/villa/issues/191) |
| Kaggle surface-detection release | `hf://buckets/scrollprize/datasets/surfaces/kaggle` | 892 CT/label pairs at 320³ |

---

## 1. A random split over these patches is not a valid split

**Command**

```bash
labelscope leakage --labels Dataset059_s1_s4_s5_patches_frangiedt/labelsTr \
                   --patch-size 300 --k 5 --out findings/dataset059/
```

**What it found**

The 1,754 patches are cut from three scrolls on a sliding window whose stride is
smaller than the patch. In `s1` the stride goes down to 64 voxels on a 300-voxel
patch; in `s4` it is a uniform 256. The result:

| | |
|---|---|
| overlapping patch pairs | **7,527** |
| patches touching at least one other patch | **1,621 / 1,754 (92.4%)** |
| median shared volume with the nearest neighbour | **36.0%** |
| worst shared volume | **57.3%** |

No `splits_final.json` ships with the dataset, so nnU-Net generates its own.
`nnUNetTrainer.do_split()` calls `generate_crossval_split(sorted_keys, seed=12345,
n_splits=5)`, which is `sklearn KFold(5, shuffle=True, random_state=12345)` over
the sorted case names — a random split, which assumes the cases are independent.
Reproducing that exact split:

| | |
|---|---|
| validation patches sharing voxels with a training patch | **1,612 / 1,754 (91.9%)** |
| mean shared volume per validation patch | **23.1%** |
| worst | **57.3%** |

Averaged over 200 random shuffles the figure is 91.9% ± 0.2, so this is a
property of the data, not of one unlucky seed.

**Why it matters**

The validation Dice from that split is not measuring generalisation to unseen
papyrus. It is measuring, in large part, reconstruction of voxels already seen in
training. That number is what selects the best checkpoint and what loss-variant
comparisons — medial-surface loss, skeleton recall, and the rest of
[villa#191](https://github.com/ScrollPrize/villa/issues/191)'s territory — are
decided on. A leaked metric does not merely read too high; it can rank two
methods in the wrong order, because a method that memorises local texture is
rewarded exactly where the leak is largest.

This says nothing about whether the trained models are good. It says the number
used to compare them is not measuring what it is assumed to measure.

**The fix, included**

`findings/dataset059/splits_final.json` is a drop-in replacement in nnU-Net's own
format. Copy it to `nnUNet_preprocessed/Dataset059_.../splits_final.json` and the
next run uses it instead of generating one.

| | random 5-fold | blocked 5-fold (shipped) |
|---|---|---|
| validation fold sizes | 351/351/351/351/350 | 351/351/351/351/350 |
| validation patches leaking | 1,612 | **0** |
| training patches per fold | 1,403 | 1,055–1,152 |

Spatial block cross-validation: whole blocks of scroll go to one fold, and any
training patch still touching a validation patch is dropped into a buffer zone.
That buffer costs 16.5% of the training patches per fold. That is the price of
an honest number, and it is the standard remedy for spatially autocorrelated
data.

---

## 2. Fifteen gigabytes of the Kaggle surface release is uncompressed padding

**Command**

```bash
labelscope scan --labels kaggle/labels --remote --headers-only --out findings/kaggle_surfaces/
```

`--headers-only` over HTTP reads roughly a kilobyte per volume: the TIFF byte-order
mark, the first image directory, and the content length. That was enough to
inventory all **1,784 volumes** of the release — every image and every label —
for about **1.8 MB of transfer**, against 45 GB to download them.

**What it found**

| | files | on disk |
|---|---|---|
| labels, `COMPRESSION.NONE` | **487** | **15.52 GB** |
| labels, `COMPRESSION.LZW` | 405 | 0.36 GB (median 0.87 MB) |
| images, `COMPRESSION.LZW` | 873 | — |
| images, `COMPRESSION.NONE` | 19 | — |
| | | **45 GB total** |

The uncompressed labels are not different in content. Sampling both groups, class
1 is a ~2-voxel-thick, highly planar writing surface in both (median local
thickness 2.0, worst-component planarity 0.007) and class 2 is a single bulky
region in both. Only the writer setting differs — and the compressed half of the
same release shows what the other half would cost: a median of 0.87 MB against
32.82 MB.

**Why it matters**

Small, and worth one line of a release script rather than a discussion: re-encoding
those 487 labels with the compression the other 405 already use takes roughly
15 GB off a 45 GB release, for a project whose users already stream tens of
terabytes of CT.

---

## 3. The release mixes three patch sizes — and that is fine, but you have to know

Plane shapes across all 892 pairs:

| shape | pairs |
|---|---|
| 320 × 320 | 840 |
| 256 × 256 | 51 |
| 384 × 384 | 1 |

**Images and labels agree on shape in every single pair — zero mismatches.** So
this is deliberate structure, not corruption, and it is recorded here as a fact a
consumer needs rather than as a defect. Any loader that hard-codes 320³ will
break on 52 of the 892 pairs, and the single 384³ pair will surprise anything that
handles only the two common sizes.

**Not a finding:** 24 sample indices between 1 and 916 have neither an image nor a
label. The numbering is simply not contiguous — nothing is orphaned or missing a
pair. `labelscope scan` reports unpaired volumes precisely so this distinction can
be made rather than assumed.

---

## 4. Labels are not binary, and the surface class is not always index 1

`labelscope` detects the sheet-like class rather than assuming it: the class that
is thin, planar and does not fill the volume. In this release that is class 1
(median thickness 2.0 voxels, worst-component planarity 0.007) with class 2 a
single bulky region occupying 19–85% of the patch depending on where the patch
sits.

Any metric computed over `label > 0` on this data — foreground fraction, sheet
thickness, class balance — is averaging a 2-voxel sheet together with a region
hundreds of voxels thick, and means nothing. That is a small trap and an easy one
to walk into.
