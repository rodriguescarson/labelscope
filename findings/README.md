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

## 2. Half the Kaggle surface release ships uncompressed

**Command**

```bash
labelscope scan --labels kaggle/labels --images kaggle/images --out findings/kaggle_surfaces/
```

**What it found**

Of 892 label volumes, 458 are byte-identical in size at exactly 32,821,210 bytes
— 320³ uint8 with `COMPRESSION.NONE`. The other 434 carry the same kind of
content compressed, and average under a megabyte.

The contents are *not* different: sampling both groups, class 1 is a ~2-voxel
thick, highly planar writing surface in both, and class 2 is a single bulky
region in both. It is purely a writer setting that differs.

**Why it matters**

Small, and worth one line of a release script rather than a discussion: it is
roughly 15 GB of avoidable transfer on a release that is otherwise about 1.5 GB
of labels, for a project whose users already stream tens of terabytes of CT.

**Not a finding:** 24 sample indices between 1 and 916 have neither an image nor
a label. Numbering is simply not contiguous — nothing is missing or orphaned.
`labelscope scan` reports unpaired volumes precisely so that this distinction
can be made instead of assumed.

---

## 3. Labels are not binary, and the surface class is not always index 1

`labelscope` detects the sheet-like class rather than assuming it: the class that
is thin, planar and does not fill the volume. In this release that is class 1
(median thickness 2.0 voxels, worst-component planarity 0.007) with class 2 a
single bulky region occupying 19–85% of the patch depending on where the patch
sits.

Any metric computed over `label > 0` on this data — foreground fraction, sheet
thickness, class balance — is averaging a 2-voxel sheet together with a region
hundreds of voxels thick, and means nothing. That is a small trap and an easy one
to walk into.
