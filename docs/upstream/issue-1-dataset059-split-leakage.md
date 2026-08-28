**Repo:** ScrollPrize/villa · **Type:** issue · **Relates to:** #191

---

### Title

`Dataset059` surface patches overlap: the split nnU-Net generates puts 92% of validation patches in contact with training voxels

### Body

[#191](https://github.com/ScrollPrize/villa/issues/191) points at
[`Dataset059_s1_s4_s5_patches_frangiedt`](https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/Dataset059_s1_s4_s5_patches_frangiedt/)
as the surface-model training set. The patches in it are cut from three scrolls
on a sliding window whose stride is smaller than the patch, so neighbouring
patches share voxels:

* stride in `s1` goes down to **64** voxels on a **300**-voxel patch
* stride in `s4` is a uniform **256** on a 300-voxel patch
* stride in `s5` goes down to **190**

Measured over all 1,754 patches (coordinates come straight from the filenames):

| | |
|---|---|
| overlapping patch pairs | 7,527 |
| patches touching at least one other | 1,621 / 1,754 (92.4%) |
| median shared volume with nearest neighbour | 36.0% |
| worst | 57.3% |

No `splits_final.json` ships with the dataset, so `nnUNetTrainer.do_split()`
generates one: `generate_crossval_split(sorted_keys, seed=12345, n_splits=5)`,
i.e. `sklearn KFold(5, shuffle=True, random_state=12345)` over the sorted case
names. Reproducing that exact split against the overlap graph:

| | |
|---|---|
| validation patches sharing voxels with a training patch | **1,612 / 1,754 (91.9%)** |
| mean shared volume per validation patch | 23.1% |
| worst | 57.3% |

Averaged over 200 random shuffles it is 91.9% ± 0.2, so it is a property of the
patch grid rather than of one seed.

**Why I think this is worth fixing rather than noting.** The validation Dice
from that split is largely measuring reconstruction of voxels already seen in
training, and that number is what selects checkpoints and what decides between
loss variants — the medial-surface / skeleton-recall comparison in #191 included.
A leaked metric does not only read high; it can rank two methods in the wrong
order, because the method that memorises local texture is rewarded most exactly
where the leak is largest. Nothing here says the trained models are bad — only
that the number used to compare them is not measuring what it is taken to
measure.

**A drop-in fix.** I wrote a small MIT-licensed tool,
[`labelscope`](https://github.com/rodriguescarson/labelscope), that measures this
and emits a replacement split in nnU-Net's own format:

```bash
labelscope leakage --labels Dataset059_s1_s4_s5_patches_frangiedt/labelsTr \
                   --patch-size 300 --k 5 --out audit/
# 1754 patches, 7527 overlapping pairs (92.4% of patches)
# nnU-Net default split: 91.9% of validation patches leak (random shuffles: 91.9%)
# blocked split: val folds [351, 351, 351, 351, 350], 0 residual leaks,
#                16.5% of training patches dropped to buffer
```

Copy `audit/splits_final.json` into
`nnUNet_preprocessed/Dataset059_.../splits_final.json`:

| | random 5-fold | blocked 5-fold |
|---|---|---|
| validation fold sizes | 351/351/351/351/350 | 351/351/351/351/350 |
| validation patches leaking | 1,612 | **0** |
| training patches per fold | 1,403 | 1,055–1,152 |

It is spatial block cross-validation: whole blocks of scroll go to one fold, and
a training patch that would still touch a validation patch is dropped into a
buffer zone rather than silently leaking. The buffer costs 16.5% of the training
patches per fold — the usual price of an honest number on spatially
autocorrelated data.

The same check applies to any patch dataset whose names carry `z`/`y`/`x`, so if
other releases are cut the same way it is one command to know.

Happy to open a PR adding the generated `splits_final.json` alongside the dataset,
or a short note in the dataset README, if either is useful.
