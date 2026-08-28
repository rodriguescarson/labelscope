**Repo:** ScrollPrize/villa · **Type:** issue (data QA) · **Relates to:** #191

---

### Title

`Dataset059_s1_s4_s5_patches_frangiedt`: four patch sizes in one directory, undocumented

### Body

[#191](https://github.com/ScrollPrize/villa/issues/191) points at
[`Dataset059_s1_s4_s5_patches_frangiedt`](https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/Dataset059_s1_s4_s5_patches_frangiedt/)
as the surface-model training set. Reading the header of every volume in
`labelsTr` (1,754 of them, header-only, no decoding):

| shape | count | scroll |
|---|---|---|
| 172³ | 840 | s1 |
| 236³ | 753 | s1 (177), s4 (576) |
| 300³ | 121 | s1 |
| 170³ | 39 | s5 |
| 364³ | 1 | s1 |

Filenames encode only an origin (`s1_z10240_y2560_x2560`), so nothing in the name
or in `dataset.json` says which of these a given patch is. That is easy to trip
over — I did, and computed every patch overlap in this dataset wrong by assuming
300³ before checking. nnU-Net itself is fine with variable sizes; anything that
crops, tiles or computes patch geometry from the names is not.

**Good news from the same pass:** where patches genuinely do overlap — 28 pairs
once real sizes are used, not the 7,527 that assuming 300³ produces — their labels
agree on the shared voxels at a **median IoU of 0.999**, minimum 0.993, none below
0.9. The release does not carry two different answers for one scroll voxel.

Reproduce:

```bash
labelscope leakage --labels Dataset059_s1_s4_s5_patches_frangiedt/labelsTr \
                   --measure-seen --consistency 250 --out audit/
```

Tool: [`labelscope`](https://github.com/rodriguescarson/labelscope), MIT. It reads
each volume's real shape by default; `--assume-patch-size` is the flag that
reproduces my original mistake, kept so the difference is testable.
