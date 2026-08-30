# The corpus pass, its control, and what is not here

Produced 2026-08-29/30 on a single RunPod A40 (`8ngrs76tm6e41l`, CA-MTL-1), which
has since been torn down. Everything needed to check the claims is in this
directory; everything that is not here is regenerable by one command, and the
command is given.

## What is here

| path | what |
|---|---|
| `corpus_summary.json` | per-scroll totals and the paired-control statistics |
| `corpus_by_scroll.csv` | the same table, flat |
| `per_surface/real/*.csv` | one row per surface, measured as published |
| `per_surface/plant/*.csv` | the same surfaces with one winding planted in half of each |
| `eval_seed0.json`, `eval_seed1.json` | label-free evaluation, 351 held-out patches per seed, per-patch and summary |
| `inputs/corpus_raw.tsv` | every published tifxyz paired with the volume it was traced on (722 rows) |
| `inputs/corpus_manifest.tsv` | the 258 selected, one per segment, with the volume URL |
| `inputs/scrolls.txt` | every scroll prefix in the open-data bucket |
| `../regularise/manifest.csv` | what the regulariser did to each of the 1,754 patches |
| `../logs/` | the sweep and training logs as they ran |
| `../../experiments/runs/*/` | the four model checkpoints, configs and curves |

## What is not here, and how to get it back

**The regularised label set** (`labelsTr_reg`, 1,754 volumes, ~19 GB). Too large
to commit and not worth a release of its own given the result was negative:

```bash
labelscope regularise --images .../Dataset059/imagesTr --labels .../Dataset059/labelsTr \
                      --out .../labelsTr_reg --cell 64
```

About 20 s per patch; the sweep script `scripts/reg_shards.sh` pattern (24 shell
shards) does the set in ~25 minutes on 96 cores. `../regularise/manifest.csv`
records what the run that produced the reported numbers did to every patch, so a
regeneration can be checked against it.

**The 258 fetched meshes** (~42 GB) and **the chunk cache**. Both are pure
downloads:

```bash
python scripts/corpus_manifest.py corpus_raw.tsv --out manifest.tsv --keys keys.tsv
# then fetch each key's meta.json, x.tif, y.tif, z.tif
```

**The Dataset059 images and labels** (~37 GB) come from
`dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/Dataset059_s1_s4_s5_patches_frangiedt/`
via `scripts/fetch_dataset.py`, which verifies what it wrote rather than what was
promised.

## Reproducing the numbers

```bash
CACHE=cache JOBS=8 WINDOW=160         scripts/fleet_sheetswitch.sh --pairs manifest.tsv
CACHE=cache JOBS=8 WINDOW=160 PLANT=1 scripts/fleet_sheetswitch.sh --pairs manifest.tsv
python scripts/corpus_report.py --real corpus_real --plant corpus_plant --out findings/corpus
```

Cache per scroll and clear it between them: caching across the whole corpus needs
more than 300 GB, and running with no cache at all is about six times slower
because adjacent windings share chunks.

## Cost

The whole thing — corpus sweep both passes, 1,754-patch regularisation, four
training runs, two evaluations — ran for about $14 of GPU and disk.

## Open, for the next round

* **The triangular-mesh detector has never completed a run against a real
  volume.** The reader is verified on a real published OBJ (`PHerc0172`,
  386,108 vertices, 774,400 faces, edge length 20.13 voxels) and the detector is
  tested against planted displacements on synthetic triangular meshes, but the
  end-to-end run on `20250917143559-w062_..._normalized.obj` was interrupted three
  times and never produced a result. The natural next step is that run plus a
  cross-format agreement check: the same physical surface as tifxyz and as OBJ
  should score the same.
* **Nothing has been run on a confirmed sheet switch.** Every validation is
  against a displacement we planted ourselves. A surface someone *knows* jumps
  wraps is the single most valuable thing this tool has not been tested against.
* **The five surfaces where planting a winding lowered the score** deserve a look;
  they are the clearest lead on where the seam statistic breaks down.
