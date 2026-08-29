# Quickstart

Three commands, no GPU, no training, no account. Each one is a complete check
that answers a question about a dataset you already have, and writes CSV, JSON
and a single-file HTML report you can open offline.

## Install

```bash
pip install -e ".[zarr]"      # Python >= 3.9
labelscope --version
```

Or without touching your environment:

```bash
docker build -t labelscope .          # ~970 MB, CPU only
docker run --rm -v "$PWD/sample:/data:ro" -v "$PWD/out:/out" \
  labelscope scan --labels /data/labelsTr --images /data/imagesTr --out /out
```

On the six volumes committed in `sample/`, that prints:

```
volumes: 6
distinct label shapes: 4  {'(300, 300, 300)': 1, '(172, 172, 172)': 1,
                           '(236, 236, 236)': 2, '(170, 170, 170)': 2}
label compressions: {'LZW': 6}
modal class scheme: [0, 1] (6/6)
detected surface class: {'1': 6}
```

and writes `scan.csv`, `scan_summary.json` and a single-file `scan.html` you can
open offline. Four different patch shapes in six volumes is
[finding 2](../findings/README.md) in miniature — it is the reason this command
exists.

## 1. Is the validation score real?

Patch datasets cut on a sliding window share voxels between patches, and a random
k-fold split then puts the same voxels on both sides.

```bash
labelscope leakage --labels nnUNet_raw/DatasetXXX/labelsTr --k 5 --out audit/
```

Writes `audit/splits_final.json` — a drop-in nnU-Net split with the overlap
removed — plus the leak percentage it started from. If the number is small, you
have learned that in about a minute; if it is not, the fix is already written.

## 2. Do the labels sit where the scan says the surface is?

```bash
labelscope align --images nnUNet_raw/DatasetXXX/imagesTr \
                 --labels nnUNet_raw/DatasetXXX/labelsTr \
                 --jobs 8 --out audit/
```

Reports the signed offset from each label to the scan's own ridge, per 64-voxel
cell and globally, with a bootstrap interval. Cells whose profile has no peak in
the window, or no signal above the noise, are counted and excluded rather than
assigned a number.

## 3. Has this traced surface jumped to a neighbouring wrap?

```bash
labelscope sheetswitch --mesh seg/*.tifxyz --volume https://.../volume.zarr \
                       --remote --window 160 --out audit/
```

`--remote` streams the scan chunk by chunk, so a segment costs about 2–4 GB of
transfer regardless of how large the volume is. `.obj` and `.ply` work too.

The two questions it refuses to answer are as important as the one it answers:

* below two grid steps per winding it reports `resolution_adequate: false` and
  moves its findings to `seams_unreliable`, because the seam is not visible at
  that scale;
* where the scan is masked out and the surface reads flat, it reports
  `dip_degenerate: true` and reports no seams, because a null with no spread in
  it has no z-scores in it.

## 4. Correct what you found

```bash
labelscope regularise --images .../imagesTr --labels .../labelsTr \
                      --out .../labelsTr_reg --cell 64
```

Writes a corrected copy plus a manifest saying what happened to every patch,
including the ones it refused. Each cell's target is the patch's own global
offset, not zero, so a consistently placed label set comes back unchanged.

## Running it over a whole scroll, or every scroll

```bash
# every published surface of one volume, 8 at a time
JOBS=8 WINDOW=160 CACHE= scripts/fleet_sheetswitch.sh --list surfaces.txt

# the control: the same surfaces with a whole winding planted in half of each
JOBS=8 WINDOW=160 CACHE= PLANT=1 scripts/fleet_sheetswitch.sh --list surfaces.txt

# several scrolls, each against the scan it was traced on
scripts/fleet_sheetswitch.sh --pairs manifest.tsv
```

`scripts/corpus_manifest.py` builds that manifest from a bucket listing, and
`scripts/pair_by_extent.py` pairs the surfaces whose directory name does not
carry a volume id.
