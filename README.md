# labelscope

**A diagnostic scope for Vesuvius Challenge surface-label datasets.**

The [Open Problems](https://scrollprize.org/2026_open_problems) post names label
quality as one of the main unwrapping bottlenecks, and describes the failure
modes in words:

> Those labels often come from human-generated meshes or annotations —
> enormously valuable, but approximate. They may wiggle. They may drift slightly
> off the true surface. They may avoid the most ambiguous regions.

Those are three testable claims, and nothing in the pipeline currently measures
them. `labelscope` turns each one into a number you can compute on a dataset you
already have, on a CPU, in minutes — and, where it finds a problem, emits the
artifact that fixes it rather than just a complaint.

It answers three questions:

| Question | Command | What comes out |
|---|---|---|
| Is my train/validation split honest? | `labelscope leakage` | leak percentage + a drop-in `splits_final.json` that removes it |
| Are the labels internally consistent? | `labelscope scan` | class-scheme, shape, encoding and topology census; ranked worst volumes |
| Do the labels sit where the CT says the surface is? | `labelscope align` | signed offset from the label to the scan's own ridge, in voxels |

---

## Install

```bash
pip install -e .            # Python ≥3.9, CPU only, no GPU required
labelscope --version
```

Optional: `pip install -e '.[zarr]'` to read Zarr and OME-Zarr as well as 3-D TIFF.

---

## 1. `labelscope leakage` — is the validation score real?

Vesuvius surface datasets are cut from a handful of scroll volumes on a sliding
window whose stride is **smaller than the patch**, so neighbouring patches share
voxels. nnU-Net's default is a *random* 5-fold split over cases. Random splits
assume the cases are independent. These are not.

```bash
labelscope leakage --labels nnUNet_raw/DatasetXXX/labelsTr --patch-size 300 --k 5 --out audit/
```

```
1754 patches, 7527 overlapping pairs (92.4% of patches)
random 5-fold: 91.9% of validation patches leak
blocked split: val folds [351, 351, 351, 351, 350], 0 residual leaks, 16.5% of training patches dropped to buffer
```

The command writes `splits_final.json` in nnU-Net's own format. Drop it into
`nnUNet_preprocessed/DatasetXXX/` and the next run uses a split where no
validation patch touches a training patch.

Two strategies, both leak-free:

* `--mode block` (default) — **spatial block cross-validation**. Each source
  volume is tiled into blocks larger than the patch and whole blocks are dealt to
  folds, so validation folds stay near-equal in size. Any training patch that
  would still touch a validation patch is dropped into a buffer zone. This costs
  a few percent of the training set and is the standard remedy for spatially
  autocorrelated data.
* `--mode component` — whole connected components of the overlap graph go to one
  fold. Nothing is discarded, but fold sizes follow component sizes and can be
  very uneven.

`--buffer N` widens the definition of contamination: two patches that stop `N`
voxels short of each other still count as neighbours, because the same papyrus
sheet almost certainly runs through both.

If the volumes are not on this machine, `--names-file names.txt` works too — the
check only needs the patch names, which carry the coordinates.

---

## 2. `labelscope scan` — what is actually in this release?

```bash
labelscope scan --labels labelsTr --images imagesTr --out audit/
labelscope scan --labels labelsTr --headers-only --out audit/   # never decodes voxels
```

Per volume it reports shape, dtype, on-disk encoding, the class values actually
used, and — per class, not lumped together — foreground fraction, local
thickness, connected components, fragment fraction and planarity.

**Labels here are not binary.** The releases seen so far carry a thin
writing-surface class alongside a bulky region class. Averaging sheet thickness
over `label > 0` produces a number that means nothing, so every metric is
computed per class and the sheet-like class is *identified* rather than assumed:
the thin, planar, non-space-filling one.

What the metrics are for:

* **thickness** (twice the Euclidean distance transform) — a traced writing
  surface is a couple of voxels thick and tightly distributed. A fat upper tail
  is where a label has swallowed the gap between two windings.
* **planarity** (smallest PCA eigenvalue of a component, as a share of the total)
  — 0 for a perfect plane, 1/3 for an isotropic blob.
* **fragment fraction** — how much of the label is in pieces too small to be a
  sheet.
* **branch points** (`--deep`, skeletonises each label) — a single sheet
  skeletonises with no interior junctions. Junctions mean the label forks, most
  often because it has bridged two windings — the error the Open Problems post
  calls unrecoverable downstream, because "a small local error can send a traced
  mesh onto the wrong wrap entirely".

---

## 3. `labelscope align` — does the label sit on the ridge?

```bash
labelscope align --images imagesTr --labels labelsTr --out audit/
```

For each sampled labelled voxel, `labelscope` estimates the surface normal by
local PCA over the label point cloud, walks along that normal with trilinear
interpolation of the CT, finds the intensity extremum with sub-voxel parabolic
refinement, and records the signed distance from the label to it.

* `median_abs_offset` — how far the label is from the scan's own ridge, in voxels.
* `mean_signed_offset` — **systematic** bias. Annotator wobble averages to zero;
  a non-zero mean does not. Sign is only meaningful when an orientation
  reference is supplied (the air/void class serves), and the report says whether
  it was.
* `frac_flat_support` — share of labelled surface sitting on a ridge
  indistinguishable from noise. Prominence is judged against the volume's own
  robust noise σ, not against its dynamic range: a hazy, compressed patch has a
  *small* dynamic range, so normalising by it would hide exactly the degradation
  we are looking for.
* `hf_energy_norm` — a label-free difficulty proxy for the patch, standing in for
  the compressed-region haze the Open Problems post describes.

The last two together answer a question nobody currently measures: **are the
labels worse exactly where the scroll is hardest?** The report bins patches by
difficulty and shows offset, prominence and label coverage per bin.

### Why the offsets are trustworthy

The alignment metric is validated against planted ground truth, not just
eyeballed. `tests/test_alignment.py` builds synthetic sheets at known sub-voxel
positions and asserts that the reported offset recovers the displacement, that a
perfectly placed label reads zero, that noise widens the spread without moving
the bias, that dark-on-bright scans are handled, and that a label on featureless
data is flagged rather than given a confident offset.

```bash
pytest -q          # 21 tests
```

Writing those tests found a real bug: the first implementation took normals from
the gradient of a distance field, which is degenerate on a thin open surface —
the field has a minimum at the sheet, so its gradient there is exactly zero. The
local-PCA estimator that replaced it is what ships.

---

## Findings

`findings/` holds the reports this tool produced on public Vesuvius data, with
the exact commands used. See [findings/README.md](findings/README.md).

---

## Design notes

* **CPU only, no GPU, no training.** Every check runs on a laptop.
* **Header-only mode.** `scan --headers-only` inventories a release without
  decoding a voxel, so a multi-terabyte set can be checked before it is pulled.
* **Standard formats in, standard formats out.** Reads 3-D TIFF stacks
  (nnU-Net `imagesTr`/`labelsTr`) and Zarr/OME-Zarr; writes CSV, JSON, nnU-Net
  `splits_final.json`, and a single-file HTML report with no external assets.
* **Nothing is assumed about class indices.** The surface class is detected.
* **Unpaired volumes are reported, not skipped.** A missing label is a finding.

## Licence

MIT.
