**Repo:** ScrollPrize/villa · **Type:** issue (data / methodology)

---

### Title

Surface labels sit ~2.3 voxels off the sheet's density maximum — they mark the recto face, not a centre-line

### Body

Measuring the offset between the labelled writing surface and the local CT
density maximum, along the surface normal, on 51 patches of the Kaggle
surface-detection release (`hf://buckets/scrollprize/datasets/surfaces/kaggle`):

* 48 of 51 patches have enough sheet contrast at the labelled surface to measure
  at all (contrast ≥ 2× voxel noise).
* In those 48 the density maximum is not under the label: median |offset|
  **2.34 voxels**, interquartile range 1.79 to 3.02, ≥1 voxel in 45 of 48
  patches, per-patch 95% bootstrap intervals typically ±0.05.
* The sign is a convention — normals are oriented toward the denser side, since
  the release ships no field that can say which way is out — so "positive in
  47 of 48" is close to tautological and is not the evidence. The evidence is the
  magnitude, its consistency, and a signal-to-noise of 2.0-14.8.
* Sheets are 7.5–9.3 voxels thick (FWHM of the mean profile), and the local
  winding spacing measured from the scans runs 10.5–29.0 voxels.

So the offset is about half a sheet thickness, in a consistent direction. That is
what a label on the recto *face* should look like, and I do not think it is an
error in the labels.

It is worth stating explicitly because of what sits downstream:

* anything treating these labels as a **sheet centre-line** — meshing, surface
  fitting, normal estimation — inherits a systematic ~2.3 voxel bias over a
  winding period of 10–29 voxels;
* **ink sampling symmetric in ±t about the label** is not symmetric about the
  sheet: it reaches ~2.3 voxels further into the void on one side, and ~2.3
  voxels less far into the papyrus on the other.

Locally the variation is larger than the systematic part: per 64³ cube of
labelled surface the median absolute offset is 2.43 voxels and the 90th
percentile 4.66.

**A methodological note that cost me a day, in case it saves someone else one.**
The obvious estimator — walk the normal from each labelled voxel, take the
nearest intensity maximum — does not work on carbonised papyrus, and fails
silently. Its answer tracks the search radius rather than any displacement
(median |offset| 0.70 vx at R=2, 1.53 at R=4, 3.09 at R=9 on the same fixed
label), because along one normal there are fibre maxima, both faces of the sheet
and the neighbouring wrap. Averaging profiles over a neighbourhood before
locating the peak fixes it; the aggregated estimator moves by 0.035 vx across
R=5…16.

Three further traps, all of which produced confident wrong numbers before being
caught:

1. Normals from the gradient of a distance field are **degenerate on a thin open
   surface** — the field has a minimum at the sheet, so its gradient there is
   exactly zero. Local PCA over the label point cloud works.
2. Normals must be **oriented consistently across the surface**, not pointwise
   against a reference. Orienting each voxel against the shipped void class gave
   normals that agreed with their own 64³ cell mean only 54–64% of the time;
   averaged profiles then half-mirror and every offset is dragged toward zero.
   MST propagation (Hoppe et al. 1992) takes that to 100%.
3. **The void class in the release cannot define "outward".** It wraps the
   writing surface on both sides; scored on a 0–1 asymmetry measure it comes out
   at 0.009–0.281. Offsets oriented against it carry a per-patch arbitrary sign.
   Orienting by the scan — positive toward the denser side — gives every patch
   the same convention.

Reproduce:

```bash
labelscope align --images kaggle/images --labels kaggle/labels --overlays 10 --out audit/
```

Tool: [`labelscope`](https://github.com/rodriguescarson/labelscope), MIT. The
estimator is validated against planted sub-voxel displacements on a synthetic
built to fail the naive measure the way real data does.
