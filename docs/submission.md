# Vesuvius Challenge — August 2026 Progress Prize submission

Form: https://docs.google.com/forms/d/e/1FAIpQLSev2vJobu521iB6OuyehDktzYTEo131F4iUGwt3Qxa9a1fk6A/viewform

**Full name:** Carson Rodrigues
**Team:** individual
**URL:** https://github.com/rodriguescarson/labelscope (MIT)

---

## Field 5 — how this increases the probability of reading complete scrolls

*(draft; ~400 words)*

The Open Problems post says label quality is now one of the main unwrapping
bottlenecks, and describes the failure modes in words: labels "may wiggle, may
drift slightly off the true surface, may avoid the most ambiguous regions."
Nothing in the pipeline measures any of that. `labelscope` is a CPU tool that
turns each sentence into a number you can compute on a dataset you already have,
in minutes, without a GPU or a training run.

Running it on the public surface data produced three things worth acting on.

**1. The surface labels mark a face, not a centre-line.** On the Kaggle
surface-detection release, 48 of 51 patches have enough sheet contrast at the
labelled surface to measure. In those 48 the sheet's local CT density maximum is
not under the label: median |offset| **2.34 voxels**, IQR 1.79–3.02, ≥1 voxel in
45 of 48 patches, per-patch 95% bootstrap intervals around ±0.05, against sheets
7.5–9.3 voxels thick and a winding spacing of 10.5–29 voxels. That is not an error — a
writing surface *is* a face — but anything treating these labels as a sheet
centre-line inherits a systematic ~2.3 voxel bias, and ink sampling symmetric in
±t about the label is not symmetric about the sheet. Locally the wander is larger
than the systematic part: per 64³ cube, median |offset| 2.43 voxels, p90 4.66.

**2. `Dataset059`, the surface training set cited in villa#191, is not what its
filenames imply.** It ships four patch sizes — 170³, 172³, 236³, 300³ — plus one
364³, and nothing in a filename or in `dataset.json` says which is which. I know
this trap is easy to fall into because I fell into it: assuming a uniform 300³
turned 28 overlapping patch pairs into 7,527 and a 1.5% train/validation leak
into 91.9%. The tool now reads every volume's real shape by default, and both
that retraction and a second one — eight volumes I reported as malformed turned
out to be truncated by my own downloader — are written into the findings rather
than quietly dropped.

**3. Fifteen gigabytes of the Kaggle release is uncompressed padding** — 487 of
892 label volumes at `COMPRESSION.NONE`, 15.52 GB of a 45 GB release, where the
compressed half of the same release averages 0.87 MB. Establishing that cost
about 1.8 MB of transfer: `labelscope scan --headers-only` inventories a remote
release from roughly a kilobyte per volume, so a multi-terabyte set can be
checked before it is pulled.

The estimator behind (1) is the substance. The obvious version — walk the normal
from each labelled voxel, take the nearest intensity maximum — fails silently on
carbonised papyrus, and its answer tracks the search radius rather than any
displacement. Four separate traps had to be fixed before the number meant
anything; all four are documented, and all four are regression tests, on a
synthetic built to fail the naive measure the way real data does.

---

## Checklist before submitting

- [ ] repo public at github.com/rodriguescarson/labelscope, MIT licence present
- [ ] `pytest -q` green on a clean clone
- [ ] `pip install -e .` works from the published repo
- [ ] findings/ committed, including the retraction in §0
- [ ] upstream issues filed on ScrollPrize/villa (docs/upstream/)
- [ ] posted in the Vesuvius Discord
- [ ] form submitted before 2026-08-31 23:59 PT (2026-09-01 12:29 IST)
