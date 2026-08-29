# Draft post for the Vesuvius Discord

Carson to post — needs his account. Suggested channel: whichever of
`#tools` / `#segmentation` / `#general` the community uses for tool releases.
Keep it short; the repo does the talking.

---

**labelscope — a CPU tool for checking surface labels and traced surfaces against the scan itself**

https://github.com/rodriguescarson/labelscope (MIT)

I kept wanting to ask questions about the published data that I couldn't answer
without writing something, so here it is. It runs on a laptop, no GPU, no
training, and reads Zarr/OME-Zarr, tifxyz, and `.obj`/`.ply`.

Four questions it answers:

- **Is my train/val split honest?** Patch datasets cut on a sliding window share
  voxels; `leakage` reports the percentage and emits a drop-in `splits_final.json`
  with the overlap removed.
- **Do the labels sit where the scan says the surface is?** `align` measures the
  offset from the label to the CT's own ridge, per 64-voxel cell, with a bootstrap
  interval. Over 2,568 pairs in two independent releases it comes out at 2.285 and
  2.576 voxels — the labels mark the recto face, consistently, in both.
- **Has this traced surface jumped to a neighbouring wrap?** `sheetswitch` looks
  for the seam — the line of grid edges that has to cross the gap between two
  wraps. This is the failure the spiral satisfaction metric scores as *exactly
  zero* change (villa#1621).
- **Can I correct what I found?** `regularise` writes a corrected label set that
  removes each patch's internal disagreement while preserving the recto-face
  convention.

Two things I'd rather lead with than the features.

**It refuses to answer, twice, and reports the refusals as data.** Below two grid
steps per winding the seam is not visible and it says so. And where the scan is
masked out the surface reads flat, the spread collapses, and a robust z-score
computed on that null will report whatever rounding noise it finds — so it checks
the scan at the surface first.

**That second refusal exists because the tool's loudest result was wrong.** On the
first corpus pass the highest score anywhere was z = 12.60, past threshold, past
the resolution gate. Its median line dip was 0.000 grey levels. It was a window
sitting in masked-out air. Eleven of 56 surfaces were in the same state. Full
write-up in `findings/`, along with two earlier retractions, because they are
mistakes anyone auditing this data can make.

Every claim in `findings/` has the command next to it. Bug reports and
disagreement very welcome — particularly if you have a surface you *know* jumps
wraps, since a confirmed positive is the one thing I have not been able to test
against.

---

## Notes for Carson

- Post after the form goes in, not before — the rules say not to make a Grand
  Prize discovery public early, and while this is a Progress Prize submission the
  safe order is submit first.
- If anyone replies with a surface they believe carries a real sheet switch, that
  is the single most valuable thing we can get: every validation so far is against
  a planted displacement, never a confirmed one in the wild.
- Registering on the Discord is also a stated requirement for the Grand Prize
  track, so the account is worth having regardless.
