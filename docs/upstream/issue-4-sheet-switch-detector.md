**Repo:** ScrollPrize/villa · **Type:** issue · **Relates to:** #1621, 2026 Open Problems bottleneck table

---

### Title

A conservative sheet-switch detector: the whole-winding displacement the spiral satisfaction metric scores as zero is visible in the scan

### Body

[#1621](https://github.com/ScrollPrize/villa/issues/1621) establishes that
`get_patch_satisfied_areas` cannot see a whole-winding displacement — it derives
its target from the patch's own position, so displacing a patch by any integer
number of windings changes the satisfied-quad fraction by **exactly zero**. The
Open Problems bottleneck table lists the same failure fourth, "Meshes can jump
from one wrap to another", and asks for conservative failure detection.

Here is a detector for it, built on the observation that the displacement is
invisible in the *surface* but not in the *scan*.

**What gives it away.** A displaced surface still lies on papyrus, so its
geometry is unremarkable and its intensity is unremarkable. What is not
unremarkable is the **seam**: the one line of grid edges joining the displaced
region to the rest, which has to cross the gap between two wraps. The gap is
dark. Sampling the scan along each grid edge and taking the depth of the trough
relative to that edge's own endpoints gives a statistic that does not care how
bright this part of the scroll happens to be, and averaging it along the seam
direction lifts it out of the noise — a seam is roughly 1% of a mesh's edges, so
judging edges individually finds nothing.

**Measured on `20230702185753` against its own 2.4 µm scan**, one winding planted
over half the grid:

| | real mesh | one winding displaced |
|---|---|---|
| seam line mean darkening | 9.1 | **36.3** |
| ratio to the other grid lines | 1.13 | **4.72×** |
| **z-score** | **0.4** | **11.5** |

It fires at one, two and three windings alike, so it does not inherit the
periodic blindness #1621 documents.

**It reports when it cannot answer, which I think matters more than the
detection.** The seam is only visible if a grid edge normally stays on one wrap.
The detector measures the winding spacing from the scan and reports
`steps_per_winding`; below two it declines and moves its findings to
`seams_unreliable`. The requirement works out as **voxel size < winding spacing /
40**, and it is per scroll rather than per project:

| scroll | resolution | winding spacing | grid step | steps/winding | usable |
|---|---|---|---|---|---|
| PHercParis4 | 45.532 µm | 12.5 vx | 19.9 vx | 0.6 | no |
| PHercParis4 | 2.400 µm | 77.7 vx | 19.9 vx | 3.9 | **yes** |
| PHerc0500P2 | 9.362 µm | 17.0 vx | 19.9 vx | 0.9 | no |

My own first fleet run was at 45.5 µm and was flagging "seams" that were the
scan's ordinary roughness; the guard exists because I needed it.

**Two simpler detectors that do not work,** recorded because both look reasonable:

* *Large 3-D jumps between grid-adjacent vertices.* Published meshes are smooth —
  a switch slides onto the next wrap rather than tearing. On
  `20230702185753-on-20230205180739-7.91um` the largest grid step is 27.7 voxels
  against a median of 20.0. No signal at all.
* *An aggregate edge-darkening statistic over the whole mesh.* Real against
  one-winding-displaced: 12.29% versus 11.72% of edges dipping past threshold.
  The seam is swamped.

**Streaming.** A traced surface is a 2-D sheet threaded through the scan and
touches a small fraction of the chunks in its own bounding box, so the tool
fetches only those chunks straight from the open-data Zarr: about 4 GB per
segment against roughly 50 GB for the bounding box, with the full array —
75784 × 32693 × 32693 at 2.4 µm — never a consideration.

```bash
labelscope sheetswitch --mesh <tifxyz>... \
    --volume https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr \
    --remote --window 160 --out audit/
```

Tool: [`labelscope`](https://github.com/rodriguescarson/labelscope), MIT, CPU
only. The detector is validated against planted displacements rather than
eyeballed, and the tests include a clean surface reporting nothing, the seam
being located at the right grid line, and the resolution guard firing in both
directions.

I would happily wire this into `save_mesh` as a gate alongside the satisfaction
check, or run it across the published segments of a scroll and report, if either
is useful.
