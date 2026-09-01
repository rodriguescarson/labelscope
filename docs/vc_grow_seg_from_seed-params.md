# `vc_grow_seg_from_seed`: a worked `params.json`

There is no example params file in villa, and the keys are split across two
source files, so the working set has to be reconstructed by reading both. This
is that reconstruction, verified against villa at `volume-cartographer/` and run
against published PHercParis4 data.

Everything below was read from source or observed in a run. Where a behaviour
surprised me, it is marked **trap**.

## Invocation

```bash
vc_grow_seg_from_seed \
  -v /path/to/prediction.zarr \
  -t /path/to/output-dir \
  -p params.json \
  -s 15744 17580 56482        # seed, in x y z voxel coordinates
```

`-s` takes three separate numbers, not a comma-separated triple. The seed must
land on a non-zero voxel of the prediction; the run prints
`seed location [x, y, z] value is 255` so you can confirm it did.

## A params.json that works

```json
{
  "mode": "seed",
  "generations": 280,
  "step_size": 20.0,
  "min_area_cm": 0.05,
  "voxelsize": 2.4,
  "use_cuda": true,
  "cache_size": 8000000000,
  "thread_limit": 4
}
```

## Where each key is read

One file is the CLI wrapper, the other the tracer, and they read *the same*
JSON. A key absent from the file you happen to be reading is not necessarily
unsupported — check both.

**`apps/src/vc_grow_seg_from_seed.cpp`** — `cache_root`, `cache_size`,
`direction_fields`, `min_area_cm`, `mode`, `search_effort`, `step_size`,
`tgt_overlap_count`, `thread_limit`, `use_cuda`, `voxelsize`, and the
`neighbor_*` family.

**`core/src/GrowPatch.cpp`** — `generations`, `growth_scale`,
`growth_directions`, `normal_grid_path`, `normal_grid_level`,
`normal_grid_scale`, `step_size`, `snapshot-interval`, `inpaint`,
`umbilicus_path`, `reference_surface`, `x_min`/`x_max`/`y_min`/`y_max`/`z_min`/
`z_max`, and the `resume_*`, `sdt_*`, `patch_normal*`, `cell_reopt_*`,
`space_line_*`, `grow_extra_*` families.

## Defaults worth knowing

| key | default | note |
|---|---|---|
| `min_area_cm` | `0.3` | a patch smaller than this is discarded; too high and a run produces nothing |
| `voxelsize` | the volume's own `voxelSize()` | an explicit value **overrides** the volume metadata |
| `cache_size` | `1e9` | bytes |
| `thread_limit` | `0` (unbounded) | see below — the default is the wrong choice on a big machine |
| `mode` | `"seed"` | |
| `generations` | `100` | |

## Traps

**`thread_limit` defaults to unbounded, and unbounded is slower.** The source
comment records wall time per cm² measured on two machines: lowest at 4 threads
on both, with unbounded costing 2.2x and 3.5x respectively. The penalty grows
with core count, so the default is worst exactly on the large machines used for
batch tracing. VC3D itself passes `thread_limit=1`. On a 96-core box, leaving
this unset is a multi-fold slowdown.

**`step_size` overrides the normal grid's own spiral step.** Priority is
explicit param > normal grid > resume surface > default. If you pass a normal
grid *and* an explicit `step_size`, yours wins and the grid's preferred step is
discarded — usually not what you want when you went to the trouble of supplying
a grid.

**`normal_grid_path` must be a local directory, not a URL.** To stream the
published grids, create a directory containing a file named
`normal-grids-remote.json`:

```json
{"url": "https://…/…-th0.45.normal-grids"}
```

and point `normal_grid_path` at *that directory*. Passing the URL directly
fails with "normal-grid file not found", which does not hint at the marker
mechanism.

**`normal_grid_level` is silently ignored unless the store is multiscale.**
`NormalGridVolume` honours a requested level only when the store's
`metadata.json` declares `"format": "normal-grid-multiscale"`. The published
PHercParis4 store has no `format` key and contains only `metadata.json` — no
per-level metadata — so the level is discarded with no warning and the log
prints `Loaded normal grid level 0` regardless of what you asked for.

**`direction_fields` wants a `x`/`y`/`z` layout.** Stores are read as
`<path>/{x,y,z}/<scale>`. The published `.normal-grids` store is laid out
`xy/ xz/ yz/ xy_img/`, so it is *not* a `direction_fields` store — passing it
there 404s on all three.

## Cost, measured

Growing with the published normal grid at level 0 on an A40, `thread_limit=4`:
about 0.5 generations/minute with a warm 13 GB grid cache at a 98.5% hit rate,
against the ~280 generations a usable patch needs. Cold it is roughly 6x slower
again. GPU utilisation stayed at 0% throughout despite `use_cuda: true`, so this
phase is CPU-bound and a GPU instance buys nothing for it.

## Check the output before trusting it

A run that completes and writes a plausible surface may still have produced a
surface that is not on the papyrus. Sampling the scan along the surface normal
separates the two cheaply:

```bash
python scripts/onsheet_check.py --mesh <out.tifxyz> --baseline <published.tifxyz> \
  --volume <scroll.zarr> --remote --blocks 8 --block-size 12
```

A surface on a sheet shows a profile range near the published baseline's; one
that has wandered off shows a range near the noise floor and its density peak
tens of voxels away from where the surface sits.
