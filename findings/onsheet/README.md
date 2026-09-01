# On-sheet corpus results

Raw output behind `../terminal-patch-result.md`.

* `onsheet_corpus/` — 81 published PHercParis4 surfaces, 8 blocks each.
* `onsheet_series/` — the 14 surfaces of PHerc0139/20250108 and
  PHerc0172/20250926 added by the pre-registration.
* `logs/` — the runs that produced them, plus the two GrowPatch variants
  (`grow_nogrid.log`, `grow_grid.log`) and the level-parameter test
  (`grow_L2.log`).

Analysis scripts are in `../../scripts/onsheet/`. The rank test
(`ranktest.py`) is the pre-registered statistic and uses no baseline surface.

`grow_L2.log` records the run that established that `normal_grid_level` is
silently ignored: it requests level 2 and logs `Loaded normal grid level 0`.
That run reached 26 of 280 generations in 75 minutes and was stopped; it wrote
no surface, because `snapshot-interval` defaults to 0 and output is only written
on completion. Grid-guided growth at level 0 is not viable on published data.
