# Result: the seam detector against pscamillo's 12 eye-labelled rejects

Pre-registered in `reject-seam-preregistration.md` (commit `895214f`, 14 Sep,
before the detector ran). Run on a 4-vCPU CPU pod, 14 Sep 2026, $0.12/h, about
an hour of billed time including one restart after the chunk cache filled a
40 GB disk.

## Outcome: INCONCLUSIVE BY GATE

**0 of 12 meshes pass the resolution gate.** Every one samples the wraps at
between 0.50 and 1.05 grid steps per winding (grid step 20.0 voxels, winding
spacing 12 to 27 voxels at 9.36 um). The gate needs 2. The pre-registration
said this was the likely outcome and that a refusal would be reported as the
result, not worked around, so it is.

```
mesh                       set   step/wind  gate seams  max_z max_z+1w      MB
------------------------------------------------------------------------------
PHerc0813__z12496_w100     step       0.65 False     0   4.24     5.35    5964
PHerc0813__z11296_w100     step       0.75 False     0   4.23     4.99    5597
PHerc0813__z4704_w100      step       0.90 False     0   2.95     9.52    7441
PHerc0813__z13088_w100     step       0.70 False     0   4.03     8.49    4893
PHerc0813__z11904_w100     step       0.50 False     0  36.15    18.85    3083
PHerc0813__z11904_w080     step       1.05 False     0   6.49     5.47    3467
PHerc0211__z4912_w080      step       0.70 False     0   6.51     7.31    5086
PHerc0211__z13920_w080     step       0.80 False     0   7.28     6.73    5509
PHerc0800__z17072_w100     step       0.60 False     0   7.04     6.71    7179
PHerc0125__z6944_w080      step       0.80 False     0  19.49    24.90    6719
PHerc0211__z6720_w100      drift      0.80 False     0   7.49     7.49    3131
PHerc0211__z9120_w100      drift      0.75 False     0  40.00    44.50    2814
```

`max_z` is the strongest grid-line darkening on the delivered mesh;
`max_z+1w` the same with a whole winding planted into half of it. With the
gate failed, `n_seams` is 0 by construction (the detector refuses to call a
seam it cannot resolve), so the primary criterion was never evaluated.

## What the numbers say anyway, descriptively

* The planted copy scores higher than the delivered one on 7 of 12. On Will
  Stevens' components it was 1 of 4. That is a weak signal that the planted
  control is at least sometimes visible even at one sample per winding, and
  no more than that.
* The two largest `max_z` values, 40.0 and 36.2, are on `PHerc0211/z9120_w100`
  and `PHerc0813/z11904_w100`. The first is one of pscamillo's two *drift*
  controls, where he says no seam should be found. Without the gate the
  detector would have called a seam on a negative control. The gate is doing
  its job.
* `PHerc0125/z6944_w080` reads 19.5 delivered against 24.9 planted, the
  clearest planted-above-delivered gap in the set, and it is a step mesh. One
  mesh is not a finding.

## What this settles and what it does not

It settles that the seam detector, as shipped, cannot be validated on
pscamillo's meshes: they are fitted at a 20-voxel grid step on 9 um scans,
and the check needs to see the dark gap between wraps, which that sampling
cannot resolve. That is consistent with Sean's point on 9 September that
intensity along a coarse grid is not a switch detector, and with the planted
control on Will's components.

It does not settle whether a finer resampling of the same meshes would pass
the gate and then separate step from drift. That is a different test with a
different predictor and would need its own pre-registration. It is the obvious
next one, and it is cheap: the same 12 meshes, resampled to a 5-voxel step,
same volumes, same pod.

## Reproduce

`findings/reject_seam/out/<mesh>__plant{0,1}.json/sheetswitch.csv` are the raw
outputs; `progress.log` has the timings. The run script is
`scripts/onsheet/reject_seam_run.sh`; the rule is applied by
`scripts/onsheet/reject_seam_analysis.py --out-dir findings/reject_seam/out`.
