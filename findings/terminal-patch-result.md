# Result: the terminal-patch test

Pre-registered in `terminal-patch-preregistration.md`, committed as `8d6faad`
**before** these numbers existed. The inclusion rule, the statistic, and what
each outcome would mean were all fixed in advance.

## Outcome: 2 of 3. The general rule does not hold.

| scroll | series | n | last winding | last range | rank of last | worst in series? |
|---|---|---|---|---|---|---|
| PHerc0139 | 20250108 | 6 | 30 | 62.5 | 5 of 6 | no |
| PHercParis4 | 20260623 | 30 | 128 | 18.0 | **1 of 30** | yes |
| PHercParis4 | 20260701 | 28 | 128 | 12.2 | **1 of 28** | yes |
| PHerc0172 | 20250926 | 8 | 88 | 18.7 | 3 of 8 | no *(secondary; scroll fails the resolution gate)* |

PHerc0139's terminal patch is healthy — a range of 62.5, mid-pack in its own
series. **"The last patch of a tracing run is off-sheet" is not a general law**,
and the pre-registration said this outcome would be reported as suggestive
rather than conclusive. It is.

No combined p-value is claimed. The two series that do hit are both PHercParis4
over the same winding span (18-128), so they are not independent samples; the
naive product (1/30 x 1/28 = 1/840) is an upper bound on significance, not a
p-value.

## What does survive

Two published PHercParis4 surfaces covering **windings 128-129** score far below
every other surface measured on that scroll, in two tracings made a week apart:

* 0.18 and 0.27 of a published baseline's profile range, against 0.51-1.06 for
  the twenty neighbouring surfaces in the same winding band.
* Each is the minimum of its own 30- and 28-member series.
* The adjacent w126-127 surfaces score 0.63 and 0.82, so **the region carries
  structure** — the low score is a property of these two tracings, not of that
  part of the scroll. This is the control that distinguishes the finding from a
  degenerate-null artifact, which is what killed an earlier candidate.

The claim is therefore narrow and checkable: *these two published surfaces
appear not to sit on papyrus*. It is not a claim that they cross between
windings, which the profile measurement cannot establish.

## Reproduce

```
python scripts/onsheet_check.py --mesh <tifxyz> --volume <scroll zarr> --remote \
  --blocks 8 --block-size 12 --out result.json
```

Scored against `20230702185753` on PHercParis4 as a published baseline; the rank
test itself uses no baseline at all.
