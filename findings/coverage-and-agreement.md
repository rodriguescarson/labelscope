# What the sheet-switch detector cannot see

The August entry's headline was a sheet-switch detector. This is what happened
when an independent measurement was pointed at the same corpus, and it is partly
a negative result about our own tool.

## The detector can only assess 45% of the published corpus

`labelscope sheetswitch` needs the winding spacing, so it refuses a surface
whose scan does not resolve windings (`steps_per_winding < 2`) or whose null is
degenerate. Across the 253 published surfaces in seven scrolls:

| | surfaces |
|---|---|
| published (corrected count) | 258 |
| **clear both gates** | **114 (44% of 258)** |
| fail the resolution gate | 137 |
| degenerate null | 16 |

The on-sheet check samples the scan along the surface normal. It needs no
winding spacing and, in its rank form, no baseline surface either, so nothing in
the method excludes a surface the way these gates do.

**The full pass has now been run: 257 of 258** published surfaces across all
seven scrolls, against the sheet-switch detector's 114 (`full-population.md`).
The single exception is one 248 MB-per-axis surface that OOM-killed the reader,
which is a reader memory limit rather than a gate. Note the denominator: the
corpus is 258 surfaces, not the 253 the August entry reported — that correction
is in `full-population.md` too.

## The one verified error falls in the half the detector cannot see

The two published PHercParis4 surfaces covering windings 128-129 do not sit on
papyrus where their neighbours do (`terminal-patch-result.md`; Mann-Whitney
p=0.0031 and p=0.0043 against their own adjacent winding, pooled p=0.00004).

Their sheet-switch rows:

| surface | steps/winding | resolution ok | degenerate | seams |
|---|---|---|---|---|
| w128-129, Jun tracing | nan | no | yes | 0 |
| w128-129, Jul tracing | 1.55 | no | yes | 0 |
| w126-127, Jun tracing | nan | no | yes | 0 |
| w126-127, Jul tracing | nan | no | yes | 0 |

**Both gates reject them, so the detector never looks.** It reports zero seams
because it declined to measure, not because it measured and found none. The
healthy neighbour is rejected too — the whole region is invisible to it.

This is the concrete form of the objection raised against the August entry, and
our own corpus supports it: on the one error here that an independent method
verifies, the sheet-switch detector is silent.

## The two measurements are independent, not redundant

Of the 40 PHercParis4 surfaces that clear the gates and that the on-sheet check
scores as healthy, sheetswitch flags **9 (22%)**.

That is *not* a false-positive rate. A surface can sit squarely on papyrus and
still cross between windings — the two methods ask different questions, and a
disagreement is expected rather than disqualifying. What it does mean is that
neither can stand in for the other, and a flag from one is not corroborated by
the other by default.

## What follows

The on-sheet check is the more broadly applicable of the two: across the whole
published corpus it scores **257 of 258** where the detector can assess **114**,
it caught a real defect the detector's gates hid, and its rank form needs no
known-good reference surface. The 2x coverage gap seen on PHercParis4 holds
corpus-wide. The sheet-switch detector
remains unproven on real errors — the validation that would settle it still
needs traced surfaces with known sheet crossings, which this work has not yet
produced.

## Reproduce

```
python scripts/onsheet/agreement.py
```

Joins `findings/corpus/per_surface/real/` (August sheet-switch pass) to
`findings/onsheet/onsheet_corpus/` (this pass) on surface name.
