# Pre-registration: are terminal patches of a tracing run off-sheet?

Written **before** the measurement it describes, and committed so the timestamp
is checkable. Paul's objection to this tool was that its signals are weak and
that dark voxels and intensity peaks do not establish a sheet crossing. That
objection is fair, and the honest response to it is to fix the analysis before
seeing the numbers rather than after.

## The observation that prompted this

On PHercParis4 (Scroll 1), the on-sheet check scored all 81 published surfaces.
Two scored far below every other: both tracings of windings 128-129, from two
independent runs a week apart, at 0.18 and 0.27 of a published baseline while
their twenty neighbours in the same winding band ran 0.51 to 1.06.

Both are the **last patch of their tracing series**. Nothing above w129 exists
in either run.

## The hypothesis

A tracing run that leaves the sheet stops. If so, the terminal patch of a series
is the patch most likely to be off-sheet, and it should score worst within its
own series.

## The statistic, fixed in advance

Per series, rank all member surfaces by median normal-profile range and ask
whether the terminal patch (highest winding index) holds the minimum rank.
Under the null that profile quality is independent of position in the series,
that has probability `1/n` for a series of length `n`, and the product across
independent series is the combined p-value.

This test uses **no baseline surface**: it is a within-series rank, so it does
not assume any published surface is known-good, and it is invariant to
per-scroll differences in scan contrast.

## Inclusion rule, fixed in advance

* **Primary:** series of >= 5 surfaces whose scroll clears the resolution gate
  (`steps_per_winding >= 2`). That is PHerc0139/20250108 (n=6) and
  PHercParis4/20260623 (n=30) and /20260701 (n=28) — 3 series, 64 surfaces.
* **Secondary, reported separately:** PHerc0172/20250926 (n=8). PHerc0172 has a
  median 0.90 steps per winding and 1 of 53 surfaces clears the gate, so its
  profile sampling is not interpretable; it is reported but excluded from the
  combined p-value.
* **Excluded:** series of 3-4 surfaces (a rank test on n=3 carries almost no
  information), and PHerc0343P, PHerc0500P2, PHerc0814, whose filenames carry no
  winding index to order a series by.

Series membership is the 8-digit date prefix of the surface name; winding index
is `w<a>-<b>` (midpoint) or `w<a>`.

## What each outcome means

* **3 of 3 terminal patches worst in series** (p = 1/5040): supports the
  hypothesis; the rule "check the last patch of every tracing run" becomes a
  cheap pre-release check.
* **2 of 3:** reported as suggestive and not conclusive, with the combined
  p-value stated honestly.
* **0 or 1 of 3:** published as a negative. The w128-129 pair then stands as two
  anomalies in one scroll with no general rule behind them, and this document
  says so.

## What this does *not* claim

That a low profile range proves a sheet crossing. It measures whether the traced
surface sits on papyrus, which is a weaker and more directly checkable claim
than the sheet-switch detector makes. The PHercParis4 w128-129 pair is supported
by an additional control: its immediate neighbours w126-127 score 0.63 and 0.82,
so the region carries structure and the low score is a property of those two
tracings rather than of that part of the scroll.
