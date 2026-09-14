# Result: the on-sheet score against blind text / no-text labels of every published ink render

Pre-registered in `ink-blind-preregistration.md` (sealed 2 Sep 2026, key hash
`a126d16f…`, unchanged; verified before labelling). Labelled and analysed
14 Sep 2026.

## Outcome: FAIL

| pre-registered criterion | required | achieved |
|---|---|---|
| segments in the bottom decile of within-scroll rank | >= 10 | 15 |
| fraction of those labelled `no text` | >= 0.80 | **0.00** |
| and at least 2x the pooled base rate of `no text` | >= 0.04 | 0.00 |

Descriptive AUC of within-scroll rank against `no text`: 0.49. Chance.

## Deviations from the pre-registration, stated before the interpretation

1. **The labeller was not Carson. It was the model (Claude) that has been
   running this project, on his instruction.** The pre-registration names
   Carson. The blindness protocol was kept: the 255 renders were copied to
   files named only by their three-digit code by a script whose mapping was
   never printed, shuffled, and viewed four to a sheet with the code burned
   in. The model had seen the four w126-129 renders with their names on
   2 September; it did not see the code-to-name key until labelling was
   complete. A vision model's "eye" is not a papyrologist's, and this result
   should be read as "where a VLM sees no letterforms", not "where a human
   sees none".
2. Renders were shown at about 900 px wide, as already recorded in the
   pre-registration, not the 2000 px first written.
3. 35 of 255 were labelled `unsure` and excluded, as the pre-registration
   allows. Two of those matter, and are discussed below.

## Population

255 renders labelled: 169 `text`, 51 `no text`, 35 `unsure`. Per scroll:

| scroll | labelled | text | no text | unsure | qualifies (>= 20% text) |
|---|---|---|---|---|---|
| PHercParis4 | 80 | 74 | 4 | 2 | yes |
| PHerc0172 | 53 | 40 | 0 | 13 | yes |
| PHerc0139 | 38 | 35 | 0 | 3 | yes |
| PHerc1667 | 19 | 16 | 0 | 3 | yes |
| PHerc0500P2 | 38 | 3 | 34 | 1 | no |
| PHerc0814 | 19 | 1 | 6 | 12 | no |
| PHerc0343P | 8 | 0 | 7 | 1 | no |

Three scrolls have almost no readable renders and drop out by the
pre-registered rule. On the four that qualify, **169 segments carry a
`text`/`no text` label and only 4 of them are `no text`: a base rate of 2%.**

## What the FAIL means

The test had almost nothing to find. On the scrolls where the ink model
works, it works on nearly every published segment; `no text` is a 2% event,
and the four cases of it on PHercParis4 (`20260603145540-5753_-4`, `-5`,
`-6`, `20260602204401-5753_-7`) all score *high* on the on-sheet measure
(within-scroll rank 0.33 to 0.62). Those are surfaces with a clear sheet
under them and nothing written on it, or a model miss, which the
pre-registration named in advance as "not the tool's fault". So the bottom
decile contains no `no text` at all, and the score cannot distinguish
anything within the 98% that reads.

**The two known cases were ranked exactly where the claim says they should
be, and then excluded.** Both published tracings of PHercParis4 w128-129 are
the two lowest-ranked segments of that scroll (rank 0.01 and 0.02 of 80).
They were labelled `unsure`: at 900 px their long, thin strips of speckle
looked like several other strips that turned out to be text, and the
labeller declined to call them. Had they been labelled `no text`, bottom-decile
precision would have been 2 of 17, 0.12, still a FAIL by the 0.80 rule.
The verdict does not change. But the reader should know that the predictor
did put the two surfaces the whole September entry is about at the very
bottom of the ranking, and that the human-readable evidence for them
(`w128-129-evidence.md`: their renders, their surface volumes, the
cross-sections) is stronger than a 900 px thumbnail.

## The tool's misses, listed as promised

Fifteen bottom-decile segments show text. Nine are on PHerc0139 and PHerc0172
with per-chunk mean range 60 to 66 and 17 to 19 respectively, which is to
say they are at the bottom of their scroll's ranking while being perfectly
healthy: on those scrolls the ranking is over a population with no bad
surfaces in it, and the bottom decile is just the lowest decile of good.
The other six are PHercParis4 w122-127 (mean 37 to 48), the neighbours of
w128-129, which read and are correctly less bright than the inner windings.

## What this says, stated narrowly

Within-scroll rank on the on-sheet score does not predict where the team's
ink model finds no text on the published corpus, because on that corpus the
model almost always finds text. The measurement's use, if it has one, is as
a pre-flight on *new* surfaces before an ink run, where the base rate of
"nothing under this" is not 2%. This test cannot show that, and does not.

## Reproduce

```
python scripts/onsheet/blind_analysis.py --labels drafts/ink-labels-model.csv \
  --key drafts/ink-labeler-key.json --predictors findings/onsheet/onsheet_sv/ \
  --out findings/ink-blind-result.json
```
The labels and the key are in `drafts/` (gitignored: the key is the
unblinding). The per-segment rows, with rank and label, are in
`findings/ink-blind-result.json`.
