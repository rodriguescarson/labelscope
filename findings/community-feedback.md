# What the Vesuvius team said, and what it means for this tool

Recorded verbatim because it is the most important input this project has had, and
because two of the three points contradict claims made in the August submission.

## Paul (Vesuvius team), #general, 2026-08-30

> reminder: LLM-written posts about tools conceived and implemented by LLMs belong
> in #robots - see #rules

> note that "segment goes through dark voxels" is NOT a good measure of whether the
> segment crosses between windings (there's too much variation of intensity and
> structure), and nearness to intensity peak in the CT is only a very weak signal
> (due to fibers leaving the sheet, delaminations, etc)

> if you want to validate your tool, create some segments with GrowPatch, look at
> them in VC3D for errors, and see if your tools finds them

And earlier, closing [villa#1640](https://github.com/ScrollPrize/villa/issues/1640):

> Yes the segmentation ideally is the recto surface, since that is where the ink is.
> And yes, using intensity peaks will not reliably localise this.

## djosey (Vesuvius team), same thread

> For visual inspection, VC3D has a segment overlap tool. Might be a good place to start

## What our own data says about the first objection

It supports him, and that is worth being explicit about. From the 253-surface corpus
pass in [finding 9](README.md):

* the planted minimum (3.60) sits **below** the unplanted maximum (9.94), so a fixed
  z threshold does not transfer between surfaces;
* on **5 of 112** surfaces, planting a whole winding *lowered* the score.

That is precisely the "too much variation of intensity and structure" he describes,
measured. Where we differ is only in the conclusion: we found the signal is real but
unusable at a fixed threshold (2.93x median lift, CI 2.62-3.26, on 107 of 112), and
argued for a per-surface control. He is saying it is too weak to rely on at all.

The gap between those two positions is an empirical question, and his suggested test
is what settles it.

## The test, and what would count as a pass or a fail

1. Grow segments with `vc_grow_seg_from_seed` (GrowPatch).
2. Inspect them for real errors — by eye, in VC3D or from rendered cross-sections.
   This is the ground truth and a human has to produce it.
3. Run `labelscope sheetswitch` **blind** to those judgements and compare.

**Pass:** it flags the segments a human calls wrong, and does not flag the ones a
human calls right.

**Fail:** it misses real errors, or flags clean segments. If that is the outcome we
publish it as the fourth retraction and say the detector does not work on real
failures. Committing to that in advance is the point of writing it down here.

This replaces "turn labelscope into a check that runs on their data releases" as the
September priority. There is no value in productionising a measurement whose premise
a domain expert disputes and which has never been tested against a real defect.

## On the channel

The August post was LLM-written, which is what the rule is about. Future posts to
this server go in #robots, or are written by Carson himself.
