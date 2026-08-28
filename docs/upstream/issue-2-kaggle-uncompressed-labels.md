**Repo:** ScrollPrize/villa · **Type:** issue (data QA)

---

### Title

Kaggle surface-detection release: 487 label volumes ship uncompressed — ~15 GB of a 45 GB release

### Body

Inventorying every volume in `hf://buckets/scrollprize/datasets/surfaces/kaggle`
by its TIFF header alone (about a kilobyte per file, ~1.8 MB of transfer for all
1,784 volumes):

| | files | on disk |
|---|---|---|
| labels, `COMPRESSION.NONE` | **487** | **15.52 GB** |
| labels, `COMPRESSION.LZW` | 405 | 0.36 GB (median 0.87 MB) |
| images, `COMPRESSION.LZW` | 873 | — |
| images, `COMPRESSION.NONE` | 19 | — |
| | | 45 GB total |

The uncompressed labels are not different in content. Sampling both groups, class
1 is a ~2-voxel-thick, highly planar writing surface in both (median local
thickness 2.0 voxels, worst-component planarity 0.007) and class 2 is a single
bulky region in both. Only the writer setting differs — and the compressed half of
the same release shows what the other half would cost: a median of **0.87 MB**
against **32.82 MB**.

Re-encoding those 487 with the compression the other 405 already use takes roughly
15 GB off the release. One line in whatever wrote them, and transfer is not free
for this project's users.

Reproduce without downloading anything:

```bash
labelscope scan --labels <url-or-dir> --headers-only --out audit/
# labels: {'NONE': 487, 'LZW': 405}
```

Tool: [`labelscope`](https://github.com/rodriguescarson/labelscope), MIT.

---

Two things checked at the same time and found to be **fine**, recorded so nobody
re-checks them:

* The release mixes three patch sizes — 320³ (840 pairs), 256³ (51), 384³ (1) —
  but images and labels agree on shape in **every** pair, zero mismatches. Worth
  documenting for anyone writing a fixed-size loader, but not a defect.
* 24 sample indices between 1 and 916 have neither an image nor a label. The
  numbering is simply not contiguous; nothing is orphaned.
