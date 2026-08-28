# Full-population results

Produced on a RunPod A40 with both public datasets held locally, so every figure
here is over the whole release rather than a sample.

| directory | what | population |
|---|---|---|
| `kaggle_scan/` | header + label census, Kaggle surface release | all 892 pairs |
| `kaggle_align/` | label-to-ridge offset, Kaggle | all 892 pairs |
| `kaggle_deep/` | topology with skeleton junctions, Kaggle | 200-pair seeded sample |
| `d059_scan/` | header + label census, `Dataset059` | all 1,754 pairs |
| `d059_align/` | label-to-ridge offset, `Dataset059` | all 1,754 pairs |
| `d059_leakage/` | patch overlap, seen-fraction, split, consistency | all 1,754 pairs |

## The offset replicates across two independent releases

| | Kaggle surface release | `Dataset059` |
|---|---|---|
| pairs | 892 | 1,754 |
| measurable | 836 (93.7%) | 1,732 (98.7%) |
| median \|offset\| | **2.285 vx** | **2.576 vx** |
| interquartile range | 1.69 – 2.83 | 1.88 – 3.79 |
| ≥ 1 voxel | 89% | 98% |
| ≥ 2 voxels | 63% | 70% |
| per-cell \|offset\|, median | 2.30 vx | 2.62 vx |
| cells ≥ 1 voxel off | 83.3% | 85.2% |
| median winding spacing | 19.0 vx | 21.0 vx |

Different scrolls, different patch sizes, different production pipelines, 2,568
pairs between them, and the same answer to within a third of a voxel. The
labelled writing surface sits about two and a half voxels from the sheet's local
density maximum — it marks the recto face, not a centre-line.

`d059_leakage/splits_final.json` is a drop-in nnU-Net split for `Dataset059` in
which no validation patch touches a training patch.
