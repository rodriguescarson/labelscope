"""The leak, measured in voxels rather than in patches."""
import numpy as np
import tifffile

from labelscope.geometry import blocked_kfold, nnunet_default_split, parse_patch_names
from labelscope.leakage import (_intersection, check_overlap_consistency,
                                measure_seen_fraction)


def build(tmp_path, stride, n=4, size=60, thickness=3):
    """A run of patches cut from one synthetic scroll on a sliding window.

    The label is a plane in *scroll* coordinates, so every patch that covers a
    given scroll voxel labels it identically — which is what the real releases
    should also do.
    """
    labels = tmp_path / "labelsTr"
    labels.mkdir(exist_ok=True)
    names = []
    for i in range(n):
        for j in range(n):
            z, y, x = i * stride, j * stride, 0
            name = f"s1_z{z}_y{y}_x{x}"
            zz = np.arange(z, z + size)[:, None, None]
            yy = np.arange(y, y + size)[None, :, None]
            # sheets periodic in scroll coordinates, so every patch carries label
            # and every shared voxel gets the same answer from both patches
            sheets = ((zz + yy) % 20 < thickness) & np.ones((size, size, size), bool)
            tifffile.imwrite(str(labels / f"{name}.tif"), sheets.astype(np.uint8))
            names.append(name)
    patches, _ = parse_patch_names(names, (size, size, size))
    return patches, str(labels)


def test_intersection_box_is_in_local_coordinates():
    patches, _ = parse_patch_names(["s1_z0_y0_x0", "s1_z30_y0_x0"], (60, 60, 60))
    box = _intersection(patches[0], patches[1])
    assert box == ((30, 0, 0), (60, 60, 60))
    assert _intersection(patches[1], patches[0]) == ((0, 0, 0), (30, 60, 60))


def test_disjoint_patches_have_no_intersection():
    patches, _ = parse_patch_names(["s1_z0_y0_x0", "s1_z60_y0_x0"], (60, 60, 60))
    assert _intersection(patches[0], patches[1]) is None


def test_overlapping_patches_leak_labelled_voxels_to_the_random_split(tmp_path):
    patches, labels = build(tmp_path, stride=30)
    splits = nnunet_default_split([p.name for p in patches], k=4)
    result = measure_seen_fraction(patches, splits, labels)
    assert result["n_patches"] > 0
    assert result["patches_with_any_seen_pct"] > 50.0
    assert result["seen_fraction_mean"] > 0.2


def test_the_blocked_split_leaks_nothing(tmp_path):
    patches, labels = build(tmp_path, stride=30, n=6)
    splits, stats = blocked_kfold(patches, k=4, mode="block", block_factor=1)
    assert stats["residual_leaking_val_patches"] == 0
    result = measure_seen_fraction(patches, splits, labels)
    assert result["seen_fraction_mean"] == 0.0
    assert result["patches_with_any_seen"] == 0


def test_disjoint_patches_leak_nothing_under_any_split(tmp_path):
    patches, labels = build(tmp_path, stride=60)
    splits = nnunet_default_split([p.name for p in patches], k=4)
    result = measure_seen_fraction(patches, splits, labels)
    assert result["seen_fraction_mean"] == 0.0


def test_consistent_labels_score_iou_one(tmp_path):
    patches, labels = build(tmp_path, stride=30)
    result = check_overlap_consistency(patches, labels, max_pairs=20)
    assert result["n_pairs"] > 0
    assert result["iou_median"] > 0.999
    assert result["pairs_below_0_9_iou"] == 0


def test_a_contradictory_patch_is_caught(tmp_path):
    """Two patches covering the same scroll voxel must agree.  Corrupt one and
    the check has to notice."""
    patches, labels = build(tmp_path, stride=30)
    import os

    victim = os.path.join(labels, patches[1].name + ".tif")
    volume = tifffile.imread(victim)
    tifffile.imwrite(victim, np.roll(volume, 8, axis=0))     # shift the sheet
    result = check_overlap_consistency(patches, labels, max_pairs=40)
    assert result["iou_min"] < 0.9
    assert result["pairs_below_0_9_iou"] >= 1


def test_buffer_dropped_patches_are_not_counted_as_training_data(tmp_path):
    """A buffered split deliberately withholds the patches that touch
    validation.  Treating "not in this fold" as training makes the fix look
    like it still leaks — which is exactly what the first version of this
    measurement reported on real data."""
    patches, labels = build(tmp_path, stride=30, n=6)
    # block_factor 1 makes the blocks patch-sized, so the folds actually split
    # this small fixture spatially and the buffer zone is exercised
    splits, stats = blocked_kfold(patches, k=4, mode="block", block_factor=1)
    assert any(stats["buffer_dropped_per_fold"]), "this fixture must exercise the buffer"

    honest = measure_seen_fraction(patches, splits, labels)
    assert honest["seen_fraction_mean"] == 0.0

    # the old behaviour, reconstructed: everything outside the fold counts
    naive = [{"train": [p.name for p in patches if p.name not in set(s["val"])],
              "val": s["val"]} for s in splits]
    assert measure_seen_fraction(patches, naive, labels)["seen_fraction_mean"] > 0.0
