import json

from labelscope.geometry import (
    blocked_kfold,
    overlap_graph,
    parse_patch_names,
    simulate_random_kfold,
)


def _grid(stride, n=4, size=300, volume="s1"):
    names = [f"{volume}_z{z * stride}_y{y * stride}_x0" for z in range(n) for y in range(n)]
    patches, unparsed = parse_patch_names(names, (size, size, size))
    assert not unparsed
    return patches


def test_disjoint_grid_has_no_overlap():
    patches = _grid(stride=300)
    assert overlap_graph(patches) == {}


def test_sliding_window_overlap_is_detected():
    patches = _grid(stride=150)  # 50% stride on a 300 patch
    graph = overlap_graph(patches)
    assert graph, "overlapping patches must produce edges"
    assert max(max(v.values()) for v in graph.values()) > 0.4


def test_overlap_fraction_is_symmetric_and_bounded():
    patches = _grid(stride=150)
    graph = overlap_graph(patches)
    for a, neighbours in graph.items():
        for b, fraction in neighbours.items():
            assert 0.0 < fraction <= 1.0
            assert abs(graph[b][a] - fraction) < 1e-9


def test_different_volumes_never_overlap():
    patches, _ = parse_patch_names(["s1_z0_y0_x0", "s4_z0_y0_x0"], (300, 300, 300))
    assert overlap_graph(patches) == {}


def test_random_kfold_leaks_and_blocked_split_does_not():
    patches = _grid(stride=150, n=6)
    graph = overlap_graph(patches)
    random_cv = simulate_random_kfold(len(patches), graph, k=5, trials=20)
    assert random_cv["val_patches_contaminated_pct_mean"] > 50.0

    splits, stats = blocked_kfold(patches, k=5, mode="block")
    assert stats["residual_leaking_val_patches"] == 0
    for split in splits:
        assert not set(split["train"]) & set(split["val"])


def test_blocked_split_covers_every_patch_exactly_once_as_validation():
    patches = _grid(stride=150, n=6)
    splits, _ = blocked_kfold(patches, k=5, mode="block")
    seen = [name for split in splits for name in split["val"]]
    assert sorted(seen) == sorted(p.name for p in patches)
    assert len(seen) == len(set(seen))


def test_splits_are_json_serialisable_in_nnunet_shape():
    patches = _grid(stride=150)
    splits, _ = blocked_kfold(patches, k=5)
    payload = json.loads(json.dumps(splits))
    assert isinstance(payload, list) and set(payload[0]) == {"train", "val"}


def test_unparsable_names_are_reported_not_dropped_silently():
    patches, unparsed = parse_patch_names(["sample_00001", "s1_z0_y0_x0"], (300, 300, 300))
    assert len(patches) == 1 and unparsed == ["sample_00001"]


def test_patch_sizes_can_differ_within_one_dataset():
    """Dataset059 looks uniform from its filenames and ships 170, 172, 236 and
    300 voxel cubes in one directory.  Assuming a single size gets every overlap
    wrong."""
    from labelscope.geometry import parse_patch_names

    names = ["s1_z0_y0_x0", "s1_z200_y0_x0"]
    sizes = {"s1_z0_y0_x0": (300, 300, 300), "s1_z200_y0_x0": (172, 172, 172)}
    patches, _ = parse_patch_names(names, (300, 300, 300), sizes=sizes)
    assert patches[0].size == (300, 300, 300)
    assert patches[1].size == (172, 172, 172)

    # 300-cube at 0 and 172-cube at 200 overlap over z 200..300
    assert patches[0].overlap_fraction(patches[1]) > 0
    # the fraction is relative to each patch's own volume, so it differs
    assert patches[0].overlap_fraction(patches[1]) != patches[1].overlap_fraction(patches[0])


def test_assuming_one_size_overstates_overlap_on_a_mixed_dataset():
    from labelscope.geometry import overlap_graph, parse_patch_names

    names = ["s1_z0_y0_x0", "s1_z200_y0_x0", "s1_z400_y0_x0"]
    real = dict.fromkeys(names, (172, 172, 172))
    assumed, _ = parse_patch_names(names, (300, 300, 300))
    actual, _ = parse_patch_names(names, (300, 300, 300), sizes=real)
    assert overlap_graph(assumed), "at 300 these patches overlap"
    assert not overlap_graph(actual), "at their real 172 they do not"


def test_sizes_are_read_from_the_volumes(tmp_path):
    import numpy as np
    import tifffile

    from labelscope.geometry import sizes_from_volumes

    for name, edge in (("s1_z0_y0_x0", 8), ("s1_z100_y0_x0", 12)):
        tifffile.imwrite(
            str(tmp_path / f"{name}.tif"), np.zeros((edge, edge, edge), dtype=np.uint8)
        )
    sizes, unreadable = sizes_from_volumes(
        ["s1_z0_y0_x0", "s1_z100_y0_x0", "s1_z999_y0_x0"], str(tmp_path)
    )
    assert sizes["s1_z0_y0_x0"] == (8, 8, 8)
    assert sizes["s1_z100_y0_x0"] == (12, 12, 12)
    assert unreadable == ["s1_z999_y0_x0"]
