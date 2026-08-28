import json

from labelscope.geometry import (blocked_kfold, overlap_graph, parse_patch_names,
                                 simulate_random_kfold)


def _grid(stride, n=4, size=300, volume="s1"):
    names = [f"{volume}_z{z*stride}_y{y*stride}_x0"
             for z in range(n) for y in range(n)]
    patches, unparsed = parse_patch_names(names, (size, size, size))
    assert not unparsed
    return patches


def test_disjoint_grid_has_no_overlap():
    patches = _grid(stride=300)
    assert overlap_graph(patches) == {}


def test_sliding_window_overlap_is_detected():
    patches = _grid(stride=150)          # 50% stride on a 300 patch
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
