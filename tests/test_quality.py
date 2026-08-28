import numpy as np

from labelscope.quality import (audit_label, component_stats, junction_density,
                                label_scheme, thickness_stats)


def sheet_label(shape=(48, 48, 48), z=24, thickness=2, value=1):
    label = np.zeros(shape, dtype=np.uint8)
    label[z:z + thickness] = value
    return label


def test_label_scheme_reports_every_class_used():
    label = sheet_label()
    label[40:] = 2
    scheme = label_scheme(label)
    assert scheme["values"] == [0, 1, 2]
    assert scheme["n_values"] == 3
    assert abs(sum(scheme["value_fractions"].values()) - 1.0) < 1e-6


def test_thickness_matches_a_known_slab():
    mask = sheet_label(thickness=4) > 0
    stats = thickness_stats(mask)
    assert 3.0 <= stats["median"] <= 5.0


def test_a_flat_sheet_is_planar_and_a_cube_is_not():
    sheet = component_stats(sheet_label(thickness=2) > 0)
    cube = np.zeros((48, 48, 48), bool)
    cube[10:30, 10:30, 10:30] = True
    blob = component_stats(cube)
    assert sheet["worst_planarity"] < 0.02
    assert blob["worst_planarity"] > 0.25


def test_fragments_are_counted_separately():
    label = sheet_label()
    label[2, 2, 2] = 1                 # a one-voxel speck
    stats = component_stats(label > 0)
    assert stats["n_components"] == 2
    assert 0.0 < stats["fragment_fraction"] < 0.001


def test_a_clean_sheet_has_no_junctions():
    clean = np.zeros((48, 48, 48), bool)
    clean[16:18] = True
    assert junction_density(clean, n_samples=1500)["junction_fraction"] < 0.02


def test_a_nearby_second_winding_is_not_mistaken_for_a_junction():
    """Two separate sheets 12 voxels apart must still read as clean — this is
    the false positive the ball-shell measure exists to avoid."""
    two = np.zeros((48, 48, 48), bool)
    two[16:18], two[28:30] = True, True
    assert junction_density(two, n_samples=1500)["junction_fraction"] < 0.02


def test_a_bridge_between_two_windings_is_detected():
    bridged = np.zeros((48, 48, 48), bool)
    bridged[16:18], bridged[28:30] = True, True
    bridged[16:30, 22:26, 22:26] = True          # the merger
    clean_rate = junction_density(
        np.pad(np.ones((2, 48, 48), bool), ((16, 30), (0, 0), (0, 0))), n_samples=1500
    )["junction_fraction"]
    assert junction_density(bridged, n_samples=4000)["junction_fraction"] > clean_rate + 0.005


def test_surface_class_is_detected_not_assumed():
    """Class 2 is the thin sheet here and class 1 is the bulk — the detector
    must not simply take the lowest index."""
    label = np.zeros((48, 48, 48), dtype=np.uint8)
    label[:20] = 1                                # bulk region
    label[30:32] = 2                              # thin sheet
    result = audit_label(label)
    assert result["surface_class"] == 2
    assert result["surface_thickness_median"] <= 3.0


def test_a_space_filling_class_is_never_called_the_surface():
    label = np.zeros((32, 32, 32), dtype=np.uint8)
    label[:] = 1                                  # fills the volume
    label[15:17] = 2
    assert audit_label(label)["surface_class"] == 2


def test_empty_label_is_handled():
    result = audit_label(np.zeros((16, 16, 16), dtype=np.uint8))
    assert result["foreground_fraction"] == 0.0
    assert result["surface_class"] is None
