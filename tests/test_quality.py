import numpy as np
import pytest

from labelscope.quality import (
    audit_label,
    component_stats,
    junction_density,
    label_scheme,
    thickness_stats,
)


def sheet_label(shape=(48, 48, 48), z=24, thickness=2, value=1):
    label = np.zeros(shape, dtype=np.uint8)
    label[z : z + thickness] = value
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
    label[2, 2, 2] = 1  # a one-voxel speck
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
    bridged[16:30, 22:26, 22:26] = True  # the merger
    clean_rate = junction_density(
        np.pad(np.ones((2, 48, 48), bool), ((16, 30), (0, 0), (0, 0))), n_samples=1500
    )["junction_fraction"]
    assert junction_density(bridged, n_samples=4000)["junction_fraction"] > clean_rate + 0.005


def test_surface_class_is_detected_not_assumed():
    """Class 2 is the thin sheet here and class 1 is the bulk — the detector
    must not simply take the lowest index."""
    label = np.zeros((48, 48, 48), dtype=np.uint8)
    label[:20] = 1  # bulk region
    label[30:32] = 2  # thin sheet
    result = audit_label(label)
    assert result["surface_class"] == 2
    assert result["surface_thickness_median"] <= 3.0


def test_a_space_filling_class_is_never_called_the_surface():
    label = np.zeros((32, 32, 32), dtype=np.uint8)
    label[:] = 1  # fills the volume
    label[15:17] = 2
    assert audit_label(label)["surface_class"] == 2


def test_empty_label_is_handled():
    result = audit_label(np.zeros((16, 16, 16), dtype=np.uint8))
    assert result["foreground_fraction"] == 0.0
    assert result["surface_class"] is None


@pytest.mark.parametrize("thickness", [2, 3, 4, 6, 9])
def test_capped_thickness_agrees_with_the_exact_transform(thickness):
    """The erosion ladder replaced an exact Euclidean distance transform that
    cost ~90 s per 320³ volume — the entire budget for auditing a release. It
    only earns that if it gives the same answer on sheet-like data."""
    mask = np.zeros((40, 64, 64), dtype=bool)
    mask[20 : 20 + thickness] = True
    capped = thickness_stats(mask)
    exact = thickness_stats(mask, max_thickness=None)
    assert abs(capped["median"] - exact["median"]) < 0.6
    assert abs(capped["p95"] - exact["p95"]) < 0.6


def test_capped_thickness_agrees_on_a_curved_sheet():
    """Where the two could diverge: city-block depth against Euclidean distance.
    Across a thin sheet they do not, because the nearest background voxel lies
    along the normal."""
    zz, yy, _ = np.meshgrid(*[np.arange(s) for s in (48, 64, 64)], indexing="ij")
    radius = np.sqrt((yy - 32.0) ** 2 + (zz - 24.0) ** 2)
    mask = np.abs(radius - 14) < 1.5
    capped = thickness_stats(mask)
    exact = thickness_stats(mask, max_thickness=None)
    assert abs(capped["median"] - exact["median"]) < 0.6
    assert abs(capped["p95"] - exact["p95"]) < 0.6


def test_a_sheet_leaving_the_volume_is_not_reported_as_thinner():
    """The volume's own faces are not background: a sheet running out of the
    patch keeps its thickness."""
    mask = np.zeros((40, 64, 64), dtype=bool)
    mask[20:24] = True  # spans the full cross-section
    assert thickness_stats(mask)["median"] >= 2.0


def test_thickness_reports_saturation_rather_than_a_wrong_number():
    blob = np.zeros((64, 64, 64), dtype=bool)
    blob[8:56, 8:56, 8:56] = True  # far thicker than the cap
    stats = thickness_stats(blob, max_thickness=6)
    assert stats["saturated"] > 0.2
    assert thickness_stats(np.zeros((8, 8, 8), bool))["saturated"] == 0.0


def test_a_bulky_class_is_skipped_rather_than_measured():
    """Sheet metrics on a region that fills a quarter of the patch cost more
    than the sheet itself and mean nothing.  They are skipped, and the record
    says so rather than carrying a misleading number."""
    label = np.zeros((48, 48, 48), dtype=np.uint8)
    label[:24] = 2  # half the volume
    label[30:32] = 1  # the sheet
    result = audit_label(label)
    assert "skipped" in result["per_class"][2]
    assert "thickness" not in result["per_class"][2]
    assert "thickness" in result["per_class"][1]
    assert result["surface_class"] == 1


def test_the_surface_class_is_still_found_when_it_is_not_index_one():
    label = np.zeros((48, 48, 48), dtype=np.uint8)
    label[:30] = 1  # bulky, skipped
    label[36:38] = 2  # the sheet
    result = audit_label(label)
    assert result["surface_class"] == 2
    assert result["surface_thickness_median"] <= 3.0
