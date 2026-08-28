"""The estimator has to survive a volume that looks like a scroll, not a volume
with one convenient sheet in it.

Carbonised papyrus gives, along any surface normal: fibre maxima, both faces of
the sheet, and the neighbouring wrap about 14 voxels away.  These tests build
that, and assert the aggregated estimator recovers a planted displacement where
a per-voxel argmax cannot.
"""
import numpy as np
import pytest

from labelscope.alignment import aggregate_alignment, ridge_alignment


def scroll_like(shape=(64, 96, 96), period=14.5, sheet_z=32.0, sigma=1.6,
                amplitude=22.0, background=60.0, fibre=34.0, noise=10.0, seed=0):
    """Parallel sheets at a realistic winding period, buried in fibrous texture.

    The parameters are chosen so the naive per-voxel estimator degrades the way
    it does on real scroll CT: median |offset| runs 0.67 -> 2.63 voxels as the
    search radius goes 2 -> 9, against 0.70 -> 3.09 measured on
    ``sample_00004`` of the Kaggle surface release.  A synthetic with one clean
    sheet in it would let a broken estimator pass.
    """
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    volume = np.full(shape, background, dtype=np.float32)
    for k in range(-6, 7):
        volume += amplitude * np.exp(-0.5 * ((z - (sheet_z + k * period)) / sigma) ** 2)
    if fibre:
        texture = ndi.gaussian_filter(rng.normal(0, 1, shape).astype(np.float32), 1.6)
        texture /= texture.std() or 1.0
        volume += fibre * texture
    if noise:
        volume += rng.normal(0, noise, shape).astype(np.float32)
    return volume


def sheet_label(shape, z, thickness=1):
    """A label whose centre of mass is exactly at ``z`` for odd thicknesses."""
    mask = np.zeros(shape, dtype=bool)
    start = int(round(z)) - thickness // 2
    mask[start:start + thickness] = True
    return mask


def upward(shape):
    return np.broadcast_to(
        np.arange(shape[0], dtype=np.float32)[:, None, None], shape).copy()


def test_naive_per_voxel_offset_is_radius_dependent_on_scroll_like_data():
    """The failure that motivated the aggregated estimator, pinned as a test:
    a correctly placed label still reads as several voxels off, and the number
    grows with the search window."""
    shape = (64, 96, 96)
    volume = scroll_like(shape)
    mask = sheet_label(shape, 32)
    small = ridge_alignment(volume, mask, radius=3, n_samples=3000,
                            orient_field=upward(shape))
    large = ridge_alignment(volume, mask, radius=9, n_samples=3000,
                            orient_field=upward(shape))
    assert large["median_abs_offset"] > 2.0 * small["median_abs_offset"]


def test_aggregated_estimator_finds_a_correctly_placed_label_at_zero():
    shape = (64, 96, 96)
    result = aggregate_alignment(scroll_like(shape), sheet_label(shape, 32),
                                 orient_field=upward(shape), n_samples=8000,
                                 bootstrap=60)
    assert abs(result["global_peak_offset_raw"]) < 0.4
    assert result["global_profile_snr"] > 1.0


@pytest.mark.parametrize("displacement", [-3, -2, 2, 3])
def test_aggregated_estimator_recovers_a_planted_displacement(displacement):
    shape = (64, 96, 96)
    volume = scroll_like(shape, sheet_z=32.0)
    mask = sheet_label(shape, 32 + displacement)
    result = aggregate_alignment(volume, mask, orient_field=upward(shape),
                                 n_samples=8000, bootstrap=40)
    assert abs(result["global_peak_offset_raw"] + displacement) < 0.6


def test_aggregated_estimator_is_stable_across_search_radii():
    """The property the naive measure lacks: once the window covers the true
    offset, widening it must not change the answer."""
    shape = (64, 96, 96)
    volume = scroll_like(shape)
    mask = sheet_label(shape, 34)          # 2 voxels off
    peaks = [aggregate_alignment(volume, mask, radius=r, orient_field=upward(shape),
                                 n_samples=6000, bootstrap=20)["global_peak_offset_raw"]
             for r in (5, 7, 9)]
    assert max(peaks) - min(peaks) < 0.5


def test_cells_are_reported_and_backed_by_enough_voxels():
    shape = (64, 96, 96)
    result = aggregate_alignment(scroll_like(shape), sheet_label(shape, 32),
                                 cell=32, min_per_cell=100, orient_field=upward(shape),
                                 n_samples=8000, bootstrap=20)
    assert result["n_cells"] >= 4
    assert result["cell_voxels_median"] >= 100
    assert result["cell_abs_offset_median"] < 0.6


def test_bootstrap_interval_widens_when_the_signal_is_weaker():
    shape = (64, 96, 96)
    strong = aggregate_alignment(scroll_like(shape, noise=3.0), sheet_label(shape, 32),
                                 orient_field=upward(shape), n_samples=6000, bootstrap=120)
    weak = aggregate_alignment(scroll_like(shape, amplitude=18.0, noise=30.0, seed=3),
                               sheet_label(shape, 32),
                               orient_field=upward(shape), n_samples=6000, bootstrap=120)
    width = lambda r: r["global_peak_ci95"][1] - r["global_peak_ci95"][0]
    assert width(weak) > width(strong)
    assert strong["global_profile_snr"] > weak["global_profile_snr"]


def test_winding_spacing_is_recovered_from_the_scan():
    from labelscope.alignment import neighbour_ridge_distance

    shape = (96, 96, 96)
    volume = scroll_like(shape, period=14.5, sheet_z=48.0)
    result = neighbour_ridge_distance(volume, sheet_label(shape, 48),
                                      orient_field=upward(shape), n_samples=6000)
    assert result["winding_spacing"] is not None
    assert abs(result["winding_spacing"] - 14.5) < 3.0
    assert 3.0 <= result["recommended_radius"] <= 12.0
    assert result["recommended_radius"] < result["winding_spacing"] / 2.0


def test_auto_radius_never_reaches_the_neighbouring_wrap():
    """The whole point: an estimator allowed to search past half the winding
    spacing can lock onto a different sheet and call it a displacement."""
    from labelscope.alignment import neighbour_ridge_distance

    shape = (96, 96, 96)
    for period in (12.0, 20.0, 30.0):
        volume = scroll_like(shape, period=period, sheet_z=48.0)
        result = neighbour_ridge_distance(volume, sheet_label(shape, 48),
                                          orient_field=upward(shape), n_samples=6000)
        assert result["recommended_radius"] < 0.5 * result["winding_spacing"] + 0.01


def test_a_cell_with_no_peak_in_the_window_is_counted_not_invented():
    """A label sitting in a void has no ridge under it.  The estimator must
    report that as unresolved rather than quote the window edge."""
    shape = (64, 96, 96)
    volume = np.full(shape, 60.0, dtype=np.float32)
    volume += np.random.default_rng(0).normal(0, 4.0, shape).astype(np.float32)
    volume[10] += 200.0                               # one far-away bright plane
    result = aggregate_alignment(volume, sheet_label(shape, 40), radius=6.0,
                                 orient_field=upward(shape), n_samples=8000,
                                 bootstrap=20, min_snr=8.0)
    assert result["cell_frac_unresolved"] + result["cell_frac_low_snr"] > 0.5


def test_reported_cells_never_sit_at_the_window_edge():
    shape = (64, 96, 96)
    result = aggregate_alignment(scroll_like(shape), sheet_label(shape, 32),
                                 radius=6.0, orient_field=upward(shape),
                                 n_samples=8000, bootstrap=20)
    if result.get("n_cells"):
        assert abs(result["cell_offset_worst"]) < 6.0


def test_polarity_is_not_inferred_from_a_displaced_label():
    """Inferring polarity from the profile assumes the label is already on the
    sheet.  A label sitting in the gap between wraps reads as darker than the
    window edges, and auto-detection then measures the gap instead of the sheet.
    """
    from labelscope.alignment import sample_profiles

    shape = (64, 96, 96)
    volume = scroll_like(shape, sheet_z=32.0)
    displaced = sheet_label(shape, 37)          # 5 vx off, still nearer its own wrap
    _, _, _, guessed = sample_profiles(volume, displaced, radius=6.0, n_samples=4000,
                                       orient_field=upward(shape), polarity="auto")
    assert guessed == "dark"                    # the trap

    result = aggregate_alignment(volume, displaced, orient_field=upward(shape),
                                 n_samples=8000, bootstrap=20)
    assert result["polarity"] == "bright"       # the default does not fall for it
    assert abs(result["global_peak_offset_raw"] + 5.0) < 1.0


def test_a_patch_with_no_sheet_contrast_reports_no_offset_at_all():
    """The gate that keeps a +9.5 voxel reading out of a report.  On the Kaggle
    release every wild global offset came from a patch whose sheet contrast was
    barely above the voxel noise; one had a bootstrap interval of [-8.5, +8.4],
    the estimator flipping between two wraps."""
    shape = (64, 96, 96)
    noise_only = np.random.default_rng(0).normal(90, 20, shape).astype(np.float32)
    result = aggregate_alignment(noise_only, sheet_label(shape, 32), radius=6.0,
                                 orient_field=upward(shape), n_samples=8000,
                                 bootstrap=20, min_global_snr=2.0)
    assert result["global_offset_reliable"] is False
    assert result["global_peak_offset"] is None
    assert result["global_peak_offset_raw"] is not None      # still available


def test_a_clear_sheet_passes_the_reliability_gate():
    shape = (64, 96, 96)
    clean = scroll_like(shape, amplitude=60.0, fibre=6.0, noise=3.0)
    result = aggregate_alignment(clean, sheet_label(shape, 32), radius=6.0,
                                 orient_field=upward(shape), n_samples=8000,
                                 bootstrap=20, min_global_snr=2.0)
    assert result["global_offset_reliable"] is True
    assert result["global_peak_offset"] == result["global_peak_offset_raw"]
