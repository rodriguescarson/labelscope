"""The alignment metric is only worth reporting if it recovers an offset we
planted ourselves.  These tests build synthetic sheets with a known truth."""

import numpy as np
import pytest

from labelscope.alignment import local_contrast, ridge_alignment, surface_normals


def synthetic_sheet(
    shape=(64, 64, 64),
    sheet_z=32.0,
    sigma=1.6,
    amplitude=140.0,
    background=40.0,
    noise=0.0,
    seed=0,
):
    """A CT-like volume whose one bright sheet sits at ``sheet_z`` (float)."""
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    volume = background + amplitude * np.exp(-0.5 * ((z - sheet_z) / sigma) ** 2)
    volume = np.broadcast_to(volume, shape).astype(np.float32).copy()
    if noise:
        volume += np.random.default_rng(seed).normal(0, noise, shape).astype(np.float32)
    return volume


def planar_label(shape, label_z):
    mask = np.zeros(shape, dtype=bool)
    mask[int(round(label_z))] = True
    return mask


def upward(shape):
    """A reference field that increases with z, so normals orient to +z."""
    return np.broadcast_to(np.arange(shape[0], dtype=np.float32)[:, None, None], shape).copy()


def test_normals_of_a_flat_sheet_point_along_z():
    mask = planar_label((48, 48, 48), 24)
    normals = surface_normals(mask)
    on_sheet = normals[:, 24, 8:-8, 8:-8]
    assert np.abs(on_sheet[0]).mean() > 0.9  # z component dominates
    assert np.abs(on_sheet[1]).mean() < 0.1
    assert np.abs(on_sheet[2]).mean() < 0.1


def test_normals_of_a_tilted_sheet_follow_the_tilt():
    shape = (48, 48, 48)
    zz, yy, _ = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    mask = np.abs(zz - (20 + 0.5 * yy)) < 0.5  # plane tilted 0.5 in z per y
    normals = surface_normals(mask)
    sampled = normals[:, mask]
    ratio = np.abs(sampled[1] / np.where(sampled[0] == 0, 1e-6, sampled[0]))
    assert 0.35 < np.median(ratio) < 0.65  # dz/dy = 0.5


def test_perfectly_placed_label_reports_zero_offset():
    shape = (64, 64, 64)
    volume = synthetic_sheet(shape, sheet_z=32.0)
    result = ridge_alignment(
        volume, planar_label(shape, 32), n_samples=4000, orient_field=upward(shape)
    )
    assert result["polarity"] == "bright"
    assert abs(result["mean_signed_offset"]) < 0.15
    assert result["median_abs_offset"] < 0.15
    assert result["frac_offset_ge_1vx"] < 0.01


@pytest.mark.parametrize("displacement", [-2, -1, 1, 2])
def test_a_planted_displacement_is_recovered(displacement):
    """A label at 32+d against a sheet at 32 must read as an offset of -d:
    the ridge lies d voxels *against* the outward normal."""
    shape = (64, 64, 64)
    volume = synthetic_sheet(shape, sheet_z=32.0)
    result = ridge_alignment(
        volume,
        planar_label(shape, 32 + displacement),
        n_samples=4000,
        orient_field=upward(shape),
    )
    assert abs(result["mean_signed_offset"] + displacement) < 0.25
    assert result["frac_offset_ge_1vx"] > 0.9


def test_subvoxel_displacement_is_recovered():
    """The sheet sits at 32.5; a label on voxel 32 is half a voxel off."""
    shape = (64, 64, 64)
    volume = synthetic_sheet(shape, sheet_z=32.5)
    result = ridge_alignment(
        volume, planar_label(shape, 32), n_samples=4000, orient_field=upward(shape)
    )
    assert 0.3 < abs(result["mean_signed_offset"]) < 0.75


def test_dark_sheets_are_handled():
    shape = (64, 64, 64)
    volume = 255.0 - synthetic_sheet(shape, sheet_z=32.0)
    result = ridge_alignment(
        volume,
        planar_label(shape, 30),
        n_samples=4000,
        orient_field=upward(shape),
        polarity="auto",
    )
    assert result["polarity"] == "dark"
    assert abs(abs(result["mean_signed_offset"]) - 2.0) < 0.3


def test_label_on_featureless_volume_is_flagged_as_flat_support():
    shape = (64, 64, 64)
    volume = np.full(shape, 90.0, dtype=np.float32)
    volume += np.random.default_rng(0).normal(0, 0.05, shape).astype(np.float32)
    result = ridge_alignment(
        volume, planar_label(shape, 32), n_samples=4000, orient_field=upward(shape)
    )
    assert result["frac_flat_support"] > 0.5


def test_noise_raises_offset_spread_but_not_the_bias():
    shape = (64, 64, 64)
    clean = ridge_alignment(
        synthetic_sheet(shape, 32.0),
        planar_label(shape, 32),
        n_samples=4000,
        orient_field=upward(shape),
    )
    noisy = ridge_alignment(
        synthetic_sheet(shape, 32.0, noise=12.0),
        planar_label(shape, 32),
        n_samples=4000,
        orient_field=upward(shape),
    )
    assert noisy["median_abs_offset"] >= clean["median_abs_offset"]
    assert abs(noisy["mean_signed_offset"]) < 0.3  # unbiased despite noise


def test_hazy_volume_scores_lower_local_contrast():
    from scipy import ndimage as ndi

    shape = (64, 64, 64)
    sharp = synthetic_sheet(shape, 32.0, sigma=1.0)
    hazy = ndi.gaussian_filter(sharp, 3.0)
    assert local_contrast(hazy)["hf_energy_norm"] < local_contrast(sharp)["hf_energy_norm"]


def test_empty_mask_is_reported_not_crashed():
    shape = (32, 32, 32)
    result = ridge_alignment(synthetic_sheet(shape, 16.0), np.zeros(shape, bool))
    assert result["n_samples"] == 0


def test_orientation_propagation_makes_a_sheet_consistent():
    """Local PCA returns an axis, not a direction, so a fitted normal field
    arrives with random signs.  Propagation has to remove them."""
    from labelscope.alignment import point_normals, propagate_orientation

    rng = np.random.default_rng(0)
    coords = (
        np.stack(
            np.meshgrid(np.arange(2), np.arange(24), np.arange(24), indexing="ij"), axis=-1
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )
    normals = point_normals(coords, coords)
    flip = rng.random(coords.shape[0]) < 0.5
    normals[:, flip] *= -1.0  # scramble the signs
    before = ((normals * normals.mean(axis=1, keepdims=True)).sum(axis=0) > 0).mean()
    after_field, _ = propagate_orientation(coords, normals)
    mean = after_field.mean(axis=1, keepdims=True)
    after = ((after_field * mean).sum(axis=0) > 0).mean()
    assert before < 0.75
    assert after > 0.99


def test_orientation_propagation_spans_several_parallel_sheets():
    """Concentric wraps are separate surfaces, but their normals are parallel,
    so a single consistent orientation across all of them is both possible and
    what the measurement needs."""
    from labelscope.alignment import point_normals, propagate_orientation

    rng = np.random.default_rng(1)
    sheets = []
    for z in (4, 18, 32):
        grid = np.stack(
            np.meshgrid([z], np.arange(20), np.arange(20), indexing="ij"), axis=-1
        ).reshape(-1, 3)
        sheets.append(grid)
    coords = np.concatenate(sheets).astype(np.float32)
    normals = point_normals(coords, coords)
    normals[:, rng.random(coords.shape[0]) < 0.5] *= -1.0
    oriented, components = propagate_orientation(coords, normals)
    # bridging must fold the separate wraps into one consistently oriented set,
    # so that only a single global sign is left to decide
    assert len(np.unique(components)) == 1
    mean = oriented.mean(axis=1, keepdims=True)
    assert ((oriented * mean).sum(axis=0) > 0).mean() > 0.99


def test_global_orientation_anchor_beats_a_pointwise_one_far_from_the_reference():
    """The reference field is reliable in aggregate and useless voxel by voxel
    once it is far away; the default takes one decision for the whole surface."""
    from labelscope.alignment import orient_normals, point_normals, propagate_orientation

    shape = (64, 32, 32)
    coords = (
        np.stack(np.meshgrid([40], np.arange(32), np.arange(32), indexing="ij"), axis=-1)
        .reshape(-1, 3)
        .astype(np.float32)
    )
    normals, _ = propagate_orientation(coords, point_normals(coords, coords))
    reference = np.zeros(shape, dtype=np.float32)
    reference[:4] = 1.0  # the void, 36 voxels away
    globally = orient_normals(normals, coords.T, reference)
    assert np.allclose(np.abs((globally * globally[:, :1]).sum(axis=0)), 1.0, atol=1e-4)
