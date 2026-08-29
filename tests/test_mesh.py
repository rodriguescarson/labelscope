"""The sheet-switch detector, against planted switches.

A surface displaced by a whole winding is the case villa's own spiral
satisfaction metric scores as no change at all (ScrollPrize/villa#1621), so the
bar here is that a planted displacement is found and a clean surface is not.
"""

import json

import numpy as np
import pytest
import tifffile

from labelscope.mesh import QuadMesh, displace, edge_dip, find_sheet_switches, read_tifxyz


def wrapped_volume(
    shape=(160, 96, 160),
    spacing=24.0,
    sheet_sigma=2.5,
    amplitude=150.0,
    background=25.0,
    noise=6.0,
    seed=0,
):
    """Concentric-ish sheets stacked along z, like wraps seen in a small window."""
    rng = np.random.default_rng(seed)
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    volume = np.full(shape, background, dtype=np.float32)
    for k in range(-2, int(shape[0] / spacing) + 2):
        volume += amplitude * np.exp(-0.5 * ((z - k * spacing) / sheet_sigma) ** 2)
    volume += rng.normal(0, noise, shape).astype(np.float32)
    return np.clip(volume, 0, 255)


def mesh_on_sheet(shape=(160, 96, 160), sheet_z=72.0, rows=40, cols=40, step=3.0):
    """A flat grid lying exactly on one sheet."""
    rr = np.arange(rows, dtype=np.float32) * step + 10.0
    cc = np.arange(cols, dtype=np.float32) * step + 10.0
    yy, xx = np.meshgrid(rr, cc, indexing="ij")
    zz = np.full_like(yy, sheet_z)
    points = np.stack([zz, yy, xx], axis=-1)
    return QuadMesh(points=points, valid=np.ones(points.shape[:2], bool), meta={})


def test_tifxyz_round_trip(tmp_path):
    points = np.zeros((5, 6, 3), dtype=np.float32)
    points[..., 0] = 3.0
    points[2, 2] = -1.0
    for i, axis in enumerate(("z", "y", "x")):
        tifffile.imwrite(str(tmp_path / f"{axis}.tif"), points[..., i])
    (tmp_path / "meta.json").write_text(json.dumps({"format": "tifxyz"}))

    mesh = read_tifxyz(str(tmp_path))
    assert mesh.shape == (5, 6)
    assert mesh.meta["format"] == "tifxyz"
    assert not mesh.valid[2, 2] and mesh.valid.sum() == 29


def test_a_missing_tifxyz_channel_is_an_error(tmp_path):
    tifffile.imwrite(str(tmp_path / "x.tif"), np.zeros((3, 3), np.float32))
    with pytest.raises(FileNotFoundError):
        read_tifxyz(str(tmp_path))


def test_grid_step_and_normals_of_a_flat_sheet():
    mesh = mesh_on_sheet(step=3.0)
    assert abs(mesh.grid_step() - 3.0) < 0.2
    normals = mesh.normals()
    assert np.abs(normals[..., 0]).mean() > 0.95  # normal is along z


def test_a_clean_surface_shows_no_seam():
    volume = wrapped_volume()
    mesh = mesh_on_sheet()
    result = find_sheet_switches(mesh, volume)
    assert result["n_seams"] == 0
    assert result["max_z"] < 5.0


def test_a_whole_winding_displacement_is_detected():
    """The exact case the satisfaction metric scores as zero change."""
    volume = wrapped_volume(spacing=24.0)
    mesh = mesh_on_sheet(sheet_z=72.0)
    switched = displace(mesh, 24.0)
    result = find_sheet_switches(switched, volume)
    assert result["n_seams"] >= 1
    assert result["max_z"] > 6.0
    seam = result["seams"][0]
    assert seam["axis"] == 1
    assert abs(seam["line"] - (mesh.shape[1] // 2 - 1)) <= 1


@pytest.mark.parametrize("windings", [1, 2, 3])
def test_any_whole_number_of_windings_is_detected(windings):
    """villa#1621: acceptance depends on distance from the nearest whole winding,
    not on magnitude, so a patch 23 windings out passes.  Detection here must not
    fade the same way."""
    volume = wrapped_volume(spacing=24.0)
    mesh = mesh_on_sheet(sheet_z=48.0)
    result = find_sheet_switches(displace(mesh, 24.0 * windings), volume)
    assert result["max_z"] > 6.0, f"{windings} windings went undetected"


def test_the_seam_is_where_the_displacement_starts():
    volume = wrapped_volume(spacing=24.0)
    mesh = mesh_on_sheet(sheet_z=72.0)
    region = np.zeros(mesh.shape, dtype=bool)
    region[:, 30:] = True
    result = find_sheet_switches(displace(mesh, 24.0, region=region), volume)
    assert result["n_seams"] >= 1
    assert abs(result["seams"][0]["line"] - 29) <= 1


def test_edge_dip_is_nan_where_a_vertex_is_missing():
    volume = wrapped_volume()
    mesh = mesh_on_sheet()
    mesh.valid[5, 5] = False
    dip = edge_dip(mesh, volume)
    assert np.isnan(dip[0][4, 5]) and np.isnan(dip[1][5, 4])


def test_the_detector_reads_a_chunked_volume_the_same_way():
    """The remote path must give the same answer as the in-memory one, or the
    fleet-wide run is measuring something different from the tests."""
    from labelscope.remote_zarr import ChunkedVolume

    volume = wrapped_volume(spacing=24.0).astype(np.uint8)

    class LocalStore:
        def get(self, url, timeout=None):
            key = tuple(int(p) for p in url.rsplit("/", 3)[-3:])
            lo = np.array(key) * 32
            block = np.zeros((32, 32, 32), np.uint8)
            hi = np.minimum(lo + 32, volume.shape)
            sub = volume[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
            block[tuple(slice(0, s) for s in sub.shape)] = sub

            class R:
                status_code = 200
                content = block.tobytes()

                def raise_for_status(self):
                    pass

            return R()

    remote = ChunkedVolume(
        "http://x/0", volume.shape, (32, 32, 32), "|u1", session=LocalStore()
    )
    mesh = mesh_on_sheet(sheet_z=72.0)
    switched = displace(mesh, 24.0)

    local_result = find_sheet_switches(switched, volume.astype(np.float32))
    remote_result = find_sheet_switches(switched, remote)
    assert abs(local_result["max_z"] - remote_result["max_z"]) < 0.5
    assert local_result["n_seams"] == remote_result["n_seams"]
    assert remote.chunks_fetched > 0


def test_the_detector_refuses_to_conclude_at_an_inadequate_resolution():
    """At 45.5 um on PHercParis4 the mesh grid step is about 18 voxels against a
    12.5 voxel winding spacing: every edge already crosses a gap, and the
    statistic is measuring roughness.  The tool has to say so rather than report
    seams it cannot distinguish."""
    volume = wrapped_volume(spacing=10.0, sheet_sigma=1.5)  # wraps closer than the grid
    mesh = mesh_on_sheet(sheet_z=70.0, step=12.0)  # step 12 vs spacing 10
    result = find_sheet_switches(mesh, volume)
    assert result["resolution_adequate"] is False
    assert result["n_seams"] == 0
    assert "seams_unreliable" in result


def test_an_adequate_resolution_is_recognised():
    volume = wrapped_volume(spacing=24.0)
    mesh = mesh_on_sheet(sheet_z=72.0, step=3.0)
    result = find_sheet_switches(mesh, volume)
    assert result["resolution_adequate"] is True
    assert result["steps_per_winding"] > 4.0


def masked_volume(shape=(160, 96, 320), spacing=24.0, seed=0):
    """A scan that is masked out where this surface lies.

    The published volumes are masked: the air around the scroll is simply absent
    from the store and reads as zero.  A mesh can be perfectly well-formed over a
    region the scan does not cover, and then every edge dips by nothing.
    """
    volume = wrapped_volume(shape=shape, spacing=spacing, seed=seed)
    volume[:, :, 160:] = 0.0  # the scan covers the left half only
    return volume


def test_a_surface_over_masked_air_is_refused_not_scored():
    """The loudest false positive in a 56-surface sweep came from this case.

    Every line mean was 0.00, the robust spread collapsed, and a 0.25 grey-level
    wobble was reported as z = 12.6 -- the highest score in the whole run, made
    of nothing.
    """
    volume = masked_volume()
    mesh = mesh_on_sheet(rows=40, cols=40, step=3.0)
    # move the whole grid into the masked half
    mesh = QuadMesh(
        points=mesh.points + np.array([0.0, 0.0, 170.0], np.float32),
        valid=mesh.valid.copy(),
        meta={},
    )
    out = find_sheet_switches(mesh, volume, z_threshold=5.0)
    assert out["dip_degenerate"] is True
    assert out["surface_intensity_median"] < 1.0
    assert out["n_seams"] == 0
    assert "seams_degenerate" in out


def test_a_surface_on_real_signal_is_still_scored():
    """The guard must not silence a surface the scan actually covers."""
    volume = masked_volume()
    mesh = mesh_on_sheet(rows=40, cols=40, step=3.0)  # sits in the covered half
    out = find_sheet_switches(displace(mesh, 24.0), volume, z_threshold=5.0)
    assert out["dip_degenerate"] is False
    assert out["surface_intensity_median"] > 1.0
    assert out["n_seams"] >= 1


def test_a_collapsed_spread_scores_zero_rather_than_falling_back():
    """A degenerate null has no z-scores in it."""
    from labelscope.mesh import _line_scores

    dip = np.zeros((8, 8))
    dip[3, :] = 0.25  # one line barely above a floor of exact zeros
    scores = _line_scores(dip, along=1)
    assert np.all(scores == 0.0)
