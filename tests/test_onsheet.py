"""The on-sheet check, against surfaces that are and are not on papyrus.

The bar here is the one the corpus pass actually relies on: a surface lying on a
sheet must show real dynamic range along its normal, a surface tilted across the
wraps must not, and :func:`compare` must separate the two from block-level data.
"""

import numpy as np
import pytest

from labelscope.mesh import QuadMesh
from labelscope.onsheet import block_profiles, compare, summarise, verdict


def wrapped_volume(shape=(160, 96, 160), spacing=24.0, seed=0):
    """Concentric-ish sheets stacked along z, like wraps in a small window.

    Defined here rather than imported from test_mesh: a `tests.` import only
    resolves under `python -m pytest` and has broken CI before.
    """
    rng = np.random.default_rng(seed)
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    volume = np.full(shape, 25.0, dtype=np.float32)
    for k in range(-2, int(shape[0] / spacing) + 2):
        volume += 150.0 * np.exp(-0.5 * ((z - k * spacing) / 2.5) ** 2)
    volume += rng.normal(0, 6.0, shape).astype(np.float32)
    return np.clip(volume, 0, 255)


def mesh_on_sheet(sheet_z=72.0, rows=40, cols=40, step=3.0):
    """A flat grid lying exactly on one sheet."""
    rr = np.arange(rows, dtype=np.float32) * step + 10.0
    cc = np.arange(cols, dtype=np.float32) * step + 10.0
    yy, xx = np.meshgrid(rr, cc, indexing="ij")
    zz = np.full_like(yy, sheet_z)
    points = np.stack([zz, yy, xx], axis=-1)
    return QuadMesh(points=points, valid=np.ones(points.shape[:2], bool), meta={})


def mesh_across_sheets(rows=40, cols=40, step=3.0, spacing=24.0):
    """A grid tilted so it cuts through the wraps instead of following one.

    Ramping z across the grid means the surface passes through several sheets and
    the gaps between them, which is the failure the check exists to catch.
    """
    rr = np.arange(rows, dtype=np.float32) * step + 10.0
    cc = np.arange(cols, dtype=np.float32) * step + 10.0
    yy, xx = np.meshgrid(rr, cc, indexing="ij")
    zz = 72.0 + (xx - xx.mean()) * (spacing / (cols * step)) * 4.0
    points = np.stack([zz, yy, xx], axis=-1)
    return QuadMesh(points=points, valid=np.ones(points.shape[:2], bool), meta={})


@pytest.fixture(scope="module")
def volume():
    return wrapped_volume()


def profiles(mesh, volume, **kw):
    kw.setdefault("reach", 36.0)
    kw.setdefault("blocks", 8)
    kw.setdefault("block_size", 8)
    kw.setdefault("seed", 0)
    kw.setdefault("step", 1.0)
    return block_profiles(mesh, volume, None, **kw)


def test_on_sheet_surface_has_dynamic_range(volume):
    found = profiles(mesh_on_sheet(), volume)
    assert found, "a clean surface should yield usable blocks"
    assert summarise("clean", found)["range_median"] > 60.0


def test_surface_across_sheets_is_flatter(volume):
    clean = summarise("clean", profiles(mesh_on_sheet(), volume))
    across = summarise("across", profiles(mesh_across_sheets(), volume))
    assert across["range_median"] < clean["range_median"]


def test_compare_separates_the_two(volume):
    stats = compare(profiles(mesh_across_sheets(), volume), profiles(mesh_on_sheet(), volume))
    assert stats["median_a"] < stats["median_b"]
    assert stats["p_less"] < 0.05


def test_compare_finds_no_difference_between_two_clean_surfaces(volume):
    a = profiles(mesh_on_sheet(), volume, seed=1)
    b = profiles(mesh_on_sheet(), volume, seed=2)
    assert compare(a, b)["p_less"] > 0.05


def test_peak_sits_near_the_surface_when_on_sheet(volume):
    """Reach must stay inside one winding, or a neighbouring sheet wins the argmax.

    The synthetic wraps sit 24 voxels apart, so sampling +/-36 puts two more
    sheets in the window and the peak legitimately lands on one of them. That is
    the same effect that makes absolute peak offset a weak signal on real data.
    """
    found = profiles(mesh_on_sheet(), volume, reach=10.0)
    assert summarise("clean", found)["peak_offset_abs_median"] <= 4.0


def test_summarise_reports_error_without_blocks():
    assert "error" in summarise("empty", [])


def test_compare_reports_error_without_blocks(volume):
    assert "error" in compare([], profiles(mesh_on_sheet(), volume))


def test_verdict_thresholds():
    assert verdict(50.0, 100.0)[0] == "ON SHEET"
    assert verdict(35.0, 100.0)[0] == "marginal"
    assert verdict(20.0, 100.0)[0] == "OFF SHEET"


def test_blocks_require_enough_valid_vertices(volume):
    """A grid that is mostly holes should yield no blocks rather than junk."""
    mesh = mesh_on_sheet()
    mesh.valid[:] = False
    mesh.valid[::7, ::7] = True
    assert profiles(mesh, volume) == []
