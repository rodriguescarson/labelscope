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


def test_tiles_never_overlap(volume):
    from labelscope.onsheet import _tiles

    tiles = _tiles(40, 40, 12, np.random.default_rng(0))
    assert len(tiles) == 9  # 40 // 12 == 3 per side
    seen = set()
    for r, c in tiles:
        cells = {(r + i, c + j) for i in range(12) for j in range(12)}
        assert not (cells & seen)
        seen |= cells


def test_lazy_and_eager_readers_give_identical_blocks(tmp_path, volume):
    """The memory-mapped path must reproduce the in-memory path exactly.

    Written the way the published corpus is written -- three uncompressed
    float32 TIFFs plus meta.json -- so this exercises the real mapping.
    """
    import tifffile

    from labelscope.mesh import LazyQuadMesh, QuadMesh, read_tifxyz

    mesh = mesh_on_sheet()
    d = tmp_path / "surf.tifxyz"
    d.mkdir()
    for axis, idx in (("z", 0), ("y", 1), ("x", 2)):
        tifffile.imwrite(d / f"{axis}.tif", mesh.points[..., idx].astype(np.float32))
    (d / "meta.json").write_text("{}")

    eager = read_tifxyz(str(d))
    lazy = read_tifxyz(str(d), lazy=True)
    assert isinstance(eager, QuadMesh)
    assert isinstance(lazy, LazyQuadMesh)
    assert lazy.shape == eager.shape

    a = profiles(eager, volume)
    b = profiles(lazy, volume)
    assert [x["block"] for x in a] == [x["block"] for x in b]
    assert np.allclose([x["range"] for x in a], [x["range"] for x in b])
    assert np.allclose([x["at_zero"] for x in a], [x["at_zero"] for x in b])


def test_lazy_auto_reads_small_files_whole(tmp_path):
    import tifffile

    from labelscope.mesh import QuadMesh, read_tifxyz

    mesh = mesh_on_sheet()
    d = tmp_path / "small.tifxyz"
    d.mkdir()
    for axis, idx in (("z", 0), ("y", 1), ("x", 2)):
        tifffile.imwrite(d / f"{axis}.tif", mesh.points[..., idx].astype(np.float32))
    assert isinstance(read_tifxyz(str(d), lazy="auto"), QuadMesh)


def test_lazy_falls_back_when_not_mappable(tmp_path, volume):
    """A compressed TIFF cannot be memory-mapped; the reader must still work."""
    import tifffile

    from labelscope.mesh import QuadMesh, read_tifxyz

    mesh = mesh_on_sheet()
    d = tmp_path / "zipped.tifxyz"
    d.mkdir()
    for axis, idx in (("z", 0), ("y", 1), ("x", 2)):
        tifffile.imwrite(
            d / f"{axis}.tif", mesh.points[..., idx].astype(np.float32), compression="zlib"
        )
    got = read_tifxyz(str(d), lazy=True)
    assert isinstance(got, QuadMesh)
    assert profiles(got, volume)


def test_summarise_reports_spread():
    blocks = [
        {"range": r, "peak_offset": 0.0, "n": 100} for r in [1, 2, 3, 40, 50, 60, 70, 80]
    ]
    s = summarise("bimodal", blocks)
    assert s["columns"] == 800
    assert s["range_p10"] < 3 and s["range_p90"] > 60
    assert s["range_iqr_over_median"] > 1.0  # a broad, two-humped sample must say so


def test_surface_volume_profiles_reads_a_chunk(monkeypatch):
    """One synthetic chunk: bright at the middle layer over the surface footprint."""
    import labelscope.onsheet as on

    cube = np.zeros((on.SV_LAYERS, on.SV_SIDE, on.SV_SIDE), np.uint8)
    cube[:, :, :96] = 20  # surface covers 3/4 of the chunk
    cube[on.SV_LAYERS // 2, :, :96] = 200  # the sheet, at the middle layer
    raw = cube.tobytes()

    monkeypatch.setattr(
        on,
        "_s3_list",
        lambda url, delimiter: (
            ["http://x/0/0/7/"]
            if delimiter
            else ["http://x/0/0/7/3", "http://x/0/0/7/4", "http://x/0/0/7/5"]
        ),
    )
    monkeypatch.setattr(
        on,
        "_sv_geometry",
        lambda store: {
            "layers": on.SV_LAYERS,
            "side": (on.SV_SIDE, on.SV_SIDE),
            "dtype": np.dtype("u1"),
            "order": "C",
            "codec": None,
        },
    )

    class _Resp:
        def read(self):
            return raw

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: _Resp())
    found = on.surface_volume_profiles("http://x", chunks=3, seed=0)
    assert len(found) == 3
    assert found[0]["n"] == 128 * 96
    assert found[0]["range"] == 180.0
    assert found[0]["peak_offset"] == 0.0
    assert found[0]["block"][0] == "7"
    assert len({tuple(f["block"]) for f in found}) == 3  # never the same chunk twice


def test_surface_volume_profiles_rejects_sparse_chunks(monkeypatch):
    import labelscope.onsheet as on

    cube = np.zeros((on.SV_LAYERS, on.SV_SIDE, on.SV_SIDE), np.uint8)
    cube[:, :, :16] = 50  # only 1/8 of the chunk carries surface
    monkeypatch.setattr(
        on,
        "_s3_list",
        lambda url, delimiter: ["http://x/0/0/1/"] if delimiter else ["http://x/0/0/1/1"],
    )
    monkeypatch.setattr(
        on,
        "_sv_geometry",
        lambda store: {
            "layers": on.SV_LAYERS,
            "side": (on.SV_SIDE, on.SV_SIDE),
            "dtype": np.dtype("u1"),
            "order": "C",
            "codec": None,
        },
    )

    class _Resp:
        def read(self):
            return cube.tobytes()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: _Resp())
    assert on.surface_volume_profiles("http://x", chunks=2, seed=0) == []


def test_surface_volume_profiles_honours_a_33_layer_store(monkeypatch):
    """PHerc0172's band is 33 layers, not 109; the geometry must come from .zarray."""
    import labelscope.onsheet as on

    cube = np.zeros((33, 128, 128), np.uint8)
    cube[:, :, :] = 30
    cube[16, :, :] = 200
    monkeypatch.setattr(
        on,
        "_s3_list",
        lambda url, delimiter: ["http://x/0/0/2/"] if delimiter else ["http://x/0/0/2/9"],
    )
    monkeypatch.setattr(
        on,
        "_sv_geometry",
        lambda store: {
            "layers": 33,
            "side": (128, 128),
            "dtype": np.dtype("u1"),
            "order": "C",
            "codec": None,
        },
    )

    class _Resp:
        def read(self):
            return cube.tobytes()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: _Resp())
    found = on.surface_volume_profiles("http://x", chunks=1, seed=0)
    assert len(found) == 1 and found[0]["range"] == 170.0 and found[0]["peak_offset"] == 0.0
