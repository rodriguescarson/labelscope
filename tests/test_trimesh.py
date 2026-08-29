"""Sheet-switch detection on triangular meshes.

Same bar as the quad detector in ``test_mesh.py``: a planted whole-winding
displacement has to be found, and a clean surface has to stay clean.  The extra
question here is whether the unstructured form of the statistic -- a connected
component of flagged edges instead of a grid line -- keeps that behaviour.
"""

import numpy as np
import pytest
from tests.test_mesh import mesh_on_sheet, wrapped_volume

from labelscope import trimesh as tm
from labelscope.mesh import QuadMesh


def write_obj(path, points_zyx, faces):
    with open(path, "w") as handle:
        for p in points_zyx:
            handle.write(f"v {p[2]} {p[1]} {p[0]}\n")
        for f in faces:
            handle.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def test_obj_round_trip_preserves_zyx_order(tmp_path):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], np.float32)
    path = tmp_path / "s.obj"
    write_obj(path, points, [(0, 1, 2)])

    mesh = tm.read_obj(str(path))
    assert mesh.n_vertices == 3 and mesh.n_faces == 1
    np.testing.assert_allclose(mesh.points, points)


def test_obj_quad_faces_are_fanned_and_slashes_ignored(tmp_path):
    path = tmp_path / "q.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1 4/1/1\n"
    )
    mesh = tm.read_obj(str(path))
    assert mesh.n_vertices == 4
    assert mesh.n_faces == 2  # one quad becomes two triangles
    assert len(mesh.edges()) == 5  # four sides plus the diagonal


def test_obj_negative_indices_count_back_from_the_end(tmp_path):
    path = tmp_path / "n.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
    mesh = tm.read_obj(str(path))
    np.testing.assert_array_equal(np.sort(mesh.faces[0]), [0, 1, 2])


def test_obj_without_faces_is_an_error(tmp_path):
    path = tmp_path / "p.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n")
    with pytest.raises(ValueError, match="no faces"):
        tm.read_obj(str(path))


def test_ply_ascii_and_binary_agree(tmp_path):
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    header = (
        "ply\nformat {fmt} 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    )
    ascii_path = tmp_path / "a.ply"
    ascii_path.write_text(header.format(fmt="ascii") + "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    bin_path = tmp_path / "b.ply"
    with open(bin_path, "wb") as handle:
        handle.write(header.format(fmt="binary_little_endian").encode())
        handle.write(verts.astype("<f4").tobytes())
        handle.write(bytes([3]) + np.array([0, 1, 2], "<i4").tobytes())

    a = tm.read_ply(str(ascii_path))
    b = tm.read_ply(str(bin_path))
    np.testing.assert_allclose(a.points, b.points)
    np.testing.assert_array_equal(np.sort(a.faces), np.sort(b.faces))
    np.testing.assert_allclose(a.points[1], [0.0, 0.0, 1.0])  # x=1 lands in the x slot


def test_read_trimesh_rejects_an_unknown_extension(tmp_path):
    path = tmp_path / "s.stl"
    path.write_text("")
    with pytest.raises(ValueError, match="unsupported mesh format"):
        tm.read_trimesh(str(path))


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_from_quad_drops_missing_vertices_and_keeps_the_surface():
    quad = mesh_on_sheet(rows=12, cols=12, step=3.0)
    valid = quad.valid.copy()
    valid[0, 0] = False
    quad = QuadMesh(points=quad.points, valid=valid, meta={})

    tri = tm.from_quad(quad)
    assert tri.n_vertices == 143  # 144 minus the hole
    assert abs(tri.edge_length() - 3.0) < 0.3
    assert np.abs(tri.normals()[:, 0]).mean() > 0.95  # normal is along z


def test_edges_are_unique_and_undirected():
    tri = tm.from_quad(mesh_on_sheet(rows=4, cols=4, step=3.0))
    e = tri.edges()
    assert (e[:, 0] < e[:, 1]).all()
    assert len(np.unique(e, axis=0)) == len(e)


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #
def planted(distance, rows=40, cols=40, step=3.0, spacing=24.0, seed=0):
    volume = wrapped_volume(spacing=spacing, seed=seed)
    tri = tm.from_quad(mesh_on_sheet(rows=rows, cols=cols, step=step))
    return volume, tri, tm.displace(tri, distance)


def test_a_clean_triangular_surface_reports_no_seam():
    volume, tri, _ = planted(0.0)
    out = tm.find_sheet_switches(tri, volume, z_threshold=5.0)
    assert out["resolution_adequate"] is True
    assert out["n_seams"] == 0


@pytest.mark.parametrize("windings", [1, 2, 3])
def test_a_planted_whole_winding_displacement_is_found(windings):
    volume, _, moved = planted(24.0 * windings)
    out = tm.find_sheet_switches(moved, volume, z_threshold=5.0)
    assert out["resolution_adequate"] is True
    assert out["n_seams"] >= 1
    assert out["seams"][0]["edges"] >= 8


def test_the_seam_separates_the_displaced_half_from_the_rest():
    volume, tri, moved = planted(24.0)
    out = tm.find_sheet_switches(moved, volume, z_threshold=5.0)
    seam = out["seams"][0]
    # the cut runs along the split used by displace(), i.e. across the middle of
    # the surface's widest extent
    axis = int(np.argmax(tri.points.max(0) - tri.points.min(0)))
    assert abs(seam["centroid_zyx"][axis] - np.median(tri.points[:, axis])) < 6.0


def test_an_isolated_bad_edge_is_not_a_seam():
    volume, tri, _ = planted(0.0)
    points = tri.points.copy()
    points[0] = points[0] + np.array([24.0, 0.0, 0.0], np.float32)
    lone = tm.TriMesh(points=points, faces=tri.faces)
    out = tm.find_sheet_switches(lone, volume, z_threshold=5.0, min_edges=8)
    assert out["n_seams"] == 0
    assert out["max_z"] > 5.0  # the edge is flagged, the component is too small


def test_a_coarse_mesh_refuses_to_answer():
    # one edge per winding: every edge crosses a gap, so there is no seam to see
    volume = wrapped_volume(spacing=24.0)
    tri = tm.from_quad(mesh_on_sheet(rows=6, cols=6, step=24.0))
    out = tm.find_sheet_switches(tm.displace(tri, 24.0), volume, z_threshold=5.0)
    assert out["resolution_adequate"] is False
    assert out["n_seams"] == 0
    assert "seams_unreliable" in out


def test_quad_and_triangular_detectors_agree_on_the_same_surface():
    from labelscope.mesh import displace as quad_displace
    from labelscope.mesh import find_sheet_switches as quad_find

    volume = wrapped_volume(spacing=24.0)
    quad = mesh_on_sheet(rows=40, cols=40, step=3.0)
    quad_moved = quad_displace(quad, 24.0)

    q = quad_find(quad_moved, volume, z_threshold=5.0)
    t = tm.find_sheet_switches(tm.from_quad(quad_moved), volume, z_threshold=5.0)
    assert q["n_seams"] >= 1 and t["n_seams"] >= 1

    clean_q = quad_find(quad, volume, z_threshold=5.0)
    clean_t = tm.find_sheet_switches(tm.from_quad(quad), volume, z_threshold=5.0)
    assert clean_q["n_seams"] == 0 and clean_t["n_seams"] == 0


def test_a_compact_dark_patch_is_not_reported_as_a_seam():
    """The false positive the span rule exists to reject.

    A blob of damage darkens every edge that crosses it, so it produces a large
    connected component of flagged edges -- but it is a patch, not a cut, and a
    surface running over damage has not jumped to another wrap.
    """
    volume = wrapped_volume(spacing=24.0)
    tri = tm.from_quad(mesh_on_sheet(rows=40, cols=40, step=3.0))
    zz, yy, xx = np.ogrid[: volume.shape[0], : volume.shape[1], : volume.shape[2]]
    blob = ((zz - 72) ** 2 + (yy - 65) ** 2 + (xx - 65) ** 2) < 14**2
    damaged = volume.copy()
    damaged[blob] = 5.0

    out = tm.find_sheet_switches(tri, damaged, z_threshold=5.0)
    assert out["max_z"] > 10.0  # the edges over the damage are flagged
    assert out["n_seams"] == 0  # but the component does not cross the surface
