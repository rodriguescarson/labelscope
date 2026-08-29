"""The regulariser has to remove wobble without touching the convention.

The labels in both public releases sit a consistent ~2.3 voxels off the sheet's
density maximum, and that is deliberate: they mark the recto face, the side the
ink is on.  So the bar here is asymmetric.  A patch whose cells all agree with
each other must come back untouched however far off the ridge it sits, and a
patch whose cells disagree must come back agreeing.
"""

import numpy as np
import pytest

from labelscope.alignment import aggregate_alignment
from labelscope.regularise import delta_field, regularise_label
from test_aggregate import scroll_like, sheet_label, upward


def wobbled(shape=(64, 96, 96), base=32, shift=3, band=slice(32, 64)):
    """A label on the sheet everywhere except one band of y, where it is off."""
    mask = sheet_label(shape, base)
    mask[:, band, :] = False
    mask[base + shift, band, :] = True
    return mask


def align(volume, mask, **kw):
    # ``scroll_like`` is deliberately noisier than the real releases -- it is
    # tuned so a naive estimator fails on it -- and its global profile SNR sits
    # around 1.6 against the 2.0 default.  The floor is lowered here so these
    # tests exercise the warp; that the floor is *obeyed* is tested separately
    # by ``test_nothing_moves_when_no_cell_resolves``.
    return aggregate_alignment(
        volume,
        mask,
        cell=32,
        min_per_cell=100,
        orient_field=upward(volume.shape),
        orient_by="field",
        n_samples=12000,
        bootstrap=20,
        min_global_snr=1.0,
        return_cells=True,
        **kw,
    )


# --------------------------------------------------------------------------- #
# the field
# --------------------------------------------------------------------------- #
def test_delta_field_is_zero_where_every_cell_matches_the_patch():
    cells = [{"key": [0, k, 0], "offset": 2.3} for k in range(3)]
    field = delta_field((32, 96, 32), cells, global_offset=2.3, cell=32)
    assert np.abs(field).max() < 1e-6


def test_delta_field_is_smooth_across_a_cell_boundary():
    """A step at a cell boundary is a seam, which is the defect the rest of the
    tool exists to find.  The correction must not manufacture one."""
    cells = [
        {"key": [0, 0, 0], "offset": 0.0},
        {"key": [0, 1, 0], "offset": 3.0},
        {"key": [0, 2, 0], "offset": 0.0},
    ]
    field = delta_field((32, 96, 32), cells, global_offset=0.0, cell=32, smooth=1.0)
    line = field[16, :, 16]
    jumps = np.abs(np.diff(line))
    assert jumps.max() < 0.5  # no step; the ramp is spread over many voxels
    assert line.max() > 1.0  # but the correction is still applied


def test_delta_field_ignores_cells_with_no_measurement():
    field = delta_field((32, 96, 32), [], global_offset=1.0, cell=32)
    assert np.abs(field).max() == 0.0


# --------------------------------------------------------------------------- #
# the warp
# --------------------------------------------------------------------------- #
def test_a_consistent_label_is_left_alone_however_far_off_the_ridge_it_sits():
    shape = (64, 96, 96)
    volume = scroll_like(shape, sheet_z=32.0)
    mask = sheet_label(shape, 34)  # a whole patch consistently 2 voxels off
    out, report = regularise_label(
        volume, mask, cell=32, alignment=align(volume, mask), orient_field=upward(shape)
    )
    assert report["max_abs_shift"] < 1.0
    assert int(np.abs(out.astype(int) - mask.astype(int)).sum()) == 0


def test_a_wobbling_label_is_pulled_back_into_agreement():
    shape = (64, 96, 96)
    volume = scroll_like(shape, sheet_z=32.0)
    mask = wobbled(shape)
    before = align(volume, mask)
    out, report = regularise_label(
        volume, mask, cell=32, alignment=before, orient_field=upward(shape)
    )
    assert report["changed"]

    after = align(volume, out)
    spread = lambda r: np.std([c["offset"] for c in r["cells"]])  # noqa: E731
    assert spread(after) < 0.6 * spread(before)


def test_nothing_moves_when_no_cell_resolves():
    shape = (64, 96, 96)
    volume = np.full(shape, 50.0, dtype=np.float32)  # featureless: no ridge
    mask = sheet_label(shape, 32)
    out, report = regularise_label(volume, mask, cell=32)
    assert not report["changed"]
    assert report["reason"] in {"no resolved cells", "global offset unreliable"}
    np.testing.assert_array_equal(out, mask)


def test_the_label_keeps_its_size():
    """A warp that quietly erodes or dilates the surface would change the class
    balance a model sees, which is a different experiment than the one intended."""
    shape = (64, 96, 96)
    volume = scroll_like(shape, sheet_z=32.0)
    mask = wobbled(shape)
    out, report = regularise_label(
        volume, mask, cell=32, alignment=align(volume, mask), orient_field=upward(shape)
    )
    assert report["voxels_after"] == pytest.approx(report["voxels_before"], rel=0.15)
