import os
import struct

import numpy as np
import tifffile

from labelscope.io import (VolumeInfo, discover_pairs, discover_pairs_remote,
                           is_remote, probe_volume, read_volume)


def write_stack(path, shape=(8, 16, 16), value=3, compression=None):
    data = np.zeros(shape, dtype=np.uint8)
    data[shape[0] // 2] = value
    tifffile.imwrite(path, data, compression=compression)
    return data


def test_probe_reads_shape_and_compression_without_decoding(tmp_path):
    path = str(tmp_path / "vol.tif")
    write_stack(path, compression="lzw")
    info = probe_volume(path)
    assert info.ok
    assert info.shape == (8, 16, 16)
    assert info.dtype == "uint8"
    assert "LZW" in info.compression.upper()
    assert info.file_size == os.path.getsize(path)


def test_probe_reports_a_broken_file_instead_of_raising(tmp_path):
    path = str(tmp_path / "truncated.tif")
    with open(path, "wb") as handle:
        handle.write(b"II" + struct.pack("<H", 42) + b"\x00" * 40)
    info = probe_volume(path)
    assert not info.ok and info.error


def test_read_volume_round_trips(tmp_path):
    path = str(tmp_path / "vol.tif")
    data = write_stack(path)
    assert np.array_equal(read_volume(path), data)


def test_read_volume_can_take_a_z_band(tmp_path):
    path = str(tmp_path / "vol.tif")
    write_stack(path, shape=(12, 8, 8))
    band = read_volume(path, slice(3, 7))
    assert band.shape == (4, 8, 8)


def test_pairs_are_matched_across_the_nnunet_channel_suffix(tmp_path):
    images, labels = tmp_path / "imagesTr", tmp_path / "labelsTr"
    images.mkdir(), labels.mkdir()
    write_stack(str(images / "s1_z0_y0_x0_0000.tif"))
    write_stack(str(labels / "s1_z0_y0_x0.tif"))
    pairs = discover_pairs(str(images), str(labels))
    assert len(pairs) == 1 and pairs[0].complete
    assert pairs[0].name == "s1_z0_y0_x0"


def test_an_unpaired_volume_is_returned_not_dropped(tmp_path):
    images, labels = tmp_path / "imagesTr", tmp_path / "labelsTr"
    images.mkdir(), labels.mkdir()
    write_stack(str(images / "lonely_0000.tif"))
    pairs = discover_pairs(str(images), str(labels))
    assert len(pairs) == 1 and not pairs[0].complete and pairs[0].label is None


def test_remote_pairing_builds_urls_and_strips_extensions():
    pairs = discover_pairs_remote("https://h/img", "https://h/lab",
                                  ["sample_1", "sample_2.tif", ""])
    assert [p.name for p in pairs] == ["sample_1", "sample_2"]
    assert pairs[0].image == "https://h/img/sample_1.tif"
    assert pairs[0].label == "https://h/lab/sample_1.tif"


def test_is_remote():
    assert is_remote("https://example.com/x") and is_remote("http://x/y")
    assert not is_remote("/local/path") and not is_remote(None)


def test_volume_info_ok_requires_a_shape():
    assert not VolumeInfo(path="x").ok
    assert VolumeInfo(path="x", shape=(1, 2, 3)).ok
