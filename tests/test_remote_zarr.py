"""ChunkedVolume reads a large array one chunk at a time.

The point is that a surface threaded through a scan touches a small fraction of
the chunks in its own bounding box, so the tests check that it fetches few
chunks, samples correctly, and treats a chunk the store omits as void rather
than as an error.
"""

import os

import numpy as np
import pytest

from labelscope.remote_zarr import ChunkedVolume


class FakeStore:
    """Serves chunks of a known array, and counts what was asked for."""

    def __init__(self, array, chunks, omit=()):
        self.array = array
        self.chunks = chunks
        self.omit = set(omit)
        self.requests = []

    def get(self, url, timeout=None):
        key = tuple(int(p) for p in url.rsplit("/", 3)[-3:])
        self.requests.append(key)
        store = self

        class Response:
            status_code = 404 if key in store.omit else 200

            @property
            def content(self):
                lo = np.array(key) * np.array(store.chunks)
                block = np.zeros(store.chunks, store.array.dtype)
                hi = np.minimum(lo + np.array(store.chunks), store.array.shape)
                sel = tuple(slice(lo[a], hi[a]) for a in range(3))
                sub = store.array[sel]
                block[tuple(slice(0, s) for s in sub.shape)] = sub
                return block.tobytes()

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(self.status_code)

        return Response()


def build(shape=(64, 64, 64), chunks=(16, 16, 16), omit=(), seed=0):
    # deterministic and never zero, so "read as void" is distinguishable from
    # "read a voxel that happened to be zero"
    array = (np.arange(int(np.prod(shape))) % 254 + 1).astype(np.uint8).reshape(shape)
    store = FakeStore(array, chunks, omit)
    vol = ChunkedVolume("http://x/0", shape, chunks, "|u1", session=store)
    return array, vol, store


def test_sampling_matches_the_underlying_array():
    array, vol, _ = build()
    pts = np.array([[10.0, 20.0, 30.0], [5.0, 5.0, 5.0], [63.0, 63.0, 63.0]])
    got = vol.sample(pts)
    want = [float(array[10, 20, 30]), float(array[5, 5, 5]), float(array[63, 63, 63])]
    assert np.allclose(got, want, atol=1e-3)


def test_trilinear_interpolation_between_voxels():
    array, vol, _ = build()
    mid = vol.sample(np.array([[10.5, 20.0, 30.0]]))[0]
    lo, hi = float(array[10, 20, 30]), float(array[11, 20, 30])
    assert abs(mid - 0.5 * (lo + hi)) < 1e-3


def test_only_the_chunks_touched_are_fetched():
    """A plane through the middle must not pull the whole array."""
    _, vol, store = build(shape=(64, 64, 64), chunks=(16, 16, 16))
    yy, xx = np.meshgrid(np.arange(0, 64, 2.0), np.arange(0, 64, 2.0), indexing="ij")
    plane = np.stack([np.full_like(yy, 33.0), yy, xx], -1).reshape(-1, 3)
    vol.sample(plane)
    touched = set(store.requests)
    assert touched == {(2, j, i) for j in range(4) for i in range(4)}
    assert len(touched) == 16  # of the array's 4x4x4 = 64 chunks


def test_each_chunk_is_fetched_once():
    _, vol, store = build()
    pts = np.random.default_rng(0).uniform(0, 60, (4000, 3))
    vol.sample(pts)
    assert len(store.requests) == len(set(store.requests))


def test_a_missing_chunk_reads_as_void_not_an_error():
    _, vol, _ = build(omit=[(0, 0, 0)])
    assert vol.sample(np.array([[1.0, 1.0, 1.0]]))[0] == 0.0
    assert vol.sample(np.array([[40.0, 40.0, 40.0]]))[0] > 0.0


def test_points_outside_the_array_read_as_zero():
    _, vol, _ = build()
    assert vol.sample(np.array([[-5.0, 10.0, 10.0], [999.0, 10.0, 10.0]]))[0] == 0.0


def test_chunk_keys_include_the_interpolation_skirt():
    _, vol, _ = build(chunks=(16, 16, 16))
    keys = vol.chunk_keys(np.array([[15.5, 15.5, 15.5]]))
    assert (0, 0, 0) in keys and (1, 1, 1) in keys


def test_prefetch_warms_the_cache_so_sampling_fetches_nothing():
    _, vol, store = build()
    pts = np.random.default_rng(1).uniform(0, 60, (2000, 3))
    vol.prefetch(pts, workers=4)
    before = len(store.requests)
    vol.sample(pts)
    assert len(store.requests) == before


def test_a_compressed_store_is_refused():
    import json

    class MetaStore:
        def get(self, url, timeout=None):
            class R:
                status_code = 200
                text = json.dumps(
                    {
                        "shape": [8, 8, 8],
                        "chunks": [4, 4, 4],
                        "dtype": "|u1",
                        "compressor": {"id": "blosc"},
                    }
                )

                def raise_for_status(self):
                    pass

            return R()

    with pytest.raises(ValueError, match="uncompressed"):
        ChunkedVolume.from_store("http://x", session=MetaStore())


def test_two_readers_can_share_one_cache_directory(tmp_path, monkeypatch):
    """The fleet configuration: several workers, one cache.

    Both readers fetching the same chunk used to write the same "<chunk>.part";
    whichever renamed second raised FileNotFoundError and took its whole surface
    down with it.  This drives the same collision deterministically.
    """
    import threading

    from labelscope.remote_zarr import ChunkedVolume

    block = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)

    class Session:
        def get(self, url, timeout=None):
            class R:
                status_code = 200
                content = block.tobytes()

                def raise_for_status(self):
                    pass

            return R()

    cache = str(tmp_path / "cache")
    readers = [
        ChunkedVolume("http://x", (16, 16, 16), (8, 8, 8), cache_dir=cache, session=Session())
        for _ in range(4)
    ]
    errors = []

    def pull(reader):
        try:
            for _ in range(20):
                reader._blocks.clear()
                assert reader._fetch((0, 0, 0)) is not None
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=pull, args=(r,)) for r in readers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert not [p for p in os.listdir(os.path.join(cache, "0", "0")) if p.endswith(".part")]
