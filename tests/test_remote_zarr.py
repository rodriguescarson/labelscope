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


def test_a_compressed_store_carries_its_codec_through_from_store():
    """Compressed stores used to be refused outright.  PHerc 0172's scan is
    blosc-compressed and is a fifth of the published corpus, so the codec is
    read from .zarray and applied per chunk instead."""
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
                        "order": "C",
                        "compressor": {
                            "id": "blosc",
                            "cname": "zstd",
                            "clevel": 3,
                            "shuffle": 1,
                        },
                    }
                )

                def raise_for_status(self):
                    pass

            return R()

    vol = ChunkedVolume.from_store("http://x", session=MetaStore())
    assert vol.compressor["id"] == "blosc"
    assert vol._codec is not None


def test_a_fortran_order_store_is_refused():
    """Order is the assumption that is still load-bearing: every chunk is
    reshaped as C-order, so an F-order store would decode into nonsense."""
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
                        "order": "F",
                        "compressor": None,
                    }
                )

                def raise_for_status(self):
                    pass

            return R()

    with pytest.raises(ValueError, match="C-order"):
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
    leftovers = [
        name for root, _, names in os.walk(cache) for name in names if name.endswith(".part")
    ]
    assert not leftovers


def test_one_cache_directory_serves_two_stores_without_mixing_them():
    """A corpus pass sweeps several scrolls through one cache.

    Chunk indices are only unique within a store, so without a per-store
    namespace chunk (0, 0, 0) of the second volume would be served from the
    first one's cached bytes -- silently, and with a plausible result.
    """
    import tempfile as tf

    from labelscope.remote_zarr import ChunkedVolume

    def store(fill):
        class Session:
            def get(self, url, timeout=None):
                class R:
                    status_code = 200
                    content = np.full((8, 8, 8), fill, np.uint8).tobytes()

                    def raise_for_status(self):
                        pass

                return R()

        return Session()

    with tf.TemporaryDirectory() as cache:
        a = ChunkedVolume(
            "http://scroll-a", (16, 16, 16), (8, 8, 8), cache_dir=cache, session=store(11)
        )
        b = ChunkedVolume(
            "http://scroll-b", (16, 16, 16), (8, 8, 8), cache_dir=cache, session=store(22)
        )
        assert a._fetch((0, 0, 0))[0, 0, 0] == 11
        assert b._fetch((0, 0, 0))[0, 0, 0] == 22
        # and a fresh reader on the same store still hits the cache, not the wire
        again = ChunkedVolume(
            "http://scroll-a", (16, 16, 16), (8, 8, 8), cache_dir=cache, session=None
        )
        assert again._fetch((0, 0, 0))[0, 0, 0] == 11
        assert again.chunks_fetched == 0


def test_a_blosc_compressed_store_reads_the_same_as_a_raw_one():
    """PHerc 0172's scan is blosc-compressed, and that is 53 of the 258
    published surfaces -- a fifth of the corpus, refused outright until now."""
    import numcodecs

    from labelscope.remote_zarr import ChunkedVolume

    block = (np.arange(8 * 8 * 8, dtype=np.uint8) % 251).reshape(8, 8, 8)
    codec = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1}
    packed = numcodecs.get_codec(codec).encode(block.tobytes())

    class Session:
        def __init__(self, payload):
            self.payload = payload

        def get(self, url, timeout=None):
            payload = self.payload

            class R:
                status_code = 200
                content = payload

                def raise_for_status(self):
                    pass

            return R()

    raw = ChunkedVolume(
        "http://raw", (16, 16, 16), (8, 8, 8), session=Session(block.tobytes())
    )
    comp = ChunkedVolume(
        "http://comp", (16, 16, 16), (8, 8, 8), compressor=codec, session=Session(packed)
    )
    np.testing.assert_array_equal(raw._fetch((0, 0, 0)), comp._fetch((0, 0, 0)))
    np.testing.assert_array_equal(comp._fetch((0, 0, 0)), block)
    # and sampling agrees, not just the raw block
    points = np.array([[1.5, 2.5, 3.5], [0.0, 0.0, 0.0]], np.float32)
    np.testing.assert_allclose(raw.sample(points), comp.sample(points))


def test_a_chunk_that_will_not_decode_is_treated_as_absent():
    from labelscope.remote_zarr import ChunkedVolume

    class Session:
        def get(self, url, timeout=None):
            class R:
                status_code = 200
                content = b"not blosc at all"

                def raise_for_status(self):
                    pass

            return R()

    vol = ChunkedVolume(
        "http://x",
        (16, 16, 16),
        (8, 8, 8),
        compressor={"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1},
        session=Session(),
    )
    assert vol._fetch((0, 0, 0)) is None
