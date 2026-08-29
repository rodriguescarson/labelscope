"""Reading only the chunks a surface actually passes through.

A traced surface is a 2-D sheet threaded through a 3-D scan.  Its bounding box
can be tens of gigabytes while the surface itself touches a few percent of the
chunks inside it — and at 2.4 µm a whole scroll volume is measured in terabytes,
so reading the box is not an option and reading the whole array never was.

This fetches the chunks that a given set of sample points falls in, and nothing
else, straight from an uncompressed Zarr v2 store over HTTP.  Every chunk that
comes back is cached, so the walk along a mesh edge costs one fetch per new
chunk rather than one per sample.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np


class ChunkedVolume:
    """Random access to a large Zarr array, one chunk at a time.

    Only uncompressed stores are supported, which is what the Vesuvius open-data
    volumes are: ``"compressor": null`` and a raw C-order block per chunk.  A
    chunk that does not exist is void — the masked-out air around the scroll —
    and reads as zeros, which is what the store itself means by it.
    """

    def __init__(
        self,
        base_url: str,
        shape: Tuple[int, int, int],
        chunks: Tuple[int, int, int],
        dtype: str = "|u1",
        cache_dir: Optional[str] = None,
        session=None,
        max_cached: int = 4096,
    ):
        self.base_url = base_url.rstrip("/")
        self.shape = tuple(int(s) for s in shape)
        self.chunks = tuple(int(c) for c in chunks)
        self.dtype = np.dtype(dtype)
        self.cache_dir = cache_dir
        self.max_cached = max_cached
        self._blocks: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._missing = set()
        self.bytes_fetched = 0
        self.chunks_fetched = 0
        if session is None:
            from labelscope.io import http_session

            session = http_session()
        self._session = session
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_store(cls, base_url: str, level: str = "0", **kwargs) -> ChunkedVolume:
        """Read ``.zarray`` from a remote store and build a reader for it."""
        import json

        from labelscope.io import http_session

        session = kwargs.pop("session", None) or http_session()
        url = f"{base_url.rstrip('/')}/{level}/.zarray"
        response = session.get(url, timeout=60)
        response.raise_for_status()
        meta = json.loads(response.text)
        if meta.get("compressor") is not None:
            raise ValueError(f"{url}: only uncompressed zarr stores are supported")
        return cls(
            base_url=f"{base_url.rstrip('/')}/{level}",
            shape=meta["shape"],
            chunks=meta["chunks"],
            dtype=meta["dtype"],
            session=session,
            **kwargs,
        )

    # -- chunk access ------------------------------------------------------
    def _fetch(self, key: Tuple[int, int, int]) -> Optional[np.ndarray]:
        if key in self._blocks:
            return self._blocks[key]
        if key in self._missing:
            return None

        path = None
        if self.cache_dir:
            path = os.path.join(self.cache_dir, *(str(k) for k in key))
            if os.path.exists(path):
                block = np.fromfile(path, self.dtype).reshape(self.chunks)
                self._remember(key, block)
                return block

        url = f"{self.base_url}/{key[0]}/{key[1]}/{key[2]}"
        try:
            response = self._session.get(url, timeout=120)
        except Exception:
            self._missing.add(key)
            return None
        if response.status_code == 404:
            self._missing.add(key)  # void: the store omits empty chunks
            return None
        response.raise_for_status()
        data = response.content
        self.bytes_fetched += len(data)
        self.chunks_fetched += 1
        expected = int(np.prod(self.chunks)) * self.dtype.itemsize
        if len(data) != expected:
            self._missing.add(key)
            return None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # The temporary name has to be unique per *writer*, not per process.
            # A fleet pass shares one cache across workers and prefetches with a
            # thread pool inside each, so two writers fetching the same chunk both
            # wrote "<chunk>.part": the first rename moved it away and the second
            # raised FileNotFoundError, which killed 21 of 77 surfaces in a run
            # that was otherwise fine.  mkstemp is unique across both.
            handle_fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(path), prefix=os.path.basename(path) + ".", suffix=".part"
            )
            try:
                with os.fdopen(handle_fd, "wb") as handle:
                    handle.write(data)
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        block = np.frombuffer(data, self.dtype).reshape(self.chunks)
        self._remember(key, block)
        return block

    def _remember(self, key, block) -> None:
        if len(self._blocks) >= self.max_cached:
            self._blocks.pop(next(iter(self._blocks)))
        self._blocks[key] = block

    def prefetch(self, points: np.ndarray, workers: int = 24) -> int:
        """Pull every chunk the given (N, 3) points touch, in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        keys = self.chunk_keys(points)
        todo = [k for k in keys if k not in self._blocks and k not in self._missing]
        if todo:
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(self._fetch, todo))
        return len(todo)

    def chunk_keys(self, points: np.ndarray) -> list:
        """The distinct chunks a set of float coordinates falls in, with a
        one-voxel skirt so trilinear interpolation never straddles a gap."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = set()
        for shift in (0.0, 1.0):
            idx = np.floor(pts + shift).astype(np.int64)
            np.clip(idx, 0, np.array(self.shape) - 1, out=idx)
            blocks = idx // np.array(self.chunks)
            out.update(map(tuple, np.unique(blocks, axis=0)))
        return sorted(out)

    # -- sampling ----------------------------------------------------------
    def sample(self, points: np.ndarray) -> np.ndarray:
        """Trilinear sample at (N, 3) float coordinates in z, y, x order."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        base = np.floor(pts).astype(np.int64)
        frac = pts - base
        out = np.zeros(pts.shape[0], dtype=np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    corner = base + np.array([dz, dy, dx])
                    weight = (
                        (frac[:, 0] if dz else 1 - frac[:, 0])
                        * (frac[:, 1] if dy else 1 - frac[:, 1])
                        * (frac[:, 2] if dx else 1 - frac[:, 2])
                    )
                    out += weight.astype(np.float32) * self._read(corner)
        return out

    def _read(self, index: np.ndarray) -> np.ndarray:
        """Nearest-voxel read at integer coordinates, chunk by chunk."""
        idx = np.asarray(index, dtype=np.int64)
        inside = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
        values = np.zeros(idx.shape[0], dtype=np.float32)
        if not inside.any():
            return values
        safe = idx[inside]
        blocks = safe // np.array(self.chunks)
        local = safe % np.array(self.chunks)
        order = np.lexsort((blocks[:, 2], blocks[:, 1], blocks[:, 0]))
        got = np.zeros(safe.shape[0], dtype=np.float32)
        start = 0
        keys = blocks[order]
        while start < len(order):
            stop = start
            key = tuple(keys[start])
            while stop < len(order) and tuple(keys[stop]) == key:
                stop += 1
            block = self._fetch(key)
            if block is not None:
                sel = local[order[start:stop]]
                got[order[start:stop]] = block[sel[:, 0], sel[:, 1], sel[:, 2]]
            start = stop
        values[inside] = got
        return values
