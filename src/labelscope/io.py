"""Reading and cheaply probing volumetric image/label pairs.

Supports the two on-disk shapes the Vesuvius community actually ships:

* 3-D TIFF stacks (one file per volume, one page per z-slice) — the nnU-Net
  ``imagesTr``/``labelsTr`` layout and the Kaggle surface-detection release.
* Zarr arrays / OME-Zarr groups, when ``zarr`` is installed.

``probe_volume`` reads only the header, so a whole dataset can be inventoried
without downloading or decoding the voxels.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import tifffile


# --------------------------------------------------------------------------- #
# header-only probing
# --------------------------------------------------------------------------- #
@dataclass
class VolumeInfo:
    """What can be learned about a volume without decoding its voxels."""

    path: str
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    compression: Optional[str] = None
    n_pages: Optional[int] = None
    file_size: Optional[int] = None
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.shape is not None


def probe_volume(path: str) -> VolumeInfo:
    """Read a volume's header only.  Never decodes pixel data."""
    info = VolumeInfo(path=path)
    try:
        info.file_size = os.path.getsize(path)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        info.error = f"stat failed: {exc}"
        return info

    if path.endswith((".tif", ".tiff")):
        try:
            with tifffile.TiffFile(path) as tf:
                pages = tf.pages
                info.n_pages = len(pages)
                page = pages[0]
                info.dtype = str(page.dtype)
                info.compression = getattr(page.compression, "name", str(page.compression))
                info.shape = (len(pages),) + tuple(page.shape)
        except Exception as exc:
            info.error = f"tiff header unreadable: {type(exc).__name__}: {exc}"
        return info

    try:  # zarr
        import zarr

        arr = zarr.open(path, mode="r")
        if hasattr(arr, "shape"):
            info.shape = tuple(arr.shape)
            info.dtype = str(arr.dtype)
            info.compression = str(getattr(arr, "compressor", None))
        else:  # a group — take the highest-resolution array
            keys = sorted(arr.array_keys())
            if not keys:
                info.error = "zarr group has no arrays"
                return info
            sub = arr[keys[0]]
            info.shape = tuple(sub.shape)
            info.dtype = str(sub.dtype)
            info.compression = str(getattr(sub, "compressor", None))
    except ImportError:
        info.error = "zarr not installed (pip install 'labelscope[zarr]')"
    except Exception as exc:
        info.error = f"zarr unreadable: {type(exc).__name__}: {exc}"
    return info


# --------------------------------------------------------------------------- #
# full reads
# --------------------------------------------------------------------------- #
def read_volume(path: str, z_slice: Optional[slice] = None) -> np.ndarray:
    """Read a volume as a 3-D array, optionally only a range of z-slices."""
    if path.endswith((".tif", ".tiff")):
        with tifffile.TiffFile(path) as tf:
            if z_slice is None:
                return tf.asarray()
            idx = range(*z_slice.indices(len(tf.pages)))
            return np.stack([tf.pages[i].asarray() for i in idx])

    import zarr

    arr = zarr.open(path, mode="r")
    if not hasattr(arr, "shape"):
        arr = arr[sorted(arr.array_keys())[0]]
    return np.asarray(arr[z_slice] if z_slice is not None else arr[:])


# --------------------------------------------------------------------------- #
# pairing images with labels
# --------------------------------------------------------------------------- #
@dataclass
class VolumePair:
    """One training example: a CT volume and its label volume."""

    name: str
    image: Optional[str] = None
    label: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.image is not None and self.label is not None


#: image filenames in an nnU-Net ``imagesTr`` carry a trailing channel index
_NNUNET_CHANNEL_SUFFIXES = ("_0000", "_0001", "_0002", "_0003")


def _stem(filename: str) -> str:
    stem = filename
    for ext in (".tiff", ".tif"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    for suffix in _NNUNET_CHANNEL_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _list_volumes(directory: str) -> dict:
    if not directory or not os.path.isdir(directory):
        return {}
    out = {}
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if entry.endswith((".tif", ".tiff")) and os.path.isfile(full):
            out[_stem(entry)] = full
        elif entry.endswith(".zarr") and os.path.isdir(full):
            out[_stem(entry[: -len(".zarr")])] = full
    return out


def discover_pairs_remote(
    images_base: Optional[str], labels_base: Optional[str], names: Sequence[str]
) -> list:
    """Pair up volumes hosted over HTTP, given the filenames to look for.

    A directory listing is not something every host offers, so the names come
    from the caller: ``--names-file``, or a listing already in hand.
    """
    pairs = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        filename = name if name.endswith((".tif", ".tiff")) else name + ".tif"
        pairs.append(
            VolumePair(
                name=_stem(filename),
                image=f"{images_base.rstrip('/')}/{filename}" if images_base else None,
                label=f"{labels_base.rstrip('/')}/{filename}" if labels_base else None,
                meta={"remote": True},
            )
        )
    return pairs


def is_remote(path: Optional[str]) -> bool:
    return bool(path) and path.startswith(("http://", "https://"))


def discover_pairs(images_dir: Optional[str], labels_dir: Optional[str]) -> list:
    """Pair up an ``imagesTr``-style and a ``labelsTr``-style directory by stem.

    Volumes present on only one side are returned too, with the missing field set
    to ``None`` — an unpaired volume is itself a finding, not an error.
    """
    images = _list_volumes(images_dir) if images_dir else {}
    labels = _list_volumes(labels_dir) if labels_dir else {}
    pairs = []
    for name in sorted(set(images) | set(labels)):
        pairs.append(VolumePair(name=name, image=images.get(name), label=labels.get(name)))
    return pairs


def iter_pairs(pairs: Sequence[VolumePair], require: str = "both") -> Iterator[VolumePair]:
    for pair in pairs:
        if require == "both" and not pair.complete:
            continue
        if require == "label" and pair.label is None:
            continue
        if require == "image" and pair.image is None:
            continue
        yield pair


# --------------------------------------------------------------------------- #
# reading a volume over HTTP without downloading all of it
# --------------------------------------------------------------------------- #
class HTTPFile:
    """A minimal seekable file over HTTP range requests.

    Enough of the file protocol for ``tifffile`` to walk a TIFF's IFD chain and
    decode individual pages.  Blocks are cached in memory, so reading a band of
    z-slices out of a 32 MB volume transfers a few MB rather than all of it —
    which is what makes it practical to audit a release hosted on S3 or the
    Hugging Face bucket before pulling it down.
    """

    def __init__(self, url: str, block_size: int = 1 << 20, session=None, timeout: int = 60):
        self.url = url
        self.block_size = block_size
        self.timeout = timeout
        self._session = session or http_session()
        self._blocks: dict = {}
        self._pos = 0
        head = self._session.head(url, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
        self.size = int(head.headers["content-length"])
        self.bytes_fetched = 0

    # -- file protocol -----------------------------------------------------
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def read(self, length: int = -1) -> bytes:
        if length is None or length < 0:
            length = self.size - self._pos
        length = max(0, min(length, self.size - self._pos))
        if length == 0:
            return b""
        start, end = self._pos, self._pos + length
        first, last = start // self.block_size, (end - 1) // self.block_size
        chunks = [self._block(i) for i in range(first, last + 1)]
        buffer = b"".join(chunks)
        offset = start - first * self.block_size
        self._pos = end
        return buffer[offset:offset + length]

    def close(self) -> None:
        self._blocks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- block cache -------------------------------------------------------
    def _block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            return cached
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        response = self._session.get(
            self.url, headers={"Range": f"bytes={start}-{end}"},
            timeout=self.timeout, allow_redirects=True,
        )
        response.raise_for_status()
        data = response.content
        self.bytes_fetched += len(data)
        self._blocks[index] = data
        return data


def read_volume_http(url: str, z_slice: Optional[slice] = None, block_size: int = 1 << 20):
    """Read a remote 3-D TIFF, optionally only a band of z-slices.

    Returns ``(array, bytes_fetched)`` so callers can report what the audit
    actually cost in transfer.
    """
    handle = HTTPFile(url, block_size=block_size)
    try:
        with tifffile.TiffFile(handle) as tf:
            if z_slice is None:
                data = tf.asarray()
            else:
                idx = range(*z_slice.indices(len(tf.pages)))
                data = np.stack([tf.pages[i].asarray() for i in idx])
        return data, handle.bytes_fetched
    finally:
        handle.close()


# --------------------------------------------------------------------------- #
# probing a remote volume from its header alone
# --------------------------------------------------------------------------- #
_TIFF_TAGS = {256: "width", 257: "length", 258: "bits", 259: "compression",
              277: "samples"}
_TIFF_COMPRESSION = {1: "NONE", 5: "LZW", 7: "JPEG", 8: "ADOBE_DEFLATE",
                     32773: "PACKBITS", 32946: "DEFLATE", 34925: "LZMA",
                     50000: "ZSTD", 50001: "WEBP"}
_TIFF_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
                   11: 4, 12: 8, 16: 8, 17: 8, 18: 8}


_SESSION = None


def http_session():
    """One pooled, retrying session for the whole process.

    Opening a fresh connection per volume exhausts local ephemeral ports long
    before a real dataset is inventoried — on macOS it surfaces as
    ``OSError(49, "Can't assign requested address")`` a few hundred files in.
    """
    global _SESSION
    if _SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET", "HEAD"]))
        adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _SESSION = session
    return _SESSION


def probe_volume_http(url: str, session=None, timeout: int = 60) -> VolumeInfo:
    """Inventory a remote 3-D TIFF from roughly one kilobyte of it.

    Reads the file header and the first image directory only — enough for shape,
    dtype and compression — and infers the number of z-slices from the content
    length.  That makes it practical to check what is in a release *before*
    pulling it, which matters when the release is measured in terabytes.

    The inferred depth assumes every page has the same footprint, which holds for
    the uniform patch stacks these datasets ship.  ``depth_inferred`` records
    that the number was derived rather than read.
    """
    import struct

    import requests

    session = session or http_session()
    info = VolumeInfo(path=url)
    last = None
    for attempt in range(4):
        try:
            response = session.get(url, headers={"Range": "bytes=0-2047"},
                                   timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            head = response.content
            total = response.headers.get("content-range", "").split("/")[-1]
            info.file_size = int(total) if total.isdigit() else None
            break
        except Exception as exc:
            last = exc
            time.sleep(0.4 * (attempt + 1))
    else:
        info.error = f"range request failed: {type(last).__name__}: {last}"
        return info

    if len(head) < 8:
        info.error = "response too short for a TIFF header"
        return info
    if head[:2] == b"II":
        endian = "<"
    elif head[:2] == b"MM":
        endian = ">"
    else:
        info.error = "not a TIFF (bad byte-order mark)"
        return info

    magic = struct.unpack(endian + "H", head[2:4])[0]
    if magic != 42:
        info.error = f"unsupported TIFF variant (magic {magic}); BigTIFF is not handled"
        return info

    offset = struct.unpack(endian + "I", head[4:8])[0]
    if offset + 2 > len(head):
        info.error = "first image directory lies beyond the fetched header"
        return info

    count = struct.unpack(endian + "H", head[offset:offset + 2])[0]
    fields = {}
    for n in range(count):
        entry = offset + 2 + n * 12
        if entry + 12 > len(head):
            break
        tag, dtype, length = struct.unpack(endian + "HHI", head[entry:entry + 8])
        name = _TIFF_TAGS.get(tag)
        if name is None:
            continue
        size = _TIFF_TYPE_SIZE.get(dtype, 4) * length
        raw = head[entry + 8:entry + 12]
        if size <= 4:
            fields[name] = struct.unpack(endian + ("I" if dtype == 4 else "H"),
                                         raw[:4] if dtype == 4 else raw[:2])[0]

    width, length = fields.get("width"), fields.get("length")
    bits = fields.get("bits", 8)
    info.compression = _TIFF_COMPRESSION.get(fields.get("compression", 1),
                                             str(fields.get("compression")))
    info.dtype = {8: "uint8", 16: "uint16", 32: "float32"}.get(bits, f"{bits}-bit")
    if not (width and length):
        info.error = "first image directory carries no width/length"
        return info

    info.meta["plane_shape"] = (length, width)          # exact, read from the IFD
    if info.compression == "NONE" and info.file_size:
        # uncompressed pages have a fixed footprint, so depth follows from size.
        # ~166 bytes of per-page directory overhead is what these writers emit;
        # rounding absorbs the rest.
        page_bytes = width * length * max(1, bits // 8)
        depth = max(1, round(info.file_size / (page_bytes + 166)))
        info.n_pages = depth
        info.shape = (depth, length, width)
        info.meta["depth_inferred"] = True
    else:
        # a compressed file's size says nothing about its page count
        info.shape = (None, length, width)
        info.meta["depth_inferred"] = False
    return info
