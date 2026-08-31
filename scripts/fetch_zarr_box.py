#!/usr/bin/env python3
"""Mirror just the chunks of a remote Zarr that lie inside a box.

Tools that were written to read a Zarr from disk -- VC3D's tracer among them --
cannot stream one over HTTP, and the stores here are whole-scroll: PHercParis4's
surface prediction is 75784 x 32693 x 32693 voxels. Downloading it to grow one
patch is not an option.

But a Zarr chunk is an individually addressable file, and a store with only some
of its chunks present is still a valid store: the missing ones read as
``fill_value``. So this fetches the chunks a box touches, writes them at the same
relative paths, copies the metadata, and leaves a local store the tracer can open
and read as though the rest of the scroll simply happened to be empty.

    python scripts/fetch_zarr_box.py --store <url> --out local.zarr \
        --center 40000 16000 16000 --radius 1200 --levels 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests


def build_session(pool: int):
    """A session that survives a multi-gigabyte transfer.

    S3 resets connections under sustained concurrency, and a fetch that treats
    that as fatal dies partway through -- leaving a store with some scale levels
    and not others, which the reader then rejects for a reason that looks nothing
    like the actual cause.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get(session, url, timeout=120):
    try:
        r = session.get(url, timeout=timeout)
    except Exception:
        return None  # exhausted retries; the caller counts it rather than dying
    return r if r.status_code == 200 else None


def mirror_metadata(session, store, out, level):
    """Copy .zarray for one scale, and the group metadata once."""
    os.makedirs(os.path.join(out, str(level)), exist_ok=True)
    r = get(session, f"{store}/{level}/.zarray")
    if r is None:
        return None
    with open(os.path.join(out, str(level), ".zarray"), "wb") as fh:
        fh.write(r.content)
    return json.loads(r.text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="base URL of the remote zarr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--center", nargs=3, type=int, required=True, metavar=("Z", "Y", "X"))
    ap.add_argument(
        "--radius", type=int, required=True, help="half-width in voxels, at level 0"
    )
    ap.add_argument("--levels", nargs="+", type=int, default=[0])
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args(argv)

    session = build_session(args.jobs)
    os.makedirs(args.out, exist_ok=True)
    for name in (".zgroup", ".zattrs"):
        r = get(session, f"{args.store}/{name}")
        if r is not None:
            with open(os.path.join(args.out, name), "wb") as fh:
                fh.write(r.content)

    total_files = total_bytes = missing = total_failed = 0
    for level in args.levels:
        meta = mirror_metadata(session, args.store, args.out, level)
        if meta is None:
            print(f"level {level}: no .zarray, skipped", file=sys.stderr)
            continue
        scale = 2**level
        shape = meta["shape"]
        chunks = meta["chunks"]
        centre = [c // scale for c in args.center]
        radius = max(1, args.radius // scale)

        keys = []
        spans = [
            range(
                max(0, (centre[i] - radius) // chunks[i]),
                min((shape[i] - 1) // chunks[i], (centre[i] + radius) // chunks[i]) + 1,
            )
            for i in range(3)
        ]
        for z in spans[0]:
            for y in spans[1]:
                for x in spans[2]:
                    keys.append((z, y, x))

        def pull(key, level=level):  # bound now, not when the pool runs it
            z, y, x = key
            rel = f"{level}/{z}/{y}/{x}"
            dest = os.path.join(args.out, rel)
            if os.path.exists(dest):
                return 0
            r = get(session, f"{args.store}/{rel}")
            if r is None:
                # Either genuinely absent upstream (most of a box around a scroll
                # is empty air) or a transfer that failed every retry. A HEAD tells
                # them apart, and only the second kind is a problem.
                head = None
                try:
                    head = session.head(f"{args.store}/{rel}", timeout=60)
                except Exception:
                    return -2
                return -1 if head is not None and head.status_code == 404 else -2
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + f".{os.getpid()}.part"
            with open(tmp, "wb") as fh:
                fh.write(r.content)
            os.replace(tmp, dest)
            return len(r.content)

        with ThreadPoolExecutor(args.jobs) as pool:
            sizes = list(pool.map(pull, keys))
        got = [s for s in sizes if s > 0]
        absent = sum(1 for s in sizes if s == -1)
        failed = sum(1 for s in sizes if s == -2)
        total_files += len(got)
        total_bytes += sum(got)
        missing += absent
        total_failed += failed
        print(
            f"level {level}: {len(keys)} in box, {len(got)} fetched, "
            f"{absent} absent upstream, {failed} FAILED, {sum(got) / 1e6:.1f} MB"
        )

    print(
        f"total {total_files} chunks, {total_bytes / 1e9:.2f} GB, "
        f"{missing} absent, {total_failed} failed -> {args.out}"
    )
    if total_failed:
        # a store missing chunks it should have had is worse than no store: the
        # reader cannot tell that apart from genuinely empty scroll
        print(f"ERROR: {total_failed} chunks could not be fetched", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
