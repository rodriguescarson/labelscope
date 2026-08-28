#!/usr/bin/env python3
"""Fetch a directory of volumes over HTTP, and verify that you got all of them.

    python scripts/fetch_dataset.py <base-url> <destination> [--names names.txt]
                                    [--jobs N] [--limit N]

Written because the obvious version of this script silently truncated eight
volumes and cost a retraction.  Two rules it follows and the obvious version does
not:

* **Verify what was written, not what was promised.**  A streaming download that
  checks "bytes received >= Content-Length" passes trivially when the server
  omits that header — the comparison is against zero — and a partial file gets
  renamed as complete.  Here a missing header triggers a HEAD, and a file that
  still cannot be verified is an error rather than a success.
* **Resume by byte range,** because the hosts serving this data drop idle
  sockets, and a restart that begins from zero never finishes on a slow link.

An index page is parsed for filenames when ``--names`` is not given, which works
for the Apache-style listings on dl.ash2txt.org.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

STALL_SECONDS = 60


def listing(url: str, session: requests.Session) -> list:
    response = session.get(url if url.endswith("/") else url + "/", timeout=120)
    response.raise_for_status()
    return sorted(set(re.findall(r'href="([^"?/][^"]*\.tiff?)"', response.text)))


def declared_size(url: str, session: requests.Session) -> int:
    try:
        head = session.head(url, allow_redirects=True, timeout=60)
        return int(head.headers.get("content-length", 0))
    except Exception:
        return 0


def fetch_one(url: str, destination: str, session: requests.Session, state, lock) -> str:
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return "skip"
    partial = destination + ".part"
    for attempt in range(40):
        try:
            have = os.path.getsize(partial) if os.path.exists(partial) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with session.get(
                url, headers=headers, stream=True, timeout=(30, STALL_SECONDS)
            ) as response:
                if response.status_code == 404:
                    return "missing"
                if response.status_code == 416:
                    pass
                elif response.status_code in (200, 206):
                    if response.status_code == 200 and have:
                        have = 0
                    with open(partial, "ab" if have else "wb") as handle:
                        for chunk in response.iter_content(1 << 20):
                            handle.write(chunk)
                            with lock:
                                state["bytes"] += len(chunk)
                else:
                    response.raise_for_status()

            expected = declared_size(url, session)
            written = os.path.getsize(partial) if os.path.exists(partial) else 0
            if not expected:
                raise RuntimeError("server declares no length; completeness unverifiable")
            if written < expected:
                raise RuntimeError(f"short read: {written} of {expected}")
            os.replace(partial, destination)
            return "ok"
        except Exception as exc:
            if attempt == 39:
                print(f"FAIL {url}: {exc}", file=sys.stderr, flush=True)
            time.sleep(min(20, 2 + attempt))
    return "fail"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("base_url")
    parser.add_argument("destination")
    parser.add_argument("--names", help="file of one filename per line")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    session = requests.Session()
    if args.names:
        with open(args.names) as handle:
            names = [line.strip() for line in handle if line.strip()]
    else:
        names = listing(args.base_url, session)
    if args.limit:
        names = names[: args.limit]
    os.makedirs(args.destination, exist_ok=True)
    print(f"{len(names)} files -> {args.destination}", flush=True)

    lock = threading.Lock()
    state = {"done": 0, "bytes": 0, "t0": time.time()}
    counts: dict = {}

    def run(name):
        result = fetch_one(
            f"{args.base_url.rstrip('/')}/{name}",
            os.path.join(args.destination, name),
            requests.Session(),
            state,
            lock,
        )
        with lock:
            state["done"] += 1
            counts[result] = counts.get(result, 0) + 1
            done, mb = state["done"], state["bytes"] / 1e6
            elapsed = time.time() - state["t0"]
        if done % 50 == 0:
            print(
                f"{done}/{len(names)}  {mb:.0f} MB  {elapsed:.0f}s  "
                f"{mb / max(elapsed, 1):.1f} MB/s  {counts}",
                flush=True,
            )
        return result

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(run, names))
    print(f"done: {counts}", flush=True)
    return 1 if counts.get("fail") else 0


if __name__ == "__main__":
    raise SystemExit(main())
