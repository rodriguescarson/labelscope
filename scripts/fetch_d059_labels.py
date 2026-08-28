#!/usr/bin/env python
"""Fetch the label volumes of Dataset059_s1_s4_s5_patches_frangiedt.

Labels only: they are ~0.4 MB each against ~28 MB for the CT, and the
memorisation experiment needs nothing else.  The host is slow and drops idle
sockets, so every transfer resumes by byte range and is watched for stalls.
"""
import os, threading, time
import requests
from concurrent.futures import ThreadPoolExecutor

BASE = ("https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/"
        "Dataset059_s1_s4_s5_patches_frangiedt/labelsTr")
DEST = os.path.expanduser("~/Projects/vesuvius-label-audit/data/Dataset059/labelsTr")
STALL = 60

names = [n.strip() for n in open("/tmp/labels.txt") if n.strip()]
lock = threading.Lock()
state = {"done": 0, "bytes": 0, "t0": time.time(), "fail": 0}


def fetch(name):
    dst = os.path.join(DEST, name)
    part = dst + ".part"
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return "skip"
    session = requests.Session()
    for attempt in range(50):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with session.get(f"{BASE}/{name}", headers=headers, stream=True,
                             timeout=(30, STALL)) as r:
                if r.status_code == 416:
                    os.replace(part, dst)
                    return "ok"
                if r.status_code not in (200, 206):
                    r.raise_for_status()
                if r.status_code == 200 and have:
                    have = 0
                total = have + int(r.headers.get("content-length", 0))
                with open(part, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(128 << 10):
                        fh.write(chunk)
                        with lock:
                            state["bytes"] += len(chunk)
            if total and os.path.getsize(part) >= total:
                os.replace(part, dst)
                return "ok"
        except Exception:
            time.sleep(min(15, 2 + attempt))
    with lock:
        state["fail"] += 1
    return f"FAIL {name}"


with ThreadPoolExecutor(12) as ex:
    for res in ex.map(fetch, names):
        with lock:
            state["done"] += 1
            n, mb, dt = state["done"], state["bytes"] / 1e6, time.time() - state["t0"]
        if res.startswith("FAIL"):
            print(res, flush=True)
        if n % 50 == 0:
            print(f"{n}/{len(names)}  {mb:.0f} MB  {dt:.0f}s  {mb/max(dt,1)*1000:.0f} KB/s  "
                  f"fail={state['fail']}", flush=True)
print("COMPLETE", state["done"], "failures", state["fail"], flush=True)
