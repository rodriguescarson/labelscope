#!/usr/bin/env python
"""Fetch a stratified sample of Dataset059 CT volumes.

Enough to ask whether the label-to-density-maximum offset measured on the Kaggle
release replicates on a second, independently produced dataset.  Stratified by
scroll and by patch size, since the release mixes four of them.
"""
import collections, os, random, threading, time
import requests
from concurrent.futures import ThreadPoolExecutor

BASE = ("https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_raw/"
        "Dataset059_s1_s4_s5_patches_frangiedt/imagesTr")
DEST = os.path.expanduser("~/Projects/vesuvius-label-audit/data/Dataset059/imagesTr")
LABELS = os.path.expanduser("~/Projects/vesuvius-label-audit/data/Dataset059/labelsTr")
STALL = 60
WANT = int(os.environ.get("LS_N", "48"))

import sys
sys.path.insert(0, os.path.expanduser("~/Projects/vesuvius-label-audit/src"))
from labelscope.io import probe_volume

names = [n.strip()[:-4] for n in open("/tmp/labels.txt") if n.strip()]
groups = collections.defaultdict(list)
for name in names:
    info = probe_volume(os.path.join(LABELS, name + ".tif"))
    if info.shape and info.shape[0] == info.shape[1] == info.shape[2]:
        groups[(name.split("_")[0], info.shape)].append(name)

random.seed(0)
plan = []
per_group = max(1, WANT // max(1, len(groups)))
for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    plan += random.sample(members, min(per_group, len(members)))
plan = plan[:WANT]
print(f"groups: { {str(k): len(v) for k, v in groups.items()} }", flush=True)
print(f"fetching {len(plan)} images", flush=True)

lock = threading.Lock()
state = {"done": 0, "bytes": 0, "t0": time.time(), "fail": 0}


def fetch(name):
    dst = os.path.join(DEST, name + "_0000.tif")
    part = dst + ".part"
    session = requests.Session()
    for attempt in range(50):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with session.get(f"{BASE}/{name}_0000.tif", headers=headers, stream=True,
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
                    for chunk in r.iter_content(256 << 10):
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


with ThreadPoolExecutor(4) as ex:
    for res in ex.map(fetch, plan):
        with lock:
            state["done"] += 1
            n, mb, dt = state["done"], state["bytes"] / 1e6, time.time() - state["t0"]
        if res.startswith("FAIL"):
            print(res, flush=True)
        if n % 4 == 0:
            print(f"{n}/{len(plan)}  {mb:.0f} MB  {dt:.0f}s  {mb/max(dt,1)*1000:.0f} KB/s",
                  flush=True)
print("COMPLETE", state["done"], "failures", state["fail"], flush=True)
