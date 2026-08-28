#!/usr/bin/env python
"""Resumable, stall-proof fetch of a stratified slice of the scrollprize Kaggle
surface-detection dataset from the Hugging Face bucket.

The link this runs over is slow (~0.2-0.5 MB/s) and the server drops idle
sockets, so every transfer is resumed by byte range and watched for stalls.
"""
import json, os, random, sys, threading, time
import requests
from concurrent.futures import ThreadPoolExecutor

BUCKET = "https://huggingface.co/buckets/scrollprize/datasets/resolve/surfaces/kaggle"
ROOT = os.path.expanduser("~/Projects/vesuvius-label-audit/data/kaggle_surfaces")
DENSE = 32821210
STALL_SECONDS = 45
N_IMAGES = int(os.environ.get("LS_N_IMAGES", "50"))

sizes = {int(k): v for k, v in json.load(open("/tmp/labsizes.json")).items()}
present = {i: v for i, v in sizes.items() if v > 0}
dense = sorted(i for i, v in present.items() if v == DENSE)
sparse = sorted(i for i, v in present.items() if v != DENSE)
random.seed(0)

half = N_IMAGES // 2
img = sorted(random.sample(sparse, half) + random.sample(dense, N_IMAGES - half))
plan = [("images", i) for i in img]
plan += [("labels", i) for i in img]
plan += [("labels", i) for i in sparse]
plan += [("labels", i) for i in random.sample(dense, 60)]
seen, ordered = set(), []
for job in plan:
    if job not in seen:
        seen.add(job)
        ordered.append(job)
plan = ordered

lock = threading.Lock()
state = {"done": 0, "bytes": 0, "t0": time.time()}


def expected_size(kind, i):
    return present[i] if kind == "labels" else None


def fetch_once(url, dst, session):
    """One resumed attempt.  Returns True when the file is complete."""
    part = dst + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(url, headers=headers, stream=True, timeout=(30, STALL_SECONDS)) as r:
        if r.status_code == 416:                       # already complete
            total = have
        elif r.status_code in (200, 206):
            if r.status_code == 200 and have:
                have = 0                               # server ignored the range
                open(part, "wb").close()
            total = have + int(r.headers.get("content-length", 0))
        else:
            r.raise_for_status()
            return False
        last = time.time()
        with open(part, "ab" if have else "wb") as fh:
            for chunk in r.iter_content(256 << 10):
                if not chunk:
                    if time.time() - last > STALL_SECONDS:
                        raise TimeoutError("stalled")
                    continue
                fh.write(chunk)
                with lock:
                    state["bytes"] += len(chunk)
                last = time.time()
    got = os.path.getsize(part)
    if total and got >= total:
        os.replace(part, dst)
        return True
    return False


def fetch(job):
    kind, i = job
    dst = f"{ROOT}/{kind}/sample_{i:05d}.tif"
    want = expected_size(kind, i)
    if os.path.exists(dst) and (want is None or os.path.getsize(dst) == want):
        if kind == "images" and os.path.getsize(dst) < 20_000_000:
            os.remove(dst)                              # truncated leftover
        else:
            return "skip"
    url = f"{BUCKET}/{kind}/sample_{i:05d}.tif"
    session = requests.Session()
    for attempt in range(40):
        try:
            if fetch_once(url, dst, session):
                return "ok"
        except Exception as exc:
            if attempt % 8 == 7:
                print(f"  retry {attempt} {kind}/{i}: {type(exc).__name__}", flush=True)
        time.sleep(min(20, 2 + attempt))
    return f"FAIL {kind} {i}"


with ThreadPoolExecutor(3) as ex:
    for res in ex.map(fetch, plan):
        with lock:
            state["done"] += 1
            n, mb = state["done"], state["bytes"] / 1e6
            dt = time.time() - state["t0"]
        if res.startswith("FAIL"):
            print(res, flush=True)
        if n % 5 == 0:
            print(f"{n}/{len(plan)}  {mb:.0f} MB  {dt:.0f}s  {mb/max(dt,1)*1000:.0f} KB/s",
                  flush=True)
print("COMPLETE", state["done"], flush=True)
