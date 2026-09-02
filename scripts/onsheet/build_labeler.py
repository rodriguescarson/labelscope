#!/usr/bin/env python3
"""Build the blind labelling page for the ink-render test.

One self-contained HTML file: every segment's ink render as a thumbnail under a
random three-digit code, in random order, with three buttons.  Nothing on the
page identifies the segment, its scroll, or its on-sheet score.  Progress is
kept in localStorage; the result is copied out as CSV.

The code-to-segment key is written separately and sealed (its SHA-256 goes
into findings/ink-blind-preregistration.md before labelling starts).

    python scripts/onsheet/build_labeler.py --thumbs w3/thumbs --manifest w3/manifest.tsv \\
        --out drafts/ink-labeler.html --key drafts/ink-labeler-key.json --seed 7
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys

import numpy as np

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Ink render labelling</title>
<style>
 body{margin:0;font:15px/1.4 -apple-system,Helvetica,Arial,sans-serif;background:#111;color:#ddd}
 #top{display:flex;gap:16px;align-items:center;padding:10px 16px;background:#1b1b1b;border-bottom:1px solid #333}
 #img{display:block;max-width:100vw;max-height:78vh;margin:0 auto;object-fit:contain;background:#000}
 button{font:16px inherit;padding:10px 18px;border-radius:6px;border:1px solid #555;background:#222;color:#eee;cursor:pointer}
 button:hover{background:#333}.k{opacity:.6;font-size:13px}
 #done{padding:24px;white-space:pre;font-family:ui-monospace,Menlo,monospace;font-size:12px;display:none}
 #code{font-family:ui-monospace,Menlo,monospace;font-size:18px}
</style></head><body>
<div id="top">
 <span>Does this ink render show <b>text</b>?</span>
 <button onclick="mark('text')">text <span class=k>[1]</span></button>
 <button onclick="mark('no text')">no text <span class=k>[2]</span></button>
 <button onclick="mark('unsure')">unsure <span class=k>[3]</span></button>
 <button onclick="back()">back <span class=k>[b]</span></button>
 <span id="prog"></span><span id="code"></span>
 <button style="margin-left:auto" onclick="finish()">finish / copy CSV</button>
</div>
<img id="img"><div id="done"></div>
<script>
const ITEMS=__ITEMS__;const KEY="ink-labels-v1";
let labels=JSON.parse(localStorage.getItem(KEY)||"{}");
let i=ITEMS.findIndex(it=>!(it.code in labels)); if(i<0)i=ITEMS.length;
function show(){ if(i>=ITEMS.length){finish();return;}
 const it=ITEMS[i]; document.getElementById("img").src=it.src;
 document.getElementById("prog").textContent=(i+1)+" / "+ITEMS.length;
 document.getElementById("code").textContent="#"+it.code+(labels[it.code]?"  ("+labels[it.code]+")":""); }
function mark(v){ labels[ITEMS[i].code]=v; localStorage.setItem(KEY,JSON.stringify(labels)); i++; show(); }
function back(){ if(i>0){i--; show();} }
function finish(){ let csv="code,label\\n"; for(const it of ITEMS){ csv+=it.code+","+(labels[it.code]||"")+"\\n"; }
 const d=document.getElementById("done"); d.style.display="block"; d.textContent=csv; document.getElementById("img").style.display="none";
 navigator.clipboard&&navigator.clipboard.writeText(csv).catch(()=>{}); }
document.addEventListener("keydown",e=>{ if(e.key==="1")mark("text"); else if(e.key==="2")mark("no text"); else if(e.key==="3")mark("unsure"); else if(e.key==="b")back(); });
show();
</script></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbs", required=True)
    ap.add_argument(
        "--manifest", required=True, help="scroll\\tsegment\\tzarr\\tink\\tbase per line"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--width", type=int, default=1400)
    args = ap.parse_args(argv)

    from PIL import Image

    rows = []
    with open(args.manifest) as fh:
        lines = fh.readlines()
    for line in lines:
        p = line.rstrip("\n").split("\t")
        if len(p) < 5 or p[3] == "NONE":
            continue
        thumb = os.path.join(args.thumbs, p[4] + ".jpg")
        if os.path.exists(thumb):
            rows.append(
                {"scroll": p[0], "segment": p[1], "base": p[4], "ink": p[3], "thumb": thumb}
            )
    rng = np.random.default_rng(args.seed)
    codes = rng.permutation(np.arange(100, 1000))[: len(rows)]
    order = rng.permutation(len(rows))

    items, key = [], {}
    for n, idx in enumerate(order):
        r = rows[idx]
        code = int(codes[n])
        im = Image.open(r["thumb"]).convert("L")
        im.thumbnail((args.width, args.width))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        items.append(
            {
                "code": f"{code:03d}",
                "src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
            }
        )
        key[f"{code:03d}"] = {
            "scroll": r["scroll"],
            "segment": r["segment"],
            "base": r["base"],
            "ink": r["ink"],
        }

    html = PAGE.replace("__ITEMS__", json.dumps(items))
    with open(args.out, "w") as fh:
        fh.write(html)
    with open(args.key, "w") as fh:
        json.dump(key, fh, indent=1, sort_keys=True)
    with open(args.key, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(f"{len(items)} items -> {args.out} ({os.path.getsize(args.out) // 2**20} MB)")
    print(f"key -> {args.key}")
    print(f"SHA-256(key) = {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
