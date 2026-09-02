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


PAGE_ARTIFACT = """<title>Ink Bench __SET__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
 :root{--ground:#0d0e10;--panel:#17191c;--line:#2a2d31;--ink:#d9dbd6;--muted:#8b8f88;--accent:#c9a24a;
       --yes:#7bb07a;--no:#c7605a;--maybe:#a9a06b;--focus:#e8d08a}
 html,body{height:100%}
 body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.45 "IBM Plex Sans",-apple-system,Helvetica,Arial,sans-serif;display:flex;flex-direction:column}
 header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
 header h1{font-size:15px;font-weight:500;margin:0;color:var(--ink)}
 header h1 b{font-weight:600}
 .verdicts{display:flex;gap:8px}
 button{font:15px/1 "IBM Plex Sans",-apple-system,Helvetica,Arial,sans-serif;padding:11px 16px;border-radius:5px;border:1px solid var(--line);background:#1f2226;color:var(--ink);cursor:pointer;display:inline-flex;align-items:center;gap:8px}
 button:hover{border-color:#3a3e44;background:#24272c}
 button:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
 .dot{width:9px;height:9px;border-radius:50%;display:inline-block}
 .k{color:var(--muted);font:12px "IBM Plex Mono",ui-monospace,Menlo,monospace}
 .meta{display:flex;gap:16px;align-items:baseline;margin-left:auto;font-variant-numeric:tabular-nums}
 #code{font:16px "IBM Plex Mono",ui-monospace,Menlo,monospace;color:var(--accent);letter-spacing:.04em}
 #prog{color:var(--muted)}
 #hint{padding:6px 16px;color:var(--muted);font-size:13px;border-bottom:1px solid var(--line);background:var(--ground)}
 #stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#000}
 #img{max-width:100%;max-height:100%;object-fit:contain;display:block}
 #done{padding:20px 16px;display:flex;flex-direction:column;gap:12px}
 #done p{margin:0;max-width:65ch}
 textarea{width:100%;min-height:50vh;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:12px;font:12px/1.5 "IBM Plex Mono",ui-monospace,Menlo,monospace;resize:vertical}
 @media (prefers-reduced-motion:no-preference){button{transition:background .12s,border-color .12s}}
</style>
<header>
 <h1>Does this render show <b>text</b>?</h1>
 <div class="verdicts">
  <button onclick="mark('text')"><span class="dot" style="background:var(--yes)"></span>text <span class="k">1</span></button>
  <button onclick="mark('no text')"><span class="dot" style="background:var(--no)"></span>no text <span class="k">2</span></button>
  <button onclick="mark('unsure')"><span class="dot" style="background:var(--maybe)"></span>unsure <span class="k">3</span></button>
  <button onclick="back()">back <span class="k">b</span></button>
 </div>
 <div class="meta"><span id="prog"></span><span id="code"></span><button onclick="finish()">finish</button></div>
</header>
<div id="hint">Ink-probability render (ds8) of one published segment, bright = predicted ink. Lines or columns of letterforms, however faint, count as text. Speckle, noise or blank is no text. Set __SET__ of 2.</div>
<div id="stage"><img id="img" alt="ink render under a random code"></div>
<div id="done" hidden>
 <p>All __N__ done. Copy everything below into <span class="k">drafts/ink-labels-__SETLC__.csv</span>, or just tell Claude it is finished and paste it in the chat.</p>
 <button onclick="selectAll()">select all</button>
 <textarea id="csv" readonly></textarea>
</div>
<script>
const ITEMS=__ITEMS__;const KEY="ink-labels-v1-__SETLC__";
let labels={};try{labels=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=ITEMS.findIndex(it=>!(it.code in labels));if(i<0)i=ITEMS.length;
function save(){try{localStorage.setItem(KEY,JSON.stringify(labels))}catch(e){}}
function show(){if(i>=ITEMS.length){finish();return;}
 document.getElementById("done").hidden=true;document.getElementById("stage").hidden=false;
 const it=ITEMS[i];document.getElementById("img").src=it.src;
 document.getElementById("prog").textContent=(i+1)+" / "+ITEMS.length;
 document.getElementById("code").textContent="#"+it.code+(labels[it.code]?"  "+labels[it.code]:"");}
function mark(v){if(i>=ITEMS.length)return;labels[ITEMS[i].code]=v;save();i++;show();}
function back(){if(i>0){i--;show();}}
function finish(){let csv="code,label\n";for(const it of ITEMS){csv+=it.code+","+(labels[it.code]||"")+"\n";}
 document.getElementById("csv").value=csv;document.getElementById("stage").hidden=true;document.getElementById("done").hidden=false;}
function selectAll(){const t=document.getElementById("csv");t.focus();t.select();}
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;
 if(e.key==="1")mark("text");else if(e.key==="2")mark("no text");else if(e.key==="3")mark("unsure");else if(e.key==="b")back();});
show();
</script>"""


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
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument(
        "--pages",
        type=int,
        default=1,
        help="split into N pages (out-1.html ...); the key is unchanged",
    )
    ap.add_argument(
        "--artifact",
        action="store_true",
        help="emit page content for the Artifact host (no html/head/body)",
    )
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
        im.save(buf, format="JPEG", quality=args.quality)
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

    if args.pages > 1:
        per = -(-len(items) // args.pages)
        stem, ext = os.path.splitext(args.out)
        for k in range(args.pages):
            chunk = items[k * per : (k + 1) * per]
            if args.artifact:
                set_name = "AB"[k] if args.pages == 2 else str(k + 1)
                page = (
                    PAGE_ARTIFACT.replace("__ITEMS__", json.dumps(chunk))
                    .replace("__SETLC__", set_name.lower())
                    .replace("__SET__", set_name)
                    .replace("__N__", str(len(chunk)))
                )
            else:
                page = PAGE.replace("__ITEMS__", json.dumps(chunk)).replace(
                    'const KEY="ink-labels-v1"', f'const KEY="ink-labels-v1-p{k + 1}"'
                )
            with open(f"{stem}-{k + 1}{ext}", "w") as fh:
                fh.write(page)
            print(
                f"page {k + 1}: {len(chunk)} items -> {stem}-{k + 1}{ext} ({os.path.getsize(f'{stem}-{k + 1}{ext}') // 2**20} MB)"
            )
    else:
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
