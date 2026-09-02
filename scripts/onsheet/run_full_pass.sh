#!/bin/bash
# On-sheet pass over every published surface not yet measured.
# Disk is 20GB, chunk cache runs ~290MB/surface, so the cache is cleared
# whenever it passes 7GB rather than at the end.
set -u
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
LS=/root/labelscope
OUT=/root/out; MESH=/root/mesh; CACHE=/root/cache
mkdir -p "$OUT" "$MESH" "$CACHE"
LOG=/root/full.log; : > "$LOG"
total=$(wc -l < /root/todo_full.tsv)
echo "surfaces to measure: $total" | tee -a "$LOG"

n=0
while IFS=$'\t' read -r base vol; do
  n=$((n+1))
  [ -f "$OUT/$base.json" ] && { echo "[$n/$total] have $base" >> "$LOG"; continue; }
  scroll=${base%%__*}; rest=${base#*__}; seg=${rest%%__*}; tif=${rest#*__}
  d="$MESH/$base"
  if [ ! -f "$d/meta.json" ]; then
    mkdir -p "$d"; ok=1
    for f in meta.json x.tif y.tif z.tif; do
      curl -sfS --retry 5 --retry-delay 2 -o "$d/$f" \
        "$B/$scroll/segments/$seg/mesh/$tif/$f" || { ok=0; break; }
    done
    [ $ok -eq 1 ] || { echo "[$n/$total] FETCH FAILED $base" | tee -a "$LOG"; rm -rf "$d"; continue; }
  fi
  timeout 1800 "$LS/.venv/bin/python" "$LS/scripts/onsheet_check.py" \
      --mesh "$d" --volume "$vol" --remote --cache "$CACHE" \
      --blocks 8 --block-size 12 --out "$OUT/$base.json" >/dev/null 2>&1 \
    && echo "[$n/$total] ok $base" >> "$LOG" \
    || echo "[$n/$total] MEASURE FAILED $base" | tee -a "$LOG"
  rm -rf "$d"                                  # mesh no longer needed
  cs=$(du -sm "$CACHE" 2>/dev/null | cut -f1)
  if [ "${cs:-0}" -gt 7000 ]; then rm -rf "$CACHE"; mkdir -p "$CACHE"; echo "  (cache cleared at ${cs}MB)" >> "$LOG"; fi
done < /root/todo_full.tsv
echo "FULL PASS DONE $(date -u +%H:%M) — $(ls $OUT | wc -l) results" | tee -a "$LOG"
