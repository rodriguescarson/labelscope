#!/bin/bash
# Measure the two series the pre-registration adds. PHercParis4 is already done.
set -u
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
M=/workspace/labelscope/findings/corpus/inputs/corpus_manifest.tsv
OUT=/workspace/onsheet_series; MESH=/workspace/series_meshes
mkdir -p "$OUT" "$MESH"
LOG=/workspace/series.log; : > "$LOG"

grep -E 'PHerc0139__20250108|PHerc0172__20250926' "$M" > /tmp/todo.tsv
echo "to measure: $(wc -l < /tmp/todo.tsv)" | tee -a "$LOG"

n=0
while IFS=$'\t' read -r mpath vol; do
  n=$((n+1))
  base=$(basename "$mpath")                 # Scroll__SEGMENT__TIFXYZ
  scroll=${base%%__*}; rest=${base#*__}
  seg=${rest%%__*}; tif=${rest#*__}
  dest="$MESH/$base"
  if [ ! -f "$dest/meta.json" ]; then
    mkdir -p "$dest"
    ok=1
    for f in meta.json x.tif y.tif z.tif; do
      curl -sfS --retry 5 --retry-delay 2 -o "$dest/$f" \
        "$B/$scroll/segments/$seg/mesh/$tif/$f" || { ok=0; break; }
    done
    [ $ok -eq 1 ] || { echo "[$(date +%H:%M)] $n FETCH FAILED $base" | tee -a "$LOG"; rm -rf "$dest"; continue; }
  fi
  echo "[$(date +%H:%M)] $n measuring $base" | tee -a "$LOG"
  timeout 1200 /workspace/labelscope/.venv/bin/python /workspace/onsheet_check.py \
      --mesh "$dest" --volume "$vol" --remote --blocks 8 --block-size 12 \
      --out "$OUT/$base.json" >>"$LOG" 2>&1 \
    || echo "[$(date +%H:%M)] $n MEASURE FAILED $base" | tee -a "$LOG"
done < /tmp/todo.tsv
echo "SERIES MEASURE DONE $(date +%H:%M)" | tee -a "$LOG"
