#!/usr/bin/env bash
# Is any *published* surface off-sheet?
#
# villa#1675 asks that of grown surfaces. The same measurement, calibrated
# against a published baseline so the number is comparable across rows, can ask
# it of the published corpus instead.
#
#   OUT=findings/onsheet scripts/corpus_onsheet.sh
set -u
B="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOL="${VOL:-$B/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr}"
OUT="${OUT:-/workspace/onsheet_corpus}"
MESHES="${MESHES:-/workspace/corpus_meshes}"
PY="${PY:-/workspace/labelscope/.venv/bin/python}"
BASELINE="${BASELINE:-/workspace/pubmesh}"
mkdir -p "$OUT"

n=0
for d in "$MESHES"/*/; do
  [ -f "$d/meta.json" ] || continue
  name=$(basename "$d")
  [ -f "$OUT/$name.json" ] && continue
  n=$((n + 1))
  "$PY" /workspace/onsheet_check.py --mesh "$d" --baseline "$BASELINE" \
      --volume "$VOL" --remote --blocks 5 --block-size 12 \
      --out "$OUT/$name.json" > "$OUT/$name.txt" 2>&1
  echo "[$(date -u +%H:%M)] $n $name $(grep -oE 'ON SHEET|OFF SHEET|marginal' "$OUT/$name.txt" | head -1)"
done
echo "CORPUS ONSHEET DONE $(date -u +%H:%M)  measured: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
