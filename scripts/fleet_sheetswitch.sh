#!/bin/bash
# Run the sheet-switch detector over every locally-present tifxyz surface,
# streaming the scan.  Resumable: a surface with a result already is skipped.
set -u
LS="${LS:-./.venv/bin/labelscope}"
VOL="${VOL:-https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr}"
OUT="${OUT:-findings/fleet_p4}"
WINDOW="${WINDOW:-160}"
mkdir -p "$OUT"
for m in "$@"; do
  name=$(basename "$m")
  [ -f "$OUT/$name/sheetswitch.csv" ] && continue
  echo "[$(date +%H:%M:%S)] $name" >> "$OUT/progress.log"
  $LS sheetswitch --mesh "$m" --volume "$VOL" --remote --window "$WINDOW" \
      --out "$OUT/$name" > "$OUT/$name.out" 2>&1
  echo "[$(date +%H:%M:%S)] done $name rc=$?" >> "$OUT/progress.log"
done
echo "[$(date +%H:%M:%S)] ALL DONE" >> "$OUT/progress.log"
