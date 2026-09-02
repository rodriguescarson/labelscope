#!/bin/bash
# The headline w128-129 numbers, from a clean checkout, in one file.
#
#   pip install "git+https://github.com/rodriguescarson/labelscope#egg=labelscope[zarr]"
#   bash findings/onsheet/REPRODUCE.sh
#
# Fetches the four published PHercParis4 surfaces (two tracings each of windings
# 126-127 and 128-129), streams only the scan chunks the blocks touch, and runs
# the comparison the finding rests on: each w128-129 surface against the
# adjacent winding of the SAME tracing run, so scan, region and provenance are
# held fixed and only the winding varies.
set -euo pipefail
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
VOL=$B/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
OUT=${OUT:-reproduce_w128}; mkdir -p "$OUT/meshes" "$OUT/cache"

fetch() {  # <segment-folder> <mesh-name>
  local d="$OUT/meshes/$1"; mkdir -p "$d"
  for f in meta.json x.tif y.tif z.tif; do
    [ -f "$d/$f" ] || curl -sfS --retry 5 -o "$d/$f" "$B/PHercParis4/segments/$1/mesh/$2/$f"
  done
}
fetch 20260623171400-w126-127 20260623171400-on-20260411134726-2.4um.tifxyz
fetch 20260623171929-w128-129 20260623171929-on-20260411134726-2.4um.tifxyz
fetch 20260701183150-w126-127 20260701183150-on-20260411134726-2.4um.tifxyz
fetch 20260701183151-w128-129 20260701183151-on-20260411134726-2.4um.tifxyz

for series in 20260623 20260701; do
  sus=$(ls -d "$OUT"/meshes/${series}*-w128-129)
  nb=$(ls -d "$OUT"/meshes/${series}*-w126-127)
  echo "== series $series: w128-129 vs adjacent w126-127 =="
  labelscope onsheet --mesh "$sus" "$nb" --volume "$VOL" --remote --cache "$OUT/cache" \
    --compare --blocks 24 --block-size 12 --seed 0 --out "$OUT/compare_$series.json"
  echo
done
