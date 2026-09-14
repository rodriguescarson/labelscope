#!/usr/bin/env bash
# Seam detector on pscamillo's 12 eye-labelled rejects. Pre-registered in
# findings/reject-seam-preregistration.md. Run on a CPU pod; ~$0.06/h.
set -euo pipefail
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
declare -A VOL=(
  [PHerc0813]="$B/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr"
  [PHerc0211]="$B/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr"
  [PHerc0800]="$B/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr"
  [PHerc0125]="$B/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr"
)
MESHES="PHerc0813/z12496_w100 PHerc0813/z11296_w100 PHerc0813/z4704_w100 PHerc0813/z13088_w100 PHerc0813/z11904_w100 PHerc0813/z11904_w080 PHerc0211/z4912_w080 PHerc0211/z13920_w080 PHerc0800/z17072_w100 PHerc0125/z6944_w080 PHerc0211/z6720_w100 PHerc0211/z9120_w100"
RAW=https://raw.githubusercontent.com/pscamillo/vesuvius-eligible-meshes/main/meshes
mkdir -p meshes out cache
for m in $MESHES; do
  d="meshes/$m"; mkdir -p "$d"
  for f in x.tif y.tif z.tif meta.json; do [ -s "$d/$f" ] || curl -sfL "$RAW/$m/$f" -o "$d/$f"; done
done
echo "fetched $(ls meshes/*/ | grep -c _w) meshes" | tee -a progress.log
for m in $MESHES; do
  s=${m%%/*}; n=${m##*/}; tag="${s}__${n}"
  for plant in 0 1; do
    o="out/${tag}__plant${plant}.json"; [ -s "$o" ] && continue
    echo "START $tag plant=$plant $(date -u +%H:%M:%S)" | tee -a progress.log
    if [ "$plant" = 1 ]; then P="--plant 1"; else P=""; fi
    labelscope sheetswitch --mesh "meshes/$m" --volume "${VOL[$s]}" --remote --cache cache $P --out "$o" > "out/${tag}__plant${plant}.out" 2>&1 || echo "FAILED $tag plant=$plant" | tee -a progress.log
    echo "DONE  $tag plant=$plant $(date -u +%H:%M:%S)" | tee -a progress.log
  done
done
echo "ALL DONE $(date -u +%H:%M:%S)" | tee -a progress.log
