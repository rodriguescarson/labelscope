#!/usr/bin/env bash
# Run the sheet-switch detector over many tifxyz surfaces, streaming the scan.
#
#   scripts/fleet_sheetswitch.sh surface.tifxyz [more...]
#   scripts/fleet_sheetswitch.sh --list paths.txt      # one path per line
#   scripts/fleet_sheetswitch.sh --pairs pairs.tsv     # mesh<TAB>volume-url
#   JOBS=8 scripts/fleet_sheetswitch.sh --list paths.txt
#
# --pairs is what a whole-corpus run needs: every scroll was traced on its own
# scan, so the volume cannot be a single global setting.
#
# The work is network-bound (each surface streams only the chunks it passes
# through), so JOBS well above the core count is the right setting.
#
# Resumable: a surface that already has a result is skipped, so an interrupted
# sweep can simply be re-run.
set -u
LS="${LS:-./.venv/bin/labelscope}"
VOL="${VOL:-https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr}"
OUT="${OUT:-findings/fleet_p4}"
WINDOW="${WINDOW:-160}"
JOBS="${JOBS:-1}"
# A disk cache is not optional over a slow link: without it an interrupted
# sweep re-fetches every byte, and the chunks a neighbouring surface needs
# overlap heavily with the ones already pulled.
CACHE="${CACHE:-.cache/chunks}"
# Set PLANT=N to run the control instead: N windings displaced into half of
# each surface before it is measured.
PLANT="${PLANT:-0}"

# Read the list from a file rather than relying on word splitting: zsh does not
# split unquoted parameters, which silently passed every path as one argument,
# and mapfile needs bash 4 while macOS ships 3.2.
MESHES=()
VOLUMES=()
if [ "${1:-}" = "--pairs" ]; then
  [ -f "${2:-}" ] || { echo "no such pairs file: ${2:-}" >&2; exit 2; }
  while IFS=$'\t' read -r m v; do
    [ -n "$m" ] || continue
    MESHES[${#MESHES[@]}]="$m"
    VOLUMES[${#VOLUMES[@]}]="$v"
  done < "$2"
elif [ "${1:-}" = "--list" ]; then
  [ -f "${2:-}" ] || { echo "no such list file: ${2:-}" >&2; exit 2; }
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    MESHES[${#MESHES[@]}]="$line"
    VOLUMES[${#VOLUMES[@]}]="$VOL"
  done < "$2"
else
  for arg in "$@"; do
    MESHES[${#MESHES[@]}]="$arg"
    VOLUMES[${#VOLUMES[@]}]="$VOL"
  done
fi
[ ${#MESHES[@]} -gt 0 ] || { echo "no surfaces given" >&2; exit 2; }

mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] sweep of ${#MESHES[@]} surfaces, window $WINDOW" >> "$OUT/progress.log"
one_surface() {
  m="$1"
  v="$2"
  name=$(basename "$m")
  echo "[$(date +%H:%M:%S)] start $name" >> "$OUT/progress.log"
  "$LS" sheetswitch --mesh "$m" --volume "$v" --remote --window "$WINDOW" \
      --cache "$CACHE" --plant "$PLANT" --out "$OUT/$name" > "$OUT/$name.out" 2>&1
  echo "[$(date +%H:%M:%S)] done  $name rc=$?" >> "$OUT/progress.log"
}

# bash 3.2 has no `wait -n`, so throttle by polling the running-job count.
i=0
while [ "$i" -lt "${#MESHES[@]}" ]; do
  m="${MESHES[$i]}"
  v="${VOLUMES[$i]}"
  i=$((i + 1))
  [ -n "$m" ] || continue
  name=$(basename "$m")
  [ -f "$OUT/$name/sheetswitch.csv" ] && continue
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$JOBS" ]; do sleep 2; done
  one_surface "$m" "$v" &
done
wait
echo "[$(date +%H:%M:%S)] ALL DONE" >> "$OUT/progress.log"
