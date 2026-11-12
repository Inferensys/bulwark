#!/usr/bin/env bash
# Download UTA-RLDD (~111 GB) from the official Google Drive folder linked at
# https://sites.google.com/view/utarldd/home, one fold-part at a time: each zip is
# tested, unzipped and deleted before the next starts, so peak disk is ~111 GB + one zip.
# Resumable: finished parts leave a .done marker, and $OUT/.complete is written once all
# parts are in (scripts/prepare_local.sh ingests UTA-RLDD only after that). Each part's
# Drive id, size and SHA-256 go to $OUT/SOURCE.txt (with the UTC time) before its zip is
# deleted, so the copy can be traced to what Drive served; a part extracted before that
# record was kept gets a row with '-' for size and hash, saying so. Needs gdown
# (`uv pip install -e ".[ops]"`).
# Drive often refuses popular files for a while ("Too many users have viewed or downloaded
# this file recently"), so each pass skips refused parts and retries them an hour later,
# for up to RETRY_PASSES passes. A corrupt zip is deleted so the next pass downloads it
# again; a failed extraction of a valid zip (usually a full disk) stops the script.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/data/raw/uta_rldd}"
GDOWN="${GDOWN:-$ROOT/.venv/bin/gdown}"
RETRY_PASSES="${RETRY_PASSES:-24}"
FOLDER=https://sites.google.com/view/utarldd/home
mkdir -p "$OUT"
sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}
if [ ! -s "$OUT/SOURCE.txt" ]; then
  {
    echo "# UTA-RLDD, downloaded by scripts/fetch_uta_rldd.sh from the Google Drive folder linked at"
    echo "# $FOLDER"
    printf '# part\tdrive_id\tbytes\tsha256\trecorded_utc\n'
  } > "$OUT/SOURCE.txt"
fi
parts() { sed -n '/^PARTS_START$/,/^PARTS_END$/p' "$0" | grep -v PARTS_; }
recorded() { grep -q "^$1$(printf '\t')" "$OUT/SOURCE.txt"; }
# A part already extracted has no zip left to hash; record that instead of leaving it out.
while read -r part id; do
  if [ -f "$OUT/.$part.done" ] && ! recorded "$part"; then
    printf '%s\t%s\t-\t-\tnot recorded (zip extracted and deleted before this row was written)\n' \
      "$part" "$id" >> "$OUT/SOURCE.txt"
  fi
done < <(parts)
for pass in $(seq 1 "$RETRY_PASSES"); do
  waiting=0
  while read -r part id; do
    [ -f "$OUT/.$part.done" ] && continue
    zip="$OUT/$part.zip"
    echo "== $part (pass $pass, $(date '+%H:%M'))"
    # gdown writes to a .part file and renames it when complete, so an existing zip is whole
    # as far as the download goes; it can still be extracted while Drive is refusing.
    if [ ! -f "$zip" ] && ! "$GDOWN" --continue "$id" -O "$zip" </dev/null; then
      waiting=$((waiting + 1)); continue   # Drive refusal or network error: retry next pass
    fi
    if ! unzip -tq "$zip" >/dev/null; then
      echo "$part.zip is corrupt; deleting it so the next pass downloads it again" >&2
      rm -f "$zip"; waiting=$((waiting + 1)); continue
    fi
    need_kb=$(( $(unzip -l "$zip" | tail -1 | awk '{print $1}') / 1024 ))
    free_kb=$(df -Pk "$OUT" | awk 'NR == 2 {print $4}')
    if [ "$need_kb" -gt "$free_kb" ]; then
      echo "$part needs $((need_kb / 1048576)) GB but $OUT has $((free_kb / 1048576)) GB free; stopping" >&2
      exit 2
    fi
    if ! recorded "$part"; then
      printf '%s\t%s\t%s\t%s\t%s\n' "$part" "$id" "$(wc -c < "$zip" | tr -d ' ')" "$(sha256 "$zip")" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/SOURCE.txt"
    fi
    if ! unzip -q -o "$zip" -d "$OUT"; then
      echo "Extracting $part failed on a valid zip; stopping (check free disk: $(df -h "$OUT" | tail -1))" >&2
      exit 2   # retrying will not help
    fi
    rm -f "$zip"
    touch "$OUT/.$part.done"
  done < <(parts)
  if [ "$waiting" -eq 0 ]; then
    touch "$OUT/.complete"
    echo "UTA-RLDD ready in $OUT"
    exit 0
  fi
  echo "$waiting parts not downloaded yet (Drive refusals or corrupt downloads)"
  if [ "$pass" -lt "$RETRY_PASSES" ]; then
    echo "Retrying in 1 h"
    sleep 3600
  fi
done
echo "Gave up after $RETRY_PASSES passes; rerun later." >&2
exit 1
: <<'PARTS'
PARTS_START
Fold1_part1 16x8afSOwfbdGAjUbfRFvwGj1ie8Lf5Do
Fold1_part2 1qPoJ06jeznytP_BGJ25hxdQy7eYTTjyk
Fold2_part1 12uVMVLv_kRXzQb4QDdru9RK3G9hzfpPQ
Fold2_part2 1nYGUpqE3x5ZeRLgDeMlUN4K_ccPjM2pe
Fold3_part1 1-JhE5TfkNyr4UbQk6PSkkpI7z-SChkWS
Fold3_part2 1LZU5KfJkFMj2pIkGwb3RjHDazoUfxSYQ
Fold4_part1 1IYKdrjvawf90bG_cfeoWA4rZSymllFbD
Fold4_part2 1zJlASBwABBgVSULQLiia0R_Eqar8_B3D
Fold5_part1 16w30wEgcAbaUr4gx9o4sl_2TbIunD5yA
Fold5_part2 1loOQE1-hoNM5dIOjXHI2Ss5NjjoPZcX6
PARTS_END
PARTS
