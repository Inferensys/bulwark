#!/usr/bin/env bash
# Sync what Azure jobs read up to Blob, under the same paths (run_job.sh fetches them from
# there): the manifest, the shards, feature caches and checkpoints. Run it from the repo
# root:
#   infra/azure/upload.sh data/shards data/manifest.parquet checkpoints/heads-v0
# The container is private and in your own subscription (setup.sh), but it is still storage
# on a third-party cloud, which the registry flag `azure_storage_ok` covers (checked by
# infra/kaggle/stage_data.py --storage azure). It is decided separately from Kaggle: a
# dataset cleared for Kaggle (`kaggle_storage_ok`) is not cleared for Blob. Shards and
# feature files of a dataset whose `azure_storage_ok` is not True in
# bulwark/data/registry.py stay local unless ALLOW_UNCLEARED names the dataset. A manifest
# with rows of such a dataset is refused unless ALLOW_UNCLEARED names it or
# DROP_UNCLEARED=1 uploads the manifest without those rows (jobs then neither train nor
# score on them).
# Unfinished files never go up: a `*.tar.partial` left by a killed `bulwark shard` run (it
# can hold real frames), or a `*.tmp` or `*.tmp.npz` still being written; nor do the
# <dataset>.shards.json records, which no job reads. Paths outside data/manifest.parquet,
# data/shards, data/features and checkpoints/<dir> are refused (data/labels too: labels
# reach jobs through the manifest). Uploads never delete blobs (the SAS carries no delete
# right).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/env.sh"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[ $# -ge 1 ] || { echo "usage: upload.sh <path> ...  (data/manifest.parquet, data/shards, data/features/..., checkpoints/...)" >&2; exit 2; }
TMP=$(mktemp -d "${TMPDIR:-/tmp}/bulwark-upload.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# Datasets that stay local: not cleared for Azure Blob in the registry and not named in
# ALLOW_UNCLEARED.
UNCLEARED=$(PYTHONPATH="$ROOT" "$PYTHON" -c '
import sys

sys.path.insert(0, sys.argv[1])
from stage_data import clearance

allowed = set(sys.argv[2:])
print(" ".join(sorted(name for name, ok in clearance("azure").items() if ok is not True and name not in allowed)))
' "$ROOT/infra/kaggle" ${ALLOW_UNCLEARED:-})
EXCLUDE="*.partial;*.tmp;*.tmp.npz;*.shards.json"
for name in $UNCLEARED; do EXCLUDE+=";$name-*.tar;features-$name-*.npz"; done

uncleared_file() {  # the uncleared dataset whose shard or feature file this is
  local base name
  base=$(basename "$1")
  for name in $UNCLEARED; do
    case "$base" in "$name"-*.tar|features-"$name"-*.npz) echo "$name"; return ;; esac
  done
}
left_out() {  # the uncleared datasets with files in a directory
  local name found=()
  for name in $UNCLEARED; do
    if [ -n "$(find "$1" \( -name "$name-*.tar" -o -name "features-$name-*.npz" \) -print -quit)" ]; then
      found+=("$name")
    fi
  done
  echo "${found[*]-}"
}

# Check every path before anything is uploaded.
PATHS=()
for path in "$@"; do
  path=${path%/}
  case "$path" in
    *..*) ;;
    data/manifest.parquet|data/shards|data/shards/*|data/features|data/features/*|checkpoints/?*)
      [ -e "$path" ] || { echo "$path: not found (paths are relative to the repo root)" >&2; exit 2; }
      PATHS+=("$path"); continue ;;
  esac
  echo "$path: not uploaded; Azure jobs read only data/manifest.parquet, data/shards, data/features and checkpoints/<dir>" >&2
  exit 2
done
for path in "${PATHS[@]}"; do
  if [ -f "$path" ]; then
    case "$path" in *.partial|*.tmp|*.tmp.npz|*.shards.json) echo "$path: unfinished, or not read by any job; not uploaded" >&2; exit 1 ;; esac
    name=$(uncleared_file "$path")
    if [ -n "$name" ]; then
      echo "Refused: $path belongs to $name, which is not cleared for storage on Azure Blob (registry azure_storage_ok). Check its terms; to upload anyway, name it in ALLOW_UNCLEARED." >&2
      exit 1
    fi
  fi
  if [ "$path" = data/manifest.parquet ]; then
    ARGS=(--root . --stage "$TMP/manifest" --storage azure --manifest)
    if [ -n "${DROP_UNCLEARED:-}" ]; then ARGS+=(--drop-uncleared); fi
    for name in ${ALLOW_UNCLEARED:-}; do ARGS+=(--allow "$name"); done
    PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/infra/kaggle/stage_data.py" "${ARGS[@]}"
  fi
done

SAS=$("$HERE/sas.sh")
BASE="https://$BULWARK_STORAGE.blob.core.windows.net/$BULWARK_CONTAINER"
for path in "${PATHS[@]}"; do
  if [ "$path" = data/manifest.parquet ]; then
    azcopy copy "$TMP/manifest/manifest.parquet" "$BASE/$path?$SAS" --overwrite=true
  elif [ -d "$path" ]; then
    skipped=$(left_out "$path")
    if [ -n "$skipped" ]; then
      echo "$path: leaving out the files of $skipped (not cleared for storage on Azure Blob; ALLOW_UNCLEARED names exceptions)" >&2
    fi
    azcopy sync "$path" "$BASE/$path?$SAS" --recursive --delete-destination=false --exclude-pattern "$EXCLUDE"
  else
    azcopy copy "$path" "$BASE/$path?$SAS"
  fi
done
