#!/usr/bin/env bash
# Carry a local purge over to the copies outside this machine, and log it.
#   scripts/purge_cloud_copies.sh <dataset>                    # Azure Blob, then Kaggle steps
#   KAGGLE_SLUGS="bulwark-dmd bulwark-meta" scripts/purge_cloud_copies.sh <dataset>
#   DRY_RUN=1 scripts/purge_cloud_copies.sh <dataset>          # show what would happen
#   scripts/purge_cloud_copies.sh --reshard <dataset>          # after RESHARD in prepare_local.sh
# Run it after `bulwark purge` (bulwark.data.removals.purge_subjects) has rewritten the
# local manifests, shards, labels and feature caches. The copies below are replaced by the
# local data or deleted, so it refuses when data/manifest.parquet is missing, while a local
# manifest still holds a subject that bulwark/data/removed_subjects.yaml lists for
# <dataset>, and when that list names no subject of <dataset> (a mistyped name, or a
# removal not yet recorded); <dataset> must be a registry dataset. --reshard skips the
# removal-list check and writes no log row: it replaces the copies of a dataset whose
# shards were rebuilt, with no subject removed.
#
# A dataset counts as cleared for each storage provider as infra/azure/upload.sh and
# infra/kaggle/push_data.sh decide it, one provider at a time: for Azure Blob its registry
# entry has azure_storage_ok=True, for Kaggle kaggle_storage_ok=True
# (bulwark/data/registry.py), or ALLOW_UNCLEARED names it (for both).
#
# Azure Blob (skipped with AZURE=0): when Blob holds the shards of a dataset cleared for
# Blob, the local ones are uploaded over them first, and only then are the shard blobs that
# no local shard replaces deleted, so an interrupted run never leaves Blob without the
# dataset (it refuses when data/shards has none of them). Shards of a dataset that is not
# cleared for Blob are never uploaded: every shard blob of it is deleted. Its feature
# blobs (data/features/*/features-<dataset>-*.npz) are deleted for the cache job to
# recompute, the manifest is replaced by the local one, and the teacher-label blobs under
# data/labels/ are deleted (labels reach jobs through the manifest, and upload.sh refuses
# data/labels). The manifest goes through the same clearance check as upload.sh: rows of
# datasets that are not cleared for Blob stop the run before anything changes, unless
# ALLOW_UNCLEARED names them or DROP_UNCLEARED=1 leaves them out. When no dataset in the
# local manifest is cleared for Blob, DROP_UNCLEARED=1 deletes the manifest blob instead of
# replacing it. Deleting needs your Entra ID login (the job SAS has no delete right). With
# blob soft delete or versioning on, deleted data stays recoverable until its retention
# ends; the script warns when either is enabled.
#
# Kaggle keeps every version of a dataset, so a new version does not remove the old
# frames: each dataset holding <dataset> shards, and the one holding the manifest, must be
# deleted and created again. KAGGLE_SLUGS names the ones to delete (needs KAGGLE_USER);
# recreate them with infra/kaggle/push_data.sh, which leaves out the shards of a dataset
# that is not cleared for Kaggle. Kernel outputs keep copies too (a cache job's output
# holds features): delete those kernels with `kaggle kernels delete`. After a reshard a new
# version of each is enough, except for a dataset that is not cleared for Kaggle: delete
# every Kaggle dataset that holds its shards and create it again without them.
#
# Each purge appends a row to docs/purge-log.md (PURGE_LOG overrides it): the date, the
# dataset, how many subjects the removal list names for it, and what was done. Subject
# ids and other personal data stay out of the log.
set -euo pipefail
MODE=purge
if [ "${1:-}" = --reshard ]; then MODE=reshard; shift; fi
[ $# -eq 1 ] || { echo "usage: purge_cloud_copies.sh [--reshard] <dataset>" >&2; exit 2; }
DATASET=$1
case "$DATASET" in ''|*[!a-z0-9_]*) echo "not a dataset name: $DATASET" >&2; exit 2 ;; esac
if [ "$MODE" = reshard ] && [ -n "${KAGGLE_SLUGS:-}" ]; then
  echo "KAGGLE_SLUGS deletes datasets for a purge only; --reshard prints the Kaggle steps instead." >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
KAGGLE="${KAGGLE:-$(command -v kaggle || echo "$ROOT/.venv/bin/kaggle")}"
LOG="${PURGE_LOG:-docs/purge-log.md}"
DRY_RUN="${DRY_RUN:-}"
run() { if [ -n "$DRY_RUN" ]; then echo "would run: $*" >&2; else "$@"; fi; }

# The local purge must come first: every copy below is replaced by the local one.
SUBJECTS=$("$PYTHON" - "$DATASET" "$MODE" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

from bulwark.data.removals import load_removed_subjects

dataset, mode = sys.argv[1:]
removed = {subject for subject in load_removed_subjects() if subject.split(":", 1)[0] == dataset}
if not removed and mode == "purge":
    sys.exit(f"bulwark/data/removed_subjects.yaml lists no subject of {dataset}: check the dataset name, "
             "or record the removal and purge locally first. To replace resharded copies, use --reshard.")
manifest = Path("data/manifest.parquet")
if not manifest.exists():
    sys.exit(f"{manifest} is missing; run this where the purged local data is, since it replaces the copies.")
for path in (manifest, Path(f"data/manifests/{dataset}.parquet")):
    if path.exists() and removed & set(pq.read_table(path, columns=["subject_id"]).column("subject_id").to_pylist()):
        sys.exit(f"{path} still holds subjects removed from {dataset}; purge locally first (bulwark purge).")
print(len(removed))
PY
)
# "<azure> <kaggle> <manifest>": yes or no for each: whether $DATASET may be stored on Azure
# Blob and on Kaggle (see above), and whether any dataset in the local manifest may be
# stored on Azure Blob (if none may, the manifest blob can only be deleted).
CLEARED=$(PYTHONPATH="$ROOT" "$PYTHON" - "$ROOT/infra/kaggle" "$DATASET" ${ALLOW_UNCLEARED:-} <<'PY'
import sys

import pyarrow.parquet as pq

sys.path.insert(0, sys.argv[1])
from stage_data import clearance

dataset, allowed = sys.argv[2], set(sys.argv[3:])
storages = ("azure", "kaggle")
registry = {storage: clearance(storage) for storage in storages}
if dataset not in registry["azure"]:
    sys.exit(f"{dataset} is not a dataset in bulwark/data/registry.py.")
in_manifest = set(pq.read_table("data/manifest.parquet", columns=["dataset"]).column("dataset").to_pylist())
answers = [registry[s][dataset] is True or dataset in allowed for s in storages]
answers.append(any(registry["azure"].get(name) is True or name in allowed for name in in_manifest))
print(" ".join("yes" if ok else "no" for ok in answers))
PY
)
read -r AZURE_CLEARED KAGGLE_CLEARED MANIFEST_CLEARED <<<"$CLEARED"
AZURE_DONE="skipped (AZURE=0)"
if [ "${AZURE:-1}" != 0 ]; then
  source "$ROOT/infra/azure/env.sh"
  BLOBS=(--account-name "$BULWARK_STORAGE" --auth-mode login)
  remote() {  # remote <prefix>: names of the blobs under it
    az storage blob list "${BLOBS[@]}" -c "$BULWARK_CONTAINER" --prefix "$1" --num-results "*" --query "[].name" -o tsv
  }
  retention=$(az storage account blob-service-properties show -n "$BULWARK_STORAGE" -g "$BULWARK_RG" \
    --query "[deleteRetentionPolicy.enabled, isVersioningEnabled]" -o tsv | tr '\n' ' ')
  case "$retention" in
    *[Tt]rue*) echo "Warning: blob soft delete or versioning is on for $BULWARK_STORAGE; deleted blobs stay recoverable until their retention ends, so delete those versions too." >&2 ;;
  esac
  SHARD_NAME="^data/shards/$DATASET(-r[0-9]{3})?-[0-9]{5}\.tar$"   # as in check_shards.py
  listed=$(remote "data/shards/$DATASET-")   # a listing that fails stops the script here
  remote_shards=()
  while IFS= read -r name; do
    if [[ $name =~ $SHARD_NAME ]]; then remote_shards+=("$name"); fi
  done <<<"$listed"
  local_shards=()
  for shard in data/shards/"$DATASET"-*.tar; do
    if [[ $shard =~ $SHARD_NAME ]] && [ -f "$shard" ]; then local_shards+=("$shard"); fi
  done
  listed=$(remote "data/features/")
  features=$(grep -cE "/features-$DATASET(-r[0-9]{3})?-[0-9]{5}\.npz$" <<<"$listed" || true)
  listed=$(remote "data/labels/")
  labels=()
  while IFS= read -r name; do
    if [ -n "$name" ]; then labels+=("$name"); fi
  done <<<"$listed"
  listed=$(remote "data/manifest.parquet")
  manifest=""; MANIFEST="not in Blob"
  if [ -n "$listed" ]; then manifest=data/manifest.parquet; MANIFEST="replaced"; fi
  # Put back only what Blob held: the local shards if it had the dataset and the dataset
  # is cleared for Blob, and the manifest if it had one.
  uploads=(); stale=()
  if [ "$AZURE_CLEARED" = no ]; then
    stale=(${remote_shards[@]+"${remote_shards[@]}"})
  elif [ ${#remote_shards[@]} -gt 0 ]; then
    if [ ${#local_shards[@]} -eq 0 ]; then
      echo "Blob holds ${#remote_shards[@]} shards of $DATASET, but data/shards has none to replace them; nothing changed." >&2
      exit 1
    fi
    uploads=("${local_shards[@]}")
    for name in "${remote_shards[@]}"; do
      case " ${local_shards[*]} " in *" $name "*) ;; *) stale+=("$name") ;; esac
    done
  fi
  # Blob holds a manifest, but no dataset in the local one may be kept there: an upload
  # would leave it empty, so with DROP_UNCLEARED=1 the manifest blob is deleted instead.
  old_manifest=()
  if [ -n "$manifest" ] && [ "$MANIFEST_CLEARED" = no ]; then
    if [ -z "${DROP_UNCLEARED:-}" ]; then
      echo "Refused: Blob holds data/manifest.parquet, but no dataset in the local manifest is cleared for storage on Azure Blob (registry azure_storage_ok). Set DROP_UNCLEARED=1 to delete the manifest blob, or name the datasets it may hold in ALLOW_UNCLEARED. Nothing changed." >&2
      exit 1
    fi
    old_manifest=("$manifest"); manifest=""
    MANIFEST="deleted (no dataset in it is cleared for storage on Azure Blob)"
  fi
  STAGE=$(mktemp -d "${TMPDIR:-/tmp}/bulwark-purge.XXXXXX")
  trap 'rm -rf "$STAGE"' EXIT
  if [ -n "$manifest" ]; then  # checked before anything changes; refuses uncleared rows
    ARGS=(--root . --stage "$STAGE" --storage azure --manifest)
    if [ -n "${DROP_UNCLEARED:-}" ]; then ARGS+=(--drop-uncleared); fi
    for name in ${ALLOW_UNCLEARED:-}; do ARGS+=(--allow "$name"); done
    PYTHONPATH="$ROOT" "$PYTHON" infra/kaggle/stage_data.py "${ARGS[@]}" >&2
  fi
  # Upload first, then delete: a run that stops halfway leaves the dataset in Blob.
  if [ ${#uploads[@]} -gt 0 ] || [ -n "$manifest" ]; then
    if [ -n "$DRY_RUN" ]; then SAS="<sas>"; else SAS=$("$ROOT/infra/azure/sas.sh" 2); fi
    BASE="https://$BULWARK_STORAGE.blob.core.windows.net/$BULWARK_CONTAINER"
    for path in ${uploads[@]+"${uploads[@]}"}; do
      run azcopy copy "$path" "$BASE/$path?$SAS" --overwrite=true --log-level=ERROR >/dev/null
    done
    if [ -n "$manifest" ]; then
      run azcopy copy "$STAGE/manifest.parquet" "$BASE/$manifest?$SAS" --overwrite=true --log-level=ERROR >/dev/null
    fi
  fi
  for name in ${stale[@]+"${stale[@]}"} ${labels[@]+"${labels[@]}"} ${old_manifest[@]+"${old_manifest[@]}"}; do
    run az storage blob delete "${BLOBS[@]}" -c "$BULWARK_CONTAINER" -n "$name" -o none
  done
  run az storage blob delete-batch "${BLOBS[@]}" -s "$BULWARK_CONTAINER" --pattern "data/features/*/features-$DATASET-*.npz" -o none
  if [ "$AZURE_CLEARED" = yes ]; then
    SHARDS="${#uploads[@]} replaced by local shards, ${#stale[@]} without a local shard deleted"
  else
    SHARDS="${#stale[@]} deleted, none uploaded (not cleared for storage on Azure Blob)"
  fi
  AZURE_DONE="shard blobs: $SHARDS; feature blobs: $features deleted; manifest: $MANIFEST; label blobs: ${#labels[@]} deleted"
  echo "Azure: $AZURE_DONE"
fi

if [ "$KAGGLE_CLEARED" = yes ]; then
  GLOBS="<every glob of the slug, including $DATASET-*.tar>"
else
  GLOBS="<every glob of the slug except $DATASET-*.tar>"
  cat <<MSG
Kaggle: $DATASET is not cleared for storage on Kaggle (registry kaggle_storage_ok),
so it stays off Kaggle: push_data.sh refuses its shards unless ALLOW_UNCLEARED names it.
MSG
fi
if [ "$MODE" = reshard ]; then
  if [ "$KAGGLE_CLEARED" = yes ]; then
    cat <<MSG
Kaggle: push a new version of every dataset that holds $DATASET shards. A new version
replaces a dataset's whole contents, so SHARD_GLOB must list every source the slug holds:
  SLUG=<slug> SHARD_GLOB='$GLOBS' infra/kaggle/push_data.sh
MSG
  else
    cat <<MSG
Kaggle keeps old versions, so delete every dataset that holds $DATASET shards and create
it again without them:
  kaggle datasets delete -y \$KAGGLE_USER/<slug>
  SLUG=<slug> SHARD_GLOB='$GLOBS' infra/kaggle/push_data.sh
MSG
  fi
  if [ -n "$DRY_RUN" ]; then echo "Dry run: nothing deleted or uploaded."; else echo "Reshard: nothing logged."; fi
  exit 0
fi
KAGGLE_DONE="no dataset deleted by this script"
if [ -n "${KAGGLE_SLUGS:-}" ]; then
  : "${KAGGLE_USER:?set KAGGLE_USER to delete the datasets in KAGGLE_SLUGS}"
  for slug in $KAGGLE_SLUGS; do
    run "$KAGGLE" datasets delete -y "$KAGGLE_USER/$slug"
  done
  KAGGLE_DONE="deleted $KAGGLE_SLUGS, to be recreated with push_data.sh"
fi
cat <<MSG
Kaggle: old dataset versions keep the frames. Delete every dataset that holds $DATASET shards
and the one that holds the manifest, then create them again from the purged local data.
A dataset's contents are what one push stages, so SHARD_GLOB must list every source the
slug should hold:
  kaggle datasets delete -y \$KAGGLE_USER/<slug>
  SLUG=<slug> SHARD_GLOB='$GLOBS' infra/kaggle/push_data.sh
  SLUG=bulwark-meta SHARD_GLOB= INCLUDE='<checkpoint dirs it held>' infra/kaggle/push_data.sh
Kernels whose output holds features of $DATASET (cache jobs) keep them until deleted:
  kaggle kernels delete -y \$KAGGLE_USER/<kernel>
MSG

if [ -n "$DRY_RUN" ]; then
  echo "Dry run: nothing deleted, uploaded or logged."
  exit 0
fi
if [ ! -s "$LOG" ]; then
  mkdir -p "$(dirname "$LOG")"
  cat > "$LOG" <<'HEAD'
# Purge log

One row per purge of a dataset's copies outside the training machine, written by
`scripts/purge_cloud_copies.sh` after the local purge (`bulwark purge`).
Removed subjects are listed by their dataset subject ids (no names) in
`bulwark/data/removed_subjects.yaml`; this log holds no subject ids or other personal data.

| Date (UTC) | Dataset | Subjects on the removal list | Azure Blob | Kaggle |
|---|---|---|---|---|
HEAD
fi
printf '| %s | %s | %s | %s | %s |\n' "$(date -u +%F)" "$DATASET" "$SUBJECTS" "$AZURE_DONE" "$KAGGLE_DONE" >> "$LOG"
echo "Logged in $LOG"
