#!/usr/bin/env bash
# Upload Bulwark data as a private Kaggle Dataset (create, or new version if it exists).
#   KAGGLE_USER=<you> SLUG=bulwark-dmd SHARD_GLOB='dmd-*.tar' infra/kaggle/push_data.sh
#   KAGGLE_USER=<you> SLUG=bulwark-meta SHARD_GLOB= INCLUDE=checkpoints/heads-v0 infra/kaggle/push_data.sh
# A new version replaces the dataset's whole contents, and jobs attach the latest version,
# so stage everything the slug should hold, every time; SLUG has no default for that
# reason. One slug per source keeps this simple. SHARD_GLOB takes several globs for a slug
# that holds more than one source ('drive_and_act-*.tar dmd-*.tar'), and has no
# default either: set it to empty for a metadata-only upload.
# Only datasets cleared for a private copy on Kaggle go up: shards of a dataset without
# `kaggle_storage_ok=True` in bulwark/data/registry.py are refused unless ALLOW_UNCLEARED
# names it (a deliberate, per-dataset decision after reading its terms). Clearance for
# Azure Blob (`azure_storage_ok`) is separate and does not count here.
# The manifest goes only into META_SLUG (default bulwark-meta), which every job reads. Rows
# of uncleared datasets stop that upload unless ALLOW_UNCLEARED names those datasets or
# DROP_UNCLEARED=1 leaves their rows out. INCLUDE adds starting checkpoints from paths
# under checkpoints/: only their best.pt, final.pt and provenance.json, and exported *.onnx
# graphs (INCLUDE=checkpoints/bulwark-e1.onnx), never last.pt (a job writing to that
# directory would resume from it or refuse to start). The code
# travels inside each job (push_job.sh). Only finished <dataset>-NNNNN.tar shards are
# staged. Kaggle unpacks uploaded .tar files, so shards go up as .shard (same bytes) and
# kernel/run.py links them back as .tar.
# Needs the kaggle CLI (`uv pip install -e ".[ops]"`).
set -euo pipefail
: "${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
: "${SLUG:?set SLUG to the Kaggle dataset to create or replace (e.g. bulwark-dmd)}"
: "${SHARD_GLOB?set SHARD_GLOB to the shards to upload (one or more globs, e.g. 'mrl_eye-*.tar'), or to empty for none}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KAGGLE="${KAGGLE:-$(command -v kaggle || echo "$ROOT/.venv/bin/kaggle")}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
META_SLUG="${META_SLUG:-bulwark-meta}"
STAGE="${STAGE:-$ROOT/runs/kaggle-dataset-$SLUG}"
ARGS=(--root "$ROOT" --stage "$STAGE" --storage kaggle --shards "$SHARD_GLOB")
if [ "$SLUG" = "$META_SLUG" ]; then ARGS+=(--manifest); fi
if [ -n "${DROP_UNCLEARED:-}" ]; then ARGS+=(--drop-uncleared); fi
for path in ${INCLUDE:-}; do ARGS+=(--include "$path"); done
for name in ${ALLOW_UNCLEARED:-}; do ARGS+=(--allow "$name"); done
PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/infra/kaggle/stage_data.py" "${ARGS[@]}"
# Slugs with a dataset card (infra/kaggle/dataset_cards.py) get its title, subtitle,
# description, citation, terms, contact link, sources, update frequency, cover and a
# description of every staged file (manifest columns included); others get a plain
# title. The card and the cover go into the staging dir as dataset-metadata.json and
# dataset-cover-image.png, two names the kaggle CLI never uploads as files. A staged file
# the card cannot describe stops the upload here, before anything is sent.
# Finder leaves .DS_Store files in folders it opens; the CLI would upload them as data.
find "$STAGE" -name .DS_Store -type f -delete
CARD=$(mktemp "${TMPDIR:-/tmp}/bulwark-card.XXXXXX")
trap 'rm -f "$CARD"' EXIT
CARDS=(env PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/infra/kaggle/dataset_cards.py" --root "$ROOT")
if "${CARDS[@]}" --list-slugs | grep -qx "$SLUG"; then
  "${CARDS[@]}" --slug "$SLUG" --stage "$STAGE" --json "$CARD" --cover "$STAGE"
  "$PYTHON" -c 'import json, sys; c = json.load(open(sys.argv[1])); c["id"] = sys.argv[3]; json.dump(c, open(sys.argv[2], "w"), indent=2)' \
    "$CARD" "$STAGE/dataset-metadata.json" "$KAGGLE_USER/$SLUG"
  HAS_CARD=1
else
  cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "Bulwark data ($SLUG)", "id": "$KAGGLE_USER/$SLUG", "licenses": [{"name": "other"}]}
JSON
  HAS_CARD=
fi
# Private: the CLI makes a dataset public only when given --public, which is never passed.
if "$KAGGLE" datasets status "$KAGGLE_USER/$SLUG" >/dev/null 2>&1; then
  echo "New version of $KAGGLE_USER/$SLUG: it will hold only what was staged above." >&2
  # Kaggle scores a new version (its usability score) against the metadata in place when
  # the version is created, so set the card's sources, update frequency and cover first;
  # the update after the upload describes the new files.
  if [ -n "$HAS_CARD" ]; then
    KAGGLE="$KAGGLE" "${CARDS[@]}" --apply --slug "$SLUG"
  fi
  "$KAGGLE" datasets version -p "$STAGE" -m "update $(date -u +%F)" --dir-mode tar
else
  "$KAGGLE" datasets create -p "$STAGE" --dir-mode tar
fi
# Create and version take only the title, subtitle, description, keywords and the
# descriptions of top-level files from the card (the shards go up inside one folder
# archive, so theirs do not), and a new version keeps the old metadata otherwise. A
# metadata update sets the rest (sources, update frequency, cover, every file's
# description) and refreshes the counts; it describes the files Kaggle lists, once that
# list matches what was staged (it waits a few minutes for Kaggle to process the upload).
if [ -n "$HAS_CARD" ]; then
  KAGGLE="$KAGGLE" "${CARDS[@]}" --apply --slug "$SLUG" --stage "$STAGE" || {
    echo "Card not refreshed; rerun: KAGGLE_USER=$KAGGLE_USER infra/kaggle/dataset_cards.py --apply --slug $SLUG --stage $STAGE" >&2
    exit 1
  }
fi
