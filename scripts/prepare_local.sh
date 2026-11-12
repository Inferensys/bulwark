#!/usr/bin/env bash
# Prepare the training data on a local machine: unpack, ingest every dataset that has a
# complete local root, split, shard, run the face teacher and merge teacher labels into the
# manifest.
#   scripts/prepare_local.sh
#   SKIP_LABELS=1 scripts/prepare_local.sh   # skip running the face teacher; still merges
#                                            # the labels already in data/labels
#   RESHARD="dmd" scripts/prepare_local.sh
# Every run re-ingests each dataset and re-splits, so manifests follow adapter changes. A
# dataset the split policy excludes (zenodo_driver in v0, SplitPolicy.excluded_datasets) is
# still ingested and sharded when its root is present, in split `excluded`: it is never
# trained on or scored, and as its registry entry is cleared for neither Kaggle nor Azure
# Blob, the upload scripts keep its shards local (unless ALLOW_UNCLEARED names it).
#
# Adding a dataset to data that is already prepared: rerunning this script does it
# (`bulwark shard` adds only clips not yet sharded, and the face teacher skips clips it has
# labelled), or run the same steps for the new dataset alone. The dataset needs an adapter
# and a registry entry (CONTRIBUTING.md, "Adding a dataset") and its root in
# configs/data.yaml. Then:
#   bulwark ingest <dataset> --dry-run              # check the adapter against the files
#   bulwark ingest <dataset>                        # writes data/manifests/<dataset>.parquet
#   bulwark split                                   # merges every manifest, assigns splits
#   bulwark shard --dataset <dataset>               # shards only that dataset's clips
#   bulwark label face --pattern '<dataset>-*.tar'  # face teacher on its shards only
#   bulwark label merge                             # split dropped the teacher labels
#   scripts/check_shards.py
# The upload scripts take its shards and manifest rows only if the registry clears it for
# their provider, or ALLOW_UNCLEARED names it (infra/kaggle/push_data.sh,
# infra/azure/upload.sh).
#
# `bulwark shard` adds the manifest clips that are not in a shard yet to new
# <dataset>-NNNNN.tar files and never rewrites a shard. It refuses to run when the manifest
# now cuts already-sharded clips differently (media path, start frame, frame count, stride
# or crop; the fingerprint in data/shards/<dataset>.shards.json). Frames that changed
# under the same cut (an adapter fix such as applying EXIF rotation) are not detected:
# they keep their old pixels until RESHARD names their dataset. Its shards then move to
# data/shards-replaced/<UTC time>/ (delete them once the new ones are checked), its clips'
# teacher labels are dropped so the teachers label the new frames (this script runs the
# face teacher unless SKIP_LABELS is set; rerun any other by hand), and the shard step
# writes it again from scratch. Feature caches recompute those shards on their next run.
# The cloud copies still hold the old frames, and upload.sh never deletes a blob, so
# replace them with `scripts/purge_cloud_copies.sh --reshard <dataset>` (Azure Blob) and a
# new version of each Kaggle dataset that holds the dataset's shards (push_data.sh). For a
# dataset that is not cleared for Azure Blob, the purge script deletes its shard blobs
# instead; for one not cleared for Kaggle, it prints how to delete and recreate the Kaggle
# datasets that hold its shards.
# The last step, scripts/check_shards.py, compares the shards with the manifest and exits 1,
# naming the datasets to fix, on stale clips (sharded but no longer in the manifest),
# re-cut clips, shards without an index, or more missing clips than its --max-missing
# allows. Clips `bulwark shard` could not decode are listed in <dataset>.shards.json and do
# not count against --max-missing. Those records stay local: no job reads them.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=.venv/bin/python
BULWARK=.venv/bin/bulwark

if [ -n "${RESHARD:-}" ]; then
  $PYTHON - "$(date -u +%Y%m%dT%H%M%SZ)" $RESHARD <<'PY'
import json
import os
import re
import sys
import tarfile
from pathlib import Path

from bulwark.data.removals import purge_clips

stamp, *datasets = sys.argv[1:]
for dataset in datasets:
    name = re.compile(rf"^{re.escape(dataset)}(?:-r\d{{3}})?-\d{{5}}\.tar$")  # as in check_shards.py
    shards = sorted(p for p in Path("data/shards").glob(f"{dataset}-*.tar") if name.match(p.name))
    if not shards:
        print(f"RESHARD: no shards of {dataset} in data/shards", file=sys.stderr)
        continue
    clip_ids: set[str] = set()
    for shard in shards:
        with tarfile.open(shard) as tar:
            try:
                clip_ids.update(json.loads(tar.extractfile("index.json").read()).values())
            except KeyError:
                pass  # an unclosed shard has no index.json; check_shards.py reports it
    labels = purge_clips(clip_ids, labels=Path("data/labels")).get("label_rows", 0)
    aside = Path("data/shards-replaced") / stamp
    aside.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        os.replace(shard, aside / shard.name)
    print(f"RESHARD {dataset}: moved {len(shards)} shards to {aside}, dropped {labels} teacher label rows")
PY
fi

# Drive&Act ships one zip per camera view.
if [ -d data/raw/drive_and_act/_zips ]; then
  for z in data/raw/drive_and_act/_zips/*.zip; do
    marker="data/raw/drive_and_act/.unzipped-$(basename "$z" .zip)"
    [ -f "$marker" ] || { unzip -q -o "$z" -d data/raw/drive_and_act && touch "$marker"; }
  done
fi

# Datasets fetched by a script are ingested only once that script has marked the download
# complete, so a half-downloaded set never reaches the manifest.
for dataset in $($PYTHON - <<'PY'
import pathlib
import sys

import yaml

from bulwark.data.sources import ADAPTERS

FETCHED = {"uta_rldd": "scripts/fetch_uta_rldd.sh"}
roots = yaml.safe_load(open("configs/data.yaml"))["roots"]
ready = []
for name, path in roots.items():
    root = pathlib.Path(path)
    if name not in ADAPTERS or not root.is_dir() or not any(root.iterdir()):
        continue
    if name in FETCHED and not (root / ".complete").exists():
        print(f"{name}: download not complete ({FETCHED[name]} writes {root}/.complete); skipped", file=sys.stderr)
        continue
    ready.append(name)
print(" ".join(ready))
PY
); do
  $BULWARK ingest "$dataset"
done
$BULWARK split
$BULWARK shard
if [ -z "${SKIP_LABELS:-}" ]; then
  $BULWARK label face
elif [ -n "${RESHARD:-}" ]; then
  echo "Warning: RESHARD dropped the face labels of $RESHARD and SKIP_LABELS skipped the face teacher;" \
    "the merged manifest has no face labels for those clips until you run" \
    "\`bulwark label face --pattern '<dataset>-*.tar'\` and \`bulwark label merge\`." >&2
fi
# Always: split rebuilt the manifest without teacher labels.
$BULWARK label merge

if ! $PYTHON scripts/check_shards.py; then
  echo "Local data prepared, but some shards are out of date (see above)." >&2
  exit 1
fi
echo "Local data ready: data/manifest.parquet, data/shards/, data/labels/"
