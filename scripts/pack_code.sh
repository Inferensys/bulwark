#!/usr/bin/env bash
# Pack the repo's code (no data, weights, runs, venv or secrets) into a .tgz for Kaggle
# and Azure jobs:  scripts/pack_code.sh /tmp/bulwark-code.tgz
# Only files git tracks or would track are packed, so ignored files (ledgers, keys, local
# overrides, caches) never leave the machine. The archive carries PROVENANCE.json (commit,
# dirty flag, changed files, UTC time; bulwark/train/provenance.py records them with each
# run) so a remote run can be traced to the exact code it ran. Jobs only install and run
# the package, so tests, docs and the Kaggle dataset covers stay out: a Kaggle job carries
# the archive inside its script, and Kaggle refuses a script much over 1 MiB.
# Uncommitted changes to the packed paths are refused unless ALLOW_DIRTY=1; the changed
# files are then listed in PROVENANCE.json. A uv.lock that no longer matches
# pyproject.toml is refused too: jobs install with `uv sync --frozen` or `uv export
# --frozen`, which would fail remotely, after the GPU is paid for.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=$1
PATHS=(pyproject.toml uv.lock README.md bulwark configs infra scripts)
SKIP=(infra/kaggle/covers scripts/hf_release.py)  # the weights upload tool is never run by a job
for optional in LICENSE NOTICE CITATION.cff; do
  if [ -e "$ROOT/$optional" ]; then PATHS+=("$optional"); fi
done

COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null) || {
  echo "pack_code: $ROOT is not a git checkout; the archive could not be traced to a commit" >&2
  exit 1
}
DIRTY=$(git -C "$ROOT" status --porcelain --untracked-files=all -- "${PATHS[@]}")
if [ -n "$DIRTY" ] && [ "${ALLOW_DIRTY:-}" != 1 ]; then
  echo "pack_code: uncommitted changes would be packed; commit them or set ALLOW_DIRTY=1:" >&2
  echo "$DIRTY" >&2
  exit 1
fi

UV="${UV:-$(command -v uv || true)}"
[ -n "$UV" ] || { echo "pack_code: uv is needed to check uv.lock (https://docs.astral.sh/uv/)" >&2; exit 1; }
if ! LOCK_CHECK=$(cd "$ROOT" && "$UV" lock --check --offline 2>&1); then
  echo "$LOCK_CHECK" >&2
  echo "pack_code: uv.lock does not match pyproject.toml; run \`uv lock\` (uv 0.8.17) and commit it" >&2
  exit 1
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/bulwark-pack.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
# Tracked plus untracked-but-not-ignored files that exist on disk (a deleted file stays in
# the index until it is committed).
git -C "$ROOT" ls-files -z --cached --others --exclude-standard -- "${PATHS[@]}" \
  "${SKIP[@]/#/:(exclude)}" |
  while IFS= read -r -d '' path; do
    if [ -f "$ROOT/$path" ] && [ "$(basename "$path")" != .DS_Store ]; then printf '%s\0' "$path"; fi
  done > "$TMP/files"
COMMIT="$COMMIT" DIRTY="$DIRTY" python3 - "$TMP/PROVENANCE.json" <<'PY'
import json, os, sys
from datetime import datetime, timezone

dirty = [line[3:] for line in os.environ["DIRTY"].splitlines() if line.strip()]
record = {
    "commit": os.environ["COMMIT"],
    "dirty": bool(dirty),
    "changed": dirty,
    "packed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with open(sys.argv[1], "w") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY
COPYFILE_DISABLE=1 tar -cf "$TMP/code.tar" -C "$ROOT" --null -T "$TMP/files"
COPYFILE_DISABLE=1 tar -rf "$TMP/code.tar" -C "$TMP" PROVENANCE.json
gzip -c "$TMP/code.tar" > "$OUT"

tar -tzf "$OUT" > "$TMP/listing"
for required in bulwark/data/manifest.py uv.lock PROVENANCE.json; do
  grep -qx "$required" "$TMP/listing" || {
    echo "pack_code: $required is missing from $OUT" >&2
    exit 1
  }
done
if [ -n "$DIRTY" ]; then
  echo "pack_code: packed uncommitted changes (ALLOW_DIRTY=1); see PROVENANCE.json" >&2
fi
