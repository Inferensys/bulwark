#!/usr/bin/env bash
# Fetch everything Bulwark needs that is not pip-installable:
#   - Meta's V-JEPA 2 code at a pinned commit; MIT (V-JEPA 2 code licence; see NOTICE).
#     The package vendors the encoder definition (bulwark/model/vjepa21); the clone is
#     only for the test that compares the two (tests/test_model_vendored_vjepa.py)
#   - the official V-JEPA 2.1 ViT-B/16 checkpoint (1.7 GB)
#   - MediaPipe Face Landmarker (Apache-2.0) for the face teacher
# Downloads are checked against pinned SHA-256 hashes before they are used, so an error
# page, a truncated file or a silently replaced upstream file fails here, not mid-job.
set -euo pipefail
cd "$(dirname "$0")/.."

VJEPA_COMMIT=204698b45b3712590f06245fbfba32d3be539812
# Hashes of the files Bulwark was developed against. Neither host publishes a SHA-256, so
# the copies were checked against what the hosts do publish (2026-10-28): the V-JEPA file
# matches the S3 multipart ETag that dl.fbaipublicfiles.com serves
# (be0dc26f052ae6a7476714cd53176836-199, MD5s of 8 MiB parts) and its Content-Length; the
# face model matches the MD5 that Google Cloud Storage publishes (x-goog-hash).
VJEPA_SHA256=848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d
FACE_SHA256=64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

if [ ! -d third_party/vjepa2 ]; then
  git clone -q https://github.com/facebookresearch/vjepa2.git third_party/vjepa2
fi
git -C third_party/vjepa2 checkout -q "$VJEPA_COMMIT"

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

# fetch <url> <dest> <sha256>: skip if dest already matches; otherwise download to
# dest.part (resumable), verify, then move into place.
fetch() {
  local url=$1 dest=$2 expected=$3
  if [ -f "$dest" ] && [ "$(sha256 "$dest")" = "$expected" ]; then
    echo "$dest: present, checksum OK"
    return
  fi
  # A .part that already matches was downloaded but never moved into place.
  if [ ! -f "$dest.part" ] || [ "$(sha256 "$dest.part")" != "$expected" ]; then
    curl -fL --retry 5 --retry-all-errors -C - -o "$dest.part" "$url"
  fi
  local actual
  actual=$(sha256 "$dest.part")
  if [ "$actual" != "$expected" ]; then
    rm -f "$dest.part"
    echo "$dest: SHA-256 mismatch (expected $expected, got $actual); not using it" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
  echo "$dest: downloaded, checksum OK"
}

mkdir -p checkpoints/pretrained
fetch https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
  checkpoints/pretrained/vjepa2_1_vitb_dist_vitG_384.pt "$VJEPA_SHA256"
fetch https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  checkpoints/pretrained/face_landmarker.task "$FACE_SHA256"
echo "Pretrained assets ready."
