#!/usr/bin/env bash
# Run a Bulwark job as a background Kaggle notebook on the T4 machine (two GPUs; cache and
# lora split each batch across both with DataParallel, while the distill configs set
# multi_gpu: false and evaluate uses one GPU), up to 12 h.
#   export KAGGLE_USER=<you>
#   infra/kaggle/push_job.sh cache configs/train/cache.yaml
#   infra/kaggle/push_job.sh lora configs/train/lora-kaggle.yaml --set 'datasets=["drive_and_act"]'
#   PREVIOUS_KERNELS=bulwark-lora \
#     infra/kaggle/push_job.sh evaluate checkpoints/b1-lora-v0/best.pt --model b1 --precision fp16
# DATASETS names the private Kaggle datasets to attach: the shard slugs you pushed with
# push_data.sh and the one that holds the manifest. Its default is the five v0 slugs
# (docs/training.md): bulwark-drive-and-act bulwark-dmd bulwark-mrl bulwark-auc
# bulwark-meta. Set it when you pushed other slugs; every slug it names must exist. The
# manifest is read only from MANIFEST_DATASET (default bulwark-meta), which must be in
# DATASETS. A job uses only the manifest clips whose shards are attached (the training
# jobs log how many have none), so leaving out bulwark-auc leaves out AUC Distracted
# Driver v2's clips, as in the first full run.
# Only the checkpoint and feature directories the job reads or writes are brought in
# (infra/job_inputs.py). A job longer than one session: run the next session as a new
# kernel that attaches the previous one's output (and any other kernels it reads, such as
# a feature cache), and it resumes from last.pt:
#   NAME=bulwark-lora-2 PREVIOUS_KERNELS="bulwark-lora" infra/kaggle/push_job.sh lora ...
# `evaluate` runs `bulwark evaluate --checkpoint <checkpoint> [options]` on the val, test
# and holdout shards; the checkpoint must be under checkpoints/ in an attached input (an
# exported graph goes up with push_data.sh INCLUDE=checkpoints/bulwark-e1.onnx), and
# the report lands in runs/eval unless --out says otherwise. Name the model kind with
# --model (b1, e1 or onnx read the shards; heads or logits read --features, default
# data/features/frozen, from an attached input). --reference is refused (compare the
# downloaded predictions locally with `bulwark report`), and so are --shards and
# --manifest other than the defaults.
# Outputs land in the kernel's /kaggle/working (download with `kaggle kernels output`).
# Needs the kaggle CLI (`uv pip install -e ".[ops]"`).
set -euo pipefail
: "${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
[ $# -ge 2 ] || {
  echo "usage: push_job.sh <job> <config> [--set key=value ...] | push_job.sh evaluate <checkpoint> [options]" >&2
  exit 2
}
JOB=$1; CONFIG=$2; shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
KAGGLE="${KAGGLE:-$(command -v kaggle || echo "$ROOT/.venv/bin/kaggle")}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
NAME="${NAME:-bulwark-$JOB}"
DATASETS="${DATASETS:-bulwark-drive-and-act bulwark-dmd bulwark-mrl bulwark-auc bulwark-meta}"
MANIFEST_DATASET="${MANIFEST_DATASET:-bulwark-meta}"
PREVIOUS_KERNELS="${PREVIOUS_KERNELS:-${PREVIOUS_KERNEL:-}}"
STAGE="${STAGE:-$ROOT/runs/kaggle-kernel-$NAME}"

# Fail here, not after Kaggle has started a GPU session: an unknown job, a config that is
# not in the repo, or a path outside it.
(cd "$ROOT" && "$PYTHON" infra/job_inputs.py "$JOB" "$CONFIG" ${1+"$@"} >/dev/null)
case " $DATASETS " in
  *" $MANIFEST_DATASET "*) ;;
  *) echo "DATASETS must include $MANIFEST_DATASET, the input that holds the manifest" >&2; exit 2 ;;
esac

rm -rf "$STAGE" && mkdir -p "$STAGE"
"$ROOT/scripts/pack_code.sh" "$STAGE/code.tgz"
# json.dumps always yields a valid Python literal, and the job arguments stay a list.
python3 - "$HERE/kernel/run.py" "$STAGE/run.py" "$STAGE/code.tgz" "$JOB" "$CONFIG" "$MANIFEST_DATASET" ${1+"$@"} <<'PY'
import base64, json, sys
src, dst, code, job, config, manifest_dataset, *extra = sys.argv[1:]
text = open(src).read()
for placeholder, value in (("__JOB__", job), ("__CONFIG__", config), ("__EXTRA__", extra), ("__MANIFEST_DATASET__", manifest_dataset)):
    text = text.replace(f'"{placeholder}"', json.dumps(value))
text = text.replace("__CODE_B64__", base64.b64encode(open(code, "rb").read()).decode())
open(dst, "w").write(text)
PY
rm "$STAGE/code.tgz"
# Kaggle answers a script much over 1 MiB with a bare "400 Bad Request" (a 1.03 MB script
# went through and a 1.20 MB one did not, on 2026-10-29), so stop here with the reason.
SIZE=$(wc -c < "$STAGE/run.py")
if [ "$SIZE" -gt 1048576 ]; then
  echo "push_job: $STAGE/run.py is $SIZE bytes, over Kaggle's 1 MiB script limit; the code" >&2
  echo "archive inside it (scripts/pack_code.sh) has grown: keep large files out of it" >&2
  exit 1
fi
python3 -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' "$STAGE/run.py"
SOURCES=$(for d in $DATASETS; do printf '"%s/%s",' "$KAGGLE_USER" "$d"; done)
KERNELS=$(for k in $PREVIOUS_KERNELS; do printf '"%s/%s",' "$KAGGLE_USER" "$k"; done)
cat > "$STAGE/kernel-metadata.json" <<JSON
{
  "id": "$KAGGLE_USER/$NAME",
  "title": "$NAME",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": true,
  "dataset_sources": [${SOURCES%,}],
  "kernel_sources": [${KERNELS%,}]
}
JSON
"$KAGGLE" kernels push -p "$STAGE"
echo "Status: kaggle kernels status $KAGGLE_USER/$NAME"
