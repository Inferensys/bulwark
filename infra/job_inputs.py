"""The checkpoints and feature directories a remote job reads and writes.

The Kaggle kernel (infra/kaggle/kernel/run.py) and the Azure VM (infra/azure/run_job.sh)
bring in only these, rather than every attached or uploaded directory, so an unrelated
or stale directory can neither be picked up by a job nor stop it, and a job's output
holds only what it wrote. Paths come from the training config with any `--set`
overrides (`--set key=value` or `--set=key=value`; an empty or null value keeps the
config's), as `bulwark train` will see them:

- cache reads `init` and writes `features`;
- heads reads `features` (and `init`, if set) and writes `out`;
- lora reads `init` and writes `out`;
- distill reads `teacher` and writes `out`;
- evaluate reads the checkpoint (which must be under checkpoints/) and, for a heads or
  logits model, `--features` (default data/features/frozen), and writes `--out` (default
  runs/eval).

Both remote sides bring reads in only from checkpoints/ and data/features/, besides the
manifest and the shards, so a job that reads anything else is refused here. So is a
training job that would write nowhere, and an evaluation that the remote side could not
supply: another `--shards` or `--manifest`, a `--reference` (compare locally with
`bulwark report` instead), or `--model auto` without `--features`, since only the
checkpoint itself says whether it needs cached features.

    python infra/job_inputs.py <job> <config> [--set key=value ...]
    python infra/job_inputs.py evaluate <checkpoint> --model <kind> [bulwark evaluate options]

prints one `read <path>` or `write <path>` line per path, or exits 2 with the reason
the job would fail. Paths are relative to the repository root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

TRAIN_JOBS = ("cache", "heads", "lora", "distill")
JOBS = (*TRAIN_JOBS, "evaluate")
# TrainConfig defaults (bulwark/train/common.py) for the keys read here.
DEFAULTS = {"features": "data/features", "out": "checkpoints/run", "init": "", "teacher": ""}
READS = {"cache": ("init",), "heads": ("features", "init"), "lora": ("init",), "distill": ("teacher",)}
WRITES = {"cache": ("features",), "heads": ("out",), "lora": ("out",), "distill": ("out",)}
# `bulwark evaluate` option defaults, and the model kinds it accepts.
EVAL_DEFAULTS = {"--features": "data/features/frozen", "--shards": "data/shards", "--manifest": "data/manifest.parquet",
                 "--out": "runs/eval"}  # fmt: skip
EVAL_OUT = EVAL_DEFAULTS["--out"]
MODEL_KINDS = ("auto", "heads", "logits", "b1", "e1", "onnx")
FEATURE_KINDS = ("heads", "logits")  # the kinds that read cached features rather than shards
# Where the Kaggle kernel and the Azure VM bring the reads in from.
REMOTE_READ_ROOTS = ("checkpoints", "data/features")


class JobError(ValueError):
    """The job could not run with these arguments."""


def option_values(extra: list[str], flag: str) -> list[str]:
    """Every value of `--flag value` and `--flag=value` in a command line, in order."""
    values: list[str] = []
    index = 0
    while index < len(extra):
        item = extra[index]
        if item == flag and index + 1 < len(extra):
            values.append(extra[index + 1])
            index += 2
            continue
        if item.startswith(flag + "="):
            values.append(item[len(flag) + 1 :])
        index += 1
    return values


def option_value(extra: list[str], flag: str, default: str = "") -> str:
    """The last value of `--flag value` or `--flag=value` in a command line."""
    values = option_values(extra, flag)
    return values[-1] if values else default


def config_value(config: Path, extra: list[str], key: str) -> str:
    """A config key as `bulwark train` will see it (bulwark.cli._train_config, load_config):
    the last `--set key=value` unless its value is empty or null, then the YAML value, then
    the TrainConfig default."""
    import yaml  # a dependency of the package

    overrides = {}
    for item in option_values(extra, "--set"):
        name, _, raw = item.partition("=")
        overrides[name] = yaml.safe_load(raw)
    value = overrides.get(key)
    if value is None:  # load_config ignores None overrides
        value = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get(key, DEFAULTS.get(key, ""))
    return str(value or "")


def entries_under(paths: list[str], root: str) -> set[str] | None:
    """Entries of `root` that `paths` touch: their first component below `root`, None (all
    of them) when a path is `root` itself, nothing for paths elsewhere."""
    entries: set[str] = set()
    for path in paths:
        relative = os.path.relpath(os.path.normpath(path), root)
        if relative == ".":
            return None
        if not relative.startswith(".."):
            entries.add(relative.split(os.sep)[0])
    return entries


def inside_repo(path: str) -> str:
    """`path` normalised; refuses absolute paths and paths that leave the repository."""
    normal = os.path.normpath(path)
    if os.path.isabs(normal) or normal == ".." or normal.startswith(".." + os.sep):
        raise JobError(f"{path} must be a path inside the repo")
    return normal


def evaluate_paths(checkpoint: str, extra: list[str]) -> tuple[list[str], list[str]]:
    """(reads, writes) of `bulwark evaluate --checkpoint <checkpoint> <extra>` on a remote job."""
    if entries_under([checkpoint], "checkpoints") in (None, set()):
        raise JobError(f"the checkpoint must be a file under checkpoints/: {checkpoint}")
    if option_values(extra, "--reference"):
        raise JobError(
            "--reference is not available on remote jobs; compare the downloaded predictions locally with "
            "`bulwark report --predictions <out>/predictions --reference <dir>`"
        )
    for flag in ("--shards", "--manifest"):
        value = option_value(extra, flag, EVAL_DEFAULTS[flag])
        if os.path.normpath(value) != EVAL_DEFAULTS[flag]:
            raise JobError(f"remote jobs bring in only {EVAL_DEFAULTS[flag]}; drop {flag} {value}")
    kind = option_value(extra, "--model", "auto")
    if kind not in MODEL_KINDS:
        raise JobError(f"--model must be one of {', '.join(MODEL_KINDS)}: {kind}")
    if kind == "auto" and checkpoint.endswith(".onnx"):
        kind = "onnx"
    features = option_value(extra, "--features")
    if kind == "auto" and not features:
        raise JobError(
            "--model auto reads the model kind from the checkpoint, which is not available here; pass "
            "--model b1, e1 or onnx (shards), or --model heads or logits (cached features, --features)"
        )
    if kind in FEATURE_KINDS:
        features = features or EVAL_DEFAULTS["--features"]
    elif kind != "auto":
        features = ""  # b1, e1 and onnx stream the shards and never read --features
    return [checkpoint, features], [option_value(extra, "--out", EVAL_OUT)]


def job_inputs(job: str, target: str, extra: list[str]) -> dict[str, list[str]]:
    """{"read": [...], "write": [...]}: the paths `job` reads and writes. `target` is the
    training config, or the checkpoint for evaluate; relative to the current directory."""
    if job not in JOBS:
        raise JobError(f"unknown job {job!r}; one of {', '.join(JOBS)}")
    if job == "evaluate":
        reads, writes = evaluate_paths(inside_repo(target), extra)
    else:
        config = Path(inside_repo(target))
        if not config.is_file():
            raise JobError(f"config not found in the repo: {target}")
        reads = [config_value(config, extra, key) for key in READS[job]]
        writes = [config_value(config, extra, key) for key in WRITES[job]]
        if not any(writes):
            keys = ", ".join(WRITES[job])
            raise JobError(f"{job} would write nowhere: {keys} is empty in {target} and its --set overrides")
    written = [inside_repo(path) for path in writes if path]
    read = [inside_repo(path) for path in reads if path]
    for path in read:
        if all(entries_under([path], root) == set() for root in REMOTE_READ_ROOTS):
            roots = " or ".join(f"{root}/" for root in REMOTE_READ_ROOTS)
            raise JobError(f"remote jobs bring in reads only from {roots}: {path}")
    return {"read": [path for path in dict.fromkeys(read) if path not in written], "write": list(dict.fromkeys(written))}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        paths = job_inputs(argv[0], argv[1], argv[2:])
    except JobError as exc:
        print(exc, file=sys.stderr)
        return 2
    for kind in ("read", "write"):
        for path in paths[kind]:
            print(kind, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
