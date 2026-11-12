"""Stage data for a third party's storage, keeping only datasets cleared for that storage.

push_data.sh runs it with `--storage kaggle` (a private Kaggle dataset, the default);
infra/azure/upload.sh and scripts/purge_cloud_copies.sh run it with `--storage azure` for
Blob storage. Each provider is cleared separately: a source dataset is cleared when its
registry entry has that provider's flag set to True (`kaggle_storage_ok` or
`azure_storage_ok`, see `bulwark.data.registry.STORAGE_FIELDS`: its terms allow a private
copy there and the maintainers have approved it), or when ALLOW_UNCLEARED names it, which
is the operator's explicit decision. Shards of any other dataset are refused.

Only finished shards (`<dataset>-NNNNN.tar`) are staged: never a `.tar.partial` left by a
killed `bulwark shard` run, nor the `<dataset>.shards.json` records, which jobs do not
read. The manifest, when staged, must not silently lose rows: rows of uncleared datasets
stop the upload unless --drop-uncleared says to leave them out, and an empty manifest is
never staged. Extra paths (--include) may only come from checkpoints/, since anything
under data/ is derived from the datasets and would bypass these checks. Of a checkpoint
directory, only the finished files a later job starts from are staged (best.pt, final.pt,
provenance.json, and exported `*.onnx` graphs for `bulwark evaluate --model onnx`): never
last.pt, which would make a job writing to that directory resume or refuse to start, and
never the stale-<time>/ folders of earlier runs.

    python infra/kaggle/stage_data.py --stage DIR [--storage kaggle|azure]
        [--shards 'GLOB ...'] [--manifest] [--drop-uncleared] [--include PATH ...]
        [--allow DATASET ...]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# <dataset>-NNNNN.tar. <dataset>-rNNN-NNNNN.tar is a legacy name an earlier version of
# `bulwark shard` wrote for added clips; such shards still count. Same pattern as
# scripts/check_shards.py (a test keeps them equal).
SHARD_NAME = re.compile(r"^(?P<dataset>.+?)(?:-r\d{3})?-\d{5}\.tar$")
# --storage value -> how messages name it (the registry flags are STORAGE_FIELDS).
STORAGE_NAMES = {"kaggle": "Kaggle", "azure": "Azure Blob"}
INCLUDE_ROOT = "checkpoints"
# Files of a training `out` dir that later jobs read (bulwark/train/common.py RUN_FILES
# minus last.pt and log.jsonl), and graphs written by `bulwark export`.
INCLUDE_FILES = ("best.pt", "final.pt", "provenance.json")
INCLUDE_SUFFIX = ".onnx"


def clearance_field(storage: str) -> str:
    """The registry flag that clears a dataset for `storage` ("kaggle" or "azure")."""
    from bulwark.data.registry import STORAGE_FIELDS

    if storage not in STORAGE_FIELDS:
        raise SystemExit(f"Unknown storage {storage!r}; expected one of {', '.join(sorted(STORAGE_FIELDS))}.")
    return STORAGE_FIELDS[storage]


def clearance(storage: str) -> dict[str, bool | None]:
    """{dataset: its flag for `storage`} from the registry; None means not decided yet.

    A registry without the field is an error, not "nothing cleared": a renamed or missing
    field must not quietly turn every upload into a refusal or an empty manifest.
    """
    from bulwark.data.registry import DATASETS

    field = clearance_field(storage)
    missing = [spec.name for spec in DATASETS if not hasattr(spec, field)]
    if missing:
        which = "" if len(missing) == len(DATASETS) else f" (missing for {', '.join(missing)})"
        raise SystemExit(
            f"bulwark/data/registry.py: DatasetSpec has no `{field}` field{which}. Add it (None "
            f"until each dataset's terms are checked) before staging data for {storage}."
        )
    return {spec.name: getattr(spec, field) for spec in DATASETS}


def cleared(allow: set[str], storage: str) -> set[str]:
    return {name for name, ok in clearance(storage).items() if ok is True} | allow


def dataset_of(name: str) -> str | None:
    match = SHARD_NAME.match(name)
    return match["dataset"] if match else None


def stage_shards(paths: list[Path], stage: Path, allowed: set[str]) -> list[str]:
    """Hard-link (or copy) shards into stage/shards as .shard; returns refused file names."""
    refused = [path.name for path in sorted(paths) if dataset_of(path.name) not in allowed]
    if refused:
        return refused
    (stage / "shards").mkdir(parents=True, exist_ok=True)
    for path in sorted(paths):
        # Kaggle unpacks uploaded .tar files, so shards go up as .shard (same bytes).
        target = stage / "shards" / f"{path.stem}.shard"
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    return []


def stage_manifest(source: Path, target: Path, allowed: set[str], drop: bool) -> tuple[bool, int, Counter[str]]:
    """Copy the manifest keeping only rows of allowed datasets. Returns (written, rows kept,
    rows left out per dataset); nothing is written when rows would be left out without
    `drop`, or when no row would be left at all. A manifest that loses no rows is copied
    as is, so its sha256 (which runs record) matches the local one. A copy that loses rows
    records them in its parquet metadata (`bulwark.data.manifest.LEFT_OUT_KEY`, rows per
    dataset), so a remote evaluation reports a holdout dataset left out of the upload as
    such rather than as not ingested. The columns are unchanged."""
    import json

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from bulwark.data.manifest import LEFT_OUT_KEY

    table = pq.read_table(source)
    keep = pc.is_in(table["dataset"], value_set=pa.array(sorted(allowed), type=table.schema.field("dataset").type))
    kept = table.filter(keep)
    dropped = Counter(table.filter(pc.invert(keep))["dataset"].to_pylist())
    if kept.num_rows == 0 or (dropped and not drop):
        return False, kept.num_rows, dropped
    if dropped:
        metadata = kept.schema.metadata or {}
        left_out = Counter(json.loads(metadata.get(LEFT_OUT_KEY) or b"{}")) + dropped  # a staged copy staged again
        record = json.dumps(dict(sorted(left_out.items()))).encode()
        pq.write_table(kept.replace_schema_metadata({**metadata, LEFT_OUT_KEY: record}), target)
    else:
        shutil.copy2(source, target)
    return True, kept.num_rows, dropped


def check_include(path: str) -> str | None:
    """Why `path` may not be staged with --include, or None if it may."""
    parts = Path(path).parts
    if Path(path).is_absolute() or ".." in parts or not parts or parts[0] != INCLUDE_ROOT:
        return f"{path}: --include takes paths under {INCLUDE_ROOT}/ only (data/ goes through the clearance checks)"
    return None


def staged_file(name: str) -> bool:
    return name in INCLUDE_FILES or name.endswith(INCLUDE_SUFFIX)


def include_files(root: Path, path: str) -> list[str]:
    """The files `--include path` stages: the INCLUDE_FILES and *.onnx graphs at or under
    `path`, outside stale-<time>/ folders and pretrained/."""
    source = root / path
    if source.is_file():
        return [path] if staged_file(source.name) else []
    return [
        candidate.relative_to(root).as_posix()
        for candidate in sorted(source.rglob("*"))
        if staged_file(candidate.name) and candidate.is_file()
        and not any(part.startswith("stale-") or part == "pretrained" for part in candidate.relative_to(source).parts)
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--storage", choices=sorted(STORAGE_NAMES), default="kaggle", help="the provider whose clearance applies"
    )
    parser.add_argument("--shards", default="", help="space-separated globs under data/shards; empty = none")
    parser.add_argument("--manifest", action="store_true", help="stage data/manifest.parquet")
    parser.add_argument(
        "--drop-uncleared", action="store_true", help="stage the manifest without rows of uncleared datasets"
    )
    parser.add_argument("--include", action="append", default=[], help="extra path under checkpoints/")
    parser.add_argument("--allow", action="append", default=[], help="dataset to upload although not cleared")
    args = parser.parse_args(argv)

    problems: list[str] = []
    included: list[str] = []
    for path in args.include:
        problem = check_include(path)
        if problem is None and not (args.root / path).exists():
            problem = f"{path}: not found"
        if problem is None:
            files = include_files(args.root, path)
            included += files
            wanted = ", ".join(INCLUDE_FILES)
            problem = None if files else f"{path}: no {wanted} or *{INCLUDE_SUFFIX} to stage (last.pt never is)"
        if problem:
            problems.append(problem)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    allowed = cleared(set(args.allow), args.storage) if args.shards.strip() or args.manifest else set()
    field, where = clearance_field(args.storage), STORAGE_NAMES[args.storage]
    shutil.rmtree(args.stage, ignore_errors=True)
    args.stage.mkdir(parents=True)
    if args.shards.strip():
        paths: set[Path] = set()
        skipped: set[str] = set()
        for pattern in args.shards.split():
            matched = [path for path in (args.root / "data" / "shards").glob(pattern) if path.is_file()]
            shards = [path for path in matched if SHARD_NAME.match(path.name)]
            skipped.update(path.name for path in matched if not SHARD_NAME.match(path.name))
            if not shards:
                print(f"No shards match data/shards/{pattern}", file=sys.stderr)
                return 1
            paths.update(shards)
        if skipped:
            print(f"Not shards, so not staged: {', '.join(sorted(skipped))}", file=sys.stderr)
        refused = stage_shards(sorted(paths), args.stage, allowed)
        if refused:
            names = sorted({dataset_of(name) or name for name in refused})
            print(
                f"Refused: {', '.join(names)} not cleared for storage on {where} "
                f"(registry {field}). Check each dataset's terms; to upload anyway, "
                "name it in ALLOW_UNCLEARED.",
                file=sys.stderr,
            )
            return 1
        sources = Counter(dataset_of(path.name) for path in paths)
        print(f"Staged {len(paths)} shards: " + ", ".join(f"{name} ({n})" for name, n in sorted(sources.items())))
    if args.manifest:
        written, rows, dropped = stage_manifest(
            args.root / "data" / "manifest.parquet", args.stage / "manifest.parquet", allowed, args.drop_uncleared
        )
        left_out = ", ".join(f"{name} ({n} rows)" for name, n in sorted(dropped.items()))
        if not written:
            if not dropped:
                problem = "data/manifest.parquet has no rows."
            elif rows == 0:
                problem = (
                    f"no dataset in the manifest is cleared for storage on {where} (registry "
                    f"{field}): {left_out}. Name the ones to upload in ALLOW_UNCLEARED."
                )
            else:
                problem = (
                    f"the manifest has rows of datasets not cleared for storage on {where} (registry "
                    f"{field}): {left_out}. Remote jobs read only this manifest, so they would "
                    "silently train on fewer datasets. Name those datasets in ALLOW_UNCLEARED, or set "
                    "DROP_UNCLEARED=1 to leave their rows out."
                )
            print(f"Refused: {problem}", file=sys.stderr)
            return 1
        print(f"Staged the manifest ({rows} rows)" + (f", leaving out {left_out}" if dropped else ""))
    for path in dict.fromkeys(included):
        target = args.stage / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.root / path, target)
    if included:
        print(f"Staged {len(set(included))} checkpoint files: " + ", ".join(dict.fromkeys(included)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
