"""Subjects removed at their own or the data provider's request (for example under GDPR).

`removed_subjects.yaml` lists them, and the split leak check (`check_disjoint`) rejects
any manifest that still holds one. Applying a removal is up to the caller: filter new
records with `drop_removed_subjects` at ingest and before splitting, and delete clips that
already exist from manifests, shards, teacher labels and feature caches with
`purge_subjects`. `purge_clips` deletes any set of clip ids the same way, for example
clips whose frames changed after an adapter fix. `bulwark ingest` and `bulwark split`
drop removed subjects; `bulwark purge` purges them from the local data directories, or
with `--clips` purges a list of clip ids and leaves the manifests alone. Copies outside
the local machine (Kaggle datasets, blob storage) have to be deleted separately
(scripts/purge_cloud_copies.sh). A removal
applies from the next retrain onward; weights that were already released are not
retrained.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bulwark.data.manifest import (
    ClipRecord,
    read_manifest,
    read_manifest_metadata,
    write_manifest,
)
from bulwark.data.shards import member_bytes

REMOVED_PATH = Path(__file__).with_name("removed_subjects.yaml")


def load_removed_subjects(path: Path = REMOVED_PATH) -> dict[str, dict[str, Any]]:
    """{"<dataset>:<subject>": {"date": ..., "source": ...}}"""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    removed = data.get("removed") or {}
    for subject, entry in removed.items():
        if (
            ":" not in subject
            or not isinstance(entry, dict)
            or not {"date", "source"} <= set(entry)
        ):
            raise ValueError(
                f"{path}: {subject!r} needs a '<dataset>:<subject>' id with date and source"
            )
    return {str(subject): dict(entry) for subject, entry in removed.items()}


def drop_removed_subjects(
    records: Iterable[ClipRecord], removed: Iterable[str] | None = None
) -> tuple[list[ClipRecord], int]:
    """Records without the removed subjects, and how many were dropped."""
    blocked = set(load_removed_subjects() if removed is None else removed)
    records = list(records)
    kept = [r for r in records if r.subject_id not in blocked]
    return kept, len(records) - len(kept)


def purge_subjects(
    subjects: Iterable[str],
    *,
    manifests: Sequence[Path],
    shards: Path | None = None,
    labels: Path | None = None,
    features: Sequence[Path] = (),
) -> dict[str, int]:
    """Delete every clip of `subjects` from local manifests, shards, labels and features.

    Clip ids are read from the manifests, and the manifests are rewritten last, so an
    interrupted purge can be run again. Returns counts of what was removed.
    """
    blocked = set(subjects)
    loaded = {path: read_manifest(path) for path in manifests if path.exists()}
    clip_ids = {
        r.clip_id for records in loaded.values() for r in records if r.subject_id in blocked
    }
    counts: Counter[str] = Counter(clips=len(clip_ids))
    if clip_ids:
        counts.update(purge_clips(clip_ids, shards=shards, labels=labels, features=features))
    for path, records in loaded.items():
        kept = [r for r in records if r.subject_id not in blocked]
        if len(kept) < len(records):
            write_manifest(kept, path, metadata=read_manifest_metadata(path))
            counts["manifest_rows"] += len(records) - len(kept)
    return dict(counts)


def purge_clips(
    clip_ids: Iterable[str],
    *,
    shards: Path | None = None,
    labels: Path | None = None,
    features: Sequence[Path] = (),
) -> dict[str, int]:
    """Delete `clip_ids` from the shards (`*.tar`), teacher labels (`*.jsonl`) and feature
    caches (`*.npz`) under the given directories. Manifests are left alone. Returns how
    many shard clips, label rows and feature rows were removed.

    Unfinished shards (`*.tar.partial`, left by a shard run that was killed) are deleted
    whole: they have no index, so nothing reads them, but they may hold the clips' frames.
    Do not purge while a shard run is writing to the same directory."""
    blocked = set(clip_ids)
    if not blocked:
        return {}
    counts: Counter[str] = Counter()
    for partial in sorted(shards.glob("*.tar.partial")) if shards else ():
        partial.unlink()
        counts["partial_shards"] += 1
    for tar in sorted(shards.glob("*.tar")) if shards else ():
        counts["shard_clips"] += _purge_shard(tar, blocked)
    for jsonl in sorted(labels.glob("*.jsonl")) if labels else ():
        counts["label_rows"] += _purge_jsonl(jsonl, blocked)
    for directory in features:
        for npz in sorted(directory.glob("*.npz")):
            counts["feature_rows"] += _purge_npz(npz, blocked)
    return dict(counts)


def _replace(path: Path, write: Any) -> None:
    tmp = path.with_name(f".{path.name}.purge")
    write(tmp)
    os.replace(tmp, path)


def _purge_shard(path: Path, clip_ids: set[str]) -> int:
    with tarfile.open(path, "r") as tar:
        index: dict[str, str] = json.loads(member_bytes(tar, "index.json"))
        drop = {key for key, clip in index.items() if clip in clip_ids}
        if not drop:
            return 0

        def write(tmp: Path) -> None:
            with tarfile.open(tmp, "w") as out:
                for member in tar.getmembers():
                    if member.name != "index.json" and member.name.partition("/")[0] not in drop:
                        out.addfile(member, tar.extractfile(member))
                kept = json.dumps({k: v for k, v in index.items() if k not in drop}).encode()
                info = tarfile.TarInfo("index.json")
                info.size = len(kept)
                out.addfile(info, io.BytesIO(kept))

        _replace(path, write)
    return len(drop)


def _purge_jsonl(path: Path, clip_ids: set[str]) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if line and json.loads(line)["clip_id"] not in clip_ids]
    removed = sum(1 for line in lines if line) - len(kept)
    if removed:
        _replace(
            path,
            lambda tmp: tmp.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8"),
        )
    return removed


def _purge_npz(path: Path, clip_ids: set[str]) -> int:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if "clip_ids" not in arrays:
        return 0
    rows = len(arrays["clip_ids"])
    keep = ~np.isin(arrays["clip_ids"].astype(str), sorted(clip_ids))
    if keep.all():
        return 0
    filtered = {k: (v[keep] if v.shape[:1] == (rows,) else v) for k, v in arrays.items()}

    def write(tmp: Path) -> None:
        with tmp.open("wb") as handle:
            np.savez(handle, **filtered)

    _replace(path, write)
    return int((~keep).sum())
