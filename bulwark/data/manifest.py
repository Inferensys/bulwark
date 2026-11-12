"""The clip manifest: one row per training sample, with its licence and labels.

A manifest can also carry file-level metadata in its parquet schema (bytes keys and
values), such as the split policy that assigned its splits
(`bulwark.data.splits.policy_metadata`). Rewriting a manifest keeps it only when the
caller passes it on: `write_manifest(records, path, metadata=read_manifest_metadata(path))`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bulwark.data.taxonomy import Taxonomy, default_taxonomy

MEDIA_TYPES = {"video", "image"}
# "excluded": kept out of training and scoring (see bulwark.data.splits).
SPLITS = {"train", "val", "test", "holdout", "excluded", "unassigned"}


@dataclass(slots=True)
class ClipRecord:
    clip_id: str
    dataset: str
    subject_id: str
    media_path: str
    media_type: str
    licence: str
    commercial_ok: bool
    start_frame: int = 0
    num_frames: int = 1
    fps: float = 0.0
    frame_stride: int = 1
    view: str = ""
    crop: tuple[int, int, int, int] | None = None
    split: str = "unassigned"
    labels_gt: dict[str, Any] = field(default_factory=dict)
    labels_teacher: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self, taxonomy: Taxonomy | None = None) -> ClipRecord:
        taxonomy = taxonomy or default_taxonomy()
        if self.media_type not in MEDIA_TYPES:
            raise ValueError(f"{self.clip_id}: media_type must be one of {MEDIA_TYPES}")
        if self.split not in SPLITS:
            raise ValueError(f"{self.clip_id}: split must be one of {SPLITS}")
        if not self.subject_id.startswith(f"{self.dataset}:"):
            raise ValueError(f"{self.clip_id}: subject_id must be prefixed with '{self.dataset}:'")
        if self.num_frames < 1 or self.frame_stride < 1:
            raise ValueError(f"{self.clip_id}: num_frames and frame_stride must be >= 1")
        taxonomy.validate(self.labels_gt)
        taxonomy.validate(self.labels_teacher)
        return self

    @property
    def labels(self) -> dict[str, Any]:
        """Teacher labels filled in under the ground truth (ground truth always wins)."""
        return {**self.labels_teacher, **self.labels_gt}


_JSON_COLUMNS = ("labels_gt", "labels_teacher", "meta", "crop")


def to_frame(records: Iterable[ClipRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = asdict(record)
        for column in _JSON_COLUMNS:
            row[column] = json.dumps(row[column], sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(ClipRecord.__dataclass_fields__))


def from_frame(frame: pd.DataFrame) -> list[ClipRecord]:
    records = []
    for row in frame.to_dict(orient="records"):
        for column in _JSON_COLUMNS:
            row[column] = json.loads(row[column])
        if row["crop"] is not None:
            row["crop"] = tuple(row["crop"])
        row["commercial_ok"] = bool(row["commercial_ok"])
        records.append(ClipRecord(**row))
    return records


# Keys pandas and pyarrow write themselves; not part of a manifest's own metadata.
_FORMAT_KEYS = frozenset({b"pandas", b"ARROW:schema"})
# Metadata key of a copy staged for Kaggle or Azure without the rows of datasets not cleared
# for that storage (infra/kaggle/stage_data.py --drop-uncleared): JSON
# {dataset: rows left out}. The evaluation report reads it (`left_out_datasets`).
LEFT_OUT_KEY = b"left_out_datasets"


def write_manifest(
    records: Iterable[ClipRecord], path: Path, metadata: Mapping[bytes, bytes] | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = to_frame(records)
    duplicated = frame["clip_id"][frame["clip_id"].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Duplicate clip ids, e.g. {duplicated.iloc[0]}")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if metadata:
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
    pq.write_table(table, path)
    return path


def read_manifest(path: Path) -> list[ClipRecord]:
    return from_frame(pd.read_parquet(path))


def read_manifest_metadata(path: Path) -> dict[bytes, bytes]:
    """The manifest's own file-level metadata (empty if it has none)."""
    metadata = pq.read_schema(path).metadata or {}
    return {key: value for key, value in metadata.items() if key not in _FORMAT_KEYS}


def left_out_datasets(path: Path) -> dict[str, int]:
    """{dataset: rows} that a staged copy of the manifest left out (LEFT_OUT_KEY); empty
    for a manifest that lost no rows."""
    raw = read_manifest_metadata(path).get(LEFT_OUT_KEY)
    return {str(name): int(rows) for name, rows in json.loads(raw).items()} if raw else {}


def merge_manifests(paths: Iterable[Path]) -> list[ClipRecord]:
    records: dict[str, ClipRecord] = {}
    for path in paths:
        for record in read_manifest(path):
            records[record.clip_id] = record
    return list(records.values())
