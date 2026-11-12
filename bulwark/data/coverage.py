"""Label coverage of a manifest, and labels that cannot be learned from it.

A head trained only on positives (or only on one class) learns a constant, and its test
metrics are undefined. That happens quietly when the sources give a label only one value:
in v0, every clip's context is the vehicle cabin, DMD's derived alertness is only ever
'microsleep', and the image phone and drink labels (from AUC Distracted Driver v2) are
positives only. Training also keeps videos and
images in separate batches, so coverage is counted per media type as well as per split.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from bulwark.data.manifest import ClipRecord
from bulwark.data.taxonomy import LabelSpec, Taxonomy, default_taxonomy


@dataclass(frozen=True, slots=True)
class CoverageRow:
    label: str
    dataset: str
    split: str
    media_type: str
    source: str  # "gt" or "teacher"
    n: int
    classes: dict[str, int]  # binary: {"0": n, "1": n}; categorical: per class; regression: {}


def _class_of(spec: LabelSpec, value: Any) -> str | None:
    if spec.kind == "binary":
        return "1" if float(value) >= 0.5 else "0"
    if spec.kind == "categorical":
        return str(value)
    return None


def label_coverage(
    records: Iterable[ClipRecord], taxonomy: Taxonomy | None = None
) -> list[CoverageRow]:
    """Counts per (label, dataset, split, media type, ground truth or teacher)."""
    taxonomy = taxonomy or default_taxonomy()
    counts: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str, str, str, str]] = Counter()
    for record in records:
        for source, labels in (("gt", record.labels_gt), ("teacher", record.labels_teacher)):
            for name, value in labels.items():
                key = (name, record.dataset, record.split, record.media_type, source)
                totals[key] += 1
                if (cls := _class_of(taxonomy[name], value)) is not None:
                    counts[key][cls] += 1
    return [
        CoverageRow(*key, n=totals[key], classes=dict(sorted(counts[key].items())))
        for key in sorted(totals)
    ]


def single_class_labels(
    records: Sequence[ClipRecord], split: str = "train", taxonomy: Taxonomy | None = None
) -> dict[tuple[str, str], str]:
    """{(label, media_type): reason} for labels whose targets in `split` (ground truth and
    teacher together, as training sees them) cover only one class. Regression labels are
    flagged when every target has the same value."""
    taxonomy = taxonomy or default_taxonomy()
    seen: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        if record.split != split:
            continue
        for name, value in record.labels.items():
            spec = taxonomy[name]
            seen[(name, record.media_type)][_class_of(spec, value) or f"{float(value):g}"] += 1
    return {
        key: f"only {next(iter(classes))!r} in {sum(classes.values())} {split} targets"
        for key, classes in sorted(seen.items())
        if len(classes) == 1
    }


def missing_classes(
    records: Sequence[ClipRecord], split: str = "train", taxonomy: Taxonomy | None = None
) -> dict[tuple[str, str], tuple[str, ...]]:
    """{(label, media_type): classes without a target in `split`} for binary and
    categorical labels that have targets there. A head trained this way never learns the
    missing classes, which `single_class_labels` does not catch once two classes are
    present (for example attention_direction with no 'forward')."""
    taxonomy = taxonomy or default_taxonomy()
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        if record.split != split:
            continue
        for name, value in record.labels.items():
            if (cls := _class_of(taxonomy[name], value)) is not None:
                seen[(name, record.media_type)].add(cls)
    missing: dict[tuple[str, str], tuple[str, ...]] = {}
    for (name, media_type), classes in sorted(seen.items()):
        spec = taxonomy[name]
        expected = spec.classes if spec.kind == "categorical" else ("0", "1")
        if absent := tuple(c for c in expected if c not in classes):
            missing[(name, media_type)] = absent
    return missing


def coverage_markdown(rows: Sequence[CoverageRow]) -> str:
    lines = [
        "| label | dataset | split | media | source | n | classes |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        classes = ", ".join(f"{k}: {v}" for k, v in row.classes.items()) or "-"
        lines.append(
            f"| {row.label} | {row.dataset} | {row.split} | {row.media_type} | {row.source} | {row.n} | {classes} |"
        )
    return "\n".join(lines) + "\n"
