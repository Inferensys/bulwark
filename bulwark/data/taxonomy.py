from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.yaml")
LABEL_KINDS = {"binary", "regression", "categorical"}


@dataclass(frozen=True, slots=True)
class LabelSpec:
    name: str
    head: str
    kind: str
    classes: tuple[str, ...] = ()
    value_range: tuple[float, float] = (0.0, 1.0)
    doc: str = ""
    # Describes the recording (camera, lighting, setting) rather than the person.
    metadata: bool = False
    # A regression target times unit_scale is a value in `unit` ("" = no natural unit).
    unit: str = ""
    unit_scale: float = 1.0

    @property
    def width(self) -> int:
        """Number of logits this label needs in its head."""
        return len(self.classes) if self.kind == "categorical" else 1

    def class_index(self, value: str) -> int:
        try:
            return self.classes.index(value)
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a class of {self.name}: {self.classes}") from exc


@dataclass(frozen=True, slots=True)
class Taxonomy:
    version: int
    labels: dict[str, LabelSpec] = field(default_factory=dict)

    @property
    def heads(self) -> dict[str, list[LabelSpec]]:
        grouped: dict[str, list[LabelSpec]] = {}
        for spec in self.labels.values():
            grouped.setdefault(spec.head, []).append(spec)
        return grouped

    def __getitem__(self, name: str) -> LabelSpec:
        return self.labels[name]

    def validate(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Check a sparse label dict against the taxonomy and return it unchanged."""
        for name, value in labels.items():
            if name not in self.labels:
                raise KeyError(f"Unknown label {name!r}")
            spec = self.labels[name]
            if spec.kind == "categorical":
                spec.class_index(str(value))
            elif spec.kind == "binary":
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1], got {value}")
            else:
                low, high = spec.value_range
                if not low - 1e-6 <= float(value) <= high + 1e-6:
                    raise ValueError(f"{name} must be in {spec.value_range}, got {value}")
        return labels


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Taxonomy:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels: dict[str, LabelSpec] = {}
    for head, entries in data["heads"].items():
        for name, entry in entries.items():
            kind = entry["kind"]
            if kind not in LABEL_KINDS:
                raise ValueError(f"{name}: unknown kind {kind}")
            if name in labels:
                raise ValueError(f"Duplicate label {name}")
            value_range = tuple(entry.get("range", (0.0, 1.0)))
            labels[name] = LabelSpec(
                name=name,
                head=head,
                kind=kind,
                classes=tuple(entry.get("classes", ())),
                value_range=(float(value_range[0]), float(value_range[1])),
                doc=entry.get("doc", ""),
                metadata=bool(entry.get("metadata", False)),
                unit=str(entry.get("unit", "")),
                unit_scale=float(entry.get("unit_scale", 1.0)),
            )
    return Taxonomy(version=int(data["version"]), labels=labels)


@lru_cache(maxsize=1)
def default_taxonomy() -> Taxonomy:
    return load_taxonomy()
