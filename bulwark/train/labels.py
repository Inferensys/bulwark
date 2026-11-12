"""Which labels a training run can learn, and the targets it trains on.

A head that only ever sees one class of a label (or one value) learns a constant. That
happens when the sources give a label only one value: in v0, every clip's context is the
vehicle cabin, DMD's derived alertness is only ever 'microsleep' and AUC Distracted
Driver v2, the only image source of phone and drink, gives them positives only. Before
training, every stage
checks the targets of its training clips (`bulwark.data.coverage`, ground truth and
teacher together) per label and media type, since images and videos are batched
separately, and masks each single-class pair: its targets leave the loss, and
distillation does not copy the teacher's outputs for it. `mask_labels` masks whole
labels on top of that, for labels whose classes are present but unrepresentative (the
v0 configs mask attention_direction, whose Drive&Act targets never include 'forward',
and face_visible). Classes that no training target covers are reported, not masked.

`supervised` lists the labels left with at least one target. Every checkpoint records it
as `supervised_labels`, and `bulwark.inference.load` returns only those outputs by
default. A label can be supervised on one media type and masked on the other (phone in
v0, with video targets of both classes but image targets that are all positives), so
every checkpoint also records the masked pairs as `masked_labels` ({label: [media types]}), and best.pt
selection leaves out their val rows (`bulwark.train.validation`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from rich.console import Console

from bulwark.data.coverage import missing_classes, single_class_labels
from bulwark.data.manifest import MEDIA_TYPES, ClipRecord
from bulwark.data.taxonomy import Taxonomy
from bulwark.model.losses import Targets, encode_targets

console = Console()

ALL_MEDIA = "*"


@dataclass(frozen=True, slots=True)
class LabelPlan:
    masked: dict[tuple[str, str], str]  # (label, media type or ALL_MEDIA) -> reason
    missing_classes: dict[tuple[str, str], tuple[str, ...]]  # unmasked pairs whose targets lack classes
    supervised: tuple[str, ...]  # labels with at least one unmasked target, in taxonomy order

    def is_masked(self, label: str, media_type: str) -> bool:
        return (label, ALL_MEDIA) in self.masked or (label, media_type) in self.masked

    def targets(self, records: Sequence[ClipRecord], taxonomy: Taxonomy) -> Targets:
        """Ground-truth and teacher targets of a batch, without the masked labels."""
        gt = [{k: v for k, v in r.labels_gt.items() if not self.is_masked(k, r.media_type)} for r in records]
        teacher = [{k: v for k, v in r.labels_teacher.items() if not self.is_masked(k, r.media_type)} for r in records]
        return encode_targets(gt, teacher, taxonomy)

    def sample_mask(self, records: Sequence[ClipRecord], labels: Iterable[str]) -> dict[str, torch.Tensor]:
        """{label: (B,) 1.0 where the record's (label, media type) is not masked}, for `labels`."""
        return {
            name: torch.tensor([0.0 if self.is_masked(name, r.media_type) else 1.0 for r in records])
            for name in labels
        }

    def masked_media(self) -> dict[str, list[str]]:
        """{label: media types it was masked on}, every media type for a whole-label mask."""
        media: dict[str, set[str]] = {}
        for label, kind in self.masked:
            media.setdefault(label, set()).update(MEDIA_TYPES if kind == ALL_MEDIA else {kind})
        return {label: sorted(kinds) for label, kinds in sorted(media.items())}

    def summary(self) -> dict[str, object]:
        """JSON-ready, for the run's provenance."""
        return {
            "masked": {f"{label}:{media}": reason for (label, media), reason in sorted(self.masked.items())},
            "missing_classes": {f"{label}:{media}": list(c) for (label, media), c in sorted(self.missing_classes.items())},
            "supervised": list(self.supervised),
        }


def label_plan(records: Iterable[ClipRecord], taxonomy: Taxonomy, mask_labels: Sequence[str] = ()) -> LabelPlan:
    """The plan for the training clips among `records` (other splits are ignored)."""
    train = [r for r in records if r.split == "train"]
    unknown = sorted(set(mask_labels) - set(taxonomy.labels))
    if unknown:
        raise ValueError(f"mask_labels names labels the taxonomy does not have: {unknown}")
    masked: dict[tuple[str, str], str] = {(name, ALL_MEDIA): "mask_labels" for name in mask_labels}
    for (name, media), reason in single_class_labels(train, "train", taxonomy).items():
        if name not in mask_labels:
            masked[(name, media)] = reason
    plan = LabelPlan(masked, {}, ())
    missing = {key: classes for key, classes in missing_classes(train, "train", taxonomy).items() if not plan.is_masked(*key)}
    present = {name for r in train for name in r.labels if not plan.is_masked(name, r.media_type)}
    return LabelPlan(masked, missing, tuple(name for name in taxonomy.labels if name in present))


def print_plan(plan: LabelPlan, taxonomy: Taxonomy) -> None:
    """What the run will not learn, before it starts."""
    for (label, media), reason in sorted(plan.masked.items()):
        where = "all clips" if media == ALL_MEDIA else f"{media} clips"
        console.print(f"[yellow]Not training {label} on {where}: {reason}[/yellow]")
    for (label, media), classes in sorted(plan.missing_classes.items()):
        console.print(f"[yellow]{label} has no {media} training targets for {list(classes)}; the head cannot learn them there.[/yellow]")
    console.print(f"Training {len(plan.supervised)} of {len(taxonomy.labels)} labels: {list(plan.supervised)}")
