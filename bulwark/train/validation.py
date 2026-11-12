"""Validation passes and best.pt selection, shared by every training stage.

The score is ground truth only (teacher pseudo-labels and weak labels never count), the
same rows the evaluation report scores: the mean over labels of a higher-is-better
headline metric. It covers what the run trained: the supervised labels, without the val
rows of (label, media type) pairs the run's `LabelPlan` masked (a label trained on images
only is not scored on video). The logged metrics (`evaluate_outputs`) still cover every
label and row. The val split is small (few subjects per dataset), so best.pt records how
many clips and subjects each label's score rests on (`select.support`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader

from bulwark.data.manifest import ClipRecord
from bulwark.data.taxonomy import LabelSpec, Taxonomy
from bulwark.eval.metrics import LabelRows, evaluate_outputs, finite_rows, headline, label_rows
from bulwark.train.common import Checkpointer, ShardStream, log_line
from bulwark.train.labels import LabelPlan
from bulwark.train.provenance import jsonable

console = Console()

SELECTION = "gt_mean_headline"  # mean over labels of AUROC / macro-F1 / 1 - normalised MAE
MIN_SELECTION_SAMPLES = 10


@dataclass(slots=True)
class Validation:
    score: float | None
    per_label: dict[str, float]
    support: dict[str, dict[str, int]]
    metrics: dict[str, dict[str, Any]]

    def select(self) -> dict[str, Any]:
        """The `select` entry of best.pt."""
        return jsonable({"criterion": SELECTION, "score": self.score, "metrics": self.per_label, "support": self.support})


def selected_rows(records: Sequence[ClipRecord], spec: LabelSpec, plan: LabelPlan | None = None) -> LabelRows | None:
    """Ground-truth rows of one label, without those of a media type `plan` masked it on."""
    data = label_rows(records, spec)
    if data is None or plan is None:
        return data
    keep = np.array([not plan.is_masked(spec.name, records[i].media_type) for i in data.rows.tolist()], dtype=bool)
    if not keep.any():
        return None
    return LabelRows(data.rows[keep], data.target[keep], data.subjects[keep], data.events[keep], data.media[keep])


def selection_score(
    logits: dict[str, torch.Tensor], records: Sequence[ClipRecord], taxonomy: Taxonomy,
    labels: Iterable[str] | None = None, plan: LabelPlan | None = None,
) -> tuple[float | None, dict[str, float]]:
    """Mean over `labels` (default: all) with ground truth in val of a higher-is-better headline.

    Binary: AUROC. Categorical: macro-F1. Regression: 1 - MAE / range width. Every label
    counts equally. Rows of a (label, media type) pair `plan` masked are left out. Labels
    with fewer than MIN_SELECTION_SAMPLES ground-truth clips, or with a single class, are
    skipped, and so are rows with non-finite outputs. None when no label qualifies.
    """
    wanted = set(taxonomy.labels if labels is None else labels)
    per_label: dict[str, float] = {}
    for name, spec in taxonomy.labels.items():
        data = selected_rows(records, spec, plan) if name in wanted and name in logits else None
        if data is None:
            continue
        raw = logits[name].float().numpy()[data.rows]
        keep = finite_rows(raw)
        raw, target = raw[keep], data.target[keep]
        if len(target) < MIN_SELECTION_SAMPLES or (spec.kind != "regression" and len(np.unique(target)) < 2):
            continue
        value = headline(spec, raw, target)
        if value is None:
            continue
        if spec.kind == "regression":
            low, high = spec.value_range
            value = 1.0 - value / (high - low)
        per_label[name] = float(value)
    return (float(np.mean(list(per_label.values()))) if per_label else None), per_label


def label_support(
    records: Sequence[ClipRecord], taxonomy: Taxonomy, labels: Iterable[str], plan: LabelPlan | None = None
) -> dict[str, dict[str, int]]:
    """Ground-truth clips and distinct subjects behind each label's selection score."""
    support: dict[str, dict[str, int]] = {}
    for name in labels:
        data = selected_rows(records, taxonomy[name], plan)
        support[name] = {"clips": 0 if data is None else len(data.rows), "subjects": 0 if data is None else len(set(data.subjects.tolist()))}
    return support


def validate(
    outputs: list[dict[str, torch.Tensor]], records: list[ClipRecord], taxonomy: Taxonomy, labels: Iterable[str],
    plan: LabelPlan | None = None,
) -> Validation:
    """Score raw logits (batches aligned with `records`) over the supervised `labels`, on the
    (label, media type) pairs `plan` trained."""
    merged = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    score, per_label = selection_score(merged, records, taxonomy, labels, plan)
    support = label_support(records, taxonomy, per_label, plan)
    return Validation(score, per_label, support, evaluate_outputs(outputs, records, taxonomy))


def report(out: Path, step: int, result: Validation, *, seconds: float | None = None) -> None:
    log_line(out, {
        "event": "val", "step": step, "val_select": result.score, "val_select_labels": result.per_label,
        "val_select_support": result.support, "metrics": result.metrics,
        **({"seconds": seconds} if seconds is not None else {}),
    })
    fewest = min((s["subjects"] for s in result.support.values()), default=0)
    score = result.score if result.score is None else round(result.score, 4)
    console.print(
        f"step {step} val {SELECTION} {score} over {len(result.per_label)} labels (fewest subjects behind a label: {fewest})"
    )


def keep_best(checkpointer: Checkpointer, result: Validation, payload: dict[str, Any]) -> bool:
    """Save best.pt when this score beats the run's best so far."""
    if result.score is None or (checkpointer.best is not None and result.score <= checkpointer.best):
        return False
    checkpointer.best = result.score
    checkpointer.save("best.pt", {**payload, "select": result.select()})
    return True


@torch.no_grad()
def predict(
    net: nn.Module, model: nn.Module, stream: ShardStream, device: torch.device, dtype: torch.dtype | None, num_workers: int
) -> tuple[list[dict[str, torch.Tensor]], list[ClipRecord]]:
    """Raw logits for every clip of a sequential stream, in eval mode; `model` returns to train mode.

    The loader has its own generator, and nothing here draws from the global RNG, so a
    validation pass does not change the training run that follows it.
    """
    model.eval()
    outputs: list[dict[str, torch.Tensor]] = []
    records: list[ClipRecord] = []
    try:
        loader = DataLoader(stream, batch_size=None, num_workers=num_workers, generator=torch.Generator())
        for frames, batch in loader:
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = net(frames.to(device, non_blocking=True))
            outputs.append({k: v.float().cpu() for k, v in logits.items()})
            records.extend(batch)
    finally:
        model.train()
    return outputs, records
