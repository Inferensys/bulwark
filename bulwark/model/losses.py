"""Sparse multi-task targets and the masked, uncertainty-weighted loss."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from bulwark.data.taxonomy import Taxonomy
from bulwark.model.heads import regression_value

TEACHER_WEIGHT = 0.5


@dataclass(slots=True)
class Targets:
    values: dict[str, torch.Tensor]   # float (B,) or long (B,) for categorical (-1 = missing)
    mask: dict[str, torch.Tensor]     # float (B,) 1 = label present
    weight: dict[str, torch.Tensor]   # float (B,) 1.0 ground truth, TEACHER_WEIGHT teacher

    def to(self, device: torch.device | str) -> Targets:
        return Targets(
            {k: v.to(device) for k, v in self.values.items()},
            {k: v.to(device) for k, v in self.mask.items()},
            {k: v.to(device) for k, v in self.weight.items()},
        )


def encode_targets(
    gt: Sequence[dict[str, Any]],
    teacher: Sequence[dict[str, Any]],
    taxonomy: Taxonomy,
    teacher_weight: float = TEACHER_WEIGHT,
) -> Targets:
    batch = len(gt)
    values: dict[str, torch.Tensor] = {}
    mask: dict[str, torch.Tensor] = {}
    weight: dict[str, torch.Tensor] = {}
    for name, spec in taxonomy.labels.items():
        categorical = spec.kind == "categorical"
        v = torch.full((batch,), -1 if categorical else 0.0, dtype=torch.long if categorical else torch.float32)
        m = torch.zeros(batch)
        w = torch.zeros(batch)
        for i in range(batch):
            if name in gt[i]:
                raw, w[i] = gt[i][name], 1.0
            elif name in teacher[i]:
                raw, w[i] = teacher[i][name], teacher_weight
            else:
                continue
            m[i] = 1.0
            v[i] = spec.class_index(str(raw)) if categorical else float(raw)
        values[name], mask[name], weight[name] = v, m, w
    return Targets(values, mask, weight)


class MultiTaskLoss(nn.Module):
    """Sum over labels of masked per-label losses, weighted by learned uncertainty
    (Kendall et al. 2018): 0.5 * exp(-s) * L + 0.5 * s, with s = log(sigma^2).

    Teacher labels count at their weight (TEACHER_WEIGHT). The uncertainty term is
    scale-invariant at its optimum, so a constant factor inside L alone would be undone
    by s; the whole term is therefore scaled by the batch's mean label weight, which keeps
    an all-teacher batch at TEACHER_WEIGHT times the loss and gradient of the same batch
    labelled as ground truth.
    """

    def __init__(self, taxonomy: Taxonomy, *, learn_weights: bool = True) -> None:
        super().__init__()
        self.taxonomy = taxonomy
        self.names = list(taxonomy.labels)
        self.log_vars = nn.Parameter(torch.zeros(len(self.names)), requires_grad=learn_weights)

    def per_label(self, logits: dict[str, torch.Tensor], targets: Targets) -> dict[str, torch.Tensor]:
        """Per-label loss: sum of weighted sample losses over the number of labelled samples."""
        losses: dict[str, torch.Tensor] = {}
        for name in self.names:
            mask = targets.mask[name]
            count = mask.sum()
            if count == 0:
                continue
            spec = self.taxonomy[name]
            output = logits[name].float()
            target = targets.values[name]
            if spec.kind == "categorical":
                raw = F.cross_entropy(output, target.clamp(min=0), reduction="none")
            elif spec.kind == "binary":
                raw = F.binary_cross_entropy_with_logits(output, target, reduction="none")
            else:
                raw = F.smooth_l1_loss(regression_value(output, spec.value_range), target, reduction="none", beta=0.1)
            losses[name] = (raw * mask * targets.weight[name]).sum() / count
        return losses

    def forward(
        self, logits: dict[str, torch.Tensor], targets: Targets
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses = self.per_label(logits, targets)
        if not losses:
            zero = next(iter(logits.values())).sum() * 0.0
            return zero, {}
        total = torch.zeros((), device=next(iter(losses.values())).device)
        for name, value in losses.items():
            mask = targets.mask[name]
            label_weight = (targets.weight[name] * mask).sum() / mask.sum()
            s = self.log_vars[self.names.index(name)].clamp(-4.0, 4.0)
            # = label_weight * (0.5 * exp(-s) * value / label_weight + 0.5 * s)
            total = total + 0.5 * torch.exp(-s) * value + 0.5 * label_weight * s
        return total, {name: float(value.detach()) for name, value in losses.items()}


def distill_loss(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    taxonomy: Taxonomy,
    temperature: float = 2.0,
    *,
    mask: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Match the student's outputs to the teacher's soft outputs.

    Without `mask`, every label counts on every sample. With `mask` ({label: (B,) 0/1}),
    only the labels it lists count, each averaged over the samples it marks; labels the
    teacher never learned (or that are masked for a media type) are left out so the
    student does not copy them. The sum is divided by the taxonomy's label count either
    way, so leaving a label out does not reweight the others.
    """
    total = torch.zeros((), device=next(iter(student.values())).device)
    for name, spec in taxonomy.labels.items():
        weights = None if mask is None else mask.get(name)
        if mask is not None and (weights is None or not bool(weights.any())):
            continue
        s, t = student[name].float(), teacher[name].float()
        if spec.kind == "categorical":
            raw = F.kl_div(F.log_softmax(s / temperature, -1), F.softmax(t / temperature, -1), reduction="none").sum(-1)
            raw = raw * temperature**2
        elif spec.kind == "binary":
            raw = F.binary_cross_entropy_with_logits(s, t.sigmoid(), reduction="none")
        else:
            raw = F.smooth_l1_loss(s.sigmoid(), t.sigmoid(), beta=0.1, reduction="none")
        total = total + (raw.mean() if weights is None else (raw * weights).sum() / weights.sum())
    return total / len(taxonomy.labels)
