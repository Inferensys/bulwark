"""How well each teacher agrees with ground truth, on clips that have both.

Teacher labels train at reduced weight on clips without ground truth, so their quality
matters, but nothing else measures it. This compares each teacher's raw output (before
the merge rules in run.py) with `labels_gt`, per label and dataset: for binary labels
precision, npv, recall and F1, accuracy for categorical labels and MAE for regression
labels.

- precision: share of the teacher's positives that ground truth confirms;
- npv (negative predictive value): share of the teacher's negatives that ground truth
  confirms.

`splits` limits the clips to some splits. The face-teacher values the merge keeps
(`FACE_KEEP` in run.py: eyes_closed 0, and no blink or yawn value) were chosen from the
train split alone, so that val and test ground truth never shape training labels. The
report still scores the teacher's raw blink and yawn, which the merge drops.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

from bulwark.data.manifest import ClipRecord
from bulwark.data.taxonomy import Taxonomy, default_taxonomy
from bulwark.label.run import load_teacher_labels

Pairs = list[tuple[Any, Any]]


def teacher_agreement(
    records: Sequence[ClipRecord],
    labels_dir: Path,
    taxonomy: Taxonomy | None = None,
    splits: Collection[str] | None = None,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """{teacher: {label: {dataset: stats}}}; stats always include `n`. With `splits`, only
    records in those splits count (None: every split)."""
    taxonomy = taxonomy or default_taxonomy()
    tables = load_teacher_labels(labels_dir)
    pairs: dict[tuple[str, str, str], Pairs] = defaultdict(list)
    for record in records:
        if splits is not None and record.split not in splits:
            continue
        for teacher, table in tables.items():
            for name, value in table.get(record.clip_id, {}).items():
                if name in taxonomy.labels and name in record.labels_gt:
                    pairs[(teacher, name, record.dataset)].append((value, record.labels_gt[name]))
    report: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for (teacher, name, dataset), found in sorted(pairs.items()):
        stats = _stats(taxonomy[name].kind, found)
        report.setdefault(teacher, {}).setdefault(name, {})[dataset] = stats
    return report


def _stats(kind: str, pairs: Pairs) -> dict[str, Any]:
    stats: dict[str, Any] = {"n": len(pairs)}
    if kind == "binary":
        tp = sum(t >= 0.5 and g >= 0.5 for t, g in pairs)
        fp = sum(t >= 0.5 and g < 0.5 for t, g in pairs)
        fn = sum(t < 0.5 and g >= 0.5 for t, g in pairs)
        tn = len(pairs) - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else None
        npv = tn / (tn + fn) if tn + fn else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
        stats.update(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=precision,
            npv=npv,
            recall=recall,
            f1=f1,
        )
    elif kind == "categorical":
        stats["accuracy"] = sum(str(t) == str(g) for t, g in pairs) / len(pairs)
    else:
        stats["mae"] = sum(abs(float(t) - float(g)) for t, g in pairs) / len(pairs)
    return stats
