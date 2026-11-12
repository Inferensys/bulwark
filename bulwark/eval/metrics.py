"""Per-label metrics on sparse ground-truth labels.

Binary labels: AUROC (computed on logits, so saturated probabilities never tie), F1 at a
threshold (one for all rows, or one per media type), Brier score and ECE. Categorical
labels: accuracy, balanced accuracy, macro-F1 (sklearn's default: averaged over the
classes present in labels or predictions) and top-label ECE. Both carry per-bin
reliability data. Regression labels: MAE, also in the label's natural unit where it has one.

Metrics that are undefined on a subset (one class only, no positives anywhere) are None,
never a plausible-looking number. Rows with non-finite outputs are counted and left out.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from bulwark.data.manifest import ClipRecord
from bulwark.data.taxonomy import LabelSpec, Taxonomy

HEADLINE = {"binary": "auroc", "categorical": "macro_f1", "regression": "mae"}
ECE_BINS = 15
MIN_CI_SUBJECTS = 10  # below this a subject-level bootstrap says little
MIN_SUBJECTS = 5      # below this a result is marked as indicative only


# ---- elementary metrics ---------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def regression_value(raw: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """NumPy twin of `bulwark.model.heads.regression_value`."""
    low, high = value_range
    return low + (high - low) * sigmoid(raw)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives, negatives = labels.sum(), (1 - labels).sum()
    if positives == 0 or negatives == 0:
        return None
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def f1(pred: np.ndarray, labels: np.ndarray) -> float | None:
    """F1 of the positive class; None when there are no positives in labels or predictions."""
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    if tp + fp + fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


def macro_f1(pred: np.ndarray, labels: np.ndarray, classes: int, only: Sequence[int] | None = None) -> float | None:
    """Mean per-class F1 over classes present in labels or predictions (sklearn's default).

    `only` restricts the average to those classes, with per-class F1 still computed over
    every row (sklearn's `labels=`). None when fewer than two of the averaged classes occur
    in the labels: a constant predictor would otherwise score 1.0 on a single-class subset.
    """
    allowed = list(range(classes) if only is None else only)
    if len(np.unique(labels[np.isin(labels, allowed)])) < 2:
        return None
    present = [c for c in allowed if (labels == c).any() or (pred == c).any()]
    return float(np.mean([f1((pred == c).astype(int), (labels == c).astype(int)) or 0.0 for c in present]))


def balanced_accuracy(pred: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean recall over the classes present in labels (sklearn's balanced_accuracy_score)."""
    present = np.unique(labels)
    if len(present) < 2:
        return None
    return float(np.mean([(pred[labels == c] == c).mean() for c in present]))


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_multiclass(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(((probs - onehot) ** 2).sum(-1)))


def reliability(confidence: np.ndarray, outcome: np.ndarray, bins: int = ECE_BINS) -> dict[str, list[Any]]:
    """Per equal-width confidence bin: clip count, mean confidence and observed frequency
    (None for empty bins). Enough to draw a reliability diagram; ECE is computed from it."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    count, mean_confidence, observed = [], [], []
    for b in range(bins):
        mask = index == b
        count.append(int(mask.sum()))
        mean_confidence.append(float(confidence[mask].mean()) if mask.any() else None)
        observed.append(float(outcome[mask].mean()) if mask.any() else None)
    return {"edges": [round(float(e), 6) for e in edges], "count": count, "mean_confidence": mean_confidence, "observed_frequency": observed}


def _ece(confidence: np.ndarray, outcome: np.ndarray, bins: int) -> float:
    table = reliability(confidence, outcome, bins)
    total = sum(table["count"])
    return float(sum(
        n / total * abs(seen - conf)
        for n, conf, seen in zip(table["count"], table["mean_confidence"], table["observed_frequency"], strict=True)
        if n
    ))


def ece_binary(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error over equal-width bins of the positive-class probability."""
    return _ece(p, y.astype(np.float64), bins)


def ece_top_label(probs: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error of the top-label confidence."""
    return _ece(probs.max(-1), (probs.argmax(-1) == y).astype(np.float64), bins)


def best_threshold(p: np.ndarray, y: np.ndarray) -> float | None:
    """Probability threshold (predict positive when p >= t) that maximises F1."""
    if y.sum() == 0:
        return None
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order].astype(np.float64)
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    fn = ys.sum() - tp
    scores = 2 * tp / (2 * tp + fp + fn)
    cut = np.r_[ps[1:] != ps[:-1], True]  # only cut between distinct probabilities
    scores[~cut] = -1.0
    return float(ps[int(scores.argmax())])


def fit_temperature(logits: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    """Temperature T minimising the NLL of logits / T (binary or categorical)."""
    if len(np.unique(y)) < 2:
        return None
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
    logits = np.asarray(logits, dtype=np.float64)
    losses = []
    for t in grid:
        z = logits / t
        if kind == "binary":
            # log(1 + e^z) - y z, written stably
            losses.append(float(np.mean(np.logaddexp(0.0, z) - y * z)))
        else:
            m = z.max(-1, keepdims=True)
            log_norm = (m + np.log(np.exp(z - m).sum(-1, keepdims=True)))[:, 0]
            losses.append(float(np.mean(log_norm - z[np.arange(len(y)), y])))
    return float(grid[int(np.argmin(losses))])


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float | None], clusters: np.ndarray, draws: int, seed: int = 0
) -> np.ndarray:
    """Resample clusters (subjects) with replacement and recompute `stat` on their rows.

    Returns `draws` values; NaN where the statistic is undefined on a resample.
    """
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
    out = np.full(draws, np.nan)
    for d in range(draws):
        picks = rng.integers(0, len(members), len(members))
        value = stat(np.concatenate([members[i] for i in picks]))
        if value is not None:
            out[d] = value
    return out


def interval(draws: np.ndarray, level: float = 0.95) -> list[float] | None:
    finite = draws[np.isfinite(draws)]
    if len(finite) < 0.9 * len(draws) or len(finite) == 0:
        return None
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(finite, [tail, 100 - tail])
    return [float(low), float(high)]


# ---- one label over a set of rows ------------------------------------------------------------


@dataclass(slots=True)
class LabelRows:
    """Rows of an output array that carry a usable ground-truth value for one label."""

    rows: np.ndarray      # indices into the outputs / records
    target: np.ndarray    # class index (binary: 0/1) or float value
    subjects: np.ndarray
    events: np.ndarray
    media: np.ndarray     # the record's media_type ("image" or "video")


def event_id(record: ClipRecord) -> str:
    """Independent event a clip belongs to. Multi-view datasets record the same moment
    once per camera; their adapters store `meta["event_id"]`. Otherwise a clip is its
    own event."""
    return str(record.meta.get("event_id") or record.clip_id)


def label_rows(records: Sequence[ClipRecord], spec: LabelSpec, *, ground_truth_only: bool = True) -> LabelRows | None:
    """Ground-truth rows for one label. Weak labels (listed in `meta["weak_labels"]`, e.g.
    derived by a rule from another label) are never scored as ground truth."""
    rows, targets, subjects, events, media = [], [], [], [], []
    for index, record in enumerate(records):
        source = record.labels_gt if ground_truth_only else record.labels
        if spec.name not in source or spec.name in record.meta.get("weak_labels", ()):
            continue
        value = source[spec.name]
        if spec.kind == "binary":
            target: float | int = int(float(value) >= 0.5)
        elif spec.kind == "categorical":
            target = spec.class_index(str(value))
        else:
            target = float(value)
        rows.append(index)
        targets.append(target)
        subjects.append(record.subject_id)
        events.append(event_id(record))
        media.append(record.media_type)
    if not rows:
        return None
    dtype = np.float64 if spec.kind == "regression" else np.int64
    return LabelRows(np.array(rows), np.array(targets, dtype=dtype), np.array(subjects), np.array(events), np.array(media))


def headline(spec: LabelSpec, logits: np.ndarray, target: np.ndarray) -> float | None:
    """The one number per label the report leads with (see HEADLINE)."""
    if spec.kind == "binary":
        return auroc(logits, target)
    if spec.kind == "categorical":
        return macro_f1(logits.argmax(-1), target, len(spec.classes))
    return float(np.abs(regression_value(logits, spec.value_range) - target).mean())


def finite_rows(logits: np.ndarray) -> np.ndarray:
    finite = np.isfinite(logits)
    return finite.reshape(len(logits), -1).all(1)


def per_row(value: Mapping[str, Any], media: np.ndarray, what: str) -> list[Any]:
    """The value of each row's media type from a {media type: value} mapping."""
    missing = sorted(set(media.tolist()) - set(value))
    if missing:
        raise ValueError(f"The {what} mapping has no value for media type(s) {missing} of the rows scored")
    return [value[m] for m in media.tolist()]


def score_label(
    spec: LabelSpec,
    logits: np.ndarray,
    data: LabelRows,
    *,
    threshold: float | Mapping[str, float] = 0.5,
    temperature: float | Mapping[str, float | None] | None = None,
    prior: dict[str, Any] | None = None,
    trained_classes: Sequence[int] | None = None,
    bootstrap: int = 0,
    seed: int = 0,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Metrics for one label on its ground-truth rows; `logits` is the full output array.

    `threshold` (binary) and `temperature` are either one value for every row or a
    {media type: value} mapping, applied to each row by its media type (`LabelRows.media`):
    F1 is then computed over all rows together, each against its own threshold, and the
    entry records the mapping over the media types among the scored rows (`threshold`, and
    `temperature` for binary and categorical labels). A mapping must cover every media type
    among the rows. A temperature of None (for one media type, or for all) means none was
    fitted, and the calibrated ECE is given only when every scored row has a temperature.
    `trained_classes` (categorical) are the classes with ground-truth training labels; when
    the evaluated labels include others, they are listed and a macro-F1 over the trained
    classes only is added. Returns the JSON-ready entry and, when a bootstrap ran, the
    headline draws (so a macro-average over datasets can get an interval from the same draws).
    """
    raw = np.asarray(logits[data.rows], dtype=np.float64)
    keep = finite_rows(raw)
    raw, target = raw[keep], data.target[keep]
    subjects, events, media = data.subjects[keep], data.events[keep], data.media[keep]
    entry: dict[str, Any] = {
        "n": int(len(target)),
        "n_subjects": len(set(subjects.tolist())),
        "n_events": len(set(events.tolist())),
    }
    if not keep.all():
        entry["nonfinite"] = int((~keep).sum())
    if not len(target):
        return entry, None
    entry["indicative"] = entry["n_subjects"] < MIN_SUBJECTS
    present = sorted(set(media.tolist()))
    t: float | np.ndarray | None = None  # temperature per row; None unless every row has one
    if isinstance(temperature, Mapping):
        if spec.kind != "regression":
            row_t = per_row(temperature, media, "temperature")
            t = np.array(row_t, dtype=np.float64) if all(v is not None for v in row_t) else None
            entry["temperature"] = {m: temperature[m] for m in present}
    else:
        t = temperature

    if spec.kind == "binary":
        p = sigmoid(raw)
        cut = np.array(per_row(threshold, media, "threshold"), dtype=np.float64) if isinstance(threshold, Mapping) else threshold
        entry.update(
            auroc=auroc(raw, target),
            f1=f1((p >= cut).astype(int), target),
            threshold={m: threshold[m] for m in present} if isinstance(threshold, Mapping) else threshold,
            positive_rate=float(target.mean()),
            brier=brier_binary(p, target),
            ece=ece_binary(p, target),
            reliability={"confidence": "positive-class probability", **reliability(p, target.astype(np.float64))},
        )
        if t is not None:
            entry["ece_calibrated"] = ece_binary(sigmoid(raw / t), target)
    elif spec.kind == "categorical":
        probs = softmax(raw)
        pred = probs.argmax(-1)
        counts = np.bincount(target, minlength=len(spec.classes))
        entry.update(
            accuracy=float((pred == target).mean()),
            balanced_accuracy=balanced_accuracy(pred, target),
            macro_f1=macro_f1(pred, target, len(spec.classes)),
            classes_present=int((counts > 0).sum()),
            class_counts={spec.classes[c]: int(n) for c, n in enumerate(counts) if n},
            brier=brier_multiclass(probs, target),
            ece=ece_top_label(probs, target),
            reliability={
                "confidence": "top-label probability",
                **reliability(probs.max(-1), (pred == target).astype(np.float64)),
            },
        )
        if trained_classes is not None:
            untrained = [c for c in np.flatnonzero(counts) if c not in set(trained_classes)]
            if untrained:
                entry["untrained_classes"] = [spec.classes[c] for c in untrained]
                entry["macro_f1_trained_classes"] = macro_f1(pred, target, len(spec.classes), only=sorted(trained_classes))
        if t is not None:
            entry["ece_calibrated"] = ece_top_label(softmax(raw / (t[:, None] if isinstance(t, np.ndarray) else t)), target)
    else:
        values = regression_value(raw, spec.value_range)
        entry["mae"] = float(np.abs(values - target).mean())
        if spec.unit:
            entry.update(mae_in_unit=entry["mae"] * spec.unit_scale, unit=spec.unit)

    if prior is not None:
        entry["baseline"] = baseline(spec, target, prior)

    draws = None
    if bootstrap and entry["n_subjects"] >= MIN_CI_SUBJECTS:
        draws = cluster_bootstrap(lambda rows: headline(spec, raw[rows], target[rows]), subjects, bootstrap, seed)
        entry["ci"] = interval(draws)
    return entry, draws


def baseline(spec: LabelSpec, target: np.ndarray, prior: dict[str, Any]) -> dict[str, Any]:
    """Score of a constant predictor fitted on the training split (see `training_prior`)."""
    if spec.kind == "binary":
        majority = int(prior["positive_rate"] >= 0.5)
        return {
            "auroc": 0.5,
            "f1": f1(np.full(len(target), majority), target),
            "predicts": majority,
            "source": prior["source"],
        }
    if spec.kind == "categorical":
        pred = np.full(len(target), prior["majority"])
        return {
            "accuracy": float((pred == target).mean()),
            "macro_f1": macro_f1(pred, target, len(spec.classes)),
            "predicts": spec.classes[prior["majority"]],
            "source": prior["source"],
        }
    mae = float(np.abs(prior["mean"] - target).mean())
    out: dict[str, Any] = {"mae": mae, "predicts": prior["mean"], "source": prior["source"]}
    if spec.unit:
        out["mae_in_unit"] = mae * spec.unit_scale
    return out


def training_prior(records: Sequence[ClipRecord], spec: LabelSpec, source: str) -> dict[str, Any] | None:
    """Constant-predictor statistics from ground-truth training labels, or None if there are none."""
    data = label_rows(records, spec)
    if data is None:
        return None
    if spec.kind == "binary":
        return {"positive_rate": float(data.target.mean()), "source": source}
    if spec.kind == "categorical":
        return {"majority": int(np.bincount(data.target).argmax()), "source": source}
    return {"mean": float(data.target.mean()), "source": source}


# ---- many labels ---------------------------------------------------------------------------


def score_outputs(
    logits: dict[str, np.ndarray],
    records: Sequence[ClipRecord],
    taxonomy: Taxonomy,
    *,
    labels: Sequence[str] | None = None,
    ground_truth_only: bool = True,
    thresholds: Mapping[str, float | Mapping[str, float]] | None = None,
    temperatures: Mapping[str, float | Mapping[str, float | None] | None] | None = None,
    priors: dict[str, dict[str, Any]] | None = None,
    trained_classes: Mapping[str, Sequence[int]] | None = None,
    bootstrap: int = 0,
    seed: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Metrics per label for outputs aligned with `records`. Returns (entries, bootstrap draws).

    `thresholds` and `temperatures` give, per label, one value or a {media type: value}
    mapping (see `score_label`); a label without a threshold is scored at 0.5.
    """
    entries: dict[str, dict[str, Any]] = {}
    draws: dict[str, np.ndarray] = {}
    for name in labels if labels is not None else taxonomy.labels:
        spec = taxonomy[name]
        if name not in logits:
            continue
        data = label_rows(records, spec, ground_truth_only=ground_truth_only)
        if data is None:
            continue
        entry, label_draws = score_label(
            spec,
            logits[name],
            data,
            threshold=(thresholds or {}).get(name, 0.5),
            temperature=(temperatures or {}).get(name),
            prior=(priors or {}).get(name),
            trained_classes=(trained_classes or {}).get(name),
            bootstrap=bootstrap,
            seed=seed,
        )
        entries[name] = entry
        if label_draws is not None:
            draws[name] = label_draws
    return entries, draws


def evaluate_outputs(
    outputs: Sequence[dict[str, torch.Tensor]],
    records: Sequence[ClipRecord],
    taxonomy: Taxonomy,
    *,
    ground_truth_only: bool = True,
) -> dict[str, dict[str, Any]]:
    """Metrics per label over batches of raw logits. By default only ground-truth labels are scored."""
    merged = {
        name: torch.cat([o[name] for o in outputs]).detach().float().cpu().numpy()
        for name in taxonomy.labels
        if outputs and name in outputs[0]
    }
    entries, _ = score_outputs(merged, records, taxonomy, ground_truth_only=ground_truth_only)
    return entries
