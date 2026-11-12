"""Evaluation report for any Bulwark model: frozen heads, B1, E1 or the exported ONNX graph.

The report reads per-split predictions (see `bulwark.eval.predict`) and scores only
ground-truth labels. For each held-out split (test, holdout) it gives, per label:

* one column per dataset, plus an unweighted mean over datasets (pooling datasets would
  let a model score by telling datasets apart) and a pooled column only where that is safe;
* clip, subject and independent-event counts, a constant-predictor baseline fitted on the
  training split, and a 95% subject-level bootstrap interval where there are enough subjects;
* F1 at a threshold tuned on val, and calibration (ECE, Brier, reliability bins) before and
  after a temperature fitted on val. Both are fitted per media type (image or video), since
  a clip model's confidence differs between still images and video clips: each clip is
  scored with the threshold and temperature fitted on the val clips of its own media type
  (see `calibrate`);
* per-event headlines for datasets that film each event from several cameras (each event
  counts once), per-video scores for datasets labelled per video, and per-clip slices by
  view and condition.

Each holdout group is scored on its own (a new camera and new drivers is a different
question from a dataset never trained on), and a holdout dataset that is not in the
manifest is listed as not available. Labels with no ground-truth training data are not
scored; they are listed with their teacher-label counts. Neither are labels the checkpoint
records as untrained (not in its `supervised_labels`), and labels it masked on a media
type get a warning. Dataset-constant metadata labels go to a separate sanity table. The
manifest must pass the split leak check (`bulwark.data.splits.check_disjoint`, under the
split policy the manifest records, else the code default) before anything is scored. A
split scores only the predicted clips the manifest puts in it: predictions saved against
an earlier split are re-scored without counting clips that are now `excluded`, train or
another split (they are counted and reported instead).
"""

from __future__ import annotations

import dataclasses
import html
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from bulwark.data.manifest import MEDIA_TYPES, ClipRecord, left_out_datasets, read_manifest
from bulwark.data.splits import SplitPolicy, check_disjoint
from bulwark.data.taxonomy import LabelSpec, Taxonomy, default_taxonomy
from bulwark.eval import metrics as M
from bulwark.eval.predict import (
    SHARD_KINDS,
    Predictions,
    detect_kind,
    predict_features,
    predict_shards,
    sample_ids,
    write_predictions,
)
from bulwark.eval.provenance import (
    FEATURES_META,
    checkpoint_meta,
    features_init,
    features_meta,
    features_summary,
    public_path,
    run_meta,
    sha256_file,
    split_policy,
    taxonomy_digest,
    training_scope,
)
from bulwark.model.bulwark import load_payload

SPLITS = ("test", "holdout")            # reported
EVAL_SPLITS = ("val", "test", "holdout")  # predicted; val only tunes thresholds and temperatures
HEADLINE = M.HEADLINE
SLICE_FIELDS: dict[str, Callable[[ClipRecord], Any]] = {
    "view": lambda r: r.view or None,
    "modality": lambda r: r.labels_gt.get("modality"),
    "low_light": lambda r: r.labels_gt.get("low_light"),
    "eyeglasses": lambda r: r.labels_gt.get("eyeglasses"),
    "sunglasses": lambda r: r.labels_gt.get("sunglasses"),
    "face_occluded": lambda r: r.labels_gt.get("face_occluded"),
    "gender": lambda r: r.meta.get("gender"),
}
VIDEO_LEVEL = "video_level_self_report"  # meta label_quality of datasets labelled per video
# Binary labels a dataset gives per video through its file names: clips of a video whose
# name says the behaviour happens carry this meta flag (YawDD "Yawning" videos), while the
# other videos' clips have the label as 0 in labels_gt. The behaviour fills only part of a
# positive video, so the video is scored on its highest clip output.
VIDEO_TARGETS = {"yawn": "yawn_video"}
# A threshold or temperature is fitted only on val clips holding at least this many clips
# of each of two classes (see `_fit`); model selection likewise skips labels with fewer
# than MIN_SELECTION_SAMPLES (10) val clips.
MIN_FIT_CLIPS = 10
DEFINITIONS = {
    "auroc": "AUROC on raw logits (binary labels).",
    "f1": "F1 of the positive class, each clip at the threshold that maximised F1 on the val clips of its media type "
    f"(image or video); 0.5 where those val clips hold fewer than {MIN_FIT_CLIPS} clips of either class, or val has no "
    "clips of that media type for the label.",
    "macro_f1": "Macro-F1 over classes present in labels or predictions (sklearn's default); undefined with one class.",
    "macro_f1_trained_classes": "Macro-F1 averaged over the classes that have ground-truth training labels only "
    "(sklearn's labels=); given where the evaluated labels include classes the model was never trained on.",
    "mae": "Mean absolute error in the normalised label range; head and gaze angles also in degrees, KSS in KSS points.",
    "unit": "Per dataset, the headline counts each independent event once. Datasets that record one event from several "
    "cameras (event ids in the manifest) are scored on one prediction per event, the views' probabilities averaged; "
    "the per-clip table counts every view. Condition slices are per clip.",
    "summary": "Unweighted mean of the per-dataset headline over datasets where it is defined (labels single-class within a dataset are left out).",
    "pooled": "All datasets pooled (per event where events are recorded), each clip at the threshold and temperature of its "
    "media type; shown only when every contributing dataset defines the metric on its own.",
    "baseline": "Constant predictor from ground-truth training labels: the dataset's own when it has training data, else all datasets'.",
    "ci": "95% percentile interval from a subject-level cluster bootstrap; only with at least "
    f"{M.MIN_CI_SUBJECTS} subjects. Results from fewer than {M.MIN_SUBJECTS} subjects are marked indicative.",
    "calibration": f"ECE over {M.ECE_BINS} equal-width bins (positive-class probability for binary labels, "
    "top-label confidence for categorical); after = with the label's temperature fitted on the val clips of each clip's "
    f"media type (not given when a scored media type has none: val holds no clips of it, or fewer than {MIN_FIT_CLIPS} "
    "clips of each of two classes). metrics.json holds the per-bin reliability data.",
    "events": "Independent events: clips of the same moment from several cameras count once (when the dataset records event ids).",
    "ground_truth": "Only ground-truth labels are scored; teacher pseudo-labels and labels marked weak never are.",
    "video_max": "Per video from file-name labels (YawDD yawn): a video is positive when its file name says the "
    "behaviour happens in it; it is scored on the highest clip logit, since the behaviour fills only part of the video.",
    "holdout": "Each holdout group is scored on its own and not averaged with the others: a held-out camera view "
    "(its test-split subjects only, so camera and driver are both new) and datasets never trained on.",
    "official_splits": "Datasets with an official subject split are split by one protocol (e.g. Drive&Act split_0). "
    "The other protocols are not scored: their test subjects are training subjects of the model. "
    "`bulwark probe drive_and_act_midlevel` covers splits 0-2 with a probe per split.",
}


def is_metadata(spec: LabelSpec) -> bool:
    """A label the taxonomy flags `metadata` (camera, lighting, setting), scored only as a sanity check."""
    return spec.metadata


# ---- training split -------------------------------------------------------------------------


def training_records(records: Sequence[ClipRecord], scope: dict[str, Any]) -> list[ClipRecord]:
    """The training split the model saw, narrowed by its recorded `commercial_only` / `datasets`."""
    train = [r for r in records if r.split == "train"]
    if scope.get("commercial_only"):
        train = [r for r in train if r.commercial_ok]
    if scope.get("datasets"):
        train = [r for r in train if r.dataset in scope["datasets"]]
    return train


def training_support(train: Sequence[ClipRecord], taxonomy: Taxonomy) -> dict[str, dict[str, Any]]:
    """Ground-truth and teacher clip counts per label (and per class) in the training split."""
    gt = Counter(k for r in train for k in r.labels_gt if k not in r.meta.get("weak_labels", ()))
    teacher = Counter(k for r in train for k in r.labels_teacher if k not in r.labels_gt)
    support: dict[str, dict[str, Any]] = {}
    for name, spec in taxonomy.labels.items():
        entry: dict[str, Any] = {"gt": gt[name], "teacher": teacher[name]}
        entry["status"] = "gt" if gt[name] else ("teacher_only" if teacher[name] else "none")
        if spec.kind == "categorical":
            classes = Counter(
                str(r.labels_gt[name]) for r in train if name in r.labels_gt and name not in r.meta.get("weak_labels", ())
            )
            entry["classes_gt"] = {c: classes[c] for c in spec.classes}
        support[name] = entry
    return support


def checkpoint_labels(recorded: dict[str, Any]) -> dict[str, Any] | None:
    """The outputs a checkpoint records as trained (`supervised_labels`) and, per label, the
    media types training masked it on (`masked_labels`), or None when it records neither."""
    supervised, masked = recorded.get("supervised_labels"), recorded.get("masked_labels")
    if supervised is None and not masked:
        return None
    return {"supervised_labels": supervised, "masked_labels": dict(masked or {})}


def trained_classes(support: dict[str, dict[str, Any]], taxonomy: Taxonomy) -> dict[str, list[int]]:
    """Class indices with ground-truth training labels, per categorical label."""
    return {
        name: [taxonomy[name].class_index(c) for c, n in entry["classes_gt"].items() if n]
        for name, entry in support.items()
        if "classes_gt" in entry
    }


def _priors(train: Sequence[ClipRecord], taxonomy: Taxonomy, labels: Sequence[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """{dataset or '*': {label: prior}}: per-dataset training priors with a pooled fallback."""
    out: dict[str, dict[str, dict[str, Any]]] = {"*": {}}
    for name in labels:
        prior = M.training_prior(train, taxonomy[name], "train:all")
        if prior is not None:
            out["*"][name] = prior
    for dataset in sorted({r.dataset for r in train}):
        subset = [r for r in train if r.dataset == dataset]
        out[dataset] = {}
        for name in labels:
            prior = M.training_prior(subset, taxonomy[name], f"train:{dataset}")
            if prior is not None:
                out[dataset][name] = prior
    return out


def _dataset_priors(priors: dict[str, dict[str, dict[str, Any]]], dataset: str) -> dict[str, dict[str, Any]]:
    return {**priors["*"], **priors.get(dataset, {})}


class _Calibration(TypedDict):
    """The calibration keywords of `M.score_outputs` that one split's scores share:
    per label, {media type: threshold} and {media type: temperature or None}."""

    thresholds: dict[str, dict[str, float]]
    temperatures: dict[str, dict[str, float | None]]
    trained_classes: dict[str, list[int]] | None


class _Bootstrap(TypedDict):
    """The bootstrap keywords of `M.score_outputs`."""

    bootstrap: int
    seed: int


# ---- which predictions a split scores ------------------------------------------------------------


@dataclasses.dataclass
class SplitRows:
    """The predicted clips the manifest puts in one split, and those it does not."""

    clip_ids: list[str]
    not_in_manifest: int
    other_splits: dict[str, int]  # split the manifest gives them -> clips


def in_split(preds: Predictions, records: dict[str, ClipRecord], split: str) -> SplitRows:
    """Predicted clips that the manifest puts in `split`. Clips it puts elsewhere (re-split,
    `excluded`, train) or does not hold at all are counted, never scored."""
    keep = [c for c in preds.clip_ids if c in records and records[c].split == split]
    other = Counter(records[c].split for c in preds.clip_ids if c in records and records[c].split != split)
    missing = sum(1 for c in preds.clip_ids if c not in records)
    return SplitRows(keep, missing, dict(sorted(other.items())))


def _rows_warning(split: str, rows: SplitRows) -> str | None:
    parts = []
    if rows.other_splits:
        where = ", ".join(f"{s} {n}" for s, n in rows.other_splits.items())
        parts.append(f"{sum(rows.other_splits.values())} are in other splits of this manifest ({where})")
    if rows.not_in_manifest:
        parts.append(f"{rows.not_in_manifest} are not in it")
    if not parts:
        return None
    use = "fitted on" if split == "val" else "scored"
    return f"Of the clips predicted as {split}, {' and '.join(parts)}; they are not {use}."


# ---- val: thresholds and temperatures -----------------------------------------------------------


def calibrate(preds: Predictions | None, records: dict[str, ClipRecord], taxonomy: Taxonomy, labels: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per-label F1 threshold and temperature from ground-truth val labels, on all val clips
    and per media type.

    Each binary and categorical label gets an entry fitted on all its val clips (the top
    level, as calibration.json has always held it) and, under `media`, one entry per media
    type with val ground truth for the label, fitted the same way on that media type's val
    clips alone (sorted keys: "image", "video"). A clip model's confidence differs between
    still images and video clips, so a threshold fitted on all val clips can suit neither.
    `media` is present (possibly empty) for every binary and categorical label, so a reader
    can tell "no val clips of this media type" from a file written before media entries.
    Regression labels get the default entry only. See `_fit` for the fit itself.

    Only clips the manifest puts in val are used, so predictions saved against an earlier
    split never fit thresholds on clips that are now test or holdout clips.
    """
    out: dict[str, dict[str, Any]] = {}
    if preds is not None:
        keep = in_split(preds, records, "val").clip_ids
        preds = preds.select(keep)
        val = [records[c] for c in keep]
    for name in labels:
        spec = taxonomy[name]
        entry: dict[str, Any] = {"threshold": 0.5, "threshold_from": "default", "temperature": None, "source": "default"}
        if spec.kind != "regression":
            media: dict[str, dict[str, Any]] = {}
            data = M.label_rows(val, spec) if preds is not None and name in preds.logits else None
            if preds is not None and data is not None:
                raw = preds.logits[name][data.rows].astype(np.float64)
                ok = M.finite_rows(raw)
                raw, target, subjects, kinds = raw[ok], data.target[ok], data.subjects[ok], data.media[ok]
                entry = _fit(spec, raw, target, subjects)
                for kind in sorted(set(data.media.tolist())):
                    rows = kinds == kind
                    media[kind] = _fit(spec, raw[rows], target[rows], subjects[rows])
            entry["media"] = media
        out[name] = entry
    return out


def _fit(spec: LabelSpec, raw: np.ndarray, target: np.ndarray, subjects: np.ndarray) -> dict[str, Any]:
    """One calibration entry from finite val logits: the temperature that minimises the NLL
    and, for binary labels, the F1-maximising threshold on sigmoid(logit).

    Both are fitted only when the rows hold at least MIN_FIT_CLIPS clips of each of two
    classes. Otherwise the entry keeps the 0.5 threshold (threshold_from "default") and no
    temperature, and `not_fitted` says why: on one class the F1-best threshold is the
    lowest probability, which flags everything, and on a handful of clips both rest on
    almost nothing (a few separable clips drive the temperature to the edge of its grid).
    The counts say what the fit rests on: `n_val` rows (every camera view counted),
    `n_val_positive` (binary) and `n_val_subjects`."""
    entry: dict[str, Any] = {"threshold": 0.5, "threshold_from": "default", "temperature": None, "source": "val", "n_val": int(len(target))}
    supported = int((np.bincount(target) >= MIN_FIT_CLIPS).sum()) >= 2
    if supported:
        entry["temperature"] = M.fit_temperature(raw, target, spec.kind)
        if spec.kind == "binary":
            entry.update(threshold=M.best_threshold(M.sigmoid(raw), target), threshold_from="val")
    if spec.kind == "binary":
        entry["n_val_positive"] = int(target.sum())
    entry["n_val_subjects"] = len(set(subjects.tolist()))
    if not supported:
        entry["not_fitted"] = f"fewer than {MIN_FIT_CLIPS} clips of each of two classes"
    return entry


# Where val has no clips of a media type for a label, nothing was fitted for it there.
_NOT_FITTED: dict[str, Any] = {"threshold": 0.5, "threshold_from": "default", "temperature": None}


def calibration_for(entry: Mapping[str, Any], media_type: str) -> Mapping[str, Any]:
    """A label's calibration.json entry for clips of `media_type`.

    Its `media` entry for that media type; with a `media` section but no entry for it (val
    had no clips of that media type for the label), the default 0.5 threshold and no
    temperature, never the fit on all val clips, which is then a fit on other media alone.
    An entry without `media` (a file written before media entries, or a regression label)
    applies to every media type.
    """
    if "media" not in entry:
        return entry
    return entry["media"].get(media_type, _NOT_FITTED)


def _media_options(calibration: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float | None]]]:
    """Per label, {media type: threshold} and {media type: temperature or None} for every
    media type, as `calibration_for` chooses them."""
    thresholds: dict[str, dict[str, float]] = {}
    temperatures: dict[str, dict[str, float | None]] = {}
    for name, entry in calibration.items():
        chosen = {kind: calibration_for(entry, kind) for kind in sorted(MEDIA_TYPES)}
        thresholds[name] = {kind: float(e["threshold"]) for kind, e in chosen.items()}
        temperatures[name] = {kind: e.get("temperature") for kind, e in chosen.items()}
    return thresholds, temperatures


# ---- scoring groups -------------------------------------------------------------------------


def _subset(preds: Predictions, rows: Sequence[int]) -> dict[str, np.ndarray]:
    index = np.asarray(rows, dtype=np.int64)
    return {k: v[index] for k, v in preds.logits.items()}


def _aggregate(
    logits: dict[str, np.ndarray], records: list[ClipRecord], taxonomy: Taxonomy, key: Callable[[ClipRecord], str]
) -> tuple[dict[str, np.ndarray], list[ClipRecord]]:
    """Average clip outputs (as probabilities or values) over clips sharing a key, e.g. all
    clips of one video, or all camera views of one event. Returns logits and one record per
    group: its first clip's, whose media type then chooses the group's threshold and
    temperature, so a group that mixes media types is refused."""
    groups: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        groups.setdefault(key(record), []).append(i)
    for group, members in groups.items():
        kinds = sorted({records[i].media_type for i in members})
        if len(kinds) > 1:
            raise ValueError(f"{group} groups clips of several media types ({', '.join(kinds)}); they cannot be scored as one")
    eps = 1e-7
    out: dict[str, np.ndarray] = {}
    for name, value in logits.items():
        spec = taxonomy[name]
        rows = []
        for members in groups.values():
            v = value[members].astype(np.float64)
            if spec.kind == "categorical":
                rows.append(np.log(np.clip(M.softmax(v).mean(0), eps, 1.0)))
            else:
                if spec.kind == "binary":
                    p = M.sigmoid(v).mean()
                else:
                    low, high = spec.value_range
                    p = (M.regression_value(v, spec.value_range).mean() - low) / (high - low)
                p = float(np.clip(p, eps, 1 - eps))
                rows.append(np.log(p / (1 - p)))
        out[name] = np.array(rows)
    return out, [records[members[0]] for members in groups.values()]


def _video_max(
    logits: np.ndarray, records: Sequence[ClipRecord], name: str, flag: str
) -> tuple[dict[str, np.ndarray], list[ClipRecord]] | None:
    """One row per video with a file-name target for `name` (see VIDEO_TARGETS): the
    highest finite clip logit, and a record carrying the video's label. None when no clip
    has a target for `name`; with one class only, the metrics come out undefined (None)."""
    videos: dict[str, tuple[float, list[int]]] = {}
    for i, record in enumerate(records):
        if record.meta.get(flag):
            target = 1.0
        elif name in record.labels_gt and name not in record.meta.get("weak_labels", ()):
            target = float(float(record.labels_gt[name]) >= 0.5)
        else:
            continue
        videos.setdefault(record.media_path, (target, []))[1].append(i)
    rows, out = [], []
    for media_path, (target, members) in videos.items():
        values = logits[members].astype(np.float64)
        values = values[np.isfinite(values)]
        rows.append(float(values.max()) if len(values) else math.nan)
        first = records[members[0]]
        out.append(dataclasses.replace(first, clip_id=media_path, labels_gt={name: target}, meta={}))
    if not out:
        return None
    return {name: np.array(rows)}, out


def _counts(records: Sequence[ClipRecord]) -> dict[str, Any]:
    return {
        "clips": len(records),
        "subjects": len({r.subject_id for r in records}),
        "events": len({M.event_id(r) for r in records}),
        "event_ids_recorded": all("event_id" in r.meta for r in records),
    }


def _summary(per_dataset: dict[str, dict[str, Any]], draws: dict[str, dict[str, np.ndarray]], taxonomy: Taxonomy, labels: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Unweighted mean of each label's headline over the datasets that define it."""
    out: dict[str, dict[str, Any]] = {}
    for name in labels:
        metric = HEADLINE[taxonomy[name].kind]
        used = [d for d, entries in per_dataset.items() if entries.get(name, {}).get(metric) is not None]
        if not used:
            continue
        entries = [per_dataset[d][name] for d in used]
        entry: dict[str, Any] = {
            "metric": metric,
            "value": float(np.mean([e[metric] for e in entries])),
            "datasets": used,
            "n": sum(e["n"] for e in entries),
            "n_subjects": sum(e["n_subjects"] for e in entries),
            "n_events": sum(e["n_events"] for e in entries),
            "indicative": any(e.get("indicative") for e in entries),
        }
        baselines = [e.get("baseline", {}).get(metric) for e in entries]
        if all(b is not None for b in baselines):
            entry["baseline"] = {metric: float(np.mean(baselines))}
        untrained = sorted({c for e in entries for c in e.get("untrained_classes", ())})
        if untrained:
            # Without the flag, a mean that includes never-trained classes would read as a plain result.
            entry["untrained_classes"] = untrained
            trained_only = [e.get("macro_f1_trained_classes", e.get(metric)) for e in entries]
            defined = [v for v in trained_only if v is not None]
            entry["macro_f1_trained_classes"] = float(np.mean(defined)) if defined else None
        if all(name in draws.get(d, {}) for d in used):
            # Datasets are resampled independently (stratified by dataset).
            entry["ci"] = M.interval(np.mean([draws[d][name] for d in used], axis=0))
        out[name] = entry
    return out


def _agreement(spec: LabelSpec, model: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """How closely a model's outputs follow a reference model's (e.g. E1 against B1)."""
    model, reference = model.astype(np.float64), reference.astype(np.float64)
    ok = M.finite_rows(model) & M.finite_rows(reference)
    model, reference = model[ok], reference[ok]
    if not len(model):
        return {"n": 0}
    if spec.kind == "binary":
        pm, pr = M.sigmoid(model), M.sigmoid(reference)
        return {"n": int(len(pm)), "mean_abs_prob_diff": float(np.abs(pm - pr).mean()), "decision_agreement": float(((pm >= 0.5) == (pr >= 0.5)).mean())}
    if spec.kind == "categorical":
        pm, pr = M.softmax(model), M.softmax(reference)
        kl = (pr * (np.log(np.clip(pr, 1e-12, 1)) - np.log(np.clip(pm, 1e-12, 1)))).sum(-1)
        return {"n": int(len(pm)), "top1_agreement": float((pm.argmax(-1) == pr.argmax(-1)).mean()), "kl_reference_to_model": float(kl.mean())}
    diff = float(np.abs(M.regression_value(model, spec.value_range) - M.regression_value(reference, spec.value_range)).mean())
    entry: dict[str, Any] = {"n": int(len(model)), "mean_abs_diff": diff}
    if spec.unit:
        entry["mean_abs_diff_in_unit"] = diff * spec.unit_scale
    return entry


def score_split(
    preds: Predictions,
    records: dict[str, ClipRecord],
    taxonomy: Taxonomy,
    *,
    split: str,
    scored: Sequence[str],
    sanity: Sequence[str],
    calibration: dict[str, dict[str, Any]],
    priors: dict[str, dict[str, dict[str, Any]]],
    trained: dict[str, list[int]] | None = None,
    reference: Predictions | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
    combine: bool = True,
    video_targets: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Every table of one split: per dataset (per event where several cameras saw each
    event), mean over datasets and pooled (unless `combine` is False, as for holdout groups
    that answer different questions), per clip, per video, condition slices, metadata
    sanity checks and, optionally, a reference model. `video_targets` are the
    VIDEO_TARGETS entries to score per video. Only predicted clips that the manifest puts
    in `split` are scored (see `in_split`).

    `calibration` is the "labels" section of calibration.json (see `calibrate`). Every
    calibrated table (per dataset, per event, per clip, per video, slices, pooled) scores
    each clip with the threshold and temperature of its record's media type
    (`calibration_for`); an entry without `media` applies to every media type. The sanity
    checks, the video maxima and the reference comparison are not calibrated."""
    selected = in_split(preds, records, split)
    keep = selected.clip_ids
    preds = preds.select(keep)
    split_records = [records[c] for c in keep]
    thresholds, temperatures = _media_options(calibration)
    options: _Calibration = {"thresholds": thresholds, "temperatures": temperatures, "trained_classes": trained}

    datasets = sorted({r.dataset for r in split_records})
    rows_of = {d: [i for i, r in enumerate(split_records) if r.dataset == d] for d in datasets}
    result: dict[str, Any] = {
        "counts": {d: _counts([split_records[i] for i in rows]) for d, rows in rows_of.items()},
        "units": {},
        "datasets": {},
        "per_clip": {},
        "sanity": {},
        "videos": {},
        "video_max": {},
        "slices": {},
    }
    if selected.not_in_manifest:
        result["predictions_not_in_manifest"] = selected.not_in_manifest
    if selected.other_splits:
        result["predictions_in_other_splits"] = selected.other_splits
    draws: dict[str, dict[str, np.ndarray]] = {}
    headline_parts: list[tuple[dict[str, np.ndarray], list[ClipRecord]]] = []
    for index, (dataset, rows) in enumerate(rows_of.items()):
        group = [split_records[i] for i in rows]
        logits = _subset(preds, rows)
        dataset_priors = _dataset_priors(priors, dataset)
        boot: _Bootstrap = {"bootstrap": bootstrap, "seed": seed + index}  # independent draws per dataset
        if all("event_id" in r.meta for r in group) and len({M.event_id(r) for r in group}) < len(group):
            # Several cameras saw each event: the headline scores one fused prediction per
            # event, so an event weighs the same however many views recorded it.
            result["per_clip"][dataset], _ = M.score_outputs(logits, group, taxonomy, labels=scored, priors=dataset_priors, **options)
            level_logits, level_records = _aggregate(logits, group, taxonomy, M.event_id)
            result["units"][dataset] = "event"
        else:
            level_logits, level_records = logits, group
            result["units"][dataset] = "clip"
        entries, draws[dataset] = M.score_outputs(level_logits, level_records, taxonomy, labels=scored, priors=dataset_priors, **boot, **options)
        result["datasets"][dataset] = entries
        headline_parts.append((level_logits, level_records))
        result["sanity"][dataset], _ = M.score_outputs(logits, group, taxonomy, labels=sanity)
        if any(r.meta.get("label_quality") == VIDEO_LEVEL for r in group):
            video_logits, videos = _aggregate(logits, group, taxonomy, lambda r: r.media_path)
            result["videos"][dataset], _ = M.score_outputs(video_logits, videos, taxonomy, labels=scored, priors=dataset_priors, **boot, **options)
        for name, flag in (video_targets or {}).items():
            found = _video_max(logits[name], group, name, flag) if name in logits and any(r.meta.get(flag) for r in group) else None
            if found is not None:
                # The val threshold and temperature were fitted on clips, not on video maxima.
                entries, _ = M.score_outputs(found[0], found[1], taxonomy, labels=[name], **boot)
                result["video_max"].setdefault(dataset, {}).update(entries)
        slices: dict[str, Any] = {}
        for field, getter in SLICE_FIELDS.items():
            values = [getter(r) for r in group]
            distinct = sorted({str(v) for v in values if v is not None})
            if len(distinct) < 2:
                continue
            for value in distinct:
                picks = [i for i, v in enumerate(values) if v is not None and str(v) == value]
                sub = {k: v[picks] for k, v in logits.items()}
                slices[f"{field}={value}"], _ = M.score_outputs(
                    sub, [group[i] for i in picks], taxonomy, labels=scored, priors=dataset_priors, **options
                )
        if slices:
            result["slices"][dataset] = slices

    result["summary"] = _summary(result["datasets"], draws, taxonomy, scored) if combine else {}
    if combine and len(datasets) > 1:
        poolable = []
        for name in scored:
            metric = HEADLINE[taxonomy[name].kind]
            contributing = [entries[name] for entries in result["datasets"].values() if name in entries]
            if len(contributing) > 1 and all(e.get(metric) is not None for e in contributing):
                poolable.append(name)
        pooled_logits = {k: np.concatenate([part[k] for part, _ in headline_parts]) for k in headline_parts[0][0]}
        pooled_records = [r for _, part in headline_parts for r in part]
        result["pooled"], _ = M.score_outputs(pooled_logits, pooled_records, taxonomy, labels=poolable, priors=priors["*"], **options)

    if reference is not None:
        reference_ids = set(reference.clip_ids)
        shared = [c for c in keep if c in reference_ids]
        ref = reference.select(shared)
        mine = preds.select(shared)
        shared_records = [records[c] for c in shared]
        ref_section: dict[str, Any] = {"clips": len(shared), "datasets": {}, "agreement": {}}
        for dataset in datasets:
            picks = [i for i, r in enumerate(shared_records) if r.dataset == dataset]
            group = [shared_records[i] for i in picks]
            model_entries, _ = M.score_outputs(_subset(mine, picks), group, taxonomy, labels=scored)
            ref_entries, _ = M.score_outputs(_subset(ref, picks), group, taxonomy, labels=scored)
            ref_section["datasets"][dataset] = {
                name: {
                    "model": model_entries[name].get(HEADLINE[taxonomy[name].kind]),
                    "reference": ref_entries.get(name, {}).get(HEADLINE[taxonomy[name].kind]),
                }
                for name in model_entries
            }
        for name in scored:
            if name in mine.logits and name in ref.logits:
                ref_section["agreement"][name] = _agreement(taxonomy[name], mine.logits[name], ref.logits[name])
        result["reference"] = ref_section
    return result


# ---- the whole report -------------------------------------------------------------------------


def build_report(
    predictions: dict[str, Predictions],
    manifest: Path,
    out: Path,
    *,
    meta: dict[str, Any] | None = None,
    reference: dict[str, Predictions] | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
) -> Path:
    """Score saved predictions against the manifest; writes metrics.json, calibration.json and report.html.

    `meta` comes from the predictions (see `evaluate`); its `predicted_with` block is kept as
    recorded, and the code and manifest used for scoring go under `scored_with`.
    """
    taxonomy = default_taxonomy()
    policy = manifest_policy(manifest)
    all_records = read_manifest(manifest)
    check_splits(all_records, manifest, policy)
    records = {r.clip_id: r for r in all_records}
    meta = dict(meta or {})
    scope = training_scope(meta.get("checkpoint", {}).get("meta", {}))
    train = training_records(all_records, scope)
    support = training_support(train, taxonomy)

    split_rows = {split: in_split(preds, records, split) for split, preds in predictions.items()}
    evaluated = [records[c] for split in SPLITS if split in split_rows for c in split_rows[split].clip_ids]
    present = {k for r in evaluated for k in r.labels_gt}
    labels_record = checkpoint_labels(meta.get("checkpoint", {}).get("meta", {}))
    supervised = (labels_record or {}).get("supervised_labels")
    not_scored = {
        n: f"no ground-truth training labels ({support[n]['teacher']} teacher-labelled training clips)"
        for n in taxonomy.labels if n in present and support[n]["status"] != "gt"
    }
    if supervised is not None:
        # The training split has ground truth, but the run did not train the head (e.g. masked as one-class).
        not_scored.update({
            n: "not trained by this checkpoint (not in its supervised_labels)"
            for n in taxonomy.labels if n in present and support[n]["status"] == "gt" and n not in supervised
        })
    # Labels the checkpoint did not train are left out even without test ground truth, so
    # calibration.json never holds thresholds for them.
    gt_trained = [n for n in taxonomy.labels if support[n]["status"] == "gt" and (supervised is None or n in supervised)]
    scored = [n for n in gt_trained if not is_metadata(taxonomy[n])]
    sanity = [n for n in gt_trained if is_metadata(taxonomy[n])]
    calibration = calibrate(predictions.get("val"), records, taxonomy, scored)
    priors = _priors(train, taxonomy, scored + sanity)

    manifest_entry = manifest_record(manifest, len(all_records))
    present = {r.dataset for r in all_records}
    meta.update(
        scored_with=run_meta(),
        manifest=manifest_entry,
        taxonomy={"version": taxonomy.version, "digest": taxonomy_digest(taxonomy)},
        split_policy=split_policy(manifest),
        training_scope=scope or "not recorded for this model; the whole training split is assumed",
        bootstrap={"draws": bootstrap, "seed": seed, "unit": "subject", "min_subjects": M.MIN_CI_SUBJECTS},
        evaluated_datasets=sorted({r.dataset for r in evaluated}),
        threshold_source="val" if "val" in predictions else "default 0.5 (no val predictions)",
        # Only the datasets this manifest holds, so a policy entry for a dataset that was
        # never ingested (AUC v2 in a manifest without it) does not appear in the header.
        official_splits={
            dataset: f"split_{protocol}"
            for dataset, protocol in policy.official_splits
            if dataset in present
        },
    )
    rows_warnings = [w for split, rows in split_rows.items() if (w := _rows_warning(split, rows))]
    masked = {n: media for n, media in (labels_record or {}).get("masked_labels", {}).items() if n in scored}
    masked_warnings = [
        f"{n} was masked on {', '.join(media)} clips in training; its scores on those clips are of an output not trained on them."
        for n, media in masked.items()
    ]
    meta["warnings"] = [
        *meta.get("warnings", []), *_consistency_warnings(meta), *rows_warnings, *masked_warnings,
        *(_unfitted_warnings(evaluated, taxonomy, calibration, masked) if "val" in predictions else []),
    ]
    results: dict[str, Any] = {
        "meta": meta,
        "definitions": DEFINITIONS,
        "training_support": support,
        "checkpoint_labels": labels_record,
        "not_scored": not_scored,
        "calibration": calibration,
        "coverage": coverage(all_records, predictions),
        "holdout_groups": holdout_groups(all_records, policy, left_out_datasets(manifest)),
        "splits": {},
    }
    trained = trained_classes(support, taxonomy)
    video_targets = {
        n: flag for n, flag in VIDEO_TARGETS.items() if support[n]["status"] != "none" and (supervised is None or n in supervised)
    }
    for split in SPLITS:
        if split not in predictions:
            continue
        results["splits"][split] = score_split(
            predictions[split], records, taxonomy, split=split,
            scored=scored, sanity=sanity, calibration=calibration, priors=priors, trained=trained,
            reference=(reference or {}).get(split), bootstrap=bootstrap, seed=seed,
            combine=split != "holdout", video_targets=video_targets,
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps(calibration_file(meta, calibration), indent=2), encoding="utf-8")
    path = out / "report.html"
    path.write_text(render(results, taxonomy), encoding="utf-8")
    return path


def _unfitted_warnings(
    evaluated: Sequence[ClipRecord], taxonomy: Taxonomy, calibration: dict[str, dict[str, Any]], masked: Mapping[str, Sequence[str]]
) -> list[str]:
    """One warning per scored label and media type whose test or holdout clips have ground
    truth but got no fit: val has no clips of that media type with ground truth for the
    label (AUC v2's image test clips, say, when val holds video only, or a label without
    any val ground truth), or too few to fit (`not_fitted`). Pairs the checkpoint masked
    already get the masked warning. Called only with val predictions; without them the
    header says that every threshold is the default."""
    warnings = []
    for name, entry in calibration.items():
        if "media" not in entry:
            continue  # regression labels: nothing is fitted
        data = M.label_rows(evaluated, taxonomy[name])
        for kind in sorted(set(data.media.tolist()) if data is not None else ()):
            fit = entry["media"].get(kind)
            if kind in masked.get(name, ()) or (fit is not None and "not_fitted" not in fit):
                continue
            held = f"no {kind} clips" if fit is None else f"{fit['not_fitted']} among its {kind} clips"
            warnings.append(
                f"Val has {held} with ground truth for {name}, so no threshold or temperature was fitted "
                f"for its {kind} clips: they are scored at the 0.5 threshold and without a temperature."
            )
    return warnings


def manifest_record(manifest: Path, clips: int) -> dict[str, Any]:
    """The manifest's path, sha256 and clip count; for a copy staged without the datasets
    not cleared for third-party storage, also the rows it left out per dataset."""
    left_out = left_out_datasets(manifest)
    return {
        "path": public_path(manifest), "sha256": sha256_file(manifest), "clips": clips,
        **({"left_out_datasets": left_out} if left_out else {}),
    }


def manifest_policy(manifest: Path) -> SplitPolicy:
    """The split policy the manifest records (see `split_policy`), else the code default.

    The leak check and the holdout groups follow it, as the report header does. Removed
    subjects always include the current removed_subjects.yaml, so a subject removed after
    the manifest was split is refused as well.
    """
    recorded = split_policy(manifest)
    if recorded.get("source") != "manifest":
        return SplitPolicy()
    default = SplitPolicy()
    values: dict[str, Any] = {}
    for item in dataclasses.fields(SplitPolicy):
        if item.name not in recorded:
            continue
        value, current = recorded[item.name], getattr(default, item.name)
        if item.name == "removed_subjects":
            if isinstance(value, (list, tuple)):
                values[item.name] = frozenset(value) | current
        elif isinstance(current, tuple) and isinstance(value, list):
            values[item.name] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        else:
            values[item.name] = value
    try:
        return SplitPolicy(**values)
    except TypeError as exc:
        raise ValueError(f"{manifest} records a split policy this code cannot read: {exc}") from exc


def check_splits(records: Sequence[ClipRecord], manifest: Path, policy: SplitPolicy | None = None) -> None:
    """Refuse to score a manifest that breaks the split policy (a leak between splits,
    a holdout clip in training, a removed subject): its numbers would not mean anything."""
    try:
        check_disjoint(records, policy or SplitPolicy())
    except AssertionError as exc:
        raise ValueError(f"{manifest} breaks the split policy: {exc}. Re-run `bulwark split`.") from exc


def coverage(records: Sequence[ClipRecord], predictions: dict[str, Predictions]) -> dict[str, dict[str, dict[str, int]]]:
    """{split: {dataset: {"manifest": clips, "predicted": clips}}} for every predicted split,
    so clips that were never predicted (not sharded or cached, --limit) are visible.
    `predicted_in_other_splits` counts predicted clips the manifest puts in another split
    (not scored), when there are any."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    by_id = {r.clip_id: r for r in records}
    for split, preds in predictions.items():
        in_manifest = Counter(r.dataset for r in records if r.split == split)
        found = [by_id[c] for c in preds.clip_ids if c in by_id]
        predicted = Counter(r.dataset for r in found if r.split == split)
        elsewhere = Counter(r.dataset for r in found if r.split != split)
        out[split] = {}
        for d in sorted(set(in_manifest) | set(elsewhere)):
            out[split][d] = {"manifest": in_manifest[d], "predicted": predicted[d]}
            if elsewhere[d]:
                out[split][d]["predicted_in_other_splits"] = elsewhere[d]
    return out


def holdout_groups(
    records: Sequence[ClipRecord], policy: SplitPolicy, left_out: Mapping[str, int] | None = None
) -> dict[str, dict[str, Any]]:
    """What each holdout group measures, with its clip and subject counts. Holdout datasets
    with no clips in the manifest are listed as not available rather than left out:
    as not ingested, or, when `left_out` (`bulwark.data.manifest.left_out_datasets`) names
    them, as left out of this staged copy of the manifest. Subject ids are not listed: they
    are part of the dataset, and the report is published with the weights."""
    holdout = [r for r in records if r.split == "holdout"]
    official = dict(policy.official_splits)
    groups: dict[str, dict[str, Any]] = {}
    for dataset, view in policy.holdout_views:
        rows = [r for r in holdout if r.dataset == dataset and r.view == view]
        protocol = f" under official split_{official[dataset]}" if dataset in official else ""
        groups[dataset] = {
            "kind": f"new camera ({view}) and new subjects: the view's clips of the test-split subjects{protocol}",
            "clips": len(rows),
            "subjects": len({r.subject_id for r in rows}),
        }
        if not rows:
            groups[dataset]["status"] = "not available (no clips of this view in the manifest)"
    for dataset in policy.holdout_datasets:
        rows = [r for r in holdout if r.dataset == dataset]
        groups[dataset] = {"kind": "dataset never trained on", "clips": len(rows), "subjects": len({r.subject_id for r in rows})}
        if not rows and dataset in (left_out or {}):
            groups[dataset]["status"] = (
                "not available (not in this manifest: left out of the upload, as its terms do not "
                "clear third-party storage; score it locally)"
            )
        elif not rows:
            groups[dataset]["status"] = "not available (not ingested)"
    return groups


def read_calibration(path: Path, checkpoint_sha256: str | None = None) -> dict[str, float]:
    """Per-label temperatures fitted on all val clips, from the calibration.json `bulwark
    evaluate` writes.

    With `checkpoint_sha256`, a file fitted on other weights is refused: temperatures
    belong to one set of weights. A plain {label: {"temperature": t}} or {label: t}
    mapping is read too, without that check. The values per media type come from
    `bulwark.inference.read_calibration(path).for_media(media_type)`; `bulwark export`
    calls this only to check the weights.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = data
    if isinstance(data.get("meta"), dict) and isinstance(data.get("labels"), dict):
        labels = data["labels"]
        fitted = data["meta"].get("checkpoint") or {}
        if checkpoint_sha256 is not None and checkpoint_sha256 not in {fitted.get("sha256"), fitted.get("exported_from_sha256")}:
            raise ValueError(
                f"{path} was fitted on weights with sha256 {str(fitted.get('sha256'))[:12]}, not these "
                f"({checkpoint_sha256[:12]}); evaluate this checkpoint to get its own temperatures."
            )
    temperatures = {}
    for name, entry in labels.items():
        value = entry.get("temperature") if isinstance(entry, dict) else entry
        if value is None:
            continue
        if not (math.isfinite(float(value)) and float(value) > 0):
            raise ValueError(f"{path}: temperature for {name} must be a positive number, got {value!r}")
        temperatures[name] = float(value)
    return temperatures


def _consistency_warnings(meta: dict[str, Any]) -> list[str]:
    """Differences between how predictions were made and how they are scored."""
    predicted = meta.get("predicted_with")
    if not predicted:
        return ["The predictions record no code version or manifest; they cannot be tied to the run that made them."]
    warnings = []
    made_on, scored_on = (predicted.get("manifest") or {}).get("sha256"), meta["manifest"]["sha256"]
    if made_on != scored_on:
        warnings.append(
            f"Predictions were made against manifest sha256 {str(made_on)[:12]} and are scored against "
            f"{scored_on[:12]}; clips, splits or labels may differ."
        )
    if (predicted.get("taxonomy") or {}).get("digest") not in (None, meta["taxonomy"]["digest"]):
        warnings.append("The taxonomy changed between prediction and scoring.")
    return warnings


def calibration_file(meta: dict[str, Any], calibration: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """calibration.json: per-label thresholds and temperatures, and the model and data they fit.

    Each label's entry holds the fit on all its val clips and, under `media`, one per media
    type (see `calibrate`). A consumer takes the entry of the media type it runs on
    (`calibration_for`, `bulwark.inference.Calibration.for_media`): with a `media` section
    but no entry for that media type, no threshold or temperature applies there; an entry
    without `media` (files written before media entries) applies to every media type.
    `meta.val_clips` counts the val clips behind each label's fit on all val clips.

    Temperatures belong to one set of weights; a consumer should compare `checkpoint.sha256`
    (or, for an ONNX graph, `checkpoint.exported_from_sha256`) with the weights it loads.
    """
    checkpoint = meta.get("checkpoint") or {}
    source = {
        "path": checkpoint.get("path"),
        "sha256": checkpoint.get("sha256"),
        "source": checkpoint.get("source", "file"),
    }
    if meta.get("model") == "onnx":
        source["exported_from_sha256"] = (checkpoint.get("meta") or {}).get("sha256")
    predicted = meta.get("predicted_with") or {}
    return {
        "meta": {
            "model": meta.get("model"),
            "checkpoint": source,
            "manifest_sha256": meta["manifest"]["sha256"],
            "predicted_with_commit": (predicted.get("git") or {}).get("commit"),
            "scored_with_commit": meta["scored_with"]["git"]["commit"],
            "fitted_on": "val" if any(e.get("source") == "val" for e in calibration.values()) else "nothing (no val predictions)",
            "val_clips": {name: e["n_val"] for name, e in calibration.items() if "n_val" in e},
        },
        "labels": calibration,
    }


def evaluate(
    kind: str,
    checkpoint: Path | None,
    *,
    manifest: Path,
    out: Path,
    features: Path | None = None,
    shards: Path | None = None,
    splits: Sequence[str] = EVAL_SPLITS,
    reference: Path | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
    batch_size: int = 16,
    device: str = "auto",
    precision: str = "auto",
    limit: int = 0,
) -> Path:
    """Predict every requested split with one model, save the predictions, then report."""
    from bulwark.eval.predict import load_predictions, read_prediction_meta

    if kind == "auto":
        kind = detect_kind(checkpoint)
    records = read_manifest(manifest)
    check_splits(records, manifest, manifest_policy(manifest))  # before predicting, not only when scoring
    wanted: dict[str, list[str]] = {}
    for split in splits:
        ids = sample_ids((r.clip_id for r in records if r.split == split), limit)
        if ids:
            wanted[split] = ids
    union = {c for ids in wanted.values() for c in ids}

    taxonomy = default_taxonomy()
    meta: dict[str, Any] = {
        "model": kind,
        "splits": list(wanted),
        "limit_per_split": limit,
        "predicted_with": {
            **run_meta(),
            "manifest": manifest_record(manifest, len(records)),
            "taxonomy": {"version": taxonomy.version, "digest": taxonomy_digest(taxonomy)},
            "split_policy": split_policy(manifest),
        },
        "warnings": [],
    }
    if checkpoint is not None:
        meta["checkpoint"] = {"path": public_path(checkpoint), "sha256": sha256_file(checkpoint)}
        if kind == "onnx":
            from bulwark.export.onnx import read_contract

            meta["checkpoint"]["meta"] = read_contract(checkpoint).get("checkpoint", {})
        else:
            meta["checkpoint"]["meta"] = checkpoint_meta(load_payload(checkpoint))
    if kind in SHARD_KINDS:
        if checkpoint is None or shards is None:
            raise ValueError(f"--model {kind} needs --checkpoint and --shards")
        predicted, skipped = predict_shards(kind, checkpoint, shards, union, batch_size=batch_size, device=device, precision=precision)
        meta.update(shards=public_path(shards), skipped_clips=skipped, precision=precision)
    else:
        if features is None:
            raise ValueError(f"--model {kind} needs --features")
        predicted = predict_features(kind, checkpoint, features, union)
        meta["features"] = features_summary(features)
        _features_source(kind, checkpoint, features, meta)
        nonfinite = {c for ids in ((features_meta(features) or {}).get("nonfinite") or {}).values() for c in ids}
        if nonfinite:
            meta["nonfinite_clips"] = {split: len(nonfinite.intersection(ids)) for split, ids in wanted.items()}
    by_split = {split: predicted.select(ids) for split, ids in wanted.items()}
    meta["predicted_clips"] = {split: len(p.clip_ids) for split, p in by_split.items()}
    meta["unpredicted_clips"] = {split: len(wanted[split]) - len(p.clip_ids) for split, p in by_split.items()}
    by_split = {split: p for split, p in by_split.items() if p.clip_ids}
    if not by_split:
        raise ValueError(f"No clip of splits {list(wanted)} could be predicted; check --features / --shards against --manifest")
    write_predictions(by_split, out / "predictions", meta)
    meta["files"] = read_prediction_meta(out / "predictions")["files"]  # the hashes `bulwark report` checks
    ref = load_predictions(reference) if reference else None
    if reference:
        meta["reference"] = public_path(reference)
    return build_report(by_split, manifest, out, meta=meta, reference=ref, bootstrap=bootstrap, seed=seed)


def _features_source(kind: str, checkpoint: Path | None, features: Path, meta: dict[str, Any]) -> None:
    """Record which model a features dir holds (its init checkpoint), or warn that it is unknown.

    `predict_features` has already refused a dir whose provenance contradicts the model.
    """
    recorded = features_meta(features)
    if recorded is None:
        if kind == "logits":
            meta["warnings"].append(
                f"{FEATURES_META} is missing from the features dir, so the checkpoint behind its logits "
                "(and what it was trained on) cannot be checked."
            )
        else:
            meta["warnings"].append(
                f"{FEATURES_META} is missing from the features dir; frozen-encoder tokens are inferred from the absence of logits."
            )
        return
    init = features_init(recorded)
    run = recorded.get("run") or {}
    if kind == "logits" and run.get("with_logits") and not run.get("logit_layout_sha256"):
        meta["warnings"].append(
            "The features dir records no logit layout (cached before logit_layout_sha256 existed), so its "
            "logits cannot be checked against this taxonomy; re-cache them if the taxonomy may have changed."
        )
    if kind == "logits" and checkpoint is None and init:
        # The init identity carries the training scope that the report's support counts need.
        meta["checkpoint"] = {
            "path": public_path(recorded.get("init_path")),
            "sha256": init.get("sha256"),
            "source": f"init recorded in the features dir's {FEATURES_META}",
            "meta": {
                **{k: init.get(k) for k in ("commercial_only", "datasets", "manifest_sha256", "commit")},
                # `bulwark train cache` copies the init's label record next to its logits.
                **{k: recorded[k] for k in ("supervised_labels", "masked_labels") if recorded.get(k) is not None},
            },
        }
    if kind == "logits" and init and not init.get("has_provenance"):
        meta["warnings"].append("The checkpoint behind the cached logits records no provenance; its training data is not known.")


def evaluate_heads(checkpoint: Path, features: Path, manifest: Path, out: Path, *, bootstrap: int = 1000) -> Path:
    """Score a heads checkpoint on cached frozen-encoder features (kept for existing callers)."""
    return evaluate("heads", checkpoint, manifest=manifest, out=out, features=features, bootstrap=bootstrap)


# ---- HTML -----------------------------------------------------------------------------------


def _num(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _cell(entry: dict[str, Any] | None, spec: LabelSpec, metric: str | None = None) -> str:
    if not entry:
        return "<td>–</td>"
    metric = entry.get("metric") or metric or HEADLINE[spec.kind]
    value = entry.get("value", entry.get(metric))
    if value is None:
        return f"<td>– <small>n={entry.get('n', 0)}</small></td>"
    text = _num(value)
    if spec.kind == "regression" and spec.unit:
        text += f" <small>({value * spec.unit_scale:.1f} {spec.unit})</small>"
    if entry.get("ci"):
        low, high = entry["ci"]
        text += f" <small>[{low:.3f}, {high:.3f}]</small>"
    base = entry.get("baseline", {}).get(metric)
    details = [f"base {_num(base)}"] if base is not None else []
    if spec.kind == "binary" and entry.get("f1") is not None:
        details.append(f"F1 {entry['f1']:.3f}")
    details.append(f"{entry.get('n', 0)} clips / {entry.get('n_subjects', 0)} subj")
    if entry.get("n_events") and entry["n_events"] != entry.get("n"):
        details.append(f"{entry['n_events']} events")
    if entry.get("untrained_classes"):
        details.append("classes without training labels: " + ", ".join(entry["untrained_classes"]))
        if "macro_f1_trained_classes" in entry:
            details.append(f"trained classes only {_num(entry['macro_f1_trained_classes'])}")
    small = " · ".join(html.escape(d) for d in details)
    if entry.get("nonfinite"):
        small += f" · <b>{entry['nonfinite']} non-finite outputs excluded</b>"
    mark = " †" if entry.get("indicative") else ""
    return f"<td>{text}{mark}<br><small>{small}</small></td>"


def _table(columns: dict[str, dict[str, Any]], labels: Sequence[str], taxonomy: Taxonomy, metrics: dict[str, str] | None = None) -> str:
    """Labels as rows, groups as columns; `metrics` overrides the headline metric per label kind."""
    metrics = {**HEADLINE, **(metrics or {})}
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    rows = []
    for name in labels:
        metric = metrics[taxonomy[name].kind]
        cells = [_cell(group.get(name), taxonomy[name], metric) for group in columns.values()]
        if any(c != "<td>–</td>" for c in cells):
            rows.append(f"<tr><td>{name} <small>({metric})</small></td>{''.join(cells)}</tr>")
    if not rows:
        return "<p>No scored labels.</p>"
    return f"<table><tr><th>label</th>{head}</tr>{''.join(rows)}</table>"


def render(results: dict[str, Any], taxonomy: Taxonomy | None = None) -> str:
    taxonomy = taxonomy or default_taxonomy()
    meta = results["meta"]
    labels = list(taxonomy.labels)
    checkpoint = meta.get("checkpoint", {})
    predicted = meta.get("predicted_with") or {}
    scored = meta.get("scored_with") or {}
    checkpoint_text = "none"
    if checkpoint:
        checkpoint_text = f"{checkpoint.get('path')} (sha256 {str(checkpoint.get('sha256'))[:12]})"
        if checkpoint.get("source"):
            checkpoint_text += f"; from the {checkpoint['source']}"
    header = [
        ("model", meta.get("model")),
        ("checkpoint", checkpoint_text),
        ("manifest", f"{meta['manifest']['path']} (sha256 {meta['manifest']['sha256'][:12]}, {meta['manifest']['clips']} clips)"),
        ("code (predictions)", f"{_commit(predicted.get('git'))}, {predicted.get('created_utc', 'date not recorded')} UTC"),
        ("code (scoring)", f"{_commit(scored.get('git'))}, {scored.get('created_utc')} UTC"
         + (" (differs from the predictions)" if predicted and _commit(predicted.get("git")) != _commit(scored.get("git")) else "")),
        ("training scope", json.dumps(meta.get("training_scope"))),
        ("split policy", (meta.get("split_policy") or {}).get("source", "not recorded")),
        ("thresholds", meta.get("threshold_source")),
    ]
    recorded = checkpoint.get("meta") or {}
    trained_with = (recorded.get("provenance") or {}).get("code")
    if trained_with:
        header.append(("code (training)", _commit(trained_with)))
    if isinstance(recorded.get("select"), dict):
        header.append(("checkpoint chosen by", _selection(recorded["select"], recorded.get("step"))))
    if meta.get("official_splits"):
        header.append(("official splits", ", ".join(f"{d}: {p}" for d, p in meta["official_splits"].items())))
    header += [("warning", w) for w in meta.get("warnings", [])]
    if meta.get("limit_per_split"):
        header.append(("subset", f"at most {meta['limit_per_split']} clips per split: a smoke check, not a result"))
    missing = {k: v for k, v in meta.get("unpredicted_clips", {}).items() if v}
    if missing:
        header.append(("not predicted", ", ".join(f"{split}: {n} clips (not in features or shards)" for split, n in missing.items())))
    nonfinite = {k: v for k, v in meta.get("nonfinite_clips", {}).items() if v}
    if nonfinite:
        header.append(("non-finite features", ", ".join(
            f"{split}: {n} clips left out of the features dir (non-finite tokens or logits, listed in its {FEATURES_META})"
            for split, n in nonfinite.items()
        )))
    if meta.get("skipped_clips", {}).get("frame_shape"):
        header.append(("skipped", f"{meta['skipped_clips']['frame_shape']} clips whose frame count or size the ONNX graph does not take"))
    parts = [
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Bulwark evaluation</title>",
        "<style>:root{--bg:#fff;--fg:#111;--line:#ddd;--muted:#666}"
        "@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--line:#333;--muted:#999}}"
        "body{font:14px/1.45 system-ui,sans-serif;margin:24px;background:var(--bg);color:var(--fg)}"
        "table{border-collapse:collapse;margin:8px 0 24px;display:block;overflow-x:auto}"
        "td,th{border:1px solid var(--line);padding:4px 8px;text-align:right;vertical-align:top}"
        "th:first-child,td:first-child{text-align:left}small{color:var(--muted)}dt{font-weight:600}</style>",
        "<h1>Bulwark evaluation</h1><table>",
        *(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in header),
        "</table><details><summary>Definitions</summary><dl>",
        *(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in results["definitions"].items()),
        "</dl><p>† fewer than 5 subjects: indicative only.</p></details>",
        _thresholds_table(results["calibration"], _scored_media(results), taxonomy),
    ]
    groups = results.get("holdout_groups") or {}
    for split, section in results["splits"].items():
        parts.append(f"<h2>{html.escape(split)}</h2>")
        counts = section["counts"]
        in_manifest = (results.get("coverage") or {}).get(split, {})
        parts.append("<table><tr><th>dataset</th><th>clips scored</th><th>in manifest</th><th>subjects</th><th>events</th></tr>")
        for dataset, c in counts.items():
            events = c["events"] if c["event_ids_recorded"] else f"{c['events']} (no event ids; = clips)"
            total = in_manifest.get(dataset, {}).get("manifest", "–")
            parts.append(
                f"<tr><td>{html.escape(dataset)}</td><td>{c['clips']}</td><td>{total}</td><td>{c['subjects']}</td><td>{events}</td></tr>"
            )
        if split == "holdout":
            for name, group in groups.items():
                if name not in counts and group.get("status"):  # e.g. UTA-RLDD before it is ingested
                    parts.append(f"<tr><td>{html.escape(name)}</td><td>0</td><td colspan=3>{html.escape(group['status'])}</td></tr>")
        parts.append("</table>")
        elsewhere = section.get("predictions_in_other_splits")
        if elsewhere:
            where = ", ".join(f"{s} {n}" for s, n in elsewhere.items())
            parts.append(f"<p>{sum(elsewhere.values())} predicted clips are in other splits of this manifest ({html.escape(where)}) and are not scored.</p>")
        units = section.get("units", {})
        columns = {"mean over datasets": section["summary"]} if section["summary"] else {}
        for d, entries in section["datasets"].items():
            title = f"{d} (per {units.get(d, 'clip')})"
            if split == "holdout" and d in groups:
                title += f": {groups[d]['kind']}"
            columns[title] = entries
        if section.get("pooled"):
            columns["pooled"] = section["pooled"]
        if split == "holdout":
            parts.append("<p>Each holdout group answers its own question, so they are not averaged or pooled.</p>")
        parts.append(_table(columns, labels, taxonomy))
        if section.get("per_clip"):
            parts.append("<h3>Per clip (every camera view counted; the table above fuses views per event)</h3>")
            parts.append(_table(section["per_clip"], labels, taxonomy))
        if section["videos"]:
            parts.append("<h3>Per video (clip outputs averaged; labels are given per video)</h3>")
            parts.append(_table(section["videos"], labels, taxonomy))
        if section.get("video_max"):
            parts.append("<h3>Per video from file-name labels (highest clip output per video)</h3>")
            parts.append(_table(section["video_max"], labels, taxonomy))
        for dataset, slices in section["slices"].items():
            parts.append(f"<h3>{html.escape(dataset)} by condition</h3>")
            parts.append(_table(slices, labels, taxonomy))
        parts.append("<h3>Calibration (ECE before → after temperature; Brier)</h3>")
        parts.append(_calibration_table(section["datasets"], labels, taxonomy))
        if any(section["sanity"].values()):
            parts.append("<h3>Sanity checks (metadata labels, constant per dataset or view)</h3>")
            parts.append(_table(section["sanity"], labels, taxonomy, {"categorical": "accuracy"}))
        if "reference" in section:
            parts.append(_reference_table(section["reference"], labels, taxonomy))
    if groups:
        holdout = results["splits"].get("holdout", {}).get("datasets", {})
        parts.append("<h2>Holdout groups</h2><table><tr><th>group</th><th>measures</th><th>clips</th><th>subjects</th><th>status</th></tr>")
        for name, group in groups.items():
            subjects = str(group["subjects"])
            status = group.get("status") or ("scored above" if name in holdout else "not predicted in this run")
            parts.append(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(group['kind'])}</td><td>{group['clips']}</td>"
                f"<td>{html.escape(subjects)}</td><td>{html.escape(status)}</td></tr>"
            )
        parts.append("</table>")
    if results["not_scored"]:
        parts.append("<h2>Not scored</h2><ul>")
        parts.extend(f"<li>{html.escape(n)}: {html.escape(r)}</li>" for n, r in results["not_scored"].items())
        parts.append("</ul>")
    parts.append(_support_table(results["training_support"], results.get("checkpoint_labels")))
    return "\n".join(parts)


def _selection(select: dict[str, Any], step: Any) -> str:
    """How best.pt was chosen: its score on the val split, or, for a heads checkpoint picked
    with `select_folds`, the mean over k held-out folds of train and val subjects at the
    chosen epoch (best.pt's step is that epoch times the steps per epoch)."""
    score = f"{select.get('criterion')} = {_num(select.get('score'), 4)}"
    if select.get("folds"):
        return (
            f"{score}, mean over {select['folds']} held-out folds of train and val subjects "
            f"(epoch {select.get('epoch')}), step {step}"
        )
    return f"{score} on val, step {step}"


def _commit(git: dict[str, Any] | None) -> str:
    if not git or not git.get("commit"):
        return "not recorded"
    return f"{git['commit'][:12]}{' (uncommitted changes)' if git.get('dirty') else ''}"


def _support_table(support: dict[str, dict[str, Any]], labels_record: dict[str, Any] | None = None) -> str:
    """Training clips per label behind the model (ground truth and teacher pseudo-labels),
    and whether the checkpoint records the label as trained (see `checkpoint_labels`)."""
    supervised = (labels_record or {}).get("supervised_labels")
    masked = (labels_record or {}).get("masked_labels") or {}
    rows = []
    for name, entry in support.items():
        classes = ""
        if entry.get("classes_gt"):
            classes = ", ".join(f"{c} {n}" for c, n in entry["classes_gt"].items())
        trained = "not recorded" if supervised is None else ("yes" if name in supervised else "no")
        if name in masked:
            trained += f"; masked on {', '.join(masked[name])}"
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{entry['gt']}</td><td>{entry['teacher']}</td>"
            f"<td>{html.escape(trained)}</td><td><small>{html.escape(classes)}</small></td></tr>"
        )
    return (
        "<h2>Training support</h2><p>Training clips per label in the model's training scope: ground truth, "
        "and teacher pseudo-labels on clips without ground truth. Trained: whether the checkpoint lists the label "
        "in its supervised_labels, and the media types training masked it on. Only labels with ground truth that "
        "the checkpoint trained are scored.</p>"
        "<table><tr><th>label</th><th>ground truth</th><th>teacher</th><th>trained</th><th>ground truth per class</th></tr>"
        + "".join(rows) + "</table>"
    )


def _scored_media(results: dict[str, Any]) -> dict[str, set[str]]:
    """Per label, the media types of the clips the calibrated per-dataset tables scored
    (the keys of their entries' temperature maps)."""
    found: dict[str, set[str]] = {}
    for section in results["splits"].values():
        for entries in section["datasets"].values():
            for name, entry in entries.items():
                if isinstance(entry.get("temperature"), dict):
                    found.setdefault(name, set()).update(entry["temperature"])
    return found


def _fit_cell(entry: dict[str, Any], spec: LabelSpec) -> str:
    """A calibration.json entry: threshold (binary) and where it came from, the
    temperature, and the val clips, positives and subjects behind them; for a fit refused
    for want of val clips, the reason (`not_fitted`)."""
    if entry.get("source") != "val":
        return "<td>not fitted (no val clips): 0.5, no temperature</td>"
    counts = [f"{entry.get('n_val', 0)} clips"]
    if "n_val_positive" in entry:
        counts.append(f"{entry['n_val_positive']} positive")
    if "n_val_subjects" in entry:
        counts.append(f"{entry['n_val_subjects']} subjects")
    if "not_fitted" in entry:
        return f"<td>not fitted ({html.escape(entry['not_fitted'])}): 0.5, no temperature<br><small>{' · '.join(counts)}</small></td>"
    t = entry.get("temperature")
    details = [f"T {t:.2f}" if t is not None else "no temperature", *counts]
    text = f"{entry['threshold']:.3f} <small>({html.escape(entry['threshold_from'])})</small><br>" if spec.kind == "binary" else ""
    return f"<td>{text}<small>{' · '.join(details)}</small></td>"


def _thresholds_table(calibration: dict[str, dict[str, Any]], scored_media: dict[str, set[str]], taxonomy: Taxonomy) -> str:
    """The thresholds and temperatures the report applies: per label, the fit on all val
    clips and the fit per media type, which is what scores each clip."""
    kinds = sorted(MEDIA_TYPES)
    rows = []
    for name, entry in calibration.items():
        if "media" not in entry:
            continue  # regression labels: nothing is fitted
        cells = [_fit_cell(entry, taxonomy[name])]
        for kind in kinds:
            if kind in entry["media"]:
                cells.append(_fit_cell(entry["media"][kind], taxonomy[name]))
            elif kind in scored_media.get(name, ()):
                cells.append("<td>no val clips of this media type: 0.5, no temperature</td>")
            else:
                cells.append("<td>–</td>")
        rows.append(f"<tr><td>{html.escape(name)}</td>{''.join(cells)}</tr>")
    if not rows:
        return "<h2>Thresholds and temperatures (fitted on val)</h2><p>No calibrated labels.</p>"
    head = "".join(f"<th>{kind}</th>" for kind in kinds)
    return (
        "<h2>Thresholds and temperatures (fitted on val)</h2><p>Each clip is scored with the threshold and temperature "
        "fitted on the val clips of its media type. The fit on all val clips is shown for comparison; calibration.json "
        "keeps it as each label's top-level entry, which files written before the per-media-type entries held alone. "
        "Thresholds are on sigmoid(logit), before any temperature.</p>"
        f"<table><tr><th>label</th><th>all val clips</th>{head}</tr>{''.join(rows)}</table>"
    )


def _temperature_text(used: Any) -> str:
    """The temperatures an entry was scored with ({media type: t or None})."""
    if not isinstance(used, dict) or not used:
        return ""
    if len(used) == 1:
        (t,) = used.values()
        return " · " + (f"T {t:.2f}" if t is not None else "no temperature")
    return " · T " + ", ".join(f"{kind} {'none' if t is None else f'{t:.2f}'}" for kind, t in used.items())


def _calibration_table(datasets: dict[str, dict[str, Any]], labels: Sequence[str], taxonomy: Taxonomy) -> str:
    rows = []
    for name in labels:
        if taxonomy[name].kind == "regression":
            continue
        cells = []
        for entries in datasets.values():
            e = entries.get(name)
            if not e or "ece" not in e:
                cells.append("<td>–</td>")
                continue
            after = f" → {e['ece_calibrated']:.3f}" if "ece_calibrated" in e else ""
            cells.append(f"<td>{e['ece']:.3f}{after}<br><small>Brier {e['brier']:.3f}{_temperature_text(e.get('temperature'))}</small></td>")
        if any(c != "<td>–</td>" for c in cells):
            rows.append(f"<tr><td>{name}</td>{''.join(cells)}</tr>")
    if not rows:
        return "<p>No calibrated labels.</p>"
    head = "".join(f"<th>{html.escape(d)}</th>" for d in datasets)
    return f"<table><tr><th>label</th>{head}</tr>{''.join(rows)}</table>"


def _reference_table(section: dict[str, Any], labels: Sequence[str], taxonomy: Taxonomy) -> str:
    parts = [f"<h3>Against the reference model ({section['clips']} shared clips)</h3><table><tr><th>label</th>"]
    datasets = list(section["datasets"])
    parts.append("".join(f"<th>{html.escape(d)}: model / reference</th>" for d in datasets) + "<th>agreement</th></tr>")
    for name in labels:
        if name not in section["agreement"]:
            continue
        cells = []
        for d in datasets:
            pair = section["datasets"][d].get(name)
            cells.append(f"<td>{_num(pair['model'])} / {_num(pair['reference'])}</td>" if pair else "<td>–</td>")
        agreement = ", ".join(f"{k} {v:.3f}" for k, v in section["agreement"][name].items() if k != "n")
        parts.append(f"<tr><td>{name}</td>{''.join(cells)}<td><small>{html.escape(agreement)}</small></td></tr>")
    parts.append("</table>")
    return "".join(parts)
