from __future__ import annotations

import numpy as np
import pytest
import torch

from bulwark.data.manifest import ClipRecord
from bulwark.data.taxonomy import default_taxonomy
from bulwark.eval import metrics as M


def record(i: int, subject: str, labels: dict, **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=f"c{i}", dataset="mrl_eye", subject_id=f"mrl_eye:{subject}", media_path=f"{i}.png",
        media_type="image", licence="t", commercial_ok=False, split="test", labels_gt=labels, **kwargs,  # type: ignore[arg-type]
    )


def test_macro_f1_matches_sklearn_default_and_needs_two_classes() -> None:
    # sklearn f1_score(average="macro") averages over classes in labels or predictions: 0.5 here
    assert abs(M.macro_f1(np.array([0, 3, 1, 3, 2, 3]), np.array([0, 0, 1, 1, 2, 2]), 4) - 0.5) < 1e-9
    assert M.macro_f1(np.zeros(5, int), np.zeros(5, int), 6) is None  # was 1.0 for a constant predictor
    assert M.balanced_accuracy(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == (1.0 + 2 / 3) / 2


def test_f1_is_undefined_without_positives() -> None:
    assert M.f1(np.zeros(4, int), np.zeros(4, int)) is None
    assert M.f1(np.array([1, 0]), np.array([0, 1])) == 0.0


def test_auroc_on_logits_has_no_saturation_ties() -> None:
    logits = np.array([18.0, 19.0, 20.0, 21.0], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    assert M.auroc(torch.tensor(logits).sigmoid().numpy(), labels) == 0.5  # float32 probabilities all round to 1.0
    assert M.auroc(logits, labels) == 1.0


def test_non_finite_outputs_are_counted_and_excluded() -> None:
    spec = default_taxonomy()["eyes_closed"]
    records = [record(i, f"s{i}", {"eyes_closed": float(i % 2)}) for i in range(4)]
    data = M.label_rows(records, spec)
    assert data is not None
    entry, _ = M.score_label(spec, np.array([-1.0, np.nan, -2.0, 3.0]), data)
    assert entry["nonfinite"] == 1 and entry["n"] == 3 and entry["auroc"] == 1.0


def test_teacher_and_weak_labels_are_never_scored() -> None:
    taxonomy = default_taxonomy()
    records = [
        record(0, "a", {}, labels_teacher={"eyes_closed": 1.0}),
        record(1, "b", {"eyes_closed": 1.0}, meta={"weak_labels": ["eyes_closed"]}),
        record(2, "c", {"eyes_closed": 0.0}),
    ]
    data = M.label_rows(records, taxonomy["eyes_closed"])
    assert data is not None and data.rows.tolist() == [2]


def test_threshold_and_temperature_from_val() -> None:
    p = np.array([0.1, 0.2, 0.35, 0.4, 0.8, 0.9])
    y = np.array([0, 0, 1, 0, 1, 1])
    assert M.best_threshold(p, y) == 0.35  # F1 6/7, above 0.8 at p >= 0.8
    assert M.best_threshold(p, np.zeros(6, int)) is None
    rng = np.random.default_rng(0)
    true = rng.normal(size=4000) * 2
    labels = (rng.random(4000) < M.sigmoid(true)).astype(int)
    fitted = M.fit_temperature(true * 3.0, labels, "binary")  # logits three times too confident
    assert fitted is not None and 2.4 < fitted < 3.7


def test_cluster_bootstrap_resamples_subjects() -> None:
    values = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    subjects = np.array(["a", "a", "b", "b", "c", "c"])
    draws = M.cluster_bootstrap(lambda rows: float(values[rows].mean()), subjects, 200, seed=0)
    assert set(np.round(draws * 3).astype(int)) <= set(range(7))  # means of three subject values
    low, high = M.interval(draws)  # type: ignore[misc]
    assert 0.0 <= low <= 1.0 <= high <= 2.0


def test_score_label_reports_counts_baseline_units_and_ci() -> None:
    taxonomy = default_taxonomy()
    spec = taxonomy["head_yaw"]
    records = [record(i, f"s{i % 12}", {"head_yaw": 0.1 * (i % 3)}) for i in range(36)]
    data = M.label_rows(records, spec)
    assert data is not None
    prior = M.training_prior(records, spec, "train:all")
    entry, draws = M.score_label(spec, np.zeros(36), data, prior=prior, bootstrap=50)
    assert entry["n"] == 36 and entry["n_subjects"] == 12 and entry["n_events"] == 36
    assert entry["unit"] == "deg" and abs(entry["mae_in_unit"] - entry["mae"] * 90) < 1e-9
    assert entry["baseline"]["mae"] < entry["mae"] and entry["ci"] is not None and draws is not None
    few = [record(i, f"s{i % 3}", {"head_yaw": 0.0}) for i in range(9)]
    entry, draws = M.score_label(spec, np.zeros(9), M.label_rows(few, spec), bootstrap=50)  # type: ignore[arg-type]
    assert entry["indicative"] and "ci" not in entry and draws is None


def test_evaluate_outputs_keeps_its_interface() -> None:
    taxonomy = default_taxonomy()
    records = [record(i, f"s{i}", {"eyes_closed": float(i % 2)}) for i in range(6)]
    logits = {name: torch.zeros(6, spec.width) if spec.kind == "categorical" else torch.zeros(6) for name, spec in taxonomy.labels.items()}
    logits["eyes_closed"] = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    results = M.evaluate_outputs([logits], records, taxonomy)
    assert set(results) == {"eyes_closed"} and results["eyes_closed"]["auroc"] == 1.0


def test_macro_f1_over_trained_classes_and_reliability_bins() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 2, 1, 1, 2, 0])
    # sklearn: f1_score(labels, pred, labels=[0, 1], average="macro") = (0.5 + 1.0) / 2
    assert abs(M.macro_f1(pred, labels, 3, only=[0, 1]) - 0.75) < 1e-9
    assert M.macro_f1(pred, labels, 3, only=[0]) is None  # one averaged class: undefined
    table = M.reliability(np.array([0.05, 0.1, 0.95, 0.9]), np.array([0.0, 1.0, 1.0, 1.0]), bins=2)
    assert table["count"] == [2, 2] and table["observed_frequency"] == [0.5, 1.0]
    assert abs(table["mean_confidence"][0] - 0.075) < 1e-9 and table["edges"] == [0.0, 0.5, 1.0]


def mixed(n_image: int, n_video: int, label: str = "eyes_closed", value=lambda i: float(i % 2)) -> list[ClipRecord]:  # type: ignore[no-untyped-def]
    """`n_image` image clips then `n_video` video clips, each with a ground-truth `label`."""
    images = [record(i, f"s{i}", {label: value(i)}) for i in range(n_image)]
    videos = [
        ClipRecord(
            clip_id=f"v{i}", dataset="dmd", subject_id=f"dmd:{i}", media_path=f"{i}.mp4", media_type="video",
            licence="t", commercial_ok=False, split="test", labels_gt={label: value(i)},
        )
        for i in range(n_video)
    ]
    return images + videos


def test_label_rows_record_each_rows_media_type_and_selection_keeps_it_aligned() -> None:
    from bulwark.train.labels import LabelPlan
    from bulwark.train.validation import selected_rows

    spec = default_taxonomy()["eyes_closed"]
    records = [*mixed(2, 3)[:2], record(9, "x", {}), *mixed(2, 3)[2:]]  # a clip without the label in between
    data = M.label_rows(records, spec)
    assert data is not None and data.rows.tolist() == [0, 1, 3, 4, 5]
    assert data.media.tolist() == ["image", "image", "video", "video", "video"]
    plan = LabelPlan(masked={("eyes_closed", "image"): "one class"}, missing_classes={}, supervised=("eyes_closed",))
    kept = selected_rows(records, spec, plan)
    assert kept is not None and kept.rows.tolist() == [3, 4, 5] and kept.media.tolist() == ["video"] * 3


def test_a_threshold_per_media_type_scores_each_row_against_its_own() -> None:
    spec = default_taxonomy()["eyes_closed"]
    records = mixed(4, 4)
    data = M.label_rows(records, spec)
    assert data is not None
    # Image logits run high and video logits low, as a clip model's confidence can.
    logits = np.array([3.0, 4.0, 2.5, 4.5, -3.0, -1.0, -2.5, -0.5])
    target = data.target
    thresholds = {"image": 0.97, "video": 0.2}
    entry, _ = M.score_label(spec, logits, data, threshold=thresholds)
    cut = np.array([0.97] * 4 + [0.2] * 4)
    assert entry["f1"] == M.f1((M.sigmoid(logits) >= cut).astype(int), target)
    assert entry["threshold"] == thresholds
    # One threshold for all rows would miss every positive video clip.
    single, _ = M.score_label(spec, logits, data, threshold=0.97)
    assert single["threshold"] == 0.97 and single["f1"] < entry["f1"]

    videos = M.label_rows(records[4:], spec)
    assert videos is not None
    entry, _ = M.score_label(spec, logits[4:], videos, threshold=thresholds)
    assert entry["threshold"] == {"video": 0.2}  # the media types among the rows scored
    with pytest.raises(ValueError, match=r"threshold mapping has no value for media type\(s\) \['image'\]"):
        M.score_label(spec, logits, data, threshold={"video": 0.2})


def test_a_temperature_per_media_type_calibrates_each_row_with_its_own() -> None:
    taxonomy = default_taxonomy()
    spec = taxonomy["eyes_closed"]
    records = mixed(6, 6)
    data = M.label_rows(records, spec)
    assert data is not None
    logits = np.linspace(-4, 4, 12)
    t_row = np.array([2.0] * 6 + [0.5] * 6)
    entry, _ = M.score_label(spec, logits, data, temperature={"image": 2.0, "video": 0.5})
    assert entry["ece_calibrated"] == M.ece_binary(M.sigmoid(logits / t_row), data.target)
    assert entry["temperature"] == {"image": 2.0, "video": 0.5}
    # No temperature fitted for one media type among the rows: no calibrated ECE for the mix.
    entry, _ = M.score_label(spec, logits, data, temperature={"image": 2.0, "video": None})
    assert "ece_calibrated" not in entry and entry["temperature"] == {"image": 2.0, "video": None}
    # One temperature for all rows keeps its old form: no temperature in the entry.
    entry, _ = M.score_label(spec, logits, data, temperature=2.0)
    assert entry["ece_calibrated"] == M.ece_binary(M.sigmoid(logits / 2.0), data.target) and "temperature" not in entry

    activity = taxonomy["activity"]
    classes = [activity.classes[i % 2] for i in range(12)]
    records = mixed(6, 6, "activity", value=lambda i: classes[i])
    data = M.label_rows(records, activity)
    assert data is not None
    raw = np.random.default_rng(0).normal(size=(12, len(activity.classes))) * 3
    entry, _ = M.score_label(activity, raw, data, temperature={"image": 2.0, "video": 0.5})
    assert entry["ece_calibrated"] == M.ece_top_label(M.softmax(raw / t_row[:, None]), data.target)
    assert entry["temperature"] == {"image": 2.0, "video": 0.5}
