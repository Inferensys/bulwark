"""Which predictions a report scores: only clips the manifest puts in the split being
scored (re-split manifests, `excluded` clips), the split policy the manifest records, and
warnings for logits whose layout was not recorded."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.data.taxonomy import default_taxonomy
from bulwark.eval import metrics as M
from bulwark.eval.predict import Predictions
from bulwark.eval.provenance import SPLIT_POLICY_KEY
from bulwark.eval.report import MIN_FIT_CLIPS, _features_source, _fit, build_report, calibrate


def rec(clip_id: str, subject: str, split: str, view: str, phone: float, **meta: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset="drive_and_act", subject_id=f"drive_and_act:{subject}", media_path=f"{clip_id}.mp4",
        media_type="video", licence="t", commercial_ok=False, split=split, view=view, labels_gt={"phone": phone}, meta=dict(meta),
    )


def predict(ids: list[str], phone: list[float] | None = None) -> Predictions:
    logits = {
        name: np.zeros((len(ids), spec.width) if spec.kind == "categorical" else len(ids), dtype=np.float32)
        for name, spec in default_taxonomy().labels.items()
    }
    if phone is not None:
        logits["phone"] = np.array(phone, dtype=np.float32)
    return Predictions(ids, logits)


RECORDS = [
    rec("da/train", "vp1", "train", "kinect_color", 1.0),
    rec("da/train0", "vp2", "train", "kinect_color", 0.0),
    rec("da/val", "vp3", "val", "kinect_color", 1.0),
    rec("da/val0", "vp4", "val", "kinect_color", 0.0),
    rec("da/test", "vp5", "test", "kinect_color", 0.0),
    rec("da/hold", "vp5", "holdout", "inner_mirror", 1.0, subject_split="test"),
    rec("da/excl", "vp1", "excluded", "inner_mirror", 1.0, subject_split="train"),
    rec("da/excl2", "vp2", "excluded", "inner_mirror", 0.0, subject_split="train"),
]


def test_predictions_of_clips_now_in_another_split_are_not_scored(tmp_path: Path) -> None:
    manifest = write_manifest(RECORDS, tmp_path / "manifest.parquet")
    # Saved against an earlier split: holdout predictions include two clips now `excluded`
    # (of a training subject), and the test predictions a clip that is now val.
    predictions = {
        "test": predict(["da/test", "da/val"]),
        "holdout": predict(["da/hold", "da/excl", "da/excl2", "da/gone"]),
    }
    path = build_report(predictions, manifest, tmp_path / "eval", bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    holdout = metrics["splits"]["holdout"]
    assert holdout["counts"]["drive_and_act"]["clips"] == 1 and holdout["counts"]["drive_and_act"]["subjects"] == 1
    assert holdout["predictions_in_other_splits"] == {"excluded": 2} and holdout["predictions_not_in_manifest"] == 1
    assert holdout["datasets"]["drive_and_act"]["phone"]["n"] == 1
    assert metrics["splits"]["test"]["counts"]["drive_and_act"]["clips"] == 1
    assert metrics["coverage"]["holdout"]["drive_and_act"] == {"manifest": 1, "predicted": 1, "predicted_in_other_splits": 2}
    assert any("predicted as holdout, 2 are in other splits of this manifest (excluded 2) and 1 are not in it" in w
               for w in metrics["meta"]["warnings"])
    assert "are not scored" in path.read_text()


def test_thresholds_are_fitted_on_val_clips_only() -> None:
    records = {r.clip_id: r for r in RECORDS}
    # The test clip's prediction would flip the threshold if it were counted as val.
    preds = predict(["da/val", "da/val0", "da/test"], phone=[2.0, -2.0, 5.0])
    entry = calibrate(preds, records, default_taxonomy(), ["phone"])["phone"]
    assert entry["n_val"] == 2 and entry["source"] == "val"


def with_policy(manifest: Path, policy: dict) -> Path:
    table = pq.read_table(manifest)
    metadata = {**(table.schema.metadata or {}), SPLIT_POLICY_KEY: json.dumps(policy).encode()}
    pq.write_table(table.replace_schema_metadata(metadata), manifest)
    return manifest


def test_a_split_policy_recorded_in_the_manifest_drives_the_leak_check_and_holdout_groups(tmp_path: Path) -> None:
    records = [*RECORDS[:5], rec("da/ceil", "vp5", "holdout", "ceiling", 1.0, subject_split="test")]
    policy = {"holdout_views": [["drive_and_act", "ceiling"]], "official_splits": [["drive_and_act", "0"]]}
    manifest = with_policy(write_manifest(records, tmp_path / "manifest.parquet"), policy)
    path = build_report({"holdout": predict(["da/ceil"])}, manifest, tmp_path / "eval", bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    group = metrics["holdout_groups"]["drive_and_act"]
    assert "ceiling" in group["kind"] and group["clips"] == 1 and metrics["meta"]["split_policy"]["source"] == "manifest"

    # A ceiling clip in training leaks only under the recorded policy, and is refused.
    leaky = [*records, rec("da/ceil2", "vp1", "train", "ceiling", 0.0)]
    manifest = with_policy(write_manifest(leaky, tmp_path / "leaky.parquet"), policy)
    with pytest.raises(ValueError, match="breaks the split policy"):
        build_report({"holdout": predict(["da/ceil"])}, manifest, tmp_path / "eval2", bootstrap=0)


def test_logits_cached_without_a_layout_record_are_flagged(tmp_path: Path) -> None:
    features = tmp_path / "features"
    features.mkdir()
    (features / "meta.json").write_text(json.dumps({"run": {"init": {"sha256": "x", "has_provenance": True}, "with_logits": True}}))
    meta: dict = {"warnings": []}
    _features_source("logits", None, features, meta)
    assert any("records no logit layout" in w for w in meta["warnings"])


def media_records() -> dict[str, ClipRecord]:
    """Val clips of eyes_closed on images (MRL-like) and video (DMD-like), eyeglasses on
    images only, perclos on video; and a test clip that must not count."""
    records = []
    for i in range(40):
        records.append(ClipRecord(
            clip_id=f"mrl/{i}", dataset="mrl_eye", subject_id=f"mrl_eye:s{i % 8}", media_path=f"{i}.png", media_type="image",
            licence="t", commercial_ok=False, split="val", labels_gt={"eyes_closed": float(i % 2), "eyeglasses": float(i % 4 == 0)},
        ))
    for i in range(50):
        records.append(ClipRecord(
            clip_id=f"dmd/{i}", dataset="dmd", subject_id=f"dmd:g{i % 3}", media_path=f"{i}.mp4", media_type="video",
            licence="t", commercial_ok=False, split="val", labels_gt={"eyes_closed": float(i % 5 == 0), "perclos": 0.2},
        ))
    records.append(ClipRecord(
        clip_id="dmd/test", dataset="dmd", subject_id="dmd:g9", media_path="t.mp4", media_type="video",
        licence="t", commercial_ok=False, split="test", labels_gt={"eyes_closed": 0.0},
    ))
    return {r.clip_id: r for r in records}


def media_logits(records: dict[str, ClipRecord]) -> Predictions:
    """Image logits that separate the classes around +2, video logits around -2 (a clip
    model less sure on video), with some noise."""
    rng = np.random.default_rng(0)
    preds = predict(list(records))
    for i, r in enumerate(records.values()):
        y = r.labels_gt.get("eyes_closed", 0.0)
        centre = 2.0 if r.media_type == "image" else -2.0
        preds.logits["eyes_closed"][i] = centre + (1.5 if y else -1.5) + rng.normal()
        preds.logits["eyeglasses"][i] = 3.0 * r.labels_gt.get("eyeglasses", 0.0) - 1.5 + rng.normal()
    preds.logits["eyes_closed"][-1] = 9.0  # the test clip would move every fit
    return preds


def test_thresholds_and_temperatures_are_fitted_on_all_val_clips_and_per_media_type() -> None:
    records = media_records()
    preds = media_logits(records)
    taxonomy = default_taxonomy()
    entries = calibrate(preds, records, taxonomy, ["eyes_closed", "eyeglasses", "perclos"])
    val = [i for i, r in enumerate(records.values()) if r.split == "val"]
    raw = preds.logits["eyes_closed"][val].astype(np.float64)
    target = np.array([int(list(records.values())[i].labels_gt["eyes_closed"]) for i in val])
    kinds = np.array([list(records.values())[i].media_type for i in val])

    entry = entries["eyes_closed"]
    # The top-level entry is the fit on all val clips, as before media entries.
    assert entry["threshold"] == M.best_threshold(M.sigmoid(raw), target) and entry["threshold_from"] == "val"
    assert entry["temperature"] == M.fit_temperature(raw, target, "binary") and entry["n_val"] == 90
    assert entry["n_val_positive"] == 20 + 10 and entry["n_val_subjects"] == 8 + 3
    assert list(entry["media"]) == ["image", "video"]
    for kind, n, positive, subjects in (("image", 40, 20, 8), ("video", 50, 10, 3)):
        rows = kinds == kind
        fit = entry["media"][kind]
        assert fit["threshold"] == M.best_threshold(M.sigmoid(raw[rows]), target[rows]) and fit["threshold_from"] == "val"
        assert fit["temperature"] == M.fit_temperature(raw[rows], target[rows], "binary") and fit["source"] == "val"
        assert (fit["n_val"], fit["n_val_positive"], fit["n_val_subjects"]) == (n, positive, subjects)
    assert entry["media"]["video"]["threshold"] < 0.5 < entry["media"]["image"]["threshold"]

    # Ground truth on one media type only: one media entry, the same fit as the top level.
    eyeglasses = entries["eyeglasses"]
    assert list(eyeglasses["media"]) == ["image"]
    assert eyeglasses["media"]["image"] == {k: v for k, v in eyeglasses.items() if k != "media"}
    # Nothing is fitted for regression labels, per media type or not.
    assert entries["perclos"] == {"threshold": 0.5, "threshold_from": "default", "temperature": None, "source": "default"}


def test_a_fit_needs_enough_val_clips_of_both_classes() -> None:
    records = media_records()
    for r in records.values():  # every image clip closed: one class on images
        if r.media_type == "image":
            r.labels_gt["eyes_closed"] = 1.0
    preds = media_logits(records)
    entry = calibrate(preds, records, default_taxonomy(), ["eyes_closed"])["eyes_closed"]
    image = entry["media"]["image"]
    assert (image["threshold"], image["threshold_from"], image["temperature"]) == (0.5, "default", None)
    assert image["n_val"] == image["n_val_positive"] == 40 and image["source"] == "val"
    assert image["not_fitted"] == f"fewer than {MIN_FIT_CLIPS} clips of each of two classes"
    assert entry["threshold_from"] == "val" and entry["media"]["video"]["threshold_from"] == "val"
    assert "not_fitted" not in entry and "not_fitted" not in entry["media"]["video"]

    # All positive in val: the F1-best threshold would be the lowest probability, which
    # flags every clip, so no threshold is fitted.
    for r in records.values():
        r.labels_gt["eyes_closed"] = 1.0
    entry = calibrate(preds, records, default_taxonomy(), ["eyes_closed"])["eyes_closed"]
    assert (entry["threshold"], entry["threshold_from"], entry["temperature"]) == (0.5, "default", None)
    assert all(fit["threshold_from"] == "default" and "not_fitted" in fit for fit in entry["media"].values())
    # Without val predictions nothing is fitted, and there are no media entries.
    assert calibrate(None, records, default_taxonomy(), ["eyes_closed"])["eyes_closed"]["media"] == {}


def test_a_handful_of_val_clips_gives_no_fit() -> None:
    phone = default_taxonomy()["phone"]
    # Three separable clips of one subject would give a "val" threshold and the edge of the
    # temperature grid (T 0.05, logits sharpened 20-fold).
    entry = _fit(phone, np.array([0.4, -0.3, -2.0]), np.array([1, 0, 0]), np.array(["a", "a", "a"]))
    assert (entry["threshold"], entry["threshold_from"], entry["temperature"]) == (0.5, "default", None)
    assert (entry["n_val"], entry["n_val_positive"], entry["n_val_subjects"]) == (3, 1, 1) and "not_fitted" in entry
    # One class short of the floor is not enough; at the floor the fit is made.
    target = np.array([1] * MIN_FIT_CLIPS + [0] * (MIN_FIT_CLIPS - 1))
    raw = np.where(target == 1, 1.0, -1.0) + np.linspace(-1.5, 1.5, len(target))
    assert "not_fitted" in _fit(phone, raw, target, np.array(["a"] * len(target)))
    target, raw = np.r_[target, 0], np.r_[raw, 0.2]
    fitted = _fit(phone, raw, target, np.array(["a"] * len(target)))
    assert fitted["threshold_from"] == "val" and fitted["temperature"] == M.fit_temperature(raw, target, "binary")
    # Categorical labels need that many clips in each of two classes.
    activity = default_taxonomy()["activity"]
    logits = np.zeros((25, activity.width))
    assert "not_fitted" in _fit(activity, logits, np.array([0] * 20 + [1] * 5), np.array(["a"] * 25))
    assert _fit(activity, logits, np.array([0] * 12 + [1] * 10 + [2] * 3), np.array(["a"] * 25))["temperature"] is not None
