"""Report checks added with the official Drive&Act split: split leaks, holdout groups,
clip coverage, the YawDD video-level yawn score, calibration files and logit layouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.data.taxonomy import default_taxonomy
from bulwark.eval.predict import Predictions, predict_features
from bulwark.eval.report import build_report, read_calibration


def rec(clip_id: str, dataset: str, subject: str, split: str, labels: dict, **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset=dataset, subject_id=f"{dataset}:{subject}", media_path=kwargs.pop("media_path", f"{clip_id}.mp4"),  # type: ignore[arg-type]
        media_type="video", licence="t", commercial_ok=False, split=split, labels_gt=labels, **kwargs,  # type: ignore[arg-type]
    )


def predictions(records: list[ClipRecord], split: str, values: dict[str, float] | None = None) -> Predictions:
    ids = [r.clip_id for r in records if r.split == split]
    logits = {}
    for name, spec in default_taxonomy().labels.items():
        logits[name] = np.zeros((len(ids), spec.width) if spec.kind == "categorical" else len(ids), dtype=np.float32)
    for row, clip_id in enumerate(ids):
        logits["yawn"][row] = (values or {}).get(clip_id, -2.0)
    return Predictions(ids, logits)


def yawdd(tmp_path: Path) -> tuple[Path, list[ClipRecord]]:
    """YawDD mirror videos: Normal/Talking clips carry yawn = 0, Yawning clips only a meta flag."""
    records = []
    for split, subjects in (("train", 3), ("test", 6)):
        for s in range(subjects):
            for behaviour in ("Normal", "Talking", "Yawning"):
                video = f"{split}/{s}-Female-{behaviour}.avi"
                for w in range(3):
                    labels = {} if behaviour == "Yawning" else {"yawn": 0.0}
                    meta = {"yawn_video": True} if behaviour == "Yawning" else {}
                    records.append(rec(f"yawdd/{video}@{w}", "yawdd", f"f{split}{s}", split, labels,
                                       media_path=video, view="mirror", meta=meta))
    return write_manifest(records, tmp_path / "manifest.parquet"), records


def test_yawn_is_scored_per_video_on_its_highest_clip(tmp_path: Path) -> None:
    manifest, records = yawdd(tmp_path)
    # One clip in the middle of each Yawning video shows the yawn; the rest look like Normal.
    values = {r.clip_id: 3.0 for r in records if r.meta.get("yawn_video") and r.clip_id.endswith("@1")}
    path = build_report({"test": predictions(records, "test", values)}, manifest, tmp_path / "eval", bootstrap=0)
    test = json.loads((path.parent / "metrics.json").read_text())["splits"]["test"]
    video = test["video_max"]["yawdd"]["yawn"]
    assert video["n"] == 18 and video["n_subjects"] == 6 and video["auroc"] == 1.0 and video["positive_rate"] == 1 / 3
    assert test["datasets"]["yawdd"]["yawn"]["auroc"] is None  # clip level: Normal/Talking negatives only
    assert "highest clip output" in path.read_text()


def test_holdout_groups_coverage_and_split_leaks(tmp_path: Path) -> None:
    records = [
        rec("da/a", "drive_and_act", "vp1", "train", {"phone": 1.0}, view="kinect_color"),
        rec("da/b", "drive_and_act", "vp5", "test", {"phone": 0.0}, view="kinect_color"),
        rec("da/c", "drive_and_act", "vp5", "holdout", {"phone": 1.0}, view="inner_mirror", meta={"subject_split": "test"}),
        rec("da/d", "drive_and_act", "vp1", "excluded", {"phone": 1.0}, view="inner_mirror", meta={"subject_split": "train"}),
    ]
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    preds = {"test": predictions(records, "test"), "holdout": predictions(records, "holdout")}
    path = build_report(preds, manifest, tmp_path / "eval", bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    groups = metrics["holdout_groups"]
    assert groups["drive_and_act"]["subjects"] == 1 and "split_0" in groups["drive_and_act"]["kind"]
    assert "subject_ids" not in groups["drive_and_act"] and "drive_and_act:vp5" not in path.read_text()  # counts only
    assert groups["uta_rldd"]["status"] == "not available (not ingested)"
    assert metrics["coverage"]["test"] == {"drive_and_act": {"manifest": 1, "predicted": 1}}
    assert metrics["meta"]["official_splits"] == {"drive_and_act": "split_0"}
    html = path.read_text()
    assert "not available (not ingested)" in html and "not averaged or pooled" in html

    leaky = [*records, rec("da/e", "drive_and_act", "vp5", "train", {"phone": 1.0}, view="ceiling")]
    with pytest.raises(ValueError, match="breaks the split policy"):
        build_report(preds, write_manifest(leaky, tmp_path / "leaky.parquet"), tmp_path / "eval2", bootstrap=0)


def test_auc_test_results_rest_on_one_driver_group(tmp_path: Path) -> None:
    # AUC v2's test side is one subject group (no driver ids): its scores are indicative and
    # get no subject-level interval, however many images it has.
    records = [
        rec(f"auc_v2/{split}/{i}.jpg", "auc_v2", f"{split}_drivers", split, {"phone": float(i % 2)},
            view="cam1", meta={"official_split": {"v2": split}})
        for split in ("train", "test") for i in range(40)
    ]  # fmt: skip
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    path = build_report({"test": predictions(records, "test")}, manifest, tmp_path / "eval", bootstrap=50)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    phone = metrics["splits"]["test"]["datasets"]["auc_v2"]["phone"]
    assert phone["n"] == 40 and phone["n_subjects"] == 1 and phone["indicative"] is True
    assert "ci" not in phone
    assert metrics["meta"]["official_splits"] == {"auc_v2": "split_v2"}  # only datasets present


def test_calibration_files_belong_to_one_set_of_weights(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "meta": {"checkpoint": {"sha256": "aa", "exported_from_sha256": "bb"}},
        "labels": {"yawn": {"temperature": 2.0}, "phone": {"temperature": None}},
    }))
    assert read_calibration(path, "aa") == read_calibration(path, "bb") == {"yawn": 2.0}
    with pytest.raises(ValueError, match="fitted on weights"):
        read_calibration(path, "cc")
    path.write_text(json.dumps({"yawn": {"temperature": 1.5}, "phone": 0.5}))  # a plain mapping
    assert read_calibration(path, "cc") == {"yawn": 1.5, "phone": 0.5}
    path.write_text(json.dumps({"yawn": -1.0}))
    with pytest.raises(ValueError, match="positive"):
        read_calibration(path)


def test_cached_logits_of_another_taxonomy_are_refused(tmp_path: Path) -> None:
    features = tmp_path / "features"
    features.mkdir()
    np.savez(features / "features-0.npz", clip_ids=np.array(["a"]), tokens=np.zeros((1, 25, 8), dtype=np.float16),
             logit__yawn=np.zeros(1, dtype=np.float32))
    (features / "meta.json").write_text(json.dumps({"run": {"init": {"sha256": "x"}, "with_logits": True, "logit_layout_sha256": "old"}}))
    with pytest.raises(ValueError, match="another taxonomy"):
        predict_features("logits", None, features, {"a"})
