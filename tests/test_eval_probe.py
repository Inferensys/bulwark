from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.eval.probe import (
    cross_validate,
    drive_and_act_midlevel,
    run_probe,
    subject_folds,
    uta_rldd_folds,
)

DIM = 8


def save_features(directory: Path, ids: list[str], signal: list[np.ndarray], *, logits: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tokens = rng.normal(size=(len(ids), 25, DIM))
    tokens[:, 0, : signal[0].shape[0]] += np.stack(signal)
    extra = {"logit__yawn": np.zeros(len(ids), dtype=np.float32)} if logits else {}
    np.savez(directory / "features-0.npz", clip_ids=np.array(ids), tokens=tokens.astype(np.float16), **extra)
    return directory


def rec(clip_id: str, dataset: str, subject: str, labels: dict, **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset=dataset, subject_id=f"{dataset}:{subject}", media_path=kwargs.pop("media_path", clip_id),  # type: ignore[arg-type]
        media_type="video", licence="t", commercial_ok=False, split="train", labels_gt=labels, **kwargs,  # type: ignore[arg-type]
    )


def test_subject_folds_are_balanced_and_stable() -> None:
    folds = subject_folds([f"s{i}" for i in range(11)] * 3, 5)
    assert sorted(np.bincount(list(folds.values()))) == [2, 2, 2, 2, 3]
    assert folds == subject_folds([f"s{i}" for i in range(11)], 5)


def test_cross_validation_over_subjects(tmp_path: Path) -> None:
    records, signal = [], []
    for s in range(10):
        for i in range(8):
            closed = i % 2
            records.append(rec(f"m/{s}/{i}", "mrl_eye", f"s{s}", {"eyes_closed": float(closed), "eyeglasses": 1.0, "modality": "nir"}))
            signal.append(np.array([3.0 * closed, 0.0]))
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    features = save_features(tmp_path / "features", [r.clip_id for r in records], signal)
    result = cross_validate(features, manifest, datasets=["mrl_eye", "absent"], folds=5)
    labels = result["mrl_eye"]["labels"]
    assert labels["eyes_closed"]["auroc"]["mean"] > 0.95 and labels["eyes_closed"]["auroc"]["defined_folds"] == 5
    assert labels["eyes_closed"]["n_subjects"] == 10
    assert "skipped" in labels["eyeglasses"] and "modality" not in labels  # one class; metadata
    assert "skipped" in result["absent"]


def test_probes_refuse_fine_tuned_features(tmp_path: Path) -> None:
    records = [rec("a", "mrl_eye", "s", {"eyes_closed": 1.0})]
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    features = save_features(tmp_path / "features", ["a"], [np.zeros(1)], logits=True)
    with pytest.raises(ValueError, match="fine-tuned"):
        cross_validate(features, manifest, datasets=["mrl_eye"])


def test_drive_and_act_official_protocol(tmp_path: Path) -> None:
    parts = {  # participants per official split
        0: {"train": [1, 2, 3, 4, 5], "val": [6], "test": [7, 8]},
        1: {"train": [3, 4, 5, 6, 7], "val": [8], "test": [1, 2]},
        2: {"train": [1, 2, 6, 7, 8], "val": [3], "test": [4, 5]},
    }
    activities = ["drinking", "eating", "talking_on_phone"]
    records, signal = [], []
    for view in ("kinect_color", "ceiling"):
        for p in range(1, 9):
            for i in range(12):
                cls = i % 3
                records.append(rec(f"da/{view}/{p}/{i}", "drive_and_act", f"vp{p}", {"activity": "other"}, view=view, meta={"native_activity": activities[cls]}))
                signal.append(np.eye(3)[cls] * 3.0)
    for split, groups in parts.items():
        for part, people in groups.items():
            for view in ("kinect_color", "ceiling", "kinect_depth"):
                path = tmp_path / "root" / "activities_3s" / view / f"midlevel.chunks_90.split_{split}.{part}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = ["participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id"]
                rows += [f"{p},vp{p}/run,{i},0,90,x,0" for p in people for i in range(13)]  # one short chunk each
                path.write_text("\n".join(rows) + "\n")
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    features = save_features(tmp_path / "features", [r.clip_id for r in records], signal)
    result = drive_and_act_midlevel(features, manifest, tmp_path / "root")
    assert result["classes"] == sorted(activities) and set(result["views"]) == {"ceiling", "kinect_color"}
    view = result["views"]["kinect_color"]
    assert view["balanced_accuracy_mean"] > 0.9 and len(view["splits"]) == 3
    assert view["splits"][0]["test_clips"] == 24 and view["splits"][0]["excluded_test_chunks"] == 2


def test_uta_rldd_fold_probe_and_written_report(tmp_path: Path) -> None:
    records, signal = [], []
    states = {"0": "alert", "5": "low_vigilance", "10": "drowsy"}
    for fold in range(1, 4):
        for s in range(2):
            for k, (state, alert) in enumerate(states.items()):
                for w in range(4):
                    records.append(rec(
                        f"u/{fold}/{s}/{state}/{w}", "uta_rldd", f"{fold}{s}", {"alertness": alert},
                        media_path=f"Fold{fold}_part1/{s}/{state}.mp4", meta={"fold": f"Fold{fold}_part{1 + s}"},
                    ))
                    signal.append(np.eye(3)[k] * 3.0)
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    features = save_features(tmp_path / "features", [r.clip_id for r in records], signal)
    result = uta_rldd_folds(features, manifest)
    assert result["folds"] == ["Fold1", "Fold2", "Fold3"]
    assert result["clips"]["accuracy"]["mean"] > 0.9 and result["video_accuracy"]["mean"] > 0.9
    path = run_probe("uta_rldd_folds", features, manifest, tmp_path / "out")
    payload = json.loads((tmp_path / "out" / "probe-uta_rldd_folds.json").read_text())
    assert payload["meta"]["protocol"] == "uta_rldd_folds" and "video accuracy" in path.read_text()
