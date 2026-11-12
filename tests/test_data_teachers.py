"""Teacher inputs and merge rules, and the Drive&Act views that are ingested."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bulwark.data.manifest import ClipRecord
from bulwark.data.sources import get_adapter
from bulwark.data.taxonomy import default_taxonomy
from bulwark.label.objects_teacher import ObjectsTeacher
from bulwark.label.run import merge_teacher_labels, teacher_applies
from bulwark.label.vlm_teacher import build_prompt


def test_vlm_prompt_says_what_it_is_shown() -> None:
    taxonomy = default_taxonomy()
    still = build_prompt(taxonomy, 1)
    assert "a still image" in still and "frames" not in still
    clip = build_prompt(taxonomy, 4)
    assert "(4 frames in time order)" in clip
    assert all(name in clip for name in taxonomy["activity"].classes)


def test_objects_teacher_names_the_sizes_it_supports() -> None:
    # Checked before RF-DETR is imported, so a wrong --size fails with a clear message.
    with pytest.raises(ValueError, match="nano, small, medium"):
        ObjectsTeacher("large")


def test_face_objects_and_vlm_output_is_dropped_on_eye_crops(tmp_path: Path) -> None:
    def clip(clip_id: str, dataset: str, view: str, meta: dict[str, object]) -> ClipRecord:
        return ClipRecord(
            clip_id=clip_id,
            dataset=dataset,
            subject_id=f"{dataset}:s0001",
            media_path=f"{clip_id}.png",
            media_type="image",
            licence="test",
            commercial_ok=False,
            view=view,
            meta=meta,
        )

    eye = clip("eye", "mrl_eye", "ids", {"crop_kind": "eye"})
    face = clip("face", "dmd", "face", {})  # a frontal camera: every teacher label is kept
    teachers = {
        "vlm": {"alertness": "drowsy", "context": "other", "_caption": "a driver"},
        # Survives the off-axis rule (mrl_eye is not a frontal view), so only the eye-crop
        # rule can drop it.
        "face": {"head_yaw": 0.2, "face_visible": 1.0},
        "objects": {"phone": 0.0},
    }
    for name, labels in teachers.items():
        rows = [json.dumps({"clip_id": c, "labels": labels}) for c in ("eye", "face")]
        (tmp_path / f"{name}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    merged = {r.clip_id: r for r in merge_teacher_labels([eye, face], tmp_path)}
    assert merged["eye"].labels_teacher == {}
    assert "caption" not in merged["eye"].meta
    assert merged["face"].labels_teacher == {
        "alertness": "drowsy",
        "context": "other",
        "face_visible": 1.0,
        "head_yaw": 0.2,
        "phone": 0.0,
    }
    assert merged["face"].meta["caption"] == "a driver"
    assert {t for t in teachers if teacher_applies(t, eye)} == set()
    assert {t for t in teachers if teacher_applies(t, face)} == set(teachers)


def test_drive_and_act_skips_the_kinect_depth_view(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    header = "participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id\n"
    for view, suffix in (("inner_mirror", "ids_1"), ("kinect_depth", "kinect_depth")):
        make_video(tmp_path / view / "vp3" / f"run1_2018-05-22-15-10-41.{suffix}.mp4", frames=120)
        folder = tmp_path / "activities_3s" / view
        folder.mkdir(parents=True)
        (folder / "midlevel.chunks_90.split_0.train.csv").write_text(
            header + f"3,vp3/run1_2018-05-22-15-10-41.{suffix},7,0,90,drinking,0\n"
        )
    records = list(get_adapter("drive_and_act", tmp_path).records())
    assert {r.view for r in records} == {"inner_mirror"}
    assert all(r.labels_gt["modality"] == "nir" for r in records)


def test_perclos_doc_names_the_rolling_mean_and_its_default() -> None:
    from bulwark.inference import PERCLOS_SECONDS, RollingMean

    doc = default_taxonomy()["perclos"].doc
    assert f"bulwark.inference.{RollingMean.__name__}" in doc
    assert f"{PERCLOS_SECONDS:g} s by default" in doc
