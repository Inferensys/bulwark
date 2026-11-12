"""Metadata the dataset source adapters record: DMD derived microsleep, Drive&Act events,
MRL Eye gender and the copy it was ingested from."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bulwark.data.manifest import ClipRecord
from bulwark.data.registry import get_spec
from bulwark.data.sources import get_adapter
from bulwark.data.sources.base import copy_fingerprint
from bulwark.data.sources.dmd import longest_closure
from bulwark.data.sources.mrl_eye import MrlEyeAdapter
from bulwark.label.run import merge_teacher_labels


def dmd_drowsiness_session(root: Path, make_video, eyes: dict[int, str]) -> None:  # type: ignore[no-untyped-def]
    folder = root / "gA" / "1" / "s5"
    stem = "gA_1_s5_2019-03-14T14;26;17+01;00"
    make_video(folder / f"{stem}_rgb_face.mp4", frames=150)
    actions = {
        str(i): {
            "type": f"eyes_state/{state}",
            "frame_intervals": [{"frame_start": f, "frame_end": f}],
        }
        for i, (f, state) in enumerate(sorted(eyes.items()))
    }
    (folder / f"{stem}_rgb_ann_drowsiness.json").write_text(
        json.dumps({"openlabel": {"actions": actions}})
    )


def test_longest_closure_counts_consecutive_closed_frames() -> None:
    eyes = (
        dict.fromkeys(range(10, 25), "close")
        | {25: "closing"}
        | dict.fromkeys(range(26, 30), "close")
    )
    assert longest_closure({"eyes_state": eyes}, 0, 48) == 15
    assert longest_closure({"eyes_state": eyes}, 20, 48) == 5
    assert longest_closure({}, 0, 48) == 0


def test_dmd_microsleep_is_a_derived_label_not_ground_truth(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    # 30 fps: eyes closed for frames 0-59 (2 s), a 10-frame blink at 100-109, open otherwise.
    eyes = {f: ("close" if f < 60 or 100 <= f < 110 else "open") for f in range(150)}
    dmd_drowsiness_session(tmp_path, make_video, eyes)
    records = {r.start_frame: r for r in get_adapter("dmd", tmp_path).records()}
    # Windows start every 30 frames and cover 48 annotated frames (16 frames, stride 3).
    assert sorted(records) == [0, 30, 60, 90]
    microsleep = {
        start for start, r in records.items() if r.labels_teacher.get("alertness") == "microsleep"
    }
    assert microsleep == {0, 30}  # 48 and 30 closed frames in a row; not the blink (90)
    for record in records.values():
        record.validate()
        assert "alertness" not in record.labels_gt
    assert records[0].meta["derived_labels"] == {"alertness": "microsleep"}
    assert (
        records[0].labels_gt["eyes_closed"] == 1.0 and records[30].labels_gt["eyes_closed"] == 0.0
    )


def test_derived_labels_survive_a_merge_and_outrank_teachers(tmp_path: Path) -> None:
    def record(clip_id: str, **kwargs: object) -> ClipRecord:
        return ClipRecord(
            clip_id=clip_id, dataset="dmd", subject_id="dmd:gA_1", media_path="v.mp4",
            media_type="video", licence="t", commercial_ok=False, view="face", **kwargs,  # type: ignore[arg-type]
        )  # fmt: skip

    derived = {"derived_labels": {"alertness": "microsleep"}}
    weak = record("weak", meta=dict(derived), labels_teacher={"alertness": "microsleep"})
    graded = record("graded", meta=dict(derived), labels_gt={"alertness": "drowsy"})
    labels = tmp_path / "labels"
    labels.mkdir()
    rows = [
        {"clip_id": c, "labels": {"alertness": "alert", "activity": "talking"}}
        for c in ("weak", "graded")
    ]
    (labels / "vlm.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    merge_teacher_labels([weak, graded], labels)
    assert weak.labels_teacher == {"alertness": "microsleep", "activity": "talking"}
    assert "alertness" not in graded.labels_teacher  # ground truth wins


def drive_and_act_views(root: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    header = "participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id\n"
    for view, suffix, fps, scale in (
        ("inner_mirror", "ids_1", 30.0, 1),
        ("kinect_color", "kinect_color", 15.0, 2),
    ):
        make_video(
            root / view / "vp3" / f"run1_2018-05-22-15-10-41.{suffix}.mp4",
            frames=200 // scale,
            fps=fps,
        )
        folder = root / "activities_3s" / view
        folder.mkdir(parents=True)
        (folder / "midlevel.chunks_90.split_0.train.csv").write_text(
            header
            + f"3,vp3/run1_2018-05-22-15-10-41.{suffix},7,0,{90 // scale},talking_on_phone,0\n"
            + f"3,vp3/run1_2018-05-22-15-10-41.{suffix},7,{90 // scale},{180 // scale},talking_on_phone,1\n"
        )


def test_drive_and_act_event_id_is_shared_across_views(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    drive_and_act_views(tmp_path, make_video)
    records = list(get_adapter("drive_and_act", tmp_path).records())
    by_event: dict[str, set[str]] = {}
    for record in records:
        by_event.setdefault(record.meta["event_id"], set()).add(record.view)
    assert by_event == {
        "drive_and_act/vp3/run1_2018-05-22-15-10-41/7/0": {"inner_mirror", "kinect_color"},
        "drive_and_act/vp3/run1_2018-05-22-15-10-41/7/1": {"inner_mirror", "kinect_color"},
    }
    assert len({r.clip_id for r in records}) == 4


@pytest.mark.filterwarnings("ignore:MRL Eye")  # the fixture is not the full release
def test_mrl_eye_gender_comes_from_the_file_name(tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    make_image(tmp_path / "s0001" / "s0001_00001_0_0_1_0_1_01.png")
    make_image(tmp_path / "s0014" / "s0014_00002_1_0_1_0_1_02.png")
    genders = {r.subject_id: r.meta["gender"] for r in get_adapter("mrl_eye", tmp_path).records()}
    assert genders == {"mrl_eye:s0001": "male", "mrl_eye:s0014": "female"}


@pytest.mark.filterwarnings("ignore:MRL Eye")  # the fixture is not the full release
def test_mrl_eye_names_the_registry_copy_only_when_the_fingerprint_matches(
    tmp_path: Path, make_image, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    make_image(tmp_path / "s0001" / "s0001_00001_0_0_1_0_1_01.png")
    fingerprint, _ = copy_fingerprint(tmp_path)
    registered = get_spec("mrl_eye")
    assert registered.source_copy and registered.source_copy_fingerprint != fingerprint
    # The same files in another folder layout (as in a mirror) fingerprint the same.
    monkeypatch.setattr(
        MrlEyeAdapter, "spec", property(lambda _: replace(registered, source_copy_fingerprint=fingerprint))
    )
    make_image(tmp_path / "mirror" / "s0001_00001_0_0_1_0_1_01.png")
    (record,) = get_adapter("mrl_eye", tmp_path / "mirror").records()
    assert record.meta["source_copy"] == registered.source_copy and "copy_fingerprint" not in record.meta
