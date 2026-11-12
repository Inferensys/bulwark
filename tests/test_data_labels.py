"""Label definitions, adapter label fixes and teacher merge rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bulwark.data.label_rules import eyes_closed_from_mask
from bulwark.data.manifest import ClipRecord
from bulwark.data.sources import get_adapter
from bulwark.data.sources.uta_rldd import STATE_LABELS
from bulwark.label.agreement import teacher_agreement
from bulwark.label.face_teacher import FrameFace, summarise
from bulwark.label.objects_teacher import summarise_detections
from bulwark.label.run import merge_teacher_labels


def face(closure: float = 0.0, jaw: float = 0.0, yaw: float = 0.0, pitch: float = 0.0) -> FrameFace:
    return FrameFace(closure=closure, jaw=jaw, yaw=yaw, pitch=pitch, roll=0.0)


def record(
    clip_id: str,
    dataset: str = "dmd",
    view: str = "face",
    media_type: str = "video",
    **kwargs: object,
) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        dataset=dataset,
        subject_id=f"{dataset}:s1",
        media_path=f"{clip_id}.mp4",
        media_type=media_type,
        licence="t",
        commercial_ok=False,
        view=view,
        **kwargs,  # type: ignore[arg-type]
    )


def write_teacher(labels_dir: Path, teacher: str, rows: dict[str, dict[str, object]]) -> None:
    labels_dir.mkdir(exist_ok=True)
    lines = [json.dumps({"clip_id": clip_id, "labels": labels}) for clip_id, labels in rows.items()]
    (labels_dir / f"{teacher}.jsonl").write_text("\n".join(lines) + "\n")


# ---- eyes_closed: one definition for adapters and the face teacher ------------------------


ONSET = [False] * 8 + [True] * 8  # eyes close half way through
RECOVERY = [True] * 10 + [False] * 6  # closed for most of the clip, open at the end


def test_eyes_closed_uses_the_last_half_of_the_clip() -> None:
    assert eyes_closed_from_mask(ONSET) == 1.0
    assert eyes_closed_from_mask(RECOVERY) == 0.0
    assert eyes_closed_from_mask([]) == 0.0


def test_face_teacher_eyes_closed_matches_the_shared_rule() -> None:
    for mask in (ONSET, RECOVERY):
        labels = summarise([face(0.9 if closed else 0.0) for closed in mask])
        assert labels["eyes_closed"] == eyes_closed_from_mask(mask)


def test_face_teacher_emits_no_events_for_a_still_image() -> None:
    still = summarise([face(jaw=0.9)])
    assert not {"blink", "head_nod", "yawn"} & set(still)
    assert still["mouth_open"] == 0.9 and "eyes_closed" in still


# ---- adapter label fixes ------------------------------------------------------------------


def test_uta_rldd_kss_is_the_midpoint_of_each_class_range() -> None:
    def kss(level: float) -> float:
        return (level - 1) / 8

    assert STATE_LABELS["0"]["kss"] == kss(2)  # KSS 1-3
    assert STATE_LABELS["5"]["kss"] == kss(6.5)  # KSS 6-7
    assert STATE_LABELS["10"]["kss"] == kss(8.5)  # KSS 8-9


def test_zenodo_turning_has_no_direction(tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    make_image(tmp_path / "turning" / "IMG_0003.jpg")
    (turning,) = get_adapter("zenodo_driver", tmp_path).records()
    assert "attention_direction" not in turning.labels_gt
    assert turning.meta["native_class"] == "turning"


def test_nthu_refuses_ambiguous_annotation_files(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "001" / "glasses"
    make_video(folder / "yawning.avi", frames=60)
    (folder / "001_glasses_yawning_mouth.txt").write_text("1" * 60)
    (folder / "002_glasses_yawning_mouth.txt").write_text("0" * 60)
    with pytest.raises(ValueError, match="Ambiguous"):
        list(get_adapter("nthu_ddd", tmp_path).records())


# ---- merge rules ---------------------------------------------------------------------------


def test_face_direction_is_kept_only_for_frontal_cameras(tmp_path: Path) -> None:
    face_out = {"attention_direction": "left", "head_yaw": 0.5, "face_visible": 1.0}
    side = record("side", dataset="state_farm", view="side", media_type="image")
    frontal = record("front")
    cabin = record("cabin", dataset="drive_and_act", view="a_column_driver")
    missed = record("missed", dataset="drive_and_act", view="ceiling")
    write_teacher(
        tmp_path,
        "face",
        {"side": face_out, "front": face_out, "cabin": face_out, "missed": {"face_visible": 0.0}},
    )
    merged = {r.clip_id: r for r in merge_teacher_labels([side, frontal, cabin, missed], tmp_path)}
    assert merged["front"].labels_teacher == face_out
    assert merged["side"].labels_teacher == {"head_yaw": 0.5, "face_visible": 1.0}
    assert "attention_direction" not in merged["cabin"].labels_teacher
    assert merged["missed"].labels_teacher == {}  # a missed face on a cabin camera says little


def test_face_teacher_is_ignored_on_eye_crops(tmp_path: Path) -> None:
    eye = record(
        "eye", dataset="mrl_eye", view="ids", media_type="image", meta={"crop_kind": "eye"}
    )
    # Output the off-axis rule keeps (mrl_eye is not a frontal view): only the eye-crop
    # rule drops it.
    write_teacher(tmp_path, "face", {"eye": {"head_yaw": 0.2, "face_visible": 1.0}})
    assert merge_teacher_labels([eye], tmp_path)[0].labels_teacher == {}


def test_face_teacher_keeps_open_eyes_and_drops_blink_and_yawn(tmp_path: Path) -> None:
    eyes_open = {"eyes_closed": 0.0, "eye_openness": 0.9, "perclos": 0.1}
    closed = {"eyes_closed": 1.0, "eye_openness": 0.3, "perclos": 0.6, "blink": 0.0, "yawn": 0.0}
    opened = {**eyes_open, "blink": 1.0, "yawn": 1.0}
    write_teacher(
        tmp_path,
        "face",
        {
            "closed": {**closed, "mouth_open": 0.1, "face_visible": 1.0},
            "open": opened,
            "cabin": {**opened, "head_yaw": 0.2},
        },
    )
    clips = [
        record("closed"),
        record("open"),
        record("cabin", dataset="drive_and_act", view="a_column_driver"),
    ]
    merged = {r.clip_id: r for r in merge_teacher_labels(clips, tmp_path)}
    # "Closed" goes with the eye_openness and perclos read from the same closure measure.
    # Blink and yawn go whatever the teacher says, 0 or 1: the face teacher labels neither.
    assert merged["closed"].labels_teacher == {"mouth_open": 0.1, "face_visible": 1.0}
    assert merged["open"].labels_teacher == eyes_open
    # The kept values were measured on a camera facing the person: none survive a cabin
    # camera.
    assert merged["cabin"].labels_teacher == {"head_yaw": 0.2}


def test_ground_truth_still_wins_over_kept_face_values(tmp_path: Path) -> None:
    gt = {"eyes_closed": 1.0, "blink": 0.0, "perclos": 0.7}
    clip = record("gt", labels_gt=dict(gt))
    write_teacher(
        tmp_path,
        "face",
        {"gt": {"eyes_closed": 0.0, "eye_openness": 0.9, "perclos": 0.1, "blink": 1.0}},
    )
    (merged,) = merge_teacher_labels([clip], tmp_path)
    assert merged.labels_gt == gt
    # Ground truth says closed, so the teacher's open-eye reading is wrong here: its
    # eye_openness, from the same closure measure, goes with it.
    assert merged.labels_teacher == {}
    # Where ground truth agrees, it still replaces the teacher's kept eyes_closed and
    # perclos; only the teacher's eye_openness, which no ground truth gives, is added.
    agreed = {"eyes_closed": 0.0, "blink": 1.0, "yawn": 0.0, "perclos": 0.0}
    said = {"eyes_closed": 0.0, "eye_openness": 0.9, "perclos": 0.1, "blink": 1.0, "yawn": 1.0}
    write_teacher(tmp_path, "face", {"ok": said})
    (merged,) = merge_teacher_labels([record("ok", labels_gt=dict(agreed))], tmp_path)
    assert merged.labels_gt == agreed
    assert merged.labels_teacher == {"eye_openness": 0.9}


def test_ground_truth_eyes_closed_decides_the_face_eye_openness(tmp_path: Path) -> None:
    def clip(clip_id: str, truth: float) -> ClipRecord:
        return record(clip_id, labels_gt={"eyes_closed": truth})

    def reading(closed: float) -> dict[str, object]:
        return {"eyes_closed": closed, "eye_openness": 0.9 - 0.7 * closed, "perclos": closed}

    write_teacher(
        tmp_path,
        "face",
        {"tp": reading(1.0), "fp": reading(1.0), "fn": reading(0.0), "tn": reading(0.0)},
    )
    clips = [clip("tp", 1.0), clip("fp", 0.0), clip("fn", 1.0), clip("tn", 0.0)]
    merged = {r.clip_id: r.labels_teacher for r in merge_teacher_labels(clips, tmp_path)}
    # Closed eyes that ground truth confirms keep their eye_openness and perclos, even
    # though FACE_KEEP never keeps the teacher's "closed" on its own.
    assert merged["tp"] == {"eye_openness": pytest.approx(0.2), "perclos": 1.0}
    assert merged["fp"] == {}  # looking down read as closed
    assert merged["fn"] == {}
    assert merged["tn"] == {"eye_openness": 0.9, "perclos": 0.0}
    # Without ground truth, FACE_KEEP decides: "closed" goes with both.
    write_teacher(tmp_path, "face", {"x": reading(1.0)})
    (alone,) = merge_teacher_labels([record("x")], tmp_path)
    assert alone.labels_teacher == {}


def test_event_labels_are_dropped_for_images(tmp_path: Path) -> None:
    # FACE_KEEP drops the face teacher's blink and yawn before the image rule runs, so they
    # come from sources it does not filter here: the prompt and a derived label.
    image = record(
        "img",
        dataset="dmd",
        media_type="image",
        meta={"prompted_labels": {"blink": 1.0, "yawn": 1.0}, "derived_labels": {"yawn": 1.0}},
    )
    write_teacher(tmp_path, "face", {"img": {"head_nod": 0.0, "eyes_closed": 0.0}})
    assert merge_teacher_labels([image], tmp_path)[0].labels_teacher == {"eyes_closed": 0.0}


def test_old_yawn_promotion_is_undone(tmp_path: Path) -> None:
    yawning = record(
        "y", dataset="yawdd", view="dash", labels_gt={"yawn": 1.0}, meta={"yawn_video": True}
    )
    talking = record("t", dataset="yawdd", view="mirror", labels_gt={"yawn": 0.0})
    write_teacher(tmp_path, "face", {"y": {"yawn": 1.0}, "t": {"yawn": 1.0}})
    merged = {r.clip_id: r for r in merge_teacher_labels([yawning, talking], tmp_path)}
    # The copied yawn leaves labels_gt, and the face teacher's yawn is not merged either.
    assert merged["y"].labels_gt == {} and merged["y"].labels_teacher == {}
    assert merged["t"].labels_gt == {"yawn": 0.0}  # file-name ground truth stays


def test_prompted_labels_are_weak_and_teachers_override_them(tmp_path: Path) -> None:
    synthetic = record(
        "s",
        dataset="synthetic_wan",
        view="",
        meta={"prompted_labels": {"alertness": "microsleep", "head_nod": 1.0}},
    )
    write_teacher(tmp_path, "face", {"s": {"head_nod": 0.0}})
    (merged,) = merge_teacher_labels([synthetic], tmp_path)
    assert merged.labels_gt == {}
    assert merged.labels_teacher == {"alertness": "microsleep", "head_nod": 0.0}


def test_prompted_labels_survive_where_the_face_teacher_value_is_dropped(
    tmp_path: Path,
) -> None:
    # The prompt is a separate source: dropping the face teacher's values must not drop
    # it, and the teacher's kept eyes_closed still overrides it.
    prompted = {"eyes_closed": 1.0, "blink": 0.0, "yawn": 1.0}
    face_out = {"eyes_closed": 1.0, "perclos": 0.8, "eye_openness": 0.2, "blink": 1.0, "yawn": 0.0}
    frontal = record("f", dataset="dmd", view="face", meta={"prompted_labels": dict(prompted)})
    synthetic = record(
        "s", dataset="synthetic_wan", view="", meta={"prompted_labels": dict(prompted)}
    )
    other = {"eyes_closed": 1.0, "blink": 1.0, "yawn": 0.0}
    kept = record("k", dataset="dmd", view="face", meta={"prompted_labels": dict(other)})
    write_teacher(
        tmp_path,
        "face",
        {"f": face_out, "s": face_out, "k": {"eyes_closed": 0.0, "blink": 0.0, "yawn": 1.0}},
    )
    merged = {r.clip_id: r for r in merge_teacher_labels([frontal, synthetic, kept], tmp_path)}
    # The teacher's blink and yawn, 1 or 0, never replace the prompt's.
    assert merged["f"].labels_teacher == prompted  # no teacher perclos or eye_openness
    assert merged["s"].labels_teacher == prompted
    assert merged["k"].labels_teacher == {**other, "eyes_closed": 0.0}


# ---- teacher quality -----------------------------------------------------------------------


def test_objects_teacher_only_writes_negatives_with_a_margin() -> None:
    frames = [{"phone": 0.9, "drink": 0.3}, {"phone": 0.8}, {}, {"phone": 0.7}]
    # A phone in view is not a phone in use, so no detection makes a positive; a weak
    # drink detection blocks the negative.
    assert summarise_detections(frames, negative_threshold=0.2) == {"food": 0.0}
    assert summarise_detections([], 0.2) == {}


def test_teacher_agreement_against_ground_truth(tmp_path: Path) -> None:
    clips = [
        record("a", labels_gt={"yawn": 1.0, "activity": "texting"}),
        record("b", labels_gt={"yawn": 0.0}),
        record("c", labels_gt={"yawn": 1.0}),
    ]
    write_teacher(tmp_path, "face", {"a": {"yawn": 1.0}, "b": {"yawn": 1.0}, "c": {"yawn": 0.0}})
    write_teacher(tmp_path, "vlm", {"a": {"activity": "texting", "_caption": "x"}})
    report = teacher_agreement(clips, tmp_path)
    yawn = report["face"]["yawn"]["dmd"]
    assert (yawn["n"], yawn["tp"], yawn["fp"], yawn["fn"]) == (3, 1, 1, 1)
    assert yawn["precision"] == 0.5 and yawn["recall"] == 0.5
    assert yawn["npv"] == 0.0  # the teacher's one "no yawn" was wrong
    assert report["vlm"]["activity"]["dmd"] == {"n": 1, "accuracy": 1.0}


def test_teacher_agreement_npv_and_split_filter(tmp_path: Path) -> None:
    gt = {"t1": 0.0, "t2": 0.0, "t3": 1.0, "t4": 1.0, "v1": 1.0, "v2": 0.0}
    said = {"t1": 0.0, "t2": 0.0, "t3": 0.0, "t4": 1.0, "v1": 0.0, "v2": 1.0}
    clips = [
        record(c, split="train" if c.startswith("t") else "val", labels_gt={"blink": g})
        for c, g in gt.items()
    ]
    write_teacher(tmp_path, "face", {c: {"blink": v} for c, v in said.items()})
    train = teacher_agreement(clips, tmp_path, splits={"train"})["face"]["blink"]["dmd"]
    assert (train["n"], train["tn"], train["fn"], train["tp"], train["fp"]) == (4, 2, 1, 1, 0)
    assert train["npv"] == 2 / 3 and train["precision"] == 1.0
    every = teacher_agreement(clips, tmp_path)["face"]["blink"]["dmd"]
    assert every["n"] == 6 and every["npv"] == 2 / 4 and every["precision"] == 1 / 2
    assert teacher_agreement(clips, tmp_path, splits={"test"}) == {}
    # No negatives from the teacher: npv is undefined.
    write_teacher(tmp_path, "face", {c: {"blink": 1.0} for c in said})
    assert teacher_agreement(clips, tmp_path)["face"]["blink"]["dmd"]["npv"] is None


def test_taxonomy_marks_recording_metadata_and_units() -> None:
    from bulwark.data.taxonomy import default_taxonomy

    taxonomy = default_taxonomy()
    flagged = {name for name, spec in taxonomy.labels.items() if spec.metadata}
    assert flagged == {"context", "modality", "low_light", "face_visible"}
    assert (taxonomy["head_yaw"].unit, taxonomy["head_yaw"].unit_scale) == ("deg", 90.0)
    assert (taxonomy["kss"].unit, taxonomy["kss"].unit_scale) == ("KSS points", 8.0)
    assert all(spec.kind == "regression" for spec in taxonomy.labels.values() if spec.unit)
