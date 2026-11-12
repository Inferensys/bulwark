"""Each dataset source adapter (`bulwark.data.sources`) against a tiny fake copy of its
dataset's native layout."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from bulwark.data.sources import get_adapter
from bulwark.data.sources.auc_v2 import CLASS_LABELS as AUC_CLASSES
from bulwark.data.sources.base import clip_stride, copy_fingerprint
from bulwark.data.sources.dmd import absent_object_labels as dmd_absent
from bulwark.data.sources.dmd import load_tracks as dmd_tracks
from bulwark.data.sources.dmd import window_labels as dmd_window
from bulwark.data.sources.nthu_ddd import window_labels as nthu_window


def labels_of(records):  # type: ignore[no-untyped-def]
    return [r.validate().labels_gt for r in records]


def test_state_farm(tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "driver_imgs_list.csv").write_text(
        "subject,classname,img\np002,c0,a.jpg\np002,c4,b.jpg\np012,c6,c.jpg\n"
    )
    for cls, name in (("c0", "a"), ("c4", "b"), ("c6", "c")):
        make_image(tmp_path / "imgs" / "train" / cls / f"{name}.jpg")
    records = list(get_adapter("state_farm", tmp_path).records())
    labels = labels_of(records)
    assert [r.subject_id for r in records] == [
        "state_farm:p002",
        "state_farm:p002",
        "state_farm:p012",
    ]
    assert labels[0]["activity"] == "attentive"
    assert labels[1] | {"activity": "phone_call", "phone_at_ear": 1.0} == labels[1]
    assert labels[2]["drink"] == 1.0


def test_auc_v2(tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    archive = tmp_path / "auc_v2"
    root = archive / "v2_cam1_cam2_ split_by_driver"  # the archive's name, space included
    for camera, split, cls, name in (
        ("Camera 1", "train", "c0", "1000.jpg"),
        ("Camera 1", "train", "c2", "1.jpg"),
        ("Camera 1", "test", "c1", "100.jpg"),
        ("Camera 1", "test", "c0", "321.jpg"),
        ("Camera 1", "test", "c0", "322.jpg"),
        ("Camera 1", "test", "c0", "415.jpg"),
        ("Camera 1", "test", "c1", "322.jpg"),
        ("Camera 2", "train", "c6", "0190.jpg"),
        ("Camera 2", "test", "c4", "03784.jpg"),
        ("Camera 2", "test", "c7", "012673.jpg"),
    ):
        make_image(root / camera / split / cls / name)
    (root / "skin_nonskin_pixels").mkdir()
    (root / "skin_nonskin_pixels" / "read_me.txt").write_text("not images")
    make_image(archive / "v1_cam1_no_split" / "Drive Safe" / "1000.jpg")  # v1: never read
    records = list(get_adapter("auc_v2", root).records())
    by_path = {r.media_path: r for r in records}
    assert sorted(by_path) == [
        "Camera 1/test/c0/321.jpg",
        "Camera 1/test/c0/322.jpg",
        "Camera 1/test/c0/415.jpg",
        "Camera 1/test/c1/100.jpg",
        "Camera 1/test/c1/322.jpg",
        "Camera 1/train/c0/1000.jpg",
        "Camera 1/train/c2/1.jpg",
        "Camera 2/test/c4/03784.jpg",
        "Camera 2/test/c7/012673.jpg",
        "Camera 2/train/c6/0190.jpg",
    ]
    labels = {path: r.validate().labels_gt for path, r in by_path.items()}
    common = {"context": "vehicle_cabin", "modality": "rgb"}
    # The class gives the object it shows in use, never a 0 for one it does not: drivers in
    # other classes hold a phone (safe driving and reaching behind included).
    assert labels["Camera 1/train/c0/1000.jpg"] == {**common, "activity": "attentive"}
    assert labels["Camera 1/test/c1/100.jpg"] == {
        **common, "activity": "texting", "phone": 1.0, "phone_at_ear": 0.0
    }  # fmt: skip
    assert labels["Camera 1/train/c2/1.jpg"] == {
        **common, "activity": "phone_call", "phone": 1.0, "phone_at_ear": 1.0
    }  # fmt: skip
    assert labels["Camera 2/train/c6/0190.jpg"] == {**common, "activity": "drinking", "drink": 1.0}
    assert labels["Camera 2/test/c7/012673.jpg"] == {**common, "activity": "reaching"}
    assert not any(
        value == 0.0 for given in AUC_CLASSES.values() for key, value in given.items()
        if key in {"phone", "drink"}
    )  # fmt: skip
    # No food (no eating class), no face_visible or attention_direction (see the adapter).
    assert not {"food", "face_visible", "attention_direction"} & {k for v in labels.values() for k in v}
    train, test = by_path["Camera 1/train/c0/1000.jpg"], by_path["Camera 2/test/c4/03784.jpg"]
    assert (train.subject_id, test.subject_id) == ("auc_v2:train_drivers", "auc_v2:test_drivers")
    assert (train.view, test.view) == ("cam1", "cam2") and test.media_type == "image"
    assert test.clip_id == "auc_v2/Camera 2/test/c4/03784.jpg"
    assert test.meta == {
        "official_split": {"v2": "test"}, "camera": "Camera 2", "native_class": "c4"
    }  # fmt: skip
    # Camera 1's test/c0 322-415 show a driver whose other classes are in train: they go to
    # train, and meta keeps the authors' side. Neighbouring numbers and other classes stay.
    for path in ("Camera 1/test/c0/322.jpg", "Camera 1/test/c0/415.jpg"):
        moved = by_path[path]
        assert moved.subject_id == "auc_v2:train_drivers" and moved.clip_id == f"auc_v2/{path}"
        assert moved.meta["official_split"] == {"v2": "train"} and moved.meta["authors_split"] == "test"
    for path in ("Camera 1/test/c0/321.jpg", "Camera 1/test/c1/322.jpg"):
        kept = by_path[path]
        assert kept.subject_id == "auc_v2:test_drivers" and kept.meta["official_split"] == {"v2": "test"}
        assert "authors_split" not in kept.meta
    # Pointed at the archive instead of its v2 folder, it says which folder to use.
    with pytest.raises(FileNotFoundError, match="v2_cam1_cam2_ split_by_driver"):
        list(get_adapter("auc_v2", archive).records())


def test_mrl_eye(tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    make_image(tmp_path / "mrlEyes_2018_01" / "s0001" / "s0001_00001_0_1_0_0_0_01.png")
    make_image(tmp_path / "mrlEyes_2018_01" / "s0002" / "s0002_00002_1_0_1_2_1_03.png")
    make_image(tmp_path / "mrlEyes_2018_01" / "s0002" / "not_a_sample.png")
    with pytest.warns(UserWarning, match="official release"):  # 2 of 84,898 images, 1 skipped
        closed, opened = list(get_adapter("mrl_eye", tmp_path).records())
    assert closed.labels_gt["eyes_closed"] == 1.0 and closed.labels_gt["eyeglasses"] == 1.0
    assert closed.labels_gt["low_light"] == 1.0 and closed.view == "realsense"
    assert opened.labels_gt["eyes_closed"] == 0.0 and opened.meta["reflections"] == 2
    # Not the copy the registry describes: the records name this copy, not that source.
    fingerprint, count = copy_fingerprint(tmp_path)
    assert "source_copy" not in opened.meta and count == 3
    assert opened.meta["copy_fingerprint"] == fingerprint and opened.meta["copy_image_count"] == 3


def test_uta_rldd_windows_and_states(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    make_video(tmp_path / "Fold1_part1" / "01" / "0.mp4", frames=30 * 25)
    make_video(tmp_path / "Fold1_part1" / "01" / "10.mp4", frames=30 * 25)
    records = list(get_adapter("uta_rldd", tmp_path).records())
    states = Counter(r.labels_gt["alertness"] for r in records)
    assert states == {"alert": 3, "drowsy": 3}  # 25 s at a 10 s hop, 2 s windows
    first = records[0]
    assert first.num_frames == 16 and first.frame_stride == 3 and first.start_frame == 0


def test_yawdd_filename_labels(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    mirror = tmp_path / "YawDD dataset" / "Mirror" / "Female_mirror"
    make_video(mirror / "3-FemaleSunGlasses-Talking.avi", frames=90)
    make_video(mirror / "3-FemaleSunGlasses-Yawning.avi", frames=90)
    make_video(tmp_path / "YawDD dataset" / "Dash" / "Male" / "7-MaleGlasses.avi", frames=90)
    records = list(get_adapter("yawdd", tmp_path).records())
    by_file = {Path(r.media_path).stem: r for r in records}
    talking = by_file["3-FemaleSunGlasses-Talking"]
    assert talking.labels_gt["activity"] == "talking" and talking.labels_gt["sunglasses"] == 1.0
    assert by_file["3-FemaleSunGlasses-Yawning"].meta == {"yawn_video": True}
    dash = by_file["7-MaleGlasses"]
    assert (
        dash.view == "dash"
        and dash.subject_id == "yawdd:male07"
        and "activity" not in dash.labels_gt
    )


def test_nthu_window_labels() -> None:
    annotations = {
        "drowsiness": "1" * 20 + "0" * 10,
        "eye": "1" * 18 + "0" * 12,
        "head": "0" * 10 + "1" * 2 + "0" * 18,
        "mouth": "2" * 30,
    }
    labels = nthu_window(annotations, 0, 30)
    # The eye track marks "sleepy eyes", not closure: no perclos / eyes_closed from it.
    assert labels["alertness"] == "drowsy" and not {"eyes_closed", "perclos"} & set(labels)
    assert labels["head_nod"] == 1.0 and labels["yawn"] == 0.0 and labels["activity"] == "talking"
    assert nthu_window({"drowsiness": "1" * 5}, 0, 30) == {}  # truncated annotation -> no label


def test_nthu_adapter(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "Training Dataset" / "004" / "night_noglasses"
    make_video(folder / "sleepyCombination.avi", frames=60)
    (folder / "004_night_noglasses_sleepyCombination_drowsiness.txt").write_text("1" * 60)
    (folder / "004_night_noglasses_sleepyCombination_mouth.txt").write_text("0" * 30 + "1" * 30)
    (folder / "004_night_noglasses_sleepyCombination_eye.txt").write_text("1" * 60)
    # "*sleepyCombination*" would also match this file, which sorts first.
    make_video(folder / "nonsleepyCombination.avi", frames=60)
    (folder / "004_night_noglasses_nonsleepyCombination_drowsiness.txt").write_text("0" * 60)
    records = list(get_adapter("nthu_ddd", tmp_path).records())
    sleepy = [r for r in records if r.meta["behaviour"] == "sleepyCombination"]
    awake = [r for r in records if r.meta["behaviour"] == "nonsleepyCombination"]
    assert sleepy and awake and all(r.labels_gt["modality"] == "nir" for r in records)
    assert {r.labels_gt["alertness"] for r in sleepy} == {"drowsy"}
    assert {r.labels_gt["alertness"] for r in awake} == {"alert"}
    assert sleepy[0].subject_id == "nthu_ddd:004" and sleepy[-1].labels_gt["yawn"] == 1.0
    assert sleepy[0].meta["sleepy_eye_fraction"] == 1.0 and "eyes_closed" not in sleepy[0].labels_gt


def test_drive_and_act(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    view = "inner_mirror"
    make_video(tmp_path / view / "vp3" / f"run1_2018-05-22-15-10-41.{view}.mp4", frames=200)
    csv_dir = tmp_path / "activities_3s" / view
    csv_dir.mkdir(parents=True)
    header = "participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id\n"
    (csv_dir / "midlevel.chunks_90.split_0.train.csv").write_text(
        header
        + f"3,vp3/run1_2018-05-22-15-10-41.{view},7,0,90,talking_on_phone,0\n"
        + f"3,vp3/run1_2018-05-22-15-10-41.{view},8,90,180,unknown_activity,1\n"
        + f"3,vp3/run1_2018-05-22-15-10-41.{view},9,100,148,looking_or_moving_around (e.g. searching),2\n"
        + f"3,vp3/run1_2018-05-22-15-10-41.{view},10,150,160,drinking,0\n"  # tail shorter than a clip
        + f"3,vp3/run1_2018-05-22-15-10-41.{view},11,199,199,eating,0\n"  # zero-length (ceiling bug)
        + f"3,vp3/run1_2018-05-22-15-10-41.{view},7,0,90,talking_on_phone,0\n"  # exact duplicate row
    )
    record, looking = list(get_adapter("drive_and_act", tmp_path).records())
    assert record.labels_gt["activity"] == "phone_call" and record.labels_gt["modality"] == "nir"
    # Same frame rate as every other video clip (2 s over 16 frames at 30 fps), centred.
    assert record.subject_id == "drive_and_act:vp3" and record.frame_stride == 3
    assert record.start_frame == 45 - 46 // 2
    assert looking.labels_gt["activity"] == "looking_away" and looking.frame_stride == 3
    # The id names the window read, so caches built with another sampling cannot match.
    assert record.clip_id.endswith(f".{view}.mp4@22s3:0") and looking.clip_id.endswith("@101s3:2")
    assert record.meta["official_split"] == {"0": "train"}
    # No object-level file for the view: no 0s from it.
    assert "drink" not in record.labels_gt and "phone" not in looking.labels_gt


def test_drive_and_act_object_negatives(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    view = "a_column_driver"
    run, bare = "vp3/run1_2018-05-22-15-10-41", "vp3/run2_2018-05-22-15-40-12"
    make_video(tmp_path / view / f"{run}.{view}.mp4", frames=300)
    make_video(tmp_path / view / f"{bare}.{view}.mp4", frames=100)
    csv_dir = tmp_path / "activities_3s" / view
    csv_dir.mkdir(parents=True)
    (csv_dir / "midlevel.chunks_90.split_0.train.csv").write_text(
        "participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id\n"
        f"3,{run}.{view},1,0,90,sitting_still,0\n"  # reads frames [22, 68)
        f"3,{run}.{view},2,90,180,sitting_still,0\n"  # [112, 158)
        f"3,{run}.{view},3,180,240,fetching_an_object,0\n"  # [187, 233)
        f"3,{run}.{view},4,240,300,eating,0\n"  # [247, 293)
        f"3,{bare}.{view},1,0,90,sitting_still,0\n"  # a run without object annotations
    )
    (csv_dir / "objectlevel.chunks_90.csv").write_text(
        "participant_id,file_id,annotation_id,frame_start,frame_end,activity,object,location,chunk_id\n"
        f"3,{run}.{view},0,60,70,interacting,bottle,no_location,0\n"
        f"3,{run}.{view},1,90,112,placing_moving_to,phone,co_driver_seat,0\n"  # ends where a clip starts
        f"3,{run}.{view},2,200,220,reaching_for,phone,co_driver_seat,0\n"
        f"3,{run}.{view},3,240,270,interacting,food,no_location,0\n"
        f"3,{run}.{view},3,270,300,interacting,food,no_location,1\n"
    )
    records = {r.start_frame: r.labels_gt for r in get_adapter("drive_and_act", tmp_path).records() if run in r.clip_id}
    objects = ("phone", "phone_at_ear", "drink", "food")
    # A bottle in hand during the clip: no drink 0.
    assert {k: records[22].get(k) for k in objects} == {"phone": 0.0, "phone_at_ear": 0.0, "drink": None, "food": 0.0}
    assert {k: records[112].get(k) for k in objects} == dict.fromkeys(objects, 0.0)
    # Fetching the phone: unlabelled for phone, not a positive.
    assert {k: records[187].get(k) for k in objects} == {"phone": None, "phone_at_ear": None, "drink": 0.0, "food": 0.0}
    assert {k: records[247].get(k) for k in objects} == {"phone": 0.0, "phone_at_ear": 0.0, "drink": 0.0, "food": 1.0}
    (other,) = [r for r in get_adapter("drive_and_act", tmp_path).records() if bare in r.clip_id]
    assert not set(objects) & set(other.labels_gt)


def test_drive_and_act_object_negatives_keep_activity_positives(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    # An activity's 1 stands where no annotation of its object overlaps the clip: Drive&Act
    # has eating stretches annotated with a bottle only.
    view = "a_column_driver"
    run = "vp3/run1_2018-05-22-15-10-41"
    make_video(tmp_path / view / f"{run}.{view}.mp4", frames=270)
    csv_dir = tmp_path / "activities_3s" / view
    csv_dir.mkdir(parents=True)
    (csv_dir / "midlevel.chunks_90.split_0.train.csv").write_text(
        "participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id\n"
        f"3,{run}.{view},1,0,90,talking_on_phone,0\n"  # reads frames [22, 68)
        f"3,{run}.{view},2,90,180,eating,0\n"  # [112, 158)
        f"3,{run}.{view},3,180,270,drinking,0\n"  # [202, 248)
    )
    (csv_dir / "objectlevel.chunks_90.csv").write_text(  # no phone or food, no bottle while drinking
        "participant_id,file_id,annotation_id,frame_start,frame_end,activity,object,location,chunk_id\n"
        f"3,{run}.{view},0,100,170,interacting,bottle,no_location,0\n"
    )
    records = {r.start_frame: r.labels_gt for r in get_adapter("drive_and_act", tmp_path).records()}
    objects = ("phone", "phone_at_ear", "drink", "food")
    assert {k: records[22].get(k) for k in objects} == {"phone": 1.0, "phone_at_ear": 1.0, "drink": 0.0, "food": 0.0}
    assert {k: records[112].get(k) for k in objects} == {"phone": 0.0, "phone_at_ear": 0.0, "drink": None, "food": 1.0}
    assert {k: records[202].get(k) for k in objects} == {"phone": 0.0, "phone_at_ear": 0.0, "drink": 1.0, "food": 0.0}


def test_clip_stride_is_one_rule_floored_to_whole_frames() -> None:
    assert clip_stride(30.0, 2.0, 16) == 3  # 16 frames cover about 1.5 s
    assert clip_stride(15.0, 2.0, 16) == 1  # about 1.07 s: Drive&Act kinect_color
    assert clip_stride(24.0, 5.0, 16) == 7


def test_dmd_window_labels() -> None:
    tracks = {
        "driver_actions": dict.fromkeys(range(0, 30), "texting_right"),
        "gaze_zone": dict.fromkeys(range(0, 30), "infotainment"),
        "eyes_state": {f: ("close" if f < 20 else "open") for f in range(0, 30)},
        "yawning": {12: "yawning without hand"},
    }
    labels = dmd_window(tracks, 0, 30)
    assert labels["activity"] == "texting" and labels["phone"] == 1.0
    assert labels["attention_direction"] == "down_center"
    # Closed for 2/3 of the window but open at the end: a recovery, not closed eyes.
    assert labels["eyes_closed"] == 0.0 and labels["perclos"] == round(20 / 30, 4)
    assert labels["yawn"] == 1.0
    onset = {"eyes_state": {f: ("close" if f >= 15 else "open") for f in range(0, 30)}}
    assert dmd_window(onset, 0, 30)["eyes_closed"] == 1.0  # closing: the onset counts


def test_dmd_adapter(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "gA" / "1" / "s1"
    stem = "gA_1_s1_2019-03-08T09;31;15+01;00"
    make_video(folder / f"{stem}_rgb_face.mp4", frames=120)
    actions = {
        "0": {
            "type": "driver_actions/drinking",
            "frame_intervals": [{"frame_start": 0, "frame_end": 119}],
        }
    }
    (folder / f"{stem}_rgb_ann_distraction.json").write_text(
        json.dumps({"openlabel": {"actions": actions}})
    )
    records = list(get_adapter("dmd", tmp_path).records())
    assert records and {r.labels_gt["activity"] for r in records} == {"drinking"}
    assert records[0].subject_id == "dmd:gA_1" and records[0].meta["session"] == "distraction"


def test_dmd_merges_gaze_and_hands_files(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "gB" / "6" / "s6"
    stem = "gB_6_s6_2019-03-08T09;31;15+01;00"
    make_video(folder / f"{stem}_rgb_face.mp4", frames=120)
    gaze = {"0": {"type": "gaze_zone/infotainment", "frame_intervals": [{"frame_start": 0, "frame_end": 119}]}}
    hands = {"0": {"type": "hands_on_wheel/both_hands", "frame_intervals": [{"frame_start": 0, "frame_end": 119}]}}
    (folder / f"{stem}_rgb_ann_gaze.json").write_text(json.dumps({"openlabel": {"actions": gaze}}))
    (folder / f"{stem}_rgb_ann_hands.json").write_text(json.dumps({"openlabel": {"actions": hands}}))
    records = list(get_adapter("dmd", tmp_path).records())
    assert len({r.clip_id for r in records}) == len(records)  # one clip per window, not one per file
    assert all(r.labels_gt["attention_direction"] == "down_center" for r in records)
    assert all("hands_on_controls" not in r.labels_gt for r in records)  # face camera: no wheel
    assert records[0].meta["session"] == "gaze+hands" and records[0].view == "face"


def test_dmd_hands_only_off_the_face_view() -> None:
    tracks = {"hands_using_wheel": dict.fromkeys(range(30), "only_left")}
    assert dmd_window(tracks, 0, 30, view="face") == {}
    assert dmd_window(tracks, 0, 30, view="mosaic") == {"hands_on_controls": "one"}


def test_dmd_phone_and_drink_negatives() -> None:
    safe = dict.fromkeys(range(48), "safe_drive")
    assert dmd_absent({"driver_actions": safe}, 0, 48) == {"phone": 0.0, "phone_at_ear": 0.0, "drink": 0.0}
    # The objects are annotated while handled (picked up, held, put back): no 0 then.
    held = {"driver_actions": safe, "object/cellphone": dict.fromkeys(range(40, 60), "cellphone")}
    assert dmd_absent(held, 0, 48) == {"drink": 0.0}
    sip = {"driver_actions": safe | {47: "drinking"}, "object/bottle": {47: "bottle"}}
    assert dmd_absent(sip, 0, 48) == {"phone": 0.0, "phone_at_ear": 0.0}
    # The action alone rules the 0 out too, should an object track miss its frames.
    assert dmd_absent({"driver_actions": safe | {47: "drinking"}}, 0, 48) == {"phone": 0.0, "phone_at_ear": 0.0}
    assert dmd_absent({"driver_actions": safe | {0: "texting_left"}}, 0, 48) == {"drink": 0.0}
    assert dmd_absent({"driver_actions": safe | {5: "phonecall_right"}}, 0, 48) == {"drink": 0.0}
    # One transition frame, or one frame without an action: nothing.
    assert dmd_absent({"driver_actions": safe | {10: "unclassified"}}, 0, 48) == {}
    assert dmd_absent({"driver_actions": dict.fromkeys(range(1, 48), "safe_drive")}, 0, 48) == {}
    assert dmd_absent({"eyes_state": dict.fromkeys(range(48), "open")}, 0, 48) == {}


def dmd_distraction_file(path: Path, actions: dict[str, list[int]], objects: dict[str, list[int]]) -> None:
    """An OpenLABEL distraction file shaped like DMD's ({type: [first, last]}, inclusive)."""

    def intervals(first: int, last: int) -> list[dict[str, int]]:
        return [{"frame_start": first, "frame_end": last}]

    openlabel = {
        "frame_intervals": intervals(0, 149),
        "objects": {
            str(i): {
                "name": "1" if kind == "driver" else "",
                "type": kind,
                "frame_intervals": intervals(*span),
                "ontology_uid": "0",
                "object_data_pointers": {"annotated": {"type": "text", "frame_intervals": intervals(*span)}},
            }
            for i, (kind, span) in enumerate(objects.items())
        },
        "actions": {
            str(i): {"name": "", "type": kind, "frame_intervals": intervals(*span), "ontology_uid": "0"}
            for i, (kind, span) in enumerate(actions.items())
        },
        "streams": {"face_camera": {"type": "camera", "stream_properties": {"sync": {"frame_shift": 0}}}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"openlabel": openlabel}))


def test_dmd_negatives_from_openlabel_objects(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "gA" / "1" / "s2"
    stem = "gA_1_s2_2019-03-08T09;31;15+01;00"
    make_video(folder / f"{stem}_rgb_face.mp4", frames=150)
    annotation = folder / f"{stem}_rgb_ann_distraction.json"
    dmd_distraction_file(
        annotation,
        actions={"driver_actions/safe_drive": [0, 53], "driver_actions/texting_right": [54, 149]},
        objects={"driver": [0, 149], "hair_comb": [0, 10], "bottle": [20, 28], "cellphone": [50, 149]},
    )
    tracks = dmd_tracks(annotation)
    assert sorted(t for t in tracks if t.startswith("object/")) == ["object/bottle", "object/cellphone"]
    assert min(tracks["object/cellphone"]) == 50 and max(tracks["object/cellphone"]) == 149
    records = {r.start_frame: r.labels_gt for r in get_adapter("dmd", tmp_path).records()}
    # Windows read 48 frames every 30. The one at 30 is half safe driving, half texting:
    # no majority action, so it is dropped as before, although nothing in it rules out
    # drink 0.
    assert sorted(records) == [0, 60, 90]
    assert records[0]["activity"] == "attentive"
    assert records[0]["phone"] == 0.0 and records[0]["phone_at_ear"] == 0.0
    assert "drink" not in records[0]  # the bottle is held during the window
    for start in (60, 90):
        assert records[start]["phone"] == 1.0 and records[start]["phone_at_ear"] == 0.0
        assert records[start]["drink"] == 0.0
    assert not any("food" in labels for labels in records.values())


def test_dmd_perclos_uses_annotated_frames_only() -> None:
    half = {"eyes_state": dict.fromkeys(range(24), "close")}  # 24 of 48 frames annotated, all closed
    assert "perclos" not in dmd_window(half, 0, 48)
    eyes = {f: ("close" if f < 30 else "open") for f in range(40)} | dict.fromkeys(range(40, 48), "undefined")
    labels = dmd_window({"eyes_state": eyes}, 0, 48)  # 40 defined frames (83%), 30 of them closed
    # The last half of the defined frames is half closed, half open: not closed.
    assert labels["perclos"] == 0.75 and labels["eyes_closed"] == 0.0
    eyes |= dict.fromkeys(range(36, 48), "undefined")  # only 36 defined frames (75%): too few
    assert "perclos" not in dmd_window({"eyes_state": eyes}, 0, 48)


def test_dmd_skips_sessions_without_a_face_video(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "gC" / "13" / "s1"
    stem = "gC_13_s1_2019-03-08T09;31;15+01;00"
    make_video(folder / f"{stem}_rgb_body.mp4", frames=120)
    actions = {"0": {"type": "driver_actions/drinking", "frame_intervals": [{"frame_start": 0, "frame_end": 119}]}}
    (folder / f"{stem}_rgb_ann_distraction.json").write_text(json.dumps({"openlabel": {"actions": actions}}))
    assert list(get_adapter("dmd", tmp_path).records()) == []


def test_cew_zenodo_sh17(tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    make_image(tmp_path / "cew" / "dataset_B_FacialImages" / "ClosedFace" / "closed_eye_0001.jpg")
    make_image(tmp_path / "cew" / "dataset_B_FacialImages" / "OpenFace" / "Abdullah_0001.jpg")
    make_image(tmp_path / "cew" / "dataset_B_Eye_Images" / "closedLeftEyes" / "x.jpg")
    cew = list(get_adapter("cew", tmp_path / "cew").records())
    assert sorted(r.labels_gt["eyes_closed"] for r in cew) == [0.0, 1.0]

    make_image(tmp_path / "zen" / "Texting" / "IMG_0001.jpg")
    make_image(tmp_path / "zen" / "Safe Driving" / "IMG_0002.jpg")
    zen = {
        r.labels_gt["activity"] for r in get_adapter("zenodo_driver", tmp_path / "zen").records()
    }
    assert zen == {"texting", "attentive"}

    make_image(tmp_path / "sh17" / "images" / "pexels-1.jpg", size=(400, 400))
    (tmp_path / "sh17" / "labels").mkdir(parents=True)
    # person box, helmet inside it, a vest far outside it
    (tmp_path / "sh17" / "labels" / "pexels-1.txt").write_text(
        "0 0.3 0.5 0.4 0.8\n10 0.3 0.15 0.1 0.08\n16 0.9 0.9 0.05 0.05\n"
    )
    (person,) = list(get_adapter("sh17", tmp_path / "sh17").records())
    assert person.labels_gt["helmet"] == 1.0 and person.labels_gt["safety_vest"] == 0.0
    assert person.crop is not None and person.clip_id.endswith("#person0")


def test_synthetic_wan(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    make_video(tmp_path / "scene00_take00.mp4", frames=24 * 5, fps=24)
    (tmp_path / "scene00_take00.json").write_text(
        json.dumps({"prompt": "p", "labels": {"alertness": "microsleep", "modality": "nir"}})
    )
    (record,) = list(get_adapter("synthetic_wan", tmp_path).records())  # one sample per clip
    assert record.meta["label_quality"] == "prompted" and record.labels_gt == {}
    assert record.meta["prompted_labels"]["alertness"] == "microsleep"
    assert record.start_frame == 0 and record.frame_stride == 120 // 16
