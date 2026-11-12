from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from bulwark.data.clips import load_clip
from bulwark.data.manifest import ClipRecord, read_manifest, write_manifest
from bulwark.data.shards import ShardWriter, read_shard
from bulwark.data.taxonomy import default_taxonomy
from bulwark.label.face_teacher import FrameFace, euler_from_matrix, summarise
from bulwark.label.run import merge_teacher_labels
from bulwark.label.vlm_teacher import parse_answer
from bulwark.model.pooling import NUM_SUMMARY_TOKENS
from bulwark.train.common import load_config


def clip(clip_id: str, split: str = "train", media_path: str = "x.jpg", **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset="state_farm", subject_id=f"state_farm:{clip_id}", media_path=media_path,
        media_type="image", licence="test", commercial_ok=False, split=split, **kwargs,  # type: ignore[arg-type]
    )


def test_load_clip_letterboxes_images_and_samples_videos(tmp_path: Path, make_image, make_video) -> None:  # type: ignore[no-untyped-def]
    make_image(tmp_path / "a.jpg", size=(200, 100))
    image = load_clip(clip("a", media_path="a.jpg"), tmp_path, size=64)  # type: ignore[arg-type]
    assert image.shape == (1, 64, 64, 3) and image[0, 0, 0].sum() == 0  # padded border
    make_video(tmp_path / "v.mp4", frames=40)
    record = ClipRecord(
        clip_id="v", dataset="yawdd", subject_id="yawdd:1", media_path="v.mp4", media_type="video",
        licence="t", commercial_ok=False, start_frame=10, num_frames=16, frame_stride=2,
    )
    frames = load_clip(record, tmp_path, size=32)
    assert frames.shape == (16, 32, 32, 3)


def test_shard_roundtrip(tmp_path: Path) -> None:
    ramp = np.linspace(0, 200, 32, dtype=np.float32)
    frames = np.stack([np.broadcast_to((ramp[None, :, None] + i * 3), (32, 32, 3)) for i in range(16)]).astype(np.uint8)
    with ShardWriter(tmp_path, prefix="t", clips_per_shard=2) as writer:
        writer.add("dmd/a@0", frames)
        writer.add("dmd/a@16", frames)
        writer.add("img/b.jpg", frames[:1])
    assert len(writer.written) == 2
    clips = dict(read_shard(writer.written[0]))
    assert set(clips) == {"dmd/a@0", "dmd/a@16"}
    assert clips["dmd/a@0"].shape == (16, 32, 32, 3)
    assert np.abs(clips["dmd/a@0"].astype(int) - frames).mean() < 12  # JPEG loss only
    assert dict(read_shard(writer.written[1]))["img/b.jpg"].shape == (1, 32, 32, 3)


def face(closure: float = 0.0, jaw: float = 0.0, yaw: float = 0.0, pitch: float = 0.0) -> FrameFace:
    return FrameFace(closure=closure, jaw=jaw, yaw=yaw, pitch=pitch, roll=0.0)


def test_face_summary_blink_yawn_direction_and_missing_face() -> None:
    blink = summarise([face(), face(0.9), face(0.9), face(), face(), face(), face(), face()])
    assert blink["blink"] == 1.0 and blink["eyes_closed"] == 0.0 and blink["attention_direction"] == "forward"
    yawn = summarise([face(jaw=0.7)] * 6 + [face()] * 2)
    assert yawn["yawn"] == 1.0
    closed = summarise([face(0.9)] * 8)
    assert closed["eyes_closed"] == 1.0 and closed["blink"] == 0.0 and "attention_direction" not in closed
    left = summarise([face(yaw=40.0)] * 8)
    assert left["attention_direction"] == "left" and 0.4 < left["head_yaw"] < 0.5
    assert summarise([None] * 8) == {"face_visible": 0.0}
    default_taxonomy().validate(blink)


def test_pitch_is_positive_when_the_head_tips_down() -> None:
    # MediaPipe's metric space: x right, y up, z towards the camera. Tipping the face
    # down by 30 degrees turns its forward axis (+z) towards -y.
    angle = math.radians(30)
    tipped_down = np.eye(4)
    tipped_down[:3, :3] = [
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)],
    ]
    assert np.allclose(tipped_down[:3, :3] @ [0, 0, 1], [0, -math.sin(angle), math.cos(angle)])
    _, pitch, _ = euler_from_matrix(tipped_down)
    assert pitch == pytest.approx(30.0)

    down = summarise([face(pitch=40.0)] * 8)
    assert down["attention_direction"] == "down" and 0.4 < down["head_pitch"] < 0.5
    assert summarise([face(pitch=-40.0)] * 8)["attention_direction"] == "up"


def test_head_nod_is_the_head_dropping_and_coming_back_up() -> None:
    nod = summarise([face(pitch=p) for p in (0, 5, 25, 35, 20, 5, 0, 0)])
    assert nod["head_nod"] == 1.0
    # The head rising and coming back down is not a nod (the rule before 0.1.0 said it was).
    assert summarise([face(pitch=p) for p in (0, -5, -25, -35, -20, -5, 0, 0)])["head_nod"] == 0.0
    # Dropping without coming back up inside the clip is not one either.
    assert summarise([face(pitch=p) for p in (0, 5, 15, 25, 35, 35, 35, 35)])["head_nod"] == 0.0


def test_merge_teacher_labels_respects_ground_truth_and_priority(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "vlm.jsonl").write_text(json.dumps({"clip_id": "a", "labels": {"activity": "eating", "phone": 1.0, "_caption": "eats"}}) + "\n")
    (labels / "objects.jsonl").write_text(json.dumps({"clip_id": "a", "labels": {"phone": 0.0}}) + "\n")
    (labels / "face.jsonl").write_text(json.dumps({"clip_id": "y", "labels": {"yawn": 1.0, "mouth_open": 0.8}}) + "\n")
    a = clip("a", labels_gt={"activity": "texting"})
    y = ClipRecord(
        clip_id="y", dataset="yawdd", subject_id="yawdd:female01", media_path="y.avi", media_type="video",
        licence="t", commercial_ok=False, view="dash", meta={"yawn_video": True},
    )
    merged = {r.clip_id: r for r in merge_teacher_labels([a, y], labels)}
    assert merged["a"].labels_teacher == {"phone": 0.0}  # objects beats vlm; GT activity kept
    assert merged["a"].meta["caption"] == "eats"
    # The face teacher's yawn, 1 or 0, is never merged (FACE_KEEP).
    assert merged["y"].labels_gt == {} and merged["y"].labels_teacher == {"mouth_open": 0.8}
    # A new teacher file replaces the old labels.
    (labels / "face.jsonl").write_text(json.dumps({"clip_id": "y", "labels": {"yawn": 0.0, "mouth_open": 0.1}}) + "\n")
    assert merge_teacher_labels([y], labels)[0].labels_teacher == {"mouth_open": 0.1}


def test_parse_vlm_answer() -> None:
    taxonomy = default_taxonomy()
    text = 'Sure. {"context": "vehicle_cabin", "activity": "texting", "alertness": "unknown", "objects": ["phone"], "caption": "A driver texts."}'
    labels = parse_answer(text, taxonomy)
    # Objects are positives only: an object left out of the list is not evidence of absence.
    assert labels["activity"] == "texting" and labels["phone"] == 1.0 and "drink" not in labels
    assert "alertness" not in labels and labels["_caption"] == "A driver texts."
    assert parse_answer("no json here", taxonomy) == {}


def test_train_heads_and_evaluate_on_fake_features(tmp_path: Path) -> None:
    from bulwark.eval.report import evaluate_heads
    from bulwark.train.train_heads import run

    rng = np.random.default_rng(0)
    records, tokens = [], []
    for i in range(120):
        yawning = i % 2 == 0
        split = "train" if i < 80 else ("val" if i < 100 else "test")
        records.append(clip(f"c{i}", split=split, labels_gt={"yawn": float(yawning), "activity": "talking" if yawning else "attentive"}))
        base = rng.normal(size=(NUM_SUMMARY_TOKENS, 16))
        base[:, 0] += 3.0 if yawning else -3.0  # a learnable signal
        tokens.append(base)
    write_manifest(records, tmp_path / "manifest.parquet")
    features = tmp_path / "features"
    features.mkdir()
    np.savez(features / "features-a.npz", clip_ids=np.array([r.clip_id for r in records]), tokens=np.array(tokens, dtype=np.float16))

    config = load_config(None, manifest=str(tmp_path / "manifest.parquet"), features=str(features), out=str(tmp_path / "run"),
                         epochs=15, batch_size=16, lr=3e-3, warmup_steps=5, device="cpu", log_every=5)
    out = run(config)
    assert (out / "best.pt").exists() and (out / "final.pt").exists()
    report = evaluate_heads(out / "best.pt", features, tmp_path / "manifest.parquet", tmp_path / "eval")
    metrics = json.loads((report.parent / "metrics.json").read_text())
    assert metrics["splits"]["test"]["datasets"]["state_farm"]["yawn"]["auroc"] > 0.9
    assert metrics["splits"]["test"]["summary"]["yawn"]["value"] > 0.9
    assert read_manifest(tmp_path / "manifest.parquet")[0].split == "train"


def test_geometric_yaw_sign() -> None:
    from bulwark.label.face_teacher import geometric_yaw_sign

    # camera faces the subject: subject's right eye corner at image x=0.4, left at 0.6
    assert geometric_yaw_sign(0.56, 0.4, 0.6, fallback=-5.0) == 1.0   # nose to image right: turned to their left
    assert geometric_yaw_sign(0.44, 0.4, 0.6, fallback=5.0) == -1.0
    assert geometric_yaw_sign(0.501, 0.4, 0.6, fallback=-5.0) == -1.0  # frontal: keep the matrix sign
