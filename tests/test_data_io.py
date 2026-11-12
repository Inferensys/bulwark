"""Image decoding, de-duplication, shards and subject removal."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bulwark.data.clips import load_clip
from bulwark.data.dedup import dhash, drop_cross_dataset_duplicates
from bulwark.data.manifest import ClipRecord, read_manifest, write_manifest
from bulwark.data.removals import (
    drop_removed_subjects,
    load_removed_subjects,
    purge_clips,
    purge_subjects,
)
from bulwark.data.shards import ShardWriter, read_shard
from bulwark.data.sources.base import copy_fingerprint
from bulwark.data.sources.images import _image_size


def image(
    clip_id: str, dataset: str, media_path: str, commercial_ok: bool, subject: str = "s1"
) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        dataset=dataset,
        subject_id=f"{dataset}:{subject}",
        media_path=media_path,
        media_type="image",
        licence=f"{dataset} terms",
        commercial_ok=commercial_ok,
    )


def textured(seed: int, shape: tuple[int, int] = (96, 128)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random(shape) * 255).astype(np.uint8), (0, 0), 6)


def test_exif_orientation_is_applied_everywhere(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # displayed rotated 90 degrees clockwise
    stored = np.zeros((40, 80, 3), dtype=np.uint8)
    stored[:, :40] = 255  # stored 80 wide, 40 tall, bright left half
    Image.fromarray(stored).save(path, exif=exif)
    # Displayed: 40 wide, 80 tall, bright top half and dark bottom half.
    assert _image_size(path) == (80, 40)
    hashed = dhash(path)
    assert hashed is not None and hashed[1] == pytest.approx(0.5)
    record = image("r", "sh17", "rotated.jpg", False)
    column = load_clip(record, tmp_path, size=60)[0, :, 30, 0]
    assert column[5:25].min() > 200 and column[35:55].max() < 50
    record.crop = (0, 50, 40, 30)  # bottom of the displayed frame; the stored one is 40 tall
    assert load_clip(record, tmp_path, size=60)[0, 10:50, 30].max() < 50


def test_dedup_keeps_the_licence_restriction(tmp_path: Path) -> None:
    photo = textured(0)
    cv2.imwrite(str(tmp_path / "sf.jpg"), photo)
    cv2.imwrite(
        str(tmp_path / "copy.jpg"), cv2.resize(photo, (256, 192)), [cv2.IMWRITE_JPEG_QUALITY, 70]
    )
    original = image("sf", "state_farm", "sf.jpg", commercial_ok=False)
    copy = image("rf", "roboflow_forklift_ppe", "copy.jpg", commercial_ok=True)
    kept, dropped = drop_cross_dataset_duplicates(
        [original, copy],
        {"state_farm": tmp_path, "roboflow_forklift_ppe": tmp_path},
        prefer_evaluable=["roboflow_forklift_ppe"],
    )
    assert dropped == 1 and [r.clip_id for r in kept] == ["rf"]
    assert kept[0].commercial_ok is False and "near-duplicate of state_farm" in kept[0].licence
    assert kept[0].meta["near_duplicate_of"] == ["state_farm"]


def test_dedup_refuses_to_run_blind(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "a.jpg"), textured(1))
    records = [
        image("a", "state_farm", "a.jpg", False),
        image("z", "zenodo_driver", "z.jpg", False),
    ]
    with pytest.raises(ValueError, match="zenodo_driver"):
        drop_cross_dataset_duplicates(records, {"state_farm": tmp_path})  # no zenodo root
    with pytest.raises(ValueError, match="zenodo_driver"):
        drop_cross_dataset_duplicates(
            records, {"state_farm": tmp_path, "zenodo_driver": tmp_path}
        )  # unreadable


def test_shards_keep_frame_order_past_100_frames(tmp_path: Path) -> None:
    frames = np.stack([np.full((8, 8, 3), i * 2, dtype=np.uint8) for i in range(120)])
    with ShardWriter(tmp_path, prefix="long") as writer:
        writer.add("long/clip", frames)
    ((_, decoded),) = list(read_shard(writer.written[0]))
    assert np.abs(decoded[:, 0, 0, 0].astype(int) - np.arange(120) * 2).max() <= 3


def test_removed_subjects_are_dropped_and_purged(tmp_path: Path) -> None:
    assert load_removed_subjects() == {}  # nothing has been removed so far
    removal = tmp_path / "removed.yaml"
    removal.write_text("removed:\n  dmd:gA_3: {date: 2026-01-01, source: test}\n")
    assert set(load_removed_subjects(removal)) == {"dmd:gA_3"}

    gone = image("dmd/gone", "dmd", "gone.jpg", False, subject="gA_3")
    kept = image("dmd/kept", "dmd", "kept.jpg", False, subject="gA_1")
    assert drop_removed_subjects([gone, kept], ["dmd:gA_3"]) == ([kept], 1)

    manifest = write_manifest([gone, kept], tmp_path / "manifest.parquet")
    shards = tmp_path / "shards"
    with ShardWriter(shards, prefix="dmd") as writer:
        writer.add("dmd/gone", np.zeros((2, 8, 8, 3), dtype=np.uint8))
        writer.add("dmd/kept", np.zeros((2, 8, 8, 3), dtype=np.uint8))
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "face.jsonl").write_text(
        "".join(
            json.dumps({"clip_id": c, "labels": {"yawn": 0.0}}) + "\n"
            for c in ("dmd/gone", "dmd/kept")
        )
    )
    features = tmp_path / "features"
    features.mkdir()
    np.savez(
        features / "features-dmd-00000.npz",
        clip_ids=np.array(["dmd/gone", "dmd/kept"]),
        tokens=np.ones((2, 3, 4)),
    )

    counts = purge_subjects(
        ["dmd:gA_3"], manifests=[manifest], shards=shards, labels=labels, features=[features]
    )
    assert counts == {
        "clips": 1,
        "shard_clips": 1,
        "label_rows": 1,
        "feature_rows": 1,
        "manifest_rows": 1,
    }
    assert [r.clip_id for r in read_manifest(manifest)] == ["dmd/kept"]
    assert [c for c, _ in read_shard(writer.written[0])] == ["dmd/kept"]
    assert "dmd/gone" not in (labels / "face.jsonl").read_text()
    with np.load(features / "features-dmd-00000.npz") as data:
        assert list(data["clip_ids"]) == ["dmd/kept"] and data["tokens"].shape == (1, 3, 4)
    assert purge_subjects(["dmd:gA_3"], manifests=[manifest], shards=shards) == {"clips": 0}


def test_purge_clips_removes_stale_clips_from_every_cache(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    with ShardWriter(shards, prefix="drive_and_act") as writer:
        writer.add("v.mp4@11237:4", np.zeros((2, 8, 8, 3), dtype=np.uint8))
        writer.add("v.mp4@11250s3:4", np.zeros((2, 8, 8, 3), dtype=np.uint8))
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "face.jsonl").write_text(
        json.dumps({"clip_id": "v.mp4@11237:4", "labels": {"yawn": 0.0}}) + "\n"
    )
    counts = purge_clips(["v.mp4@11237:4"], shards=shards, labels=labels)
    assert counts == {"shard_clips": 1, "label_rows": 1}
    assert [c for c, _ in read_shard(writer.written[0])] == ["v.mp4@11250s3:4"]
    assert (labels / "face.jsonl").read_text() == ""
    assert purge_clips([], shards=shards) == {}


def test_copy_fingerprint_ignores_folders_but_not_content(tmp_path: Path) -> None:
    for root, folders in (("mirror", ("train/awake", "val/sleepy")), ("official", ("s1", "s1"))):
        for seed, (folder, name) in enumerate(zip(folders, ("s1_a.png", "s1_b.png"), strict=True)):
            (tmp_path / root / folder).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp_path / root / folder / name), textured(seed))
    fingerprint, count = copy_fingerprint(tmp_path / "mirror")
    assert count == 2 and copy_fingerprint(tmp_path / "official") == (fingerprint, 2)
    cv2.imwrite(str(tmp_path / "official" / "s1" / "s1_a.png"), textured(99))
    assert copy_fingerprint(tmp_path / "official")[0] != fingerprint
