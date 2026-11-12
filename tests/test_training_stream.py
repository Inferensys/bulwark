from __future__ import annotations

import multiprocessing
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

from bulwark.data.manifest import ClipRecord
from bulwark.data.shards import ShardWriter
from bulwark.train.common import find_shards
from bulwark.train.stream import (
    PHOTOMETRIC_POLICY,
    ShardStream,
    _index_cached,
    augment_clip,
    index_shards,
    shard_fingerprint,
    shard_index,
)


def record(clip_id: str, dataset: str, subject: str, split: str = "train", frames: int = 1) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset=dataset, subject_id=f"{dataset}:{subject}", media_path=f"{clip_id}.jpg",
        media_type="image" if frames == 1 else "video", licence="test", commercial_ok=False, split=split,
        num_frames=frames,
    )


def write_dataset(out: Path, dataset: str, records: list[ClipRecord], per_shard: int, size: int = 8) -> list[Path]:
    """Shards in manifest order, like `bulwark shard` writes them."""
    with ShardWriter(out, prefix=dataset, clips_per_shard=per_shard) as writer:
        for index, r in enumerate(records):
            writer.add(r.clip_id, np.full((r.num_frames, size, size, 3), index % 250, dtype=np.uint8))
    return writer.written


def sequential(dataset: str, subjects: int, per_subject: int, frames: int = 1) -> list[ClipRecord]:
    return [record(f"{dataset}/s{s}@{i}", dataset, f"s{s}", frames=frames) for s in range(subjects) for i in range(per_subject)]


def test_batches_mix_subjects_like_random_sampling(tmp_path: Path) -> None:
    # 800 consecutive clips of 8 subjects in two shards: reading in order would give
    # about 1-2 subjects per batch of 16; random sampling gives about 7.3.
    records = sequential("drive_and_act", subjects=8, per_subject=100)
    shards = write_dataset(tmp_path, "drive_and_act", records, per_shard=400)
    stream = ShardStream(shards, {r.clip_id: r for r in records}, "train", 16, seed=3)
    distinct = [len({r.subject_id for r in batch}) for _, batch in stream]
    assert len(distinct) == 50
    assert np.mean(distinct) >= 6.5


def test_datasets_are_balanced_by_square_root_of_size_without_repeats(tmp_path: Path) -> None:
    big = sequential("big", subjects=4, per_subject=100)
    small = sequential("small", subjects=2, per_subject=50)
    shards = write_dataset(tmp_path, "big", big, 200) + write_dataset(tmp_path, "small", small, 100)
    records = {r.clip_id: r for r in big + small}
    stream = ShardStream(shards, records, "train", 10, seed=0)
    assert stream.counts == {"big": 400, "small": 100}
    assert stream.quotas() == {"big": 333, "small": 167}  # 500 clips split 20:10
    epochs = [[r for _, batch in _epoch(stream, e) for r in batch] for e in (0, 1)]
    shares = Counter(r.dataset for r in epochs[0])
    assert abs(shares["big"] - 333) <= 10 and abs(shares["small"] - 167) <= 10
    # The big dataset is undersampled; epochs walk through it without repeats, so two
    # epochs cover nearly all of it (sampling with replacement would cover about 81%).
    first = [r.clip_id for r in epochs[0] if r.dataset == "big"]
    assert len(set(first)) == len(first)
    assert len(first + [r.clip_id for r in epochs[1] if r.dataset == "big"]) > 600
    assert len({r.clip_id for e in epochs for r in e if r.dataset == "big"}) >= 390


def test_buckets_by_clip_length_and_resumes_at_the_same_batch(tmp_path: Path) -> None:
    images = sequential("img", subjects=2, per_subject=40)
    videos = sequential("vid", subjects=2, per_subject=24, frames=4)
    shards = write_dataset(tmp_path, "img", images, 40) + write_dataset(tmp_path, "vid", videos, 24)
    records = {r.clip_id: r for r in images + videos}
    stream = ShardStream(shards, records, "train", 8, seed=1, augment=True)
    full = list(_epoch(stream, 2))
    assert {frames.shape[1] for frames, _ in full} == {1, 4}
    assert all(len({r.num_frames for r in batch}) == 1 for _, batch in full)
    stream.skip = 5
    tail = list(_epoch(stream, 2))
    assert [_ids(b) for _, b in tail] == [_ids(b) for _, b in full[5:]]
    assert all(np.array_equal(a.numpy(), b.numpy()) for (a, _), (b, _) in zip(tail, full[5:], strict=True))
    stream.skip = 0
    with_workers = DataLoader(stream, batch_size=None, num_workers=2)
    assert [_ids(b) for _, b in with_workers] == [_ids(b) for _, b in full]


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="no fork start method")
def test_workers_forked_after_the_parent_read_open_their_own_shards(tmp_path: Path) -> None:
    # DataLoader workers are forked on Linux. Shards the parent still had open came with
    # them, one file offset shared by every process, so workers seeking and reading at once
    # read each other's bytes ("JPEG decoding failed" in CI). Fork here on macOS too.
    records = sequential("vid", subjects=4, per_subject=32, frames=4)
    shards = write_dataset(tmp_path, "vid", records, 32)
    stream = ShardStream(shards, {r.clip_id: r for r in records}, "train", 4, seed=2)
    in_parent = list(_epoch(stream, 0))
    assert len(stream._files) == len(shards) == 4  # open when the workers fork
    for _ in range(3):
        forked = list(DataLoader(stream, batch_size=None, num_workers=4, multiprocessing_context="fork"))
        assert [_ids(b) for _, b in forked] == [_ids(b) for _, b in in_parent]
        assert all(np.array_equal(a.numpy(), b.numpy()) for (a, _), (b, _) in zip(forked, in_parent, strict=True))


def test_sequential_mode_yields_every_present_clip_once(tmp_path: Path) -> None:
    records = sequential("img", subjects=3, per_subject=7)
    shards = write_dataset(tmp_path, "img", records, 10)
    table = {r.clip_id: r for r in records}
    table["img/not-sharded"] = record("img/not-sharded", "img", "s9")
    table["img/s0@0"] = record("img/s0@0", "img", "s0", split="test")
    stream = ShardStream(shards, table, "train", 4, shuffle=False, drop_last=False)
    seen = [r.clip_id for _, batch in stream for r in batch]
    assert sorted(seen) == sorted(r.clip_id for r in records if r.clip_id != "img/s0@0")
    assert stream.missing == 1 and stream.num_clips == 20


def test_empty_selection_yields_nothing(tmp_path: Path) -> None:
    records = sequential("img", subjects=1, per_subject=4)
    shards = write_dataset(tmp_path, "img", records, 4)
    stream = ShardStream(shards, {r.clip_id: r for r in records}, "val", 2)
    assert stream.num_clips == 0 and stream.num_batches() == 0 and list(stream) == []


def test_shard_fingerprint_tracks_contents(tmp_path: Path) -> None:
    records = sequential("img", subjects=1, per_subject=4)
    (shard,) = write_dataset(tmp_path / "a", "img", records, 4)
    (same,) = write_dataset(tmp_path / "b", "img", records, 4)
    (other,) = write_dataset(tmp_path / "c", "img", records[:3], 4)
    assert shard_fingerprint(shard) == shard_fingerprint(same) != shard_fingerprint(other)


def test_find_shards_indexes_every_shard_in_threads_like_one_by_one(tmp_path: Path) -> None:
    # On a Kaggle dataset mount, listing the tar headers of the 170 shards without AUC
    # Distracted Driver v2 one after another was most of a setup of up to 55.7 minutes.
    records = sequential("vid", subjects=3, per_subject=5, frames=3)
    shards = write_dataset(tmp_path / "shards", "vid", records, 2)
    assert len(shards) == 8
    one_by_one = {p.name: _index_cached.__wrapped__(str(p.resolve()), 0, 0) for p in shards}
    _index_cached.cache_clear()
    assert find_shards(tmp_path / "shards") == shards
    assert _index_cached.cache_info().currsize == len(shards)  # filled by the threads, not on first use
    for path in shards:
        threaded = shard_index(path)
        assert threaded.clip_ids == one_by_one[path.name].clip_ids
        assert threaded.index_sha256 == one_by_one[path.name].index_sha256
        assert all(np.array_equal(a, b) for a, b in zip(threaded.offsets, one_by_one[path.name].offsets, strict=True))
    assert _index_cached.cache_info().misses == len(shards)  # nothing was indexed twice
    index_shards(shards[:1])  # one shard: no pool
    index_shards([])


def test_augmentation_is_per_clip_deterministic_and_keeps_shape() -> None:
    ramp = np.linspace(0, 255, 32 * 32 * 3).reshape(32, 32, 3)
    frames = np.stack([ramp] * 4).astype(np.uint8)
    one = augment_clip(frames, np.random.default_rng(5))
    two = augment_clip(frames, np.random.default_rng(5))
    assert one.shape == frames.shape and one.dtype == np.uint8
    assert np.array_equal(one, two) and not np.array_equal(one, frames)
    assert all(np.array_equal(one[0], f) for f in one)  # the same transform on every frame


def _epoch(stream: ShardStream, epoch: int):  # type: ignore[no-untyped-def]
    stream.epoch = epoch
    return iter(stream)


def _ids(batch: list[ClipRecord]) -> list[str]:
    return [r.clip_id for r in batch]


@pytest.fixture(autouse=True)
def _quiet_cv2() -> None:
    import cv2

    cv2.setNumThreads(1)


def test_photometric_policy_keeps_the_view() -> None:
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    frames[:, :, 8:] = 200  # left half dark, right half bright
    out = augment_clip(frames, np.random.default_rng(0), PHOTOMETRIC_POLICY)
    # No crop: every pixel keeps its side, so each half stays uniform and the edge stays put.
    assert len(np.unique(out[:, :, :8])) == 1 and len(np.unique(out[:, :, 8:])) == 1
    assert out[0, 0, 8, 0] > out[0, 0, 7, 0]
