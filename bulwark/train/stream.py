"""Training batches straight from tar shards, mixed across datasets.

`bulwark shard` writes one shard set per dataset in manifest order, so neighbouring clips
in a shard are usually consecutive chunks of one video. Reading shards front to back
would give batches of one dataset and a few subjects. Instead, each shard's header is
indexed once (member offsets, no decoding) and every epoch is planned up front:

- each dataset contributes clips in proportion to size**dataset_power (0.5 by default,
  the same mix as the heads stage), cycling through a seeded permutation of its clips so
  every clip is seen before any repeats;
- clips are bucketed by length (images, T=1, and videos go through different patch
  embeddings), shuffled, cut into batches, and the batches shuffled;
- batches are dealt to DataLoader workers in turn, so the loader returns them in plan
  order, and `skip` resumes mid-epoch without reading the skipped batches.

Clips are then read by offset and decoded only when their batch is produced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
import zlib
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from bulwark.data.manifest import ClipRecord
from bulwark.data.shards import decode_jpeg, member_bytes

Batch = tuple[torch.Tensor, list[ClipRecord]]
INDEX_THREADS = 16  # shards indexed at once by index_shards

# Light, label-safe augmentation: the same parameters for every frame of a clip. No
# horizontal flip, since that would have to swap left/right labels (attention_direction)
# and negate yaw and roll.
AUGMENT_POLICY: dict[str, Any] = {
    "crop_area": (0.8, 1.0),
    "crop_aspect": (0.9, 1.1),
    "gamma": (0.8, 1.25),
    "contrast": (0.8, 1.2),
    "brightness": (-0.08, 0.08),
    "grayscale_p": 0.2,
    "hflip": False,
}
# The same without the crop, for distillation: its soft targets were cached on full,
# unaugmented clips, and a crop would change what they describe.
PHOTOMETRIC_POLICY: dict[str, Any] = {**AUGMENT_POLICY, "crop_area": (1.0, 1.0), "crop_aspect": (1.0, 1.0)}


@dataclass(frozen=True, slots=True)
class ShardIndex:
    """Where each clip's frames sit inside one tar shard."""

    clip_ids: tuple[str, ...]
    offsets: tuple[np.ndarray, ...]  # per clip: data offset of each frame member, frame order
    sizes: tuple[np.ndarray, ...]
    index_sha256: str  # hash of the shard's index.json (its clip list)

    def fingerprint(self, size: int) -> str:
        return hashlib.sha256(f"{size}:{self.index_sha256}".encode()).hexdigest()[:16]


@cache
def _index_cached(path: str, size: int, mtime_ns: int) -> ShardIndex:
    groups: dict[str, list[tuple[str, int, int]]] = {}
    index_bytes: bytes | None = None
    with tarfile.open(path, "r") as tar:
        for member in tar.getmembers():  # headers only; frame data is skipped over
            if member.name == "index.json":
                index_bytes = member_bytes(tar, member)
                continue
            key, _, _ = member.name.partition("/")
            groups.setdefault(key, []).append((member.name, member.offset_data, member.size))
    if index_bytes is None:
        raise ValueError(f"{path} has no index.json; the shard is incomplete and must be rebuilt")
    names: dict[str, str] = json.loads(index_bytes)
    clip_ids, offsets, sizes = [], [], []
    for key, members in groups.items():
        members.sort()
        clip_ids.append(names[key])
        offsets.append(np.array([m[1] for m in members], dtype=np.int64))
        sizes.append(np.array([m[2] for m in members], dtype=np.int64))
    return ShardIndex(tuple(clip_ids), tuple(offsets), tuple(sizes), hashlib.sha256(index_bytes).hexdigest())


def shard_index(path: Path) -> ShardIndex:
    stat = Path(path).stat()
    return _index_cached(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)


def index_shards(paths: Sequence[Path], threads: int = INDEX_THREADS) -> None:
    """Index every shard up front, several at a time, into `shard_index`'s cache.

    Listing a shard reads one tar header per frame (about 16,000 per Drive&Act shard). On a
    Kaggle dataset mount, listing the 170 shards without AUC Distracted Driver v2 one after
    another was most of a setup of 2.7 to 55.7 minutes depending on the session, most of
    the listing waiting on reads, which threads overlap; parsing the headers holds the GIL,
    so a session with fast reads gains less (1.6 times, measured). In threads, the one
    Kaggle session measured so far listed the 185 v0 shards in 202 s.
    """
    if len(paths) > 1 and threads > 1:
        with ThreadPoolExecutor(min(threads, len(paths))) as pool:
            list(pool.map(shard_index, paths))


def shard_fingerprint(path: Path) -> str:
    """Fast content fingerprint of a shard: its size and the hash of its clip index."""
    return shard_index(path).fingerprint(Path(path).stat().st_size)


def shards_identity(paths: Sequence[Path]) -> dict[str, str]:
    return {Path(p).name: shard_fingerprint(Path(p)) for p in paths}


def augment_clip(frames: np.ndarray, rng: np.random.Generator, policy: dict[str, Any] = AUGMENT_POLICY) -> np.ndarray:
    """Apply `policy` (AUGMENT_POLICY by default) to one clip (T, H, W, 3) uint8, identically on every frame."""
    _, height, width, _ = frames.shape
    area = rng.uniform(*policy["crop_area"])
    aspect = math.exp(rng.uniform(*np.log(policy["crop_aspect"])))
    crop_h = min(height, round(height * math.sqrt(area / aspect)))
    crop_w = min(width, round(width * math.sqrt(area * aspect)))
    top = int(rng.integers(0, height - crop_h + 1))
    left = int(rng.integers(0, width - crop_w + 1))
    if (crop_h, crop_w) != (height, width):
        frames = np.stack([
            cv2.resize(frame[top : top + crop_h, left : left + crop_w], (width, height), interpolation=cv2.INTER_LINEAR)
            for frame in frames
        ])
    x = frames.astype(np.float32) / 255.0
    x = x ** math.exp(rng.uniform(*np.log(policy["gamma"])))
    mean = float(x.mean())
    x = (x - mean) * rng.uniform(*policy["contrast"]) + mean + rng.uniform(*policy["brightness"])
    if rng.random() < policy["grayscale_p"]:  # RGB clips that look like NIR ones
        gray = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        x = np.repeat(gray[..., None], 3, axis=-1)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


class ShardStream(IterableDataset[Batch]):
    """Batches of one split from tar shards. Use with `DataLoader(stream, batch_size=None)`.

    With `shuffle=False` every present clip is yielded once, in shard order, bucketed by
    clip length (evaluation). Set `epoch` before each pass and `skip` to resume mid-epoch.
    """

    def __init__(
        self,
        shard_paths: Sequence[Path],
        records: dict[str, ClipRecord],
        split: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        dataset_power: float = 0.5,
        augment: bool = False,
        policy: dict[str, Any] | None = None,
    ) -> None:
        self.shard_paths = [Path(p) for p in shard_paths]
        self.records = records
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.dataset_power = dataset_power
        self.augment = augment
        self.policy = policy or AUGMENT_POLICY
        self.epoch = 0
        self.skip = 0
        clip_ids: list[str] = []
        shard: list[int] = []
        lengths: list[int] = []
        self._offsets: list[np.ndarray] = []
        self._sizes: list[np.ndarray] = []
        for number, path in enumerate(self.shard_paths):
            index = shard_index(path)
            for clip_id, offsets, sizes in zip(index.clip_ids, index.offsets, index.sizes, strict=True):
                record = records.get(clip_id)
                if record is None or record.split != split:
                    continue
                clip_ids.append(clip_id)
                shard.append(number)
                lengths.append(len(offsets))
                self._offsets.append(offsets)
                self._sizes.append(sizes)
        self.clip_ids = clip_ids
        self._shard = np.array(shard, dtype=np.int32)
        self._length = np.array(lengths, dtype=np.int32)
        by_dataset: dict[str, list[int]] = {}
        for position, clip_id in enumerate(clip_ids):
            by_dataset.setdefault(records[clip_id].dataset, []).append(position)
        self._by_dataset = {name: np.array(by_dataset[name], dtype=np.int64) for name in sorted(by_dataset)}
        self.counts = {name: len(positions) for name, positions in self._by_dataset.items()}
        present = set(clip_ids)
        self.missing = sum(1 for r in records.values() if r.split == split and r.clip_id not in present)
        self._files: OrderedDict[int, BinaryIO] = OrderedDict()
        self._files_pid = os.getpid()  # the process that opened _files

    @property
    def num_clips(self) -> int:
        return len(self.clip_ids)

    def quotas(self) -> dict[str, int]:
        """Clips per dataset in one epoch: total clips split in proportion to size**power."""
        weights = {name: count**self.dataset_power for name, count in self.counts.items()}
        total = sum(weights.values())
        return {name: max(1, round(self.num_clips * w / total)) for name, w in weights.items()}

    def plan(self, epoch: int) -> list[np.ndarray]:
        """The epoch's batches, each an array of clip positions. Deterministic in (seed, epoch)."""
        if not self.shuffle:
            return self._sequential_plan()
        rng = np.random.default_rng([self.seed, epoch])
        chosen = [self._dataset_slice(name, epoch, quota) for name, quota in self.quotas().items()]
        picks = np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)
        batches: list[np.ndarray] = []
        for length in sorted(set(self._length[picks].tolist())):
            bucket = rng.permutation(picks[self._length[picks] == length])
            for start in range(0, len(bucket), self.batch_size):
                chunk = bucket[start : start + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)
        return [batches[i] for i in rng.permutation(len(batches))]

    def num_batches(self, epoch: int = 0) -> int:
        return len(self.plan(epoch))

    def _dataset_slice(self, name: str, epoch: int, quota: int) -> np.ndarray:
        """Clips [epoch*quota, (epoch+1)*quota) of an endless sequence of permutations."""
        positions = self._by_dataset[name]
        size = len(positions)
        start, stop = epoch * quota, (epoch + 1) * quota
        cycles = [
            np.random.default_rng([self.seed, zlib.crc32(name.encode()), cycle]).permutation(positions)
            for cycle in range(start // size, (stop - 1) // size + 1)
        ]
        offset = start - (start // size) * size
        return np.concatenate(cycles)[offset : offset + quota]

    def _sequential_plan(self) -> list[np.ndarray]:
        batches: list[np.ndarray] = []
        buckets: dict[int, list[int]] = {}
        for position, length in enumerate(self._length.tolist()):
            bucket = buckets.setdefault(length, [])
            bucket.append(position)
            if len(bucket) == self.batch_size:
                batches.append(np.array(bucket, dtype=np.int64))
                bucket.clear()
        if not self.drop_last:
            batches.extend(np.array(b, dtype=np.int64) for b in buckets.values() if b)
        return batches

    def __iter__(self) -> Iterator[Batch]:
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        epoch = self.epoch
        batches = self.plan(epoch)
        # Batches after `skip` are dealt to workers in turn, which is the order the
        # DataLoader collects them in, so the loader yields them in plan order.
        for number in range(self.skip + worker_id, len(batches), workers):
            yield self._load(batches[number], epoch, number)

    def _load(self, positions: np.ndarray, epoch: int, number: int) -> Batch:
        rng = np.random.default_rng([self.seed, epoch, number, 1]) if self.augment else None
        clips = []
        for position in positions.tolist():
            frames = self._read(position)
            clips.append(augment_clip(frames, rng, self.policy) if rng is not None else frames)
        records = [self.records[self.clip_ids[p]] for p in positions.tolist()]
        return torch.from_numpy(np.stack(clips)), records

    def _read(self, position: int) -> np.ndarray:
        offsets, sizes = self._offsets[position], self._sizes[position]
        handle = self._file(int(self._shard[position]))
        start = int(offsets[0])
        handle.seek(start)
        blob = handle.read(int(offsets[-1] + sizes[-1]) - start)
        return np.stack([decode_jpeg(blob[o - start : o - start + s]) for o, s in zip(offsets.tolist(), sizes.tolist(), strict=True)])

    def _file(self, number: int) -> BinaryIO:
        if self._files_pid != os.getpid():
            # A forked DataLoader worker (Linux) inherits the parent's open handles, each
            # sharing one file offset with the parent and the other workers, whose
            # seek-then-read calls would interleave and read the wrong bytes. Open its own.
            self._files, self._files_pid = OrderedDict(), os.getpid()
        handle = self._files.get(number)
        if handle is None:
            if len(self._files) >= 32:  # stay well under per-process open-file limits
                self._files.popitem(last=False)[1].close()
            handle = self._files[number] = open(self.shard_paths[number], "rb")  # noqa: SIM115
        else:
            self._files.move_to_end(number)
        return handle

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_files"] = OrderedDict()  # file handles do not cross into worker processes
        return state
