"""Tar shards of JPEG frame stacks.

Shards hold pixels only, keyed by clip id. Labels stay in the manifest so relabelling
never means re-sharding. Each member is `<key>/<frame:02d>.jpg`; keys are a hash of the
clip id (clip ids contain slashes and `@`), and `index.json` in each shard maps key ->
clip id.

A shard is written as `<prefix>-NNNNN.tar.partial` and renamed to `.tar` only once its
index is written, so every `.tar` is complete. A write interrupted by an exception
deletes the shard it was writing (the shards it finished stay); one killed outright leaves
a `.partial` file, which the next writer of that shard number overwrites and
`removals.purge_clips` deletes. Readers and uploads should only ever take `*.tar`.
`first_shard` numbers a writer's first file, so a run that adds clips can continue an
existing set; a writer never overwrites a `.tar`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def clip_key(clip_id: str) -> str:
    return hashlib.sha1(clip_id.encode()).hexdigest()[:16]


def encode_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
    if frame is None:
        raise ValueError("JPEG decoding failed")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def member_bytes(tar: tarfile.TarFile, member: str | tarfile.TarInfo) -> bytes:
    """The contents of one regular file in an open tar (KeyError when there is no such member)."""
    handle = tar.extractfile(member)
    if handle is None:
        name = member if isinstance(member, str) else member.name
        raise ValueError(f"Tar member {name!r} is not a regular file")
    return handle.read()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@dataclass
class ShardWriter:
    out_dir: Path
    prefix: str = "shard"
    clips_per_shard: int = 1000
    quality: int = 90
    # Number of the first shard file, to add shards after an existing set without
    # overwriting it.
    first_shard: int = 0

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._tar: tarfile.TarFile | None = None
        self._index: dict[str, str] = {}
        self._shard = self.first_shard
        self.written: list[Path] = []  # finished shards, in order

    def _path(self) -> Path:
        return self.out_dir / f"{self.prefix}-{self._shard:05d}.tar"

    def _open(self) -> tarfile.TarFile:
        if self._tar is None:
            path = self._path()
            if path.exists():
                raise FileExistsError(f"{path} exists; shards are never overwritten")
            self._tar = tarfile.open(_partial(path), "w")  # noqa: SIM115 - stays open across add() calls
        return self._tar

    def add(self, clip_id: str, frames: np.ndarray) -> None:
        # Encode first, so a frame that fails to encode leaves nothing of the clip behind.
        encoded = [encode_jpeg(frame, self.quality) for frame in frames]
        tar = self._open()
        key = clip_key(clip_id)
        for index, data in enumerate(encoded):
            _add_bytes(tar, f"{key}/{index:02d}.jpg", data)
        self._index[key] = clip_id
        if len(self._index) >= self.clips_per_shard:
            self._close()

    def _close(self) -> None:
        if self._tar is not None:
            _add_bytes(self._tar, "index.json", json.dumps(self._index).encode())
            self._tar.close()
            path = self._path()
            os.replace(_partial(path), path)
            self.written.append(path)
            self._tar = None
            self._index = {}
            self._shard += 1

    def discard(self) -> None:
        """Drop the shard being written (its clips are in no finished shard)."""
        if self._tar is not None:
            self._tar.close()
            _partial(self._path()).unlink(missing_ok=True)
            self._tar = None
            self._index = {}

    def close(self) -> list[Path]:
        self._close()
        return self.written

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.discard()


def _partial(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def shard_index(path: Path) -> dict[str, str]:
    """{member key: clip id} from a shard's index.json."""
    with tarfile.open(path, "r") as tar:
        return _read_index(tar, path)


def _read_index(tar: tarfile.TarFile, path: Path) -> dict[str, str]:
    try:
        member = tar.extractfile("index.json")
    except KeyError:
        member = None
    if member is None:
        raise ValueError(f"{path} has no index.json; the shard is incomplete and must be rebuilt")
    return json.loads(member.read())


def read_shard(path: Path, only: Collection[str] | None = None) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (clip_id, frames[T, H, W, 3]) in shard order; with `only`, just those clip
    ids, decoding nothing else."""
    with tarfile.open(path, "r") as tar:
        index = _read_index(tar, path)
        if only is not None:
            index = {key: clip_id for key, clip_id in index.items() if clip_id in only}
        groups: dict[str, list[tarfile.TarInfo]] = {}
        for member in tar.getmembers():
            key, _, _ = member.name.partition("/")
            if member.name != "index.json" and key in index:
                groups.setdefault(key, []).append(member)
        for key, frames in groups.items():
            frames.sort(key=_frame_index)  # numeric: "100.jpg" must follow "99.jpg"
            stack = [decode_jpeg(member_bytes(tar, m)) for m in frames]
            yield index[key], np.stack(stack)


def _frame_index(member: tarfile.TarInfo) -> int:
    return int(member.name.rsplit("/", 1)[1].split(".", 1)[0])


def read_shards(
    paths: Iterable[Path], only: Collection[str] | None = None
) -> Iterator[tuple[str, np.ndarray]]:
    for path in paths:
        yield from read_shard(path, only)
