"""Shard writing is all-or-nothing per file; reads can pick clips without decoding the rest."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from bulwark.data import shards as shard_module
from bulwark.data.removals import purge_clips
from bulwark.data.shards import ShardWriter, read_shard, shard_index
from bulwark.label.run import run_teacher


def frames(value: int = 0, count: int = 2) -> np.ndarray:
    return np.full((count, 8, 8, 3), value, dtype=np.uint8)


def test_a_shard_appears_only_when_it_is_complete(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, prefix="t")
    writer.add("a", frames())
    assert [p.name for p in tmp_path.iterdir()] == ["t-00000.tar.partial"]
    assert writer.close() == [tmp_path / "t-00000.tar"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t-00000.tar"]
    assert shard_index(tmp_path / "t-00000.tar") == {shard_module.clip_key("a"): "a"}


def test_an_exception_discards_the_unfinished_shard(tmp_path: Path) -> None:
    with (
        pytest.raises(RuntimeError),
        ShardWriter(tmp_path, prefix="t", clips_per_shard=2) as writer,
    ):
        writer.add("a", frames())
        writer.add("b", frames())  # finishes t-00000
        writer.add("c", frames())
        raise RuntimeError("interrupted")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t-00000.tar"]
    assert [c for c, _ in read_shard(tmp_path / "t-00000.tar")] == ["a", "b"]


def test_first_shard_continues_a_set_and_never_overwrites(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, prefix="t") as writer:
        writer.add("a", frames())
    with ShardWriter(tmp_path, prefix="t", first_shard=1) as writer:
        writer.add("b", frames())
    assert writer.written == [tmp_path / "t-00001.tar"]
    with pytest.raises(FileExistsError), ShardWriter(tmp_path, prefix="t") as writer:
        writer.add("c", frames())
    assert [c for c, _ in read_shard(tmp_path / "t-00000.tar")] == ["a"]
    assert not list(tmp_path.glob("*.partial"))


def test_a_frame_that_fails_to_encode_leaves_no_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encode = shard_module.encode_jpeg
    calls = {"n": 0}

    def flaky(frame: np.ndarray, quality: int = 90) -> bytes:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("JPEG encoding failed")
        return encode(frame, quality)

    monkeypatch.setattr(shard_module, "encode_jpeg", flaky)
    with ShardWriter(tmp_path, prefix="t") as writer:
        with pytest.raises(ValueError):
            writer.add("bad", frames())
        writer.add("good", frames())
    with tarfile.open(writer.written[0]) as tar:
        members = set(tar.getnames())
    good = shard_module.clip_key("good")
    assert members == {f"{good}/00.jpg", f"{good}/01.jpg", "index.json"}
    assert [c for c, _ in read_shard(writer.written[0])] == ["good"]


def test_purge_deletes_unfinished_shards(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, prefix="t") as writer:
        writer.add("a", frames())
        writer.add("b", frames())
    # A shard run killed outright leaves a .partial with frames and no index.
    with tarfile.open(tmp_path / "t-00001.tar.partial", "w") as tar:
        tar.add(writer.written[0], arcname=f"{shard_module.clip_key('a')}/00.jpg")
    assert purge_clips(["a"], shards=tmp_path) == {"partial_shards": 1, "shard_clips": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t-00000.tar"]
    assert [c for c, _ in read_shard(writer.written[0])] == ["b"]


def test_read_shard_decodes_only_the_clips_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ShardWriter(tmp_path, prefix="t") as writer:
        for value, clip_id in enumerate(("a", "b", "c")):
            writer.add(clip_id, frames(value * 50, count=3))
    decoded = {"n": 0}
    decode = shard_module.decode_jpeg

    def counting(data: bytes) -> np.ndarray:
        decoded["n"] += 1
        return decode(data)

    monkeypatch.setattr(shard_module, "decode_jpeg", counting)
    ((clip_id, stack),) = list(read_shard(writer.written[0], only={"b", "missing"}))
    assert clip_id == "b" and stack.shape == (3, 8, 8, 3) and decoded["n"] == 3
    assert [c for c, _ in read_shard(writer.written[0], only=set())] == []


def test_shard_without_index_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "t-00000.tar"
    with tarfile.open(broken, "w"):
        pass
    with pytest.raises(ValueError, match="no index.json"):
        shard_index(broken)
    with pytest.raises(ValueError, match="no index.json"):
        list(read_shard(broken))


def test_teacher_runs_resume_without_decoding_labelled_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with ShardWriter(tmp_path / "shards", prefix="t") as writer:
        for clip_id in ("a", "b", "c"):
            writer.add(clip_id, frames())
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "face.jsonl").write_text(
        json.dumps({"clip_id": "a", "labels": {"blink": 0.0}}) + "\n"
    )
    decoded = {"n": 0}
    decode = shard_module.decode_jpeg

    def counting(data: bytes) -> np.ndarray:
        decoded["n"] += 1
        return decode(data)

    monkeypatch.setattr(shard_module, "decode_jpeg", counting)
    seen: list[int] = []
    run_teacher(
        "face",
        lambda x: seen.append(len(x)) or {"blink": 1.0},
        writer.written,
        labels,
        only={"a", "b"},
    )
    assert seen == [2] and decoded["n"] == 2  # only "b": "a" is done and "c" is not wanted
    rows = [
        json.loads(line)["clip_id"] for line in (labels / "face.jsonl").read_text().splitlines()
    ]
    assert rows == ["a", "b"]


def test_undecodable_frames_and_non_file_members_raise_value_errors(tmp_path: Path) -> None:
    for data in (b"", b"not a jpeg"):
        with pytest.raises(ValueError, match="JPEG decoding failed"):
            shard_module.decode_jpeg(data)
    path = tmp_path / "t.tar"
    with tarfile.open(path, "w") as tar:
        tar.addfile(tarfile.TarInfo("index.json"), io.BytesIO(b""))  # an empty regular file
        folder = tarfile.TarInfo("key")
        folder.type = tarfile.DIRTYPE
        tar.addfile(folder)
    with tarfile.open(path) as tar:
        assert shard_module.member_bytes(tar, "index.json") == b""
        with pytest.raises(ValueError, match="'key' is not a regular file"):
            shard_module.member_bytes(tar, "key")
        with pytest.raises(KeyError):
            shard_module.member_bytes(tar, "missing")
