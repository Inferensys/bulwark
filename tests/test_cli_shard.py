"""`bulwark shard` (runs that add clips, --sample, interrupted shards, clips that fail to
decode) and CLI-wide checks: export parity clips, help text on every option and
`bulwark evaluate` errors."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pytest
from conftest import plain
from typer.testing import CliRunner

from bulwark.cli import app
from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.data.shards import shard_index

runner = CliRunner()


def image(name: str, split: str = "train") -> ClipRecord:
    return ClipRecord(
        clip_id=f"state_farm/{name}", dataset="state_farm", subject_id=f"state_farm:{name}", media_path=f"{name}.jpg",
        media_type="image", licence="t", commercial_ok=False, split=split,
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_image) -> Path:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data.yaml").write_text("roots:\n  state_farm: raw\n")
    monkeypatch.setattr("bulwark.cli.DATA_CONFIG", tmp_path / "configs" / "data.yaml")
    for name in "abcde":
        make_image(tmp_path / "raw" / f"{name}.jpg")
    return tmp_path


def sharded_ids(directory: Path) -> dict[str, set[str]]:
    return {p.name: set(shard_index(p).values()) for p in sorted(directory.glob("*.tar"))}


def test_shard_adds_only_missing_clips_and_reports_stale(workspace: Path) -> None:
    write_manifest([image("a"), image("b")], workspace / "m.parquet")
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert result.exit_code == 0, result.output
    first = sharded_ids(workspace / "shards")
    assert first == {"state_farm-00000.tar": {"state_farm/a", "state_farm/b"}}

    write_manifest([image("b"), image("c"), image("d")], workspace / "m.parquet")  # a removed, c and d new
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert result.exit_code == 0, result.output
    assert "2 to add, 1 stale" in result.output
    second = sharded_ids(workspace / "shards")
    assert second["state_farm-00000.tar"] == first["state_farm-00000.tar"]  # never rewritten
    assert second["state_farm-00001.tar"] == {"state_farm/c", "state_farm/d"}  # numbering continues

    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert result.exit_code == 0 and "0 to add" in result.output
    assert len(list((workspace / "shards").glob("*.tar"))) == 2


def test_shard_rerun_with_a_clip_that_never_decodes_succeeds(workspace: Path) -> None:
    (workspace / "configs" / "data.yaml").write_text("roots:\n  state_farm: raw\n  mrl_eye: raw\n")
    eye = ClipRecord(
        clip_id="mrl_eye/e", dataset="mrl_eye", subject_id="mrl_eye:s1", media_path="e.jpg",
        media_type="image", licence="t", commercial_ok=False, split="train",
    )
    write_manifest([image("a"), image("missing"), eye], workspace / "m.parquet")  # missing.jpg does not exist
    first = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert first.exit_code == 0 and "(1 failed)" in first.output, first.output

    second = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert second.exit_code == 0, second.output
    assert "1 clips still fail to decode" in second.output and "every clip failed" not in second.output
    assert "mrl_eye: 1 of 1 clips already sharded" in second.output  # later datasets are still reached
    assert sorted(p.name for p in (workspace / "shards").glob("*.tar")) == ["mrl_eye-00000.tar", "state_farm-00000.tar"]


def test_parity_clips_match_the_graph_input(tmp_path: Path) -> None:
    from bulwark.cli import _parity_clips
    from bulwark.data.shards import ShardWriter

    rng = np.random.default_rng(0)
    with ShardWriter(tmp_path, prefix="t", clips_per_shard=10) as writer:
        writer.add("big", rng.integers(0, 256, (4, 96, 96, 3), dtype=np.uint8))
        writer.add("small", rng.integers(0, 256, (4, 80, 80, 3), dtype=np.uint8))
        writer.add("image", rng.integers(0, 256, (1, 80, 80, 3), dtype=np.uint8))
    clips = _parity_clips(tmp_path, frames=4, input_size=80)
    assert clips is not None and clips.shape == (1, 4, 80, 80, 3)
    assert _parity_clips(tmp_path, frames=4, input_size=64) is None


def test_shard_sample_goes_to_its_own_directory(workspace: Path) -> None:
    write_manifest([image("a"), image("b", "test")], workspace / "m.parquet")
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--sample", "1"])
    assert result.exit_code == 0, result.output
    assert list((workspace / "data" / "shards-sample").glob("*.tar")) and not (workspace / "data" / "shards").exists()


def test_shard_refuses_interrupted_shards_and_fails_when_every_clip_fails(workspace: Path) -> None:
    write_manifest([image("x"), image("y")], workspace / "m.parquet")  # no such images
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert result.exit_code == 1 and "every clip failed" in result.output

    (workspace / "shards").mkdir(exist_ok=True)
    with tarfile.open(workspace / "shards" / "state_farm-00000.tar", "w"):
        pass  # a .tar without index.json, as an older writer left one when it was interrupted
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert result.exit_code == 1 and "no index" in result.output


def test_every_option_has_help_text() -> None:
    import click
    import typer.main

    command = typer.main.get_command(app)

    def walk(cmd: click.Command, path: str) -> list[str]:
        missing = [f"{path} {p.opts[0]}" for p in cmd.params if isinstance(p, click.Option) and not p.help]
        for name, sub in getattr(cmd, "commands", {}).items():
            missing += walk(sub, f"{path} {name}")
        return missing

    assert walk(command, "bulwark") == []


def test_evaluate_reports_errors_without_a_traceback(tmp_path: Path) -> None:
    write_manifest([image("a", "test")], tmp_path / "m.parquet")
    (tmp_path / "features").mkdir()
    np.savez(tmp_path / "features" / "features-0.npz", clip_ids=np.array(["state_farm/a"]), tokens=np.zeros((1, 25, 8), dtype=np.float16))
    result = runner.invoke(app, ["evaluate", "--model", "logits", "--features", str(tmp_path / "features"), "--manifest", str(tmp_path / "m.parquet"), "--out", str(tmp_path / "eval")])
    assert result.exit_code == 1 and "no logit" in plain(result.output) and "Traceback" not in result.output
