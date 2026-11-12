"""Local preparation scripts: the UTA-RLDD fetch loop, the pretrained pins and the shard coverage check."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bulwark.data.shards import ShardWriter

REPO = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash scripts")


GDOWN_STUB = """#!/usr/bin/env python3
import os, sys, zipfile
file_id, out = sys.argv[sys.argv.index("--continue") + 1], sys.argv[sys.argv.index("-O") + 1]
if file_id in os.environ.get("REFUSE_IDS", "").split():
    sys.exit(1)
if file_id in os.environ.get("CORRUPT_IDS", "").split():
    open(out, "wb").write(b"not a zip")
    sys.exit(0)
with zipfile.ZipFile(out, "w") as archive:
    archive.writestr(f"{file_id}/video.txt", "frames")
"""


def run_uta_fetch(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    stub = tmp_path / "gdown"
    stub.write_text(GDOWN_STUB)
    stub.chmod(0o755)
    command = ["bash", str(REPO / "scripts" / "fetch_uta_rldd.sh"), str(tmp_path / "uta")]
    env = {**os.environ, "GDOWN": str(stub), "RETRY_PASSES": "1", **env}
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)


def test_uta_fetch_deletes_corrupt_zips_and_marks_completion(tmp_path: Path) -> None:
    corrupt, refused = "12uVMVLv_kRXzQb4QDdru9RK3G9hzfpPQ", "1-JhE5TfkNyr4UbQk6PSkkpI7z-SChkWS"
    first = run_uta_fetch(tmp_path, CORRUPT_IDS=corrupt, REFUSE_IDS=refused)
    out = tmp_path / "uta"
    assert first.returncode == 1  # gave up after the only pass, without sleeping an hour
    assert "corrupt" in first.stderr
    assert not (out / "Fold2_part1.zip").exists()  # deleted, so the next pass downloads it again
    assert len(list(out.glob(".*.done"))) == 8
    assert not (out / ".complete").exists()

    second = run_uta_fetch(tmp_path)
    assert second.returncode == 0, second.stderr
    assert (out / ".complete").exists() and (out / corrupt / "video.txt").exists()
    assert not list(out.glob("*.zip"))

    # Every part's Drive id and hash were recorded before its zip was deleted, once.
    rows = [line.split("\t") for line in (out / "SOURCE.txt").read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 10 and len({row[0] for row in rows}) == 10
    part = {row[0]: row for row in rows}["Fold2_part1"]
    assert part[1] == corrupt and int(part[2]) > 0 and re.fullmatch(r"[0-9a-f]{64}", part[3])


def test_uta_fetch_records_parts_extracted_before_hashes_were_kept(tmp_path: Path) -> None:
    out = tmp_path / "uta"
    out.mkdir()
    (out / ".Fold1_part1.done").touch()  # extracted by an earlier run, its zip long gone
    result = run_uta_fetch(tmp_path)
    assert result.returncode == 0, result.stderr
    rows = {row[0]: row for row in (line.split("\t") for line in (out / "SOURCE.txt").read_text().splitlines())
            if not row[0].startswith("#")}  # fmt: skip
    assert len(rows) == 10
    assert rows["Fold1_part1"][1:4] == ["16x8afSOwfbdGAjUbfRFvwGj1ie8Lf5Do", "-", "-"]
    assert rows["Fold1_part1"][4].startswith("not recorded")
    assert not (out / "16x8afSOwfbdGAjUbfRFvwGj1ie8Lf5Do").exists()  # not downloaded again

    run_uta_fetch(tmp_path)  # a rerun adds no second row
    assert len([line for line in (out / "SOURCE.txt").read_text().splitlines() if not line.startswith("#")]) == 10


def load_check_shards():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("check_shards", REPO / "scripts" / "check_shards.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["check_shards"] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_check_shards_reports_missing_stale_and_broken_shards(tmp_path: Path) -> None:
    check = load_check_shards()
    manifest = tmp_path / "manifest.parquet"
    ids = {"a": [f"a/{i}" for i in range(4)], "b": [f"b/{i}" for i in range(3)]}
    pq.write_table(
        pa.table({"clip_id": ids["a"] + ids["b"], "dataset": ["a"] * 4 + ["b"] * 3}), manifest
    )
    shards = tmp_path / "shards"
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with ShardWriter(shards, prefix="a", clips_per_shard=10) as writer:
        for clip_id in ids["a"][:3] + ["a/old"]:  # one clip missing, one no longer in the manifest
            writer.add(clip_id, frames)
    with ShardWriter(shards, prefix="b", clips_per_shard=10) as writer:
        for clip_id in ids["b"]:
            writer.add(clip_id, frames)
    assert check.main(["--manifest", str(manifest), "--shards", str(shards)]) == 1
    report = check.coverage(manifest, shards)
    assert report["a"].missing == {"a/3"} and report["a"].stale == {"a/old"}
    assert report["b"].ok

    # A shard whose writer never closed it has no index.json.
    with tarfile.open(shards / "b-00001.tar", "w") as tar:
        info = tarfile.TarInfo("key/00.jpg")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"jpg"))
    report = check.coverage(manifest, shards)
    assert report["b"].broken == ["b-00001.tar"] and not report["b"].ok


def write_shards(shards: Path, prefix: str, clip_ids: list[str]) -> None:
    with ShardWriter(shards, prefix=prefix, clips_per_shard=10) as writer:
        for clip_id in clip_ids:
            writer.add(clip_id, np.zeros((2, 8, 8, 3), dtype=np.uint8))


def test_check_shards_tolerates_a_few_decode_failures(tmp_path: Path) -> None:
    check = load_check_shards()
    manifest = tmp_path / "manifest.parquet"
    ids = [f"a/{i}" for i in range(400)]
    pq.write_table(pa.table({"clip_id": ids, "dataset": ["a"] * len(ids)}), manifest)
    shards = tmp_path / "shards"
    write_shards(shards, "a", ids[:300])
    write_shards(shards, "a-r001", ids[300:399])  # added by a later shard run; one clip failed
    args = ["--manifest", str(manifest), "--shards", str(shards)]
    report = check.coverage(manifest, shards)
    assert set(report) == {"a"} and report["a"].missing == {"a/399"} and not report["a"].stale
    assert check.main(args) == 0  # 1 of 400 missing: under the default 0.5%
    assert check.main([*args, "--max-missing", "0"]) == 1


def test_fetch_pretrained_pins_match_the_code() -> None:
    from bulwark.model import backbone

    script = (REPO / "scripts" / "fetch_pretrained.sh").read_text()
    pins = dict(re.findall(r"^(VJEPA_COMMIT|VJEPA_SHA256)=(\S+)$", script, re.M))
    assert pins["VJEPA_COMMIT"] == backbone.VJEPA_COMMIT
    assert backbone.VJEPA_URL in script
    if hasattr(backbone, "VJEPA_SHA256"):  # load_pretrained checks the hash too
        assert pins["VJEPA_SHA256"] == backbone.VJEPA_SHA256


def fake_checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A repo root for prepare_local.sh whose `bulwark` command only logs its arguments."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "prepare_local.sh").write_text((REPO / "scripts" / "prepare_local.sh").read_text())
    (root / "scripts" / "check_shards.py").write_text("raise SystemExit(0)\n")
    (root / "bulwark").symlink_to(REPO / "bulwark")
    (root / "configs").mkdir()
    (root / "configs" / "data.yaml").write_text("roots: {}\n")
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    log = tmp_path / "bulwark.log"
    (venv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "bulwark").write_text(f'#!/bin/sh\necho "$*" >> "{log}"\n')
    for tool in ("python", "bulwark"):
        (venv / tool).chmod(0o755)
    return root, log


def test_reshard_sets_a_dataset_aside_with_its_teacher_labels(tmp_path: Path) -> None:
    root, log = fake_checkout(tmp_path)
    shards, labels = root / "data" / "shards", root / "data" / "labels"
    write_shards(shards, "zenodo_driver", ["zenodo_driver/a", "zenodo_driver/b"])
    write_shards(shards, "zenodo_driver-r001", ["zenodo_driver/c"])
    write_shards(shards, "mrl_eye", ["mrl_eye/a"])
    labels.mkdir(parents=True)
    rows = [{"clip_id": clip, "labels": {}} for clip in ("zenodo_driver/a", "zenodo_driver/c", "mrl_eye/a")]
    (labels / "face.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    env = {**os.environ, "RESHARD": "zenodo_driver", "SKIP_LABELS": "1"}
    result = subprocess.run(["bash", "scripts/prepare_local.sh"], cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in shards.iterdir()) == ["mrl_eye-00000.tar"]
    (aside,) = (root / "data" / "shards-replaced").iterdir()
    assert sorted(p.name for p in aside.iterdir()) == ["zenodo_driver-00000.tar", "zenodo_driver-r001-00000.tar"]
    assert [json.loads(line)["clip_id"] for line in (labels / "face.jsonl").read_text().splitlines()] == ["mrl_eye/a"]
    assert log.read_text().splitlines() == ["split", "shard", "label merge"]
    # The face teacher was skipped, so the dropped labels stay missing until it runs.
    assert "dropped the face labels of zenodo_driver" in result.stderr and "bulwark label face" in result.stderr


def test_the_steps_for_adding_one_dataset_use_real_commands_and_options() -> None:
    """The header of prepare_local.sh lists how to add one dataset to prepared data; each
    step must be a `bulwark` command, with options that command has, in a working order."""
    import typer

    from bulwark.cli import app

    header = (REPO / "scripts" / "prepare_local.sh").read_text().split("\nset -euo pipefail")[0]
    steps = re.findall(r"^#\s+bulwark ([^#\n]+?)\s*(?:#.*)?$", header, re.MULTILINE)
    cli = typer.main.get_command(app)
    commands = []
    for step in steps:
        words, command, path = step.split(), cli, []
        while words and words[0] in getattr(command, "commands", {}):  # a command group
            path.append(words[0])
            command = command.commands[words.pop(0)]
        assert path and not hasattr(command, "commands"), step
        known = {option for param in command.params for option in param.opts}
        assert {word for word in words if word.startswith("--")} <= known, step
        commands.append(" ".join(path))
    assert commands == ["ingest", "ingest", "split", "shard", "label face", "label merge"]
    assert "--dataset <dataset>" in header and "--pattern '<dataset>-*.tar'" in header
