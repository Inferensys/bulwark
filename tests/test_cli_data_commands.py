"""Data commands of the CLI: config roots, split, shard records, coverage, teacher
agreement, purge, the face teacher's clip filter and training errors."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import typer
from conftest import plain
from typer.testing import CliRunner

from bulwark.cli import _roots, app
from bulwark.data.manifest import ClipRecord, read_manifest, write_manifest
from bulwark.data.shards import ShardWriter

runner = CliRunner()
REPO = Path(__file__).resolve().parents[1]


def clip(clip_id: str, dataset: str, subject: str, split: str = "train", **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset=dataset, subject_id=f"{dataset}:{subject}", media_path=kwargs.pop("media_path", f"{clip_id.split('/')[-1]}.jpg"),  # type: ignore[arg-type]
        media_type=kwargs.pop("media_type", "image"), licence="t", commercial_ok=False, split=split, **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data.yaml").write_text("roots:\n  state_farm: raw\n  mrl_eye: raw/mrl\n")
    monkeypatch.setattr("bulwark.cli.DATA_CONFIG", tmp_path / "configs" / "data.yaml")
    return tmp_path


def test_help_is_the_project_tagline() -> None:
    description = re.search(r'^description = "(.+)"$', (REPO / "pyproject.toml").read_text(), re.M)[1]  # type: ignore[index]
    assert app.info.help == description


def test_roots_are_relative_to_the_config_and_a_missing_config_fails(workspace: Path) -> None:
    roots = _roots()
    assert roots["state_farm"] == (workspace / "raw").resolve() and roots["mrl_eye"] == (workspace / "raw" / "mrl").resolve()
    with pytest.raises(typer.Exit):
        _roots(workspace / "missing.yaml")
    assert _roots(workspace / "missing.yaml", required=False) == {}


def test_split_drops_removed_subjects_and_shows_excluded_clips(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    removed = {"drive_and_act:vp9": {"date": "2026-01-01", "source": "test"}}
    monkeypatch.setattr("bulwark.data.removals.load_removed_subjects", lambda *_: removed)
    monkeypatch.setattr("bulwark.data.splits.load_removed_subjects", lambda *_: removed)
    records = []
    for subject, part in (("vp1", "train"), ("vp2", "val"), ("vp5", "test"), ("vp9", "train")):
        for view in ("kinect_color", "inner_mirror"):
            records.append(clip(f"da/{subject}/{view}", "drive_and_act", subject, "unassigned", media_type="video",
                                media_path=f"{subject}.mp4", view=view, meta={"official_split": {"0": part}}))
    (workspace / "manifests").mkdir()
    write_manifest(records, workspace / "manifests" / "drive_and_act.parquet")
    result = runner.invoke(app, ["split", "--manifests", "manifests", "--out", "m.parquet", "--no-dedup"])
    assert result.exit_code == 0, result.output
    assert "Dropped 2 clips of removed subjects" in result.output and "excluded" in result.output
    splits = {r.clip_id: r.split for r in read_manifest(workspace / "m.parquet")}
    assert "da/vp9/kinect_color" not in splits
    assert splits["da/vp5/inner_mirror"] == "holdout" and splits["da/vp1/inner_mirror"] == "excluded"


def test_split_reports_a_failed_dedup_without_a_traceback(workspace: Path) -> None:
    (workspace / "manifests").mkdir()
    write_manifest([clip("state_farm/a", "state_farm", "a")], workspace / "manifests" / "state_farm.parquet")
    write_manifest([clip("mrl_eye/b", "mrl_eye", "b")], workspace / "manifests" / "mrl_eye.parquet")
    result = runner.invoke(app, ["split", "--manifests", "manifests", "--out", "m.parquet"])  # no images on disk
    assert result.exit_code == 1 and "--no-dedup" in result.output and "Traceback" not in result.output


def test_split_prefers_datasets_with_person_ids_over_driver_groups(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_dedup(records, roots, *, prefer_evaluable=()):  # type: ignore[no-untyped-def]
        seen["prefer"] = list(prefer_evaluable)
        return list(records), 0

    monkeypatch.setattr("bulwark.data.dedup.drop_cross_dataset_duplicates", fake_dedup)
    (workspace / "manifests").mkdir()
    auc = [clip(f"auc_v2/{s}", "auc_v2", f"{s}_drivers", "unassigned", meta={"official_split": {"v2": s}})
           for s in ("train", "test")]  # fmt: skip
    write_manifest(auc, workspace / "manifests" / "auc_v2.parquet")
    write_manifest([clip("state_farm/a", "state_farm", "a", "unassigned")], workspace / "manifests" / "state_farm.parquet")
    result = runner.invoke(app, ["split", "--manifests", "manifests", "--out", "m.parquet"])
    assert result.exit_code == 0, result.output
    # AUC v2's ids name driver groups, so a copy shared with State Farm is kept in State Farm.
    assert seen["prefer"] == ["state_farm", "auc_v2"]


def load_check_shards():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("check_shards", REPO / "scripts" / "check_shards.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["check_shards"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_shard_lists_decode_failures_for_check_shards_and_keeps_one_frame_size(workspace: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    make_image(workspace / "raw" / "a.jpg")
    write_manifest([clip("state_farm/a", "state_farm", "a"), clip("state_farm/broken", "state_farm", "broken")], workspace / "m.parquet")
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert result.exit_code == 0, result.output
    record = json.loads((workspace / "shards" / "state_farm.shards.json").read_text())
    assert record["frame_size"] == 256 and list(record["failed"]) == ["state_farm/broken"]

    check = load_check_shards()
    args = ["--manifest", "m.parquet", "--shards", "shards", "--max-missing", "0"]
    assert check.main(args) == 0  # the one missing clip is a known decode failure
    assert check.coverage(Path("m.parquet"), Path("shards"))["state_farm"].undecodable == {"state_farm/broken"}

    make_image(workspace / "raw" / "c.jpg")
    write_manifest([clip("state_farm/a", "state_farm", "a"), clip("state_farm/c", "state_farm", "c")], workspace / "m.parquet")
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards", "--size", "128"])
    assert result.exit_code == 1 and "256 px" in result.output


def test_shard_fingerprints_how_clips_are_cut_and_counts_legacy_shard_names(workspace: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    for name in "abc":
        make_image(workspace / "raw" / f"{name}.jpg")
    records = [clip("state_farm/a", "state_farm", "a"), clip("state_farm/b", "state_farm", "b", crop=(0, 0, 32, 24))]
    write_manifest(records, workspace / "m.parquet")
    assert runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"]).exit_code == 0
    record = json.loads((workspace / "shards" / "state_farm.shards.json").read_text())
    assert set(record["fingerprint"]) == {"clip_ids", "cuts"}
    check = load_check_shards()
    args = ["--manifest", "m.parquet", "--shards", "shards", "--max-missing", "0"]
    assert check.main(args) == 0 and not check.coverage(Path("m.parquet"), Path("shards"))["state_farm"].recut  # same digests

    # The adapter now cuts a sharded clip differently under the same clip id.
    write_manifest([dataclasses.replace(records[0], start_frame=5), records[1]], workspace / "m.parquet")
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert result.exit_code == 1 and "cuts already-sharded clips differently" in result.output
    assert check.main(args) == 1 and check.coverage(Path("m.parquet"), Path("shards"))["state_farm"].recut

    # Clips in a <dataset>-rNNN-NNNNN.tar shard (a name an earlier version wrote) count as sharded.
    with ShardWriter(workspace / "shards", prefix="state_farm-r001", clips_per_shard=10) as writer:
        writer.add("state_farm/c", np.zeros((1, 8, 8, 3), dtype=np.uint8))
    write_manifest([*records, clip("state_farm/c", "state_farm", "c")], workspace / "m.parquet")
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert result.exit_code == 0 and "3 of 3 clips already sharded, 0 to add" in result.output
    assert check.main(args) == 0


def test_coverage_and_teacher_agreement_commands(workspace: Path) -> None:
    records = [
        clip(f"state_farm/{i}", "state_farm", str(i), labels_gt={"phone": 1.0, "activity": "phone_call"}) for i in range(3)
    ]
    write_manifest(records, workspace / "m.parquet")
    result = runner.invoke(app, ["coverage", "--manifest", "m.parquet", "--markdown", "coverage.md"])
    assert result.exit_code == 0, result.output
    assert "phone" in result.output and "only '1'" in result.output and (workspace / "coverage.md").exists()

    (workspace / "labels").mkdir()
    rows = [{"clip_id": f"state_farm/{i}", "labels": {"phone": float(i > 0)}} for i in range(3)]
    (workspace / "labels" / "objects.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = runner.invoke(app, ["label", "agreement", "--manifest", "m.parquet", "--labels", "labels", "--out", "agreement.json"])
    assert result.exit_code == 0, result.output
    written = json.loads((workspace / "agreement.json").read_text())
    assert written["splits"] == "all"
    stats = written["report"]["objects"]["phone"]["state_farm"]
    assert stats["n"] == 3 and stats["recall"] == 2 / 3 and "objects" in result.output


def test_teacher_agreement_leaves_out_excluded_datasets(workspace: Path) -> None:
    records = [clip("state_farm/a", "state_farm", "a", labels_gt={"phone": 1.0}),
               clip("zenodo_driver/a", "zenodo_driver", "a", "excluded", labels_gt={"phone": 1.0})]
    write_manifest(records, workspace / "m.parquet")  # no recorded policy: the code default excludes zenodo_driver
    (workspace / "labels").mkdir()
    rows = [{"clip_id": r.clip_id, "labels": {"phone": 1.0}} for r in records]
    (workspace / "labels" / "objects.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = runner.invoke(app, ["label", "agreement", "--manifest", "m.parquet", "--labels", "labels", "--out", "agreement.json"])
    assert result.exit_code == 0, result.output
    assert set(json.loads((workspace / "agreement.json").read_text())["report"]["objects"]["phone"]) == {"state_farm"}
    assert "zenodo_driver" not in result.output


def test_teacher_agreement_split_filter_and_npv(workspace: Path) -> None:
    records = [clip("state_farm/a", "state_farm", "a", "train", labels_gt={"phone": 0.0}),
               clip("state_farm/b", "state_farm", "b", "val", labels_gt={"phone": 1.0}),
               clip("state_farm/c", "state_farm", "c", "test", labels_gt={"phone": 1.0})]
    write_manifest(records, workspace / "m.parquet")
    (workspace / "labels").mkdir()
    rows = [{"clip_id": r.clip_id, "labels": {"phone": 0.0}} for r in records]
    (workspace / "labels" / "objects.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = ["label", "agreement", "--manifest", "m.parquet", "--labels", "labels", "--out", "agreement.json"]

    def phone(*split: str) -> dict:
        result = runner.invoke(app, [*args, *(a for s in split for a in ("--split", s))])
        assert result.exit_code == 0, result.output
        assert "npv" in result.output
        written = json.loads((workspace / "agreement.json").read_text())
        # The file records the splits it scored, so a train-only table is recognisable.
        names = sorted({n for s in split for n in s.split(",")})
        assert written["splits"] == (names or "all")
        return written["report"]["objects"]["phone"]["state_farm"]

    train = phone("train")
    assert (train["n"], train["tn"], train["fn"], train["npv"]) == (1, 1, 0, 1.0)
    every = phone()  # default: every split
    assert (every["n"], every["fn"], every["npv"]) == (3, 2, 1 / 3)
    assert phone("train,val", "test") == every  # comma-separated and repeated
    val = phone("val")
    assert (val["n"], val["fn"], val["npv"]) == (1, 1, 0.0)
    result = runner.invoke(app, [*args, "--split", "trian"])
    assert result.exit_code != 0 and "trian" in result.output
    # An empty --split (an unset shell variable) is an error, not a silent every-split run.
    for empty in ("", " , "):
        result = runner.invoke(app, [*args, "--split", empty])
        assert result.exit_code != 0 and "names no split" in result.output


def test_check_shards_lists_unreadable_shards_as_broken(workspace: Path) -> None:
    write_manifest([clip("state_farm/a", "state_farm", "a")], workspace / "m.parquet")
    (workspace / "shards").mkdir()
    (workspace / "shards" / "state_farm-00000.tar").write_bytes(b"not a tar file" * 200)
    (workspace / "shards" / "state_farm-00001.tar").write_bytes(b"abc")  # truncated
    check = load_check_shards()
    found = check.coverage(Path("m.parquet"), Path("shards"))["state_farm"]
    assert found.broken == ["state_farm-00000.tar", "state_farm-00001.tar"]
    assert check.main(["--manifest", "m.parquet", "--shards", "shards"]) == 1


def test_purge_removes_subjects_everywhere_and_logs_it(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listed = {"state_farm:a": {"date": "2026-01-01", "source": "test"}}
    monkeypatch.setattr("bulwark.data.removals.load_removed_subjects", lambda *_: listed)
    records = [clip("state_farm/a", "state_farm", "a"), clip("state_farm/b", "state_farm", "b")]
    (workspace / "data" / "manifests").mkdir(parents=True)
    write_manifest(records, workspace / "data" / "manifests" / "state_farm.parquet")
    write_manifest(records, workspace / "data" / "manifest.parquet")
    with ShardWriter(workspace / "data" / "shards", prefix="state_farm", clips_per_shard=10) as writer:
        for r in records:
            writer.add(r.clip_id, np.zeros((1, 8, 8, 3), dtype=np.uint8))
    with ShardWriter(workspace / "data" / "shards-sample", prefix="state_farm", clips_per_shard=10) as writer:
        writer.add("state_farm/a", np.zeros((1, 8, 8, 3), dtype=np.uint8))  # `bulwark shard --sample`
    (workspace / "data" / "labels").mkdir()
    (workspace / "data" / "labels" / "face.jsonl").write_text("".join(json.dumps({"clip_id": r.clip_id, "labels": {}}) + "\n" for r in records))
    (workspace / "data" / "features" / "frozen").mkdir(parents=True)
    np.savez(workspace / "data" / "features" / "frozen" / "features-0.npz", clip_ids=np.array(["state_farm/a", "state_farm/b"]), tokens=np.zeros((2, 3)))

    result = runner.invoke(app, ["purge", "state_farm:a", "--yes"])
    assert result.exit_code == 0, result.output
    assert [r.clip_id for r in read_manifest(workspace / "data" / "manifest.parquet")] == ["state_farm/b"]
    with np.load(workspace / "data" / "features" / "frozen" / "features-0.npz") as data:
        assert list(data["clip_ids"]) == ["state_farm/b"]
    log = [json.loads(line) for line in (workspace / "data" / "purges.jsonl").read_text().splitlines()]
    assert log[0]["counts"]["clips"] == 1 and log[0]["counts"]["shard_clips"] == 2 and log[0]["date"]

    result = runner.invoke(app, ["purge", "state_farm:b", "--yes"])  # not listed: ingest would bring it back
    assert result.exit_code == 1 and "removed_subjects.yaml" in result.output
    listed["state_farm:b"] = {"date": "2026-01-02", "source": "test"}
    result = runner.invoke(app, ["purge", "state_farm:b"], input="n\n")  # declined: nothing changes
    assert result.exit_code == 1 and len(read_manifest(workspace / "data" / "manifest.parquet")) == 1


def test_face_teacher_skips_eye_crops(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_manifest([clip("state_farm/a", "state_farm", "a"), clip("mrl_eye/e", "mrl_eye", "e", meta={"crop_kind": "eye"})], workspace / "m.parquet")
    seen = {}

    class Teacher:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def clip(self, frames: np.ndarray) -> dict:
            return {}

    def run_teacher(name: str, teacher: object, shards: list, labels: Path, *, only: set | None = None, limit: int = 0) -> Path:
        seen["only"] = only
        return labels / f"{name}.jsonl"

    monkeypatch.setattr("bulwark.label.face_teacher.FaceTeacher", Teacher)
    monkeypatch.setattr("bulwark.label.run.run_teacher", run_teacher)
    result = runner.invoke(app, ["label", "face", "--manifest", "m.parquet"])
    assert result.exit_code == 0, result.output
    assert seen["only"] == {"state_farm/a"} and "1 eye crops skipped" in result.output


def test_training_flags_map_to_config_and_refusals_have_no_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    from bulwark.train.common import ResumeError

    seen = {}

    def run(config: object) -> Path:
        seen["config"] = config
        raise ResumeError("out holds another run")

    monkeypatch.setattr("bulwark.train.train_heads.run", run)
    result = runner.invoke(app, ["train", "heads", "--fresh", "--allow-mismatch"])
    assert result.exit_code == 1 and "out holds another run" in result.output and "Traceback" not in result.output
    assert seen["config"].fresh is True and seen["config"].allow_mismatch is True
    result = runner.invoke(app, ["train", "heads", "--set", "no_such_key=1"])
    assert result.exit_code == 1 and "Unknown config keys" in result.output


def test_train_cache_offers_overwrite_and_refuses_the_last_pt_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def run(config: object) -> Path:
        seen["config"] = config
        return Path("features")

    monkeypatch.setattr("bulwark.train.cache_features.run", run)
    usage = plain(runner.invoke(app, ["train", "cache", "--help"], env={"COLUMNS": "200"}).output)
    assert "--overwrite" in usage and "--fresh" not in usage and "--allow-mismatch" not in usage
    assert "release_only" in plain(runner.invoke(app, ["train", "heads", "--help"], env={"COLUMNS": "200"}).output)
    result = runner.invoke(app, ["train", "cache", "--overwrite"])
    assert result.exit_code == 0 and seen["config"].overwrite is True
    seen.clear()
    for flag in ("fresh=true", "allow_mismatch=true"):
        result = runner.invoke(app, ["train", "cache", "--set", flag])
        assert result.exit_code == 1 and "--overwrite" in result.output and "Traceback" not in result.output
    assert not seen  # refused before the job ran
    assert runner.invoke(app, ["train", "cache", "--fresh"]).exit_code == 2  # no such option
