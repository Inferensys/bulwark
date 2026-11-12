"""The clips `bulwark label objects` and `bulwark label vlm` queue: eye crops are skipped,
since `bulwark label merge` drops both teachers' output on them, and --size is checked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from bulwark.cli import app
from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.label.objects_teacher import MODEL_SIZES

runner = CliRunner()


def clip(clip_id: str, dataset: str, **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset=dataset, subject_id=f"{dataset}:{clip_id}", media_path=f"{clip_id.split('/')[-1]}.jpg",
        media_type="image", licence="t", commercial_ok=False, split="train", **kwargs,  # type: ignore[arg-type]
    )


class Teacher:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def clip(self, frames: np.ndarray) -> dict:
        return {}


@pytest.fixture
def seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.chdir(tmp_path)
    write_manifest([
        clip("state_farm/a", "state_farm"),
        clip("drive_and_act/b", "drive_and_act", labels_gt={"activity": "drinking"}),
        clip("mrl_eye/e", "mrl_eye", meta={"crop_kind": "eye"}),
    ], tmp_path / "m.parquet")
    calls: dict = {}

    def run_teacher(name: str, teacher: object, shards: list, labels: Path, *, only: set | None = None, limit: int = 0) -> Path:
        calls[name] = only
        return labels / f"{name}.jsonl"

    monkeypatch.setattr("bulwark.label.objects_teacher.ObjectsTeacher", Teacher)
    monkeypatch.setattr("bulwark.label.vlm_teacher.VlmTeacher", Teacher)
    monkeypatch.setattr("bulwark.label.run.run_teacher", run_teacher)
    return calls


def test_objects_teacher_skips_eye_crops(seen: dict) -> None:
    result = runner.invoke(app, ["label", "objects", "--manifest", "m.parquet"])
    assert result.exit_code == 0, result.output
    assert seen["objects"] == {"state_farm/a", "drive_and_act/b"} and "1 eye crops skipped" in result.output


def test_vlm_teacher_skips_eye_crops_and_clips_with_ground_truth_activity(seen: dict) -> None:
    result = runner.invoke(app, ["label", "vlm", "--manifest", "m.parquet"])
    assert result.exit_code == 0, result.output
    assert seen["vlm"] == {"state_farm/a"} and "1 eye crops skipped" in result.output


def test_objects_size_is_checked_before_anything_runs(seen: dict) -> None:
    assert list(MODEL_SIZES) == ["nano", "small", "medium"]
    result = runner.invoke(app, ["label", "objects", "--manifest", "m.parquet", "--size", "large"])
    assert result.exit_code == 2 and "nano, small, medium" in result.output and "objects" not in seen
