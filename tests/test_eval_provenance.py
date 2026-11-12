from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.eval.provenance import (
    REPO_ROOT,
    SPLIT_POLICY_KEY,
    git_state,
    public_path,
    scrub_paths,
    split_policy,
)


@pytest.mark.parametrize("keys", [("commit", "dirty"), ("git_commit", "git_dirty")])
def test_packed_code_reports_the_commit_it_was_packed_from(tmp_path: Path, keys: tuple[str, str]) -> None:
    (tmp_path / "PROVENANCE.json").write_text(json.dumps({keys[0]: "0123abcd", keys[1]: False}))
    assert git_state(tmp_path) == {"commit": "0123abcd", "dirty": False, "changed": [], "source": "PROVENANCE.json"}
    (tmp_path / "PROVENANCE.json").write_text(json.dumps({keys[0]: "0123abcd", keys[1]: True, "dirty_files": ["bulwark/cli.py"]}))
    assert git_state(tmp_path)["changed"] == ["bulwark/cli.py"]


def test_published_paths_carry_no_local_layout(tmp_path: Path) -> None:
    assert public_path(REPO_ROOT / "data" / "manifest.parquet") == "data/manifest.parquet"
    assert public_path(tmp_path / "runs" / "final.pt") == "final.pt"
    assert public_path(None) is None
    scrubbed = scrub_paths({"init": "/Users/someone/b1/final.pt", "shards": "data/shards", "n": 3, "list": ["~/x/y.pt"]})
    assert scrubbed == {"init": "final.pt", "shards": "data/shards", "n": 3, "list": ["y.pt"]}


def test_split_policy_recorded_in_the_manifest_wins(tmp_path: Path) -> None:
    record = ClipRecord(clip_id="a", dataset="d", subject_id="d:1", media_path="a.jpg", media_type="image", licence="t", commercial_ok=True, split="test")
    path = write_manifest([record], tmp_path / "m.parquet")
    default = split_policy(path)
    assert default["source"].startswith("code default") and "salt" in default
    assert all(not isinstance(v, (set, frozenset)) for v in default.values())

    table = pq.read_table(path)
    metadata = {**(table.schema.metadata or {}), SPLIT_POLICY_KEY: json.dumps({"salt": "custom"}).encode()}
    pq.write_table(table.replace_schema_metadata(metadata), path)
    assert split_policy(path) == {"source": "manifest", "salt": "custom"}


def test_probe_formats_the_mean_over_views() -> None:
    from bulwark.eval.probe import _render

    payload = {
        "meta": {"protocol": "drive_and_act_midlevel", "git": {"commit": "abc", "dirty": False}, "features": "frozen", "pool": "global",
                 "manifest": {"sha256": "0" * 64}, "created_utc": "2026-01-01T00:00:00+00:00"},
        "results": {"views": {}, "classes": ["a", "b"], "mean_over_views": 0.0},
    }
    assert "Mean over views: 0.000." in _render(payload)
    payload["results"]["mean_over_views"] = 0.41234567
    assert "Mean over views: 0.412." in _render(payload)
    payload["results"]["mean_over_views"] = None
    assert "Mean over views: –." in _render(payload)


def test_report_keeps_prediction_provenance_and_flags_a_changed_manifest(tmp_path: Path) -> None:
    import numpy as np

    from bulwark.eval.predict import Predictions
    from bulwark.eval.report import build_report

    records = [
        ClipRecord(clip_id=f"{split}{i}", dataset="mrl_eye", subject_id=f"mrl_eye:{split}{i}", media_path=f"{i}.jpg", media_type="image",
                   licence="t", commercial_ok=True, split=split, labels_gt={"eyes_closed": float(i % 2)})
        for split in ("train", "test") for i in range(4)
    ]
    manifest = write_manifest(records, tmp_path / "m.parquet")
    predictions = {"test": Predictions([f"test{i}" for i in range(4)], {"eyes_closed": np.array([-1, 1, -1, 1], dtype=np.float32)})}
    predicted_with = {"git": {"commit": "f" * 40, "dirty": False}, "created_utc": "2026-01-01T00:00:00+00:00",
                      "manifest": {"path": "old.parquet", "sha256": "e" * 64, "clips": 3}}
    path = build_report(predictions, manifest, tmp_path / "eval", meta={"model": "e1", "predicted_with": predicted_with}, bootstrap=0)
    meta = json.loads((path.parent / "metrics.json").read_text())["meta"]
    assert meta["predicted_with"] == predicted_with  # scoring never overwrites how predictions were made
    assert meta["scored_with"]["git"] != predicted_with["git"]
    assert any("manifest sha256 eeeeeeeeeeee" in w for w in meta["warnings"])
    html = path.read_text()
    assert "ffffffffffff, 2026-01-01" in html and "differs from the predictions" in html and "warning" in html
