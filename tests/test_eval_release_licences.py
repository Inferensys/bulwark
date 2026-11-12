"""`bulwark release`, `bulwark export` and examples/make_release.py classify a weights
licence the same way: one that restricts use to non-commercial research is accepted for
v0 weights, whatever its wording, and any other needs the commercial release check. The
CLI and the example write the same config.json, release check included."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from bulwark.cli import app
from bulwark.data.registry import MAINTAINERS_RELEASE_DECISION
from bulwark.export.release import (
    MODEL_RELEASE,
    RESEARCH_LICENCE,
    model_card_url,
    prepare_release,
    research_only,
)
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
EDGE = {"backbone": "test_efficientnet", "img_size": 64}
runner = CliRunner()


@pytest.fixture
def v0_checkpoint(tmp_path: Path) -> Path:
    """An E1 trained on research-only data (DMD), as the v0 runs are; DMD is cleared for a
    weights release by the maintainers' decision of 2026-10-30."""
    torch.manual_seed(0)
    provenance = {
        "job": "distill", "commercial_only": False,
        "datasets": {"dmd": {"licence": "CC BY-NC-ND 4.0", "commercial_ok": False, "splits": {"train": 4}}},
        "code": {"commit": "0123456789ab" * 3, "dirty": False},
    }
    extra = {"step": 1, **EDGE, "supervised_labels": ["yawn"], "provenance": provenance}
    return save_checkpoint(tmp_path / "e1.pt", BulwarkEdge(pretrained=False, **EDGE), extra)


def test_research_only_accepts_non_commercial_wordings_and_nothing_else() -> None:
    for licence in (RESEARCH_LICENCE, "CC BY-NC 4.0 (research use only)", "CC-BY-NC-SA-4.0", "research use only"):
        assert research_only(licence), licence
    for licence in ("Apache-2.0", "MIT (research use only)", "CC BY 4.0", "commercial use allowed", "", "proprietary"):
        assert not research_only(licence), licence


def test_release_accepts_a_non_commercial_wording_and_refuses_a_commercial_one(v0_checkpoint: Path, tmp_path: Path) -> None:
    out = tmp_path / "release"
    wording = "CC BY-NC 4.0 (research use only)"
    result = runner.invoke(app, ["release", str(v0_checkpoint), str(out), "--weights-licence", wording])
    assert result.exit_code == 0, result.output
    config = json.loads((out / "config.json").read_text())
    assert config["weights_licence"] == wording and config["release_check"]["checked_for"] == "research licence"

    refused = runner.invoke(app, ["release", str(v0_checkpoint), str(tmp_path / "apache"), "--weights-licence", "Apache-2.0"])
    assert refused.exit_code == 1 and "does not restrict use to non-commercial research" in refused.output
    assert not (tmp_path / "apache").exists()
    with pytest.raises(ValueError, match="cleared for commercial release"):
        prepare_release(v0_checkpoint, "MIT", kind="e1")


def test_make_release_writes_the_same_config_as_the_cli(v0_checkpoint: Path, tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    done = subprocess.run(
        [sys.executable, "examples/make_release.py", str(v0_checkpoint), str(tmp_path / "example")],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert "Not cleared for publication" not in done.stdout  # DMD is cleared by the maintainers' decision
    result = runner.invoke(app, ["release", str(v0_checkpoint), str(tmp_path / "cli")])
    assert result.exit_code == 0, result.output

    example = json.loads((tmp_path / "example" / "config.json").read_text())
    cli = json.loads((tmp_path / "cli" / "config.json").read_text())
    assert example["weights_licence"] == cli["weights_licence"] == RESEARCH_LICENCE
    assert example["model_card"] == model_card_url("e1") and example["model_release"] == MODEL_RELEASE
    for key in ("model_card", "model_release", "release_check", "source", "commercial_only", "labels"):
        assert example[key] == cli[key], key
    assert cli["release_check"]["cleared"] is True
    assert cli["release_check"]["datasets"]["dmd"]["weights_release_basis"] == MAINTAINERS_RELEASE_DECISION
