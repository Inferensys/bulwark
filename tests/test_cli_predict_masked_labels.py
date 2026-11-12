"""`bulwark predict` on a checkpoint that records a label masked on one media type: the
label is left out for that media, in the printed lines and in the --out timeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from bulwark.cli import app
from bulwark.data.taxonomy import default_taxonomy
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint

runner = CliRunner()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An E1 that trained phone and yawn, with phone masked on video clips."""
    torch.manual_seed(0)
    model = BulwarkEdge(default_taxonomy(), pretrained=False, img_size=64)
    extra = {"img_size": 64, "supervised_labels": ["phone", "yawn"], "masked_labels": {"phone": ["video"]}}
    return save_checkpoint(tmp_path_factory.mktemp("e1") / "e1.pt", model, extra, full=True)


def test_predict_leaves_out_a_label_masked_on_video(checkpoint: Path, tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    video = make_video(tmp_path / "v.avi", frames=70)
    result = runner.invoke(app, ["predict", str(checkpoint), str(video), "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert "yawn=" in result.output and "phone=" not in result.output

    out = tmp_path / "timeline.json"
    result = runner.invoke(app, ["predict", str(checkpoint), str(video), "--device", "cpu", "--out", str(out)])
    assert result.exit_code == 0, result.output
    timeline = json.loads(out.read_text())
    assert timeline["labels"] == ["yawn"] and timeline["masked_labels"] == {"phone": ["video"]}
    assert timeline["windows"] and all(set(w["outputs"]) == {"yawn"} for w in timeline["windows"])

    # Every requested label masked on video: windows without outputs, not an error.
    result = runner.invoke(app, ["predict", str(checkpoint), str(video), "--device", "cpu", "--labels", "phone", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["labels"] == []


def test_predict_returns_both_labels_on_an_image(checkpoint: Path, tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    image = make_image(tmp_path / "face.jpg")
    out = tmp_path / "image.json"
    result = runner.invoke(app, ["predict", str(checkpoint), str(image), "--device", "cpu", "--out", str(out)])
    assert result.exit_code == 0, result.output
    timeline = json.loads(out.read_text())
    assert timeline["labels"] == ["phone", "yawn"] and set(timeline["windows"][0]["outputs"]) == {"phone", "yawn"}
