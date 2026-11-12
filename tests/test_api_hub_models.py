"""`load` and `bulwark predict` take a Hugging Face repository id in place of a local path
(bulwark.inference.resolve_model), against a fake huggingface_hub (no test touches the
network)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from typer.testing import CliRunner

from bulwark.cli import app
from bulwark.inference import (
    E1_GRAPHS,
    ONNX_RELEASE_FORMAT,
    TorchModel,
    load,
    resolve_model,
    save_release,
)
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint

EDGE = {"backbone": "test_efficientnet", "img_size": 64}
B1_ID, E1_ID = "Inferensys/bulwark-b1-driver-monitoring", "Inferensys/bulwark-e1-driver-monitoring-onnx"


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Repositories as local folders, served by fake hf_hub_download and snapshot_download
    that copy files into a cache folder, as huggingface_hub does."""
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    repos: dict[str, Path] = {}
    cache = tmp_path / "cache"

    def folder(repo_id: str) -> Path:
        if repo_id not in repos:  # as the Hub answers for a missing or unreadable repository
            raise RepositoryNotFoundError("404", response=SimpleNamespace(headers={}, request=None, status_code=404))
        return repos[repo_id]

    def hf_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
        target = cache / repo_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder(repo_id) / filename, target)
        return str(target)

    def snapshot_download(repo_id: str, allow_patterns: list[str] | None = None, **kwargs: Any) -> str:
        source = folder(repo_id)
        for path in source.iterdir():
            if allow_patterns is None or any(Path(path.name).match(p) for p in allow_patterns):
                hf_hub_download(repo_id, path.name)
        return str(cache / repo_id)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return repos


def release(path: Path) -> Path:
    torch.manual_seed(0)
    save_checkpoint(path.with_suffix(".pt"), BulwarkEdge(pretrained=False, **EDGE), {"step": 2, **EDGE})
    model = load(path.with_suffix(".pt"), device="cpu", labels=["yawn"])
    assert isinstance(model, TorchModel)
    return save_release(model, path, weights_licence="CC BY-NC 4.0 (research use only)")


def test_a_repository_id_loads_the_release_in_it(tmp_path: Path, hub: dict[str, Path]) -> None:
    hub[B1_ID] = release(tmp_path / "repo-b1")
    (hub[B1_ID] / "README.md").write_text("# card\n")  # not needed to load, so not downloaded
    model = load(B1_ID, device="cpu")
    assert model.labels == ("yawn",) and isinstance(model, TorchModel)
    assert sorted(p.name for p in (tmp_path / "cache" / B1_ID).iterdir()) == ["config.json", "model.safetensors"]


def test_e1_s_repository_gives_its_video_graph_or_the_graph_asked_for(tmp_path: Path, hub: dict[str, Path]) -> None:
    repo = hub[E1_ID] = tmp_path / "repo-e1"
    repo.mkdir()
    (repo / "config.json").write_text(json.dumps({"format": ONNX_RELEASE_FORMAT, "graphs": {}}))
    for name in E1_GRAPHS.values():
        (repo / name).write_bytes(name.encode())
    assert resolve_model(E1_ID).read_bytes() == b"bulwark-e1-video.onnx"
    assert resolve_model(f"{E1_ID}/bulwark-e1-image.onnx").read_bytes() == b"bulwark-e1-image.onnx"


def test_local_paths_win_and_other_strings_stay_paths(tmp_path: Path, hub: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Inferensys").mkdir()
    (tmp_path / "Inferensys" / "local").mkdir()
    assert resolve_model("Inferensys/local") == Path("Inferensys/local")  # exists here
    assert resolve_model(Path("Inferensys/missing")) == Path("Inferensys/missing")  # a Path is never a repository id
    assert resolve_model("runs/eval/b1/best.pt") == Path("runs/eval/b1/best.pt")  # three parts, not a graph


def test_a_missing_local_path_is_never_sent_to_the_hub(
    tmp_path: Path, hub: dict[str, Path], monkeypatch: pytest.MonkeyPatch, make_video
) -> None:  # type: ignore[no-untyped-def]
    import huggingface_hub

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: calls.append(args))
    # Under a folder that exists here, or naming a model file: a typo, not a repository id.
    for path in ("runs/b1-release", "runs/export/e1.onnx", "export/e1.onnx", "checkpoints/best.pt"):
        assert resolve_model(path) == Path(path)
    video = make_video(tmp_path / "clip.mp4", frames=40)
    result = CliRunner().invoke(app, ["predict", "runs/b1-release", str(video)])
    assert result.exit_code != 0 and "request access" not in " ".join(result.output.split())
    assert calls == []


def test_a_missing_repository_or_huggingface_hub_says_what_to_do(hub: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(FileNotFoundError, match="request access on its page, then run `hf auth login`"):
        resolve_model("Inferensys/no-such-model")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match="install huggingface_hub"):
        resolve_model("Inferensys/no-such-model")


def test_predict_takes_a_repository_id_and_fails_cleanly_without_access(tmp_path: Path, hub: dict[str, Path], make_video) -> None:  # type: ignore[no-untyped-def]
    video = make_video(tmp_path / "clip.mp4", frames=40)
    result = CliRunner().invoke(app, ["predict", "Inferensys/no-such-model", str(video)])
    assert result.exit_code == 1 and "request access on its page" in " ".join(result.output.split())
    hub[B1_ID] = release(tmp_path / "repo-b1")
    result = CliRunner().invoke(app, ["predict", B1_ID, str(video), "--device", "cpu", "--out", str(tmp_path / "timeline.json")])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "timeline.json").read_text())["model"] == B1_ID
