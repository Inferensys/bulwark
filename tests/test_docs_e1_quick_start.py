"""The quick start of the Bulwark-E1 card (release/hf/e1.md) needs only onnxruntime, NumPy,
OpenCV and huggingface_hub, and reads a video as Bulwark does: the same letterbox, the same
16 frames, and the outputs, temperatures and thresholds that bulwark.inference gives."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bulwark.data.clips import letterbox  # noqa: E402
from bulwark.export.onnx import export_edge  # noqa: E402
from bulwark.export.release import MODEL_REPOS  # noqa: E402
from bulwark.inference import load, video_windows  # noqa: E402
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "release" / "hf" / "e1.md"
EDGE = {"backbone": "test_efficientnet", "img_size": 64}
SUPERVISED = ["eyes_closed", "perclos", "yawn", "head_yaw", "phone", "activity", "modality"]
MASKED = {"phone": ["image"], "modality": ["video"]}  # withheld on that media type
TEMPERATURE = {"eyes_closed": 1.7, "yawn": 0.6, "activity": 2.0}
THRESHOLDS = {"eyes_closed": 0.3, "yawn": 0.8}  # on the uncalibrated probability


def quick_start() -> str:
    """The card's Python as published, for E1's repository."""
    code = re.findall(r"```python\n(.*?)```", CARD.read_text(encoding="utf-8"), re.S)[0]
    return code.replace("{{ repo_id }}", MODEL_REPOS["e1"])


def definitions() -> dict[str, Any]:
    """The quick start's imports and functions, without its top-level calls."""
    tree = ast.parse(quick_start())
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace: dict[str, Any] = {}
    exec(compile(tree, "e1 quick start", "exec"), namespace)
    return namespace


def load_hf_release():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("hf_release", REPO / "scripts" / "hf_release.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["hf_release"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_the_quick_start_imports_no_bulwark_code() -> None:
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.walk(ast.parse(quick_start()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported == {"json", "cv2", "numpy", "onnxruntime", "huggingface_hub"}
    install = re.search(r"pip install ([^\n]*)", CARD.read_text(encoding="utf-8"))
    assert install is not None and install.group(1).split() == ["onnxruntime", "numpy", "opencv-python-headless", "huggingface_hub"]


@pytest.mark.parametrize("shape", [(120, 160), (480, 640), (720, 1280), (300, 200), (256, 256), (100, 60), (37, 53), (1000, 10)])
def test_the_letterbox_is_bulwark_s(shape: tuple[int, int]) -> None:
    frame = np.random.default_rng(shape[0] * shape[1]).integers(0, 256, (*shape, 3), dtype=np.uint8)
    np.testing.assert_array_equal(definitions()["letterbox"](frame), letterbox(frame, 256))


@pytest.fixture(scope="module")
def staged(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """E1's video graph with temperatures, thresholds and masked outputs, and the
    config.json the stage writes next to it."""
    folder = tmp_path_factory.mktemp("e1")
    torch.manual_seed(0)
    meta = {"step": 1, **EDGE, "supervised_labels": SUPERVISED, "masked_labels": MASKED}
    checkpoint = save_checkpoint(folder / "e1.pt", BulwarkEdge(pretrained=False, **EDGE).eval(), meta)
    export_edge(checkpoint, folder / "bulwark-e1-video.onnx", temperature=TEMPERATURE, thresholds=THRESHOLDS)
    (folder / "e1.pt").unlink()
    config = load_hf_release().graphs_config(folder)
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


def test_the_quick_start_runs_a_video_as_bulwark_does(staged: Path, tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    # 25 fps (every 3rd frame) and an odd size, so sampling and letterboxing both matter.
    video = make_video(tmp_path / "clip.mp4", frames=70, fps=25.0, size=(160, 90))
    downloads: list[tuple[str, str]] = []

    def hf_hub_download(repo_id: str, filename: str) -> str:
        downloads.append((repo_id, filename))
        return str(staged / filename)

    code = quick_start()
    assert "from huggingface_hub import hf_hub_download\n" in code and '"my_clip.mp4"' in code
    code = code.replace("from huggingface_hub import hf_hub_download\n", "").replace('"my_clip.mp4"', repr(str(video)))
    namespace: dict[str, Any] = {"hf_hub_download": hf_hub_download}
    exec(compile(code, "e1 quick start", "exec"), namespace)
    assert downloads == [(MODEL_REPOS["e1"], "bulwark-e1-video.onnx"), (MODEL_REPOS["e1"], "config.json")]

    window = next(video_windows(video))
    np.testing.assert_array_equal(namespace["frames"][0], window.frames)
    model = load(staged / "bulwark-e1-video.onnx")
    expected = model.predict(window.frames)
    decoded = {o["name"]: namespace["decode"](o, logit[0], namespace["contract"]) for o, logit in zip(namespace["outputs"], namespace["logits"], strict=True)}
    assert list(decoded) == list(model.labels_for("video")) == list(expected)  # trained, minus modality on video
    thresholds = model.thresholds_for("video")
    for name, value in decoded.items():
        reference = expected[name][0]
        if isinstance(value, dict) and "probability" in value:
            assert value["probability"] == pytest.approx(float(reference), abs=1e-5), name
            assert value["flagged"] is (float(reference) >= thresholds.get(name, 0.5)), name
        elif isinstance(value, dict):
            np.testing.assert_allclose(list(value.values()), reference, atol=1e-5, err_msg=name)
        else:
            assert value == pytest.approx(float(reference), abs=1e-5), name
    assert model.temperature_for("video") == pytest.approx(TEMPERATURE) and set(thresholds) == set(THRESHOLDS)


def test_the_quick_start_withholds_untrained_and_masked_outputs_per_media_type(staged: Path) -> None:
    contract = json.loads((staged / "config.json").read_text(encoding="utf-8"))["graphs"]["bulwark-e1-video.onnx"]["bulwark_contract"]
    trained = definitions()["trained_outputs"]
    model = load(staged / "bulwark-e1-video.onnx")
    for media in ("video", "image"):
        assert [o["name"] for o in trained(contract, media)] == list(model.labels_for(media)), media
