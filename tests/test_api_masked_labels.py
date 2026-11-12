"""Labels masked on one media type in training are left out of that media type's outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bulwark.inference import RollingMean, load, predict_image, predict_video, save_release
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint

EDGE = {"backbone": "test_efficientnet", "img_size": 64}
RESEARCH = "CC BY-NC 4.0 (research use only)"
SUPERVISED = ["eyes_closed", "phone", "yawn"]
# As the training jobs record it (LabelPlan.masked_media): phone had one class on video,
# attention_direction was masked as a whole by the config's mask_labels.
MASKED = {"attention_direction": ["image", "video"], "phone": ["video"]}


def edge_checkpoint(path: Path, **extra: object) -> Path:
    torch.manual_seed(0)
    return save_checkpoint(path, BulwarkEdge(pretrained=False, **EDGE), {"step": 1, **EDGE, **extra})


def clips(frames: int, batch: int = 2) -> np.ndarray:
    return np.random.default_rng(frames).integers(0, 255, (batch, frames, 40, 40, 3), dtype=np.uint8)


def test_a_label_masked_on_video_is_returned_for_images_only(tmp_path: Path) -> None:
    path = edge_checkpoint(tmp_path / "e1.pt", supervised_labels=SUPERVISED, masked_labels=MASKED)
    model = load(path, device="cpu")
    assert model.labels == tuple(SUPERVISED)
    assert model.masked_labels == {"attention_direction": ("image", "video"), "phone": ("video",)}
    assert model.labels_for("video") == ("eyes_closed", "yawn") and model.labels_for("image") == tuple(SUPERVISED)
    with pytest.raises(ValueError, match="media_type"):
        model.labels_for("audio")

    video, image = clips(4), clips(1)
    assert set(model.predict(video)) == {"eyes_closed", "yawn"}
    assert set(model.predict(image)) == set(SUPERVISED)
    # The outputs that remain are the same as without the record, and so are the embeddings.
    unmasked = load(edge_checkpoint(tmp_path / "plain.pt", supervised_labels=SUPERVISED), device="cpu")
    np.testing.assert_allclose(model.predict(video)["yawn"], unmasked.predict(video)["yawn"], atol=1e-6)
    np.testing.assert_allclose(model.embed(video, "state"), unmasked.embed(video, "state"), atol=1e-6)

    # Naming a label does not bring it back on the media type it learned nothing on.
    phone = load(path, device="cpu", labels=["phone"])
    assert phone.predict(video) == {} and set(phone.predict(image)) == {"phone"}

    with pytest.raises(ValueError, match="unknown media types"):
        load(edge_checkpoint(tmp_path / "bad.pt", masked_labels={"phone": ["audio"]}), device="cpu")


def test_releases_keep_the_masked_labels(tmp_path: Path) -> None:
    model = load(edge_checkpoint(tmp_path / "e1.pt", supervised_labels=SUPERVISED, masked_labels=MASKED), device="cpu")
    release = save_release(model, tmp_path / "release", weights_licence=RESEARCH)
    assert json.loads((release / "config.json").read_text())["masked_labels"] == MASKED
    reloaded = load(release, device="cpu")
    assert reloaded.masked_labels == model.masked_labels and set(reloaded.predict(clips(4))) == {"eyes_closed", "yawn"}


def test_timelines_and_images_leave_out_masked_labels(tmp_path: Path, make_video, make_image) -> None:  # type: ignore[no-untyped-def]
    model = load(edge_checkpoint(tmp_path / "e1.pt", supervised_labels=SUPERVISED, masked_labels=MASKED), device="cpu")
    video = make_video(tmp_path / "v.avi", frames=60, fps=30.0)
    (window,) = predict_video(model, video)
    assert set(window["outputs"]) == {"eyes_closed", "yawn"}
    with pytest.raises(KeyError, match="labels_for"):
        RollingMean("phone", seconds=2).update(window)

    image = predict_image(model, make_image(tmp_path / "face.jpg"))
    assert image["start"] == image["end"] == 0.0 and set(image["outputs"]) == set(SUPERVISED)

    # Every requested label masked on video: windows still come, with no outputs.
    phone = load(tmp_path / "e1.pt", device="cpu", labels=["phone"])
    assert [w["outputs"] for w in predict_video(phone, video)] == [{}]


def test_onnx_graphs_read_the_masked_labels_from_the_export_contract(tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from bulwark.export.onnx import export_edge

    checkpoint = edge_checkpoint(tmp_path / "e1.pt", supervised_labels=SUPERVISED, masked_labels=MASKED)
    graph = load(export_edge(checkpoint, tmp_path / "e1.onnx", frames=2))  # the record travels in the contract's checkpoint
    assert graph.masked_labels == {"attention_direction": ("image", "video"), "phone": ("video",)}
    expected = load(checkpoint, device="cpu").predict(clips(2))
    outputs = graph.predict(clips(2))
    assert set(outputs) == {"eyes_closed", "yawn"}
    for name in outputs:
        np.testing.assert_allclose(outputs[name], expected[name], atol=1e-4)
    with pytest.raises(ValueError, match="--frames 1"):
        predict_image(graph, make_image(tmp_path / "face.jpg"))


def test_onnx_graphs_exported_with_a_manifest_return_the_checkpoints_labels(tmp_path: Path, make_image, make_video) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from bulwark.data.manifest import ClipRecord, write_manifest
    from bulwark.export.onnx import export_edge, read_contract

    # Every training video is alert, so the training job masked alertness on video and did
    # not supervise it. The manifest's label status still counts alertness as a target, but
    # the checkpoint's own record wins in the exported contract.
    records = [
        ClipRecord(clip_id=f"dmd/{i}", dataset="dmd", subject_id=f"dmd:{i}", media_path=f"{i}.mp4", media_type="video",
                   licence="t", commercial_ok=False, split="train", labels_gt={"alertness": "alert", "yawn": float(i % 2)})
        for i in range(4)
    ]
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    masked = {"alertness": ["video"], "phone": ["video"]}
    checkpoint = edge_checkpoint(tmp_path / "e1.pt", supervised_labels=["phone", "yawn"], masked_labels=masked)
    path = export_edge(checkpoint, tmp_path / "e1.onnx", frames=1, manifest=manifest)
    assert read_contract(path)["supervised_labels"] == ["phone", "yawn"]

    graph, model = load(path), load(checkpoint, device="cpu")
    assert graph.labels == model.labels == ("phone", "yawn") and graph.masked_labels == model.masked_labels
    image = make_image(tmp_path / "face.jpg")
    assert set(predict_image(graph, image)["outputs"]) == set(predict_image(model, image)["outputs"]) == {"phone", "yawn"}
    # A 1-frame graph cuts 1-frame windows from a video, which count as images.
    video = make_video(tmp_path / "v.avi", frames=60, fps=30.0)
    assert [set(w["outputs"]) for w in predict_video(graph, video)] == [{"phone", "yawn"}]
    assert [set(w["outputs"]) for w in predict_video(model, video)] == [{"yawn"}]
