"""OnnxModel reads `masked_labels` from each place `bulwark export` writes it, on its own.

`bulwark export` stores the checkpoint's record in three places: the top-level
metadata_props key `masked_labels`, `bulwark_contract["masked_labels"]` and the
contract's copy of the source checkpoint (`bulwark_contract["checkpoint"]`). A graph
that keeps any one of them (for example after a tool rewrote part of its metadata)
must still withhold a label on the media type it was masked on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bulwark.inference import load, predict_image
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint

EDGE = {"backbone": "test_efficientnet", "img_size": 64}
SUPERVISED = ["eyes_closed", "phone", "yawn"]
MASKED = {"phone": ["image"]}  # phone had one class on the image datasets


def one_frame_graph(tmp_path: Path) -> Path:
    from bulwark.export.onnx import export_edge

    torch.manual_seed(0)
    checkpoint = save_checkpoint(
        tmp_path / "e1.pt",
        BulwarkEdge(pretrained=False, **EDGE),
        {"step": 1, **EDGE, "supervised_labels": SUPERVISED, "masked_labels": MASKED},
    )
    return export_edge(checkpoint, tmp_path / "e1.onnx", frames=1)


def rewrite_metadata(source: Path, out: Path, edit: Callable[[dict[str, Any], dict[str, Any]], None]) -> Path:
    """Copy `source` to `out` with `edit(props, contract)` applied to its metadata."""
    import onnx

    graph = onnx.load(str(source))
    props = {p.key: p.value for p in graph.metadata_props}
    contract = json.loads(props.pop("bulwark_contract"))
    edit(props, contract)
    del graph.metadata_props[:]
    for key, value in {**props, **({"bulwark_contract": json.dumps(contract)} if contract else {})}.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(graph, str(out))
    return out


def only_top_level(props: dict[str, Any], contract: dict[str, Any]) -> None:
    contract.clear()  # a graph without the contract: single keys only


def only_contract(props: dict[str, Any], contract: dict[str, Any]) -> None:
    props.pop("masked_labels")
    contract["checkpoint"].pop("masked_labels")


def only_checkpoint_record(props: dict[str, Any], contract: dict[str, Any]) -> None:
    props.pop("masked_labels")
    contract.pop("masked_labels")


def nowhere(props: dict[str, Any], contract: dict[str, Any]) -> None:
    props.pop("masked_labels")
    contract.pop("masked_labels")
    contract["checkpoint"].pop("masked_labels")


@pytest.mark.parametrize("edit", [only_top_level, only_contract, only_checkpoint_record], ids=lambda f: f.__name__)
def test_each_metadata_source_alone_withholds_a_label_masked_on_images(
    tmp_path: Path, make_image, edit: Callable[[dict[str, Any], dict[str, Any]], None]  # type: ignore[no-untyped-def]
) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    path = rewrite_metadata(one_frame_graph(tmp_path), tmp_path / "edited.onnx", edit)

    graph = load(path)
    assert graph.labels == tuple(SUPERVISED) and graph.masked_labels == {"phone": ("image",)}
    assert graph.labels_for("image") == ("eyes_closed", "yawn")
    # A 1-frame clip is an image: the head learned nothing there, so its output is withheld.
    frame = np.random.default_rng(0).integers(0, 255, (2, 1, 40, 40, 3), dtype=np.uint8)
    assert set(graph.predict(frame)) == {"eyes_closed", "yawn"}
    assert set(predict_image(graph, make_image(tmp_path / "face.jpg"))["outputs"]) == {"eyes_closed", "yawn"}
    # Naming the label does not bring it back on images.
    assert set(load(path, labels=["phone", "yawn"]).predict(frame)) == {"yawn"}


def test_a_graph_that_records_no_masked_labels_returns_every_supervised_label(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    graph = load(rewrite_metadata(one_frame_graph(tmp_path), tmp_path / "plain.onnx", nowhere))
    frame = np.random.default_rng(0).integers(0, 255, (2, 1, 40, 40, 3), dtype=np.uint8)
    assert graph.masked_labels == {} and set(graph.predict(frame)) == set(SUPERVISED)


def test_a_malformed_masked_labels_entry_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    def audio(props: dict[str, Any], contract: dict[str, Any]) -> None:
        only_top_level(props, contract)
        props["masked_labels"] = json.dumps({"phone": ["audio"]})

    with pytest.raises(ValueError, match="unknown media types"):
        load(rewrite_metadata(one_frame_graph(tmp_path), tmp_path / "bad.onnx", audio))
