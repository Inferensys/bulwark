"""Release checks, the terms and label status an exported graph carries, and the
predict / release / release-check commands."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import plain
from typer.testing import CliRunner

pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from bulwark.cli import app  # noqa: E402
from bulwark.data.manifest import ClipRecord, write_manifest  # noqa: E402
from bulwark.data.registry import MAINTAINERS_RELEASE_DECISION  # noqa: E402
from bulwark.data.taxonomy import default_taxonomy  # noqa: E402
from bulwark.eval.provenance import sha256_file  # noqa: E402
from bulwark.export.onnx import export_edge, read_contract  # noqa: E402
from bulwark.export.release import (  # noqa: E402
    RESEARCH_LICENCE,
    TrainingRecord,
    check_release,
    record_from_provenance,
    training_record,
)
from bulwark.inference import load  # noqa: E402
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint  # noqa: E402

runner = CliRunner()
CLEAN_CODE = {"commit": "0123456789ab" * 3, "dirty": False}


def summary(datasets: dict[str, bool]) -> dict:
    return {name: {"licence": "t", "commercial_ok": ok, "splits": {"train": 5}} for name, ok in datasets.items()}


def provenance(datasets: dict[str, bool], *, commercial_only: bool = False, code: dict | None = None, teacher: dict[str, bool] | None = None) -> dict:
    """A distillation run's provenance: its own datasets, and the teacher's (`teacher`): the
    B1 the teacher logits were cached with, recorded as distill records it."""
    b1 = {"sha256": "ab" * 32, "has_provenance": True, "datasets": summary(teacher or {"dmd": False})}
    return {
        "commercial_only": commercial_only,
        "datasets": summary(datasets),
        "manifest_sha256": None,
        "code": code or CLEAN_CODE,
        "inputs": {"teacher": {"has_provenance": True, "with_logits": True, "init_sha256": b1["sha256"], "init": b1, "datasets": b1["datasets"]}},
    }


def record(**kwargs: object) -> TrainingRecord:
    fields = {"source": "test", "commercial_only": False, "datasets": {"dmd": {}}, "commit": "0123456789ab", "dirty": False}
    return TrainingRecord(**{**fields, **kwargs})  # type: ignore[arg-type]


def test_release_check_needs_cleared_datasets_and_clean_code(monkeypatch: pytest.MonkeyPatch) -> None:
    from bulwark.data import registry

    check = check_release(record(datasets={"yawdd": {}}))
    assert not check.ok and any("yawdd (not determined)" in p for p in check.problems)
    assert check.summary()["datasets"]["yawdd"]["weights_release_ok"] is None
    assert check.summary()["datasets"]["yawdd"]["weights_release_basis"] is None

    # DMD is cleared by the maintainers' own decision, and the check carries that basis; it
    # stays research-only (commercial_ok is False).
    decided = check_release(record())
    assert decided.ok and decided.summary()["datasets"]["dmd"]["weights_release_ok"] is True
    assert decided.summary()["datasets"]["dmd"]["weights_release_basis"] == MAINTAINERS_RELEASE_DECISION
    commercial = check_release(record(commercial_only=True), commercial=True)
    assert commercial.problems == ["Datasets not licensed for commercial use: ['dmd']."]

    cleared = dataclasses.replace(registry.BY_NAME["synthetic_wan"], weights_release_ok=True)  # commercial_ok too
    monkeypatch.setitem(registry.BY_NAME, "synthetic_wan", cleared)
    assert check_release(record(datasets={"synthetic_wan": {}})).ok
    assert check_release(record(datasets={"synthetic_wan": {}}, commercial_only=True), commercial=True).ok
    # A commercial licence needs a commercial_only run; dirty code and unknown datasets never pass.
    assert not check_release(record(datasets={"synthetic_wan": {}}), commercial=True).ok
    assert any("uncommitted" in p for p in check_release(record(datasets={"synthetic_wan": {}}, dirty=True)).problems)
    assert any("not in the registry" in p for p in check_release(record(datasets={"nowhere": {}})).problems)
    assert any("no training datasets" in p for p in check_release(record(datasets={})).problems)
    # A dataset that lost commercial_ok in this run (a near-duplicate of a restricted copy) is not commercial.
    lost = record(datasets={"synthetic_wan": {"commercial_ok": False}}, commercial_only=True)
    assert any("not licensed for commercial use" in p for p in check_release(lost, commercial=True).problems)


def test_datasets_of_upstream_models_count_as_training_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    from bulwark.data import registry

    cleared = dataclasses.replace(registry.BY_NAME["synthetic_wan"], weights_release_ok=True)
    monkeypatch.setitem(registry.BY_NAME, "synthetic_wan", cleared)
    # E1 distilled on synthetic_wan clips only, from a teacher trained on yawdd.
    distilled = record_from_provenance(provenance({"synthetic_wan": True}, commercial_only=True, teacher={"yawdd": False}), "e1.pt")
    check = check_release(distilled)
    assert not check.ok and any("yawdd (not determined)" in p for p in check.problems)
    assert check.datasets["yawdd"]["via"] == ["teacher"] and check.datasets["synthetic_wan"]["via"] == ["training"]
    assert not check_release(distilled, commercial=True).ok  # the teacher's data is not commercially usable
    own_teacher = record_from_provenance(provenance({"synthetic_wan": True}, teacher={"synthetic_wan": True}), "e1.pt")
    assert check_release(own_teacher).ok and check_release(own_teacher).datasets["synthetic_wan"]["via"] == ["training", "teacher"]

    # An upstream model without a record of its data, and a dataset the registry lacks, block a release
    # and are still listed.
    unknown = provenance({"synthetic_wan": True}, teacher={"elsewhere": True})
    unknown["inputs"]["init"] = {"sha256": "cd" * 32, "has_provenance": False}
    check = check_release(record_from_provenance(unknown, "b1.pt"))
    assert check.summary()["datasets"]["elsewhere"] == {
        "licence": "t", "commercial_ok": True, "weights_release_ok": None, "weights_release_basis": None,
        "in_registry": False, "via": ["teacher"],
    }
    assert any("not in the registry: ['elsewhere']" in p for p in check.problems)
    assert any("init (records no provenance)" in p for p in check.problems)
    # Frozen-encoder features have no Bulwark training data behind them; logits cached with a B1 do.
    frozen = provenance({"synthetic_wan": True})
    frozen["inputs"] = {"features": {"has_provenance": True, "with_logits": False, "init": None, "datasets": None}}
    assert check_release(record_from_provenance(frozen, "heads.pt")).ok
    cached = provenance({"synthetic_wan": True})
    b1 = {"sha256": "ef" * 32, "has_provenance": True, "datasets": summary({"dmd": False})}
    cached["inputs"] = {"teacher": {"has_provenance": True, "with_logits": True, "init": b1, "datasets": b1["datasets"]}}
    assert check_release(record_from_provenance(cached, "e1.pt")).datasets["dmd"]["via"] == ["teacher"]
    cached["inputs"]["teacher"].update(init={**b1, "has_provenance": False}, datasets=None)
    assert any("teacher (cached with a checkpoint that records no provenance)" in p
               for p in check_release(record_from_provenance(cached, "e1.pt")).problems)


@pytest.fixture(scope="module")
def e1(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A random-weight E1 with provenance, a training manifest, and its graph with embeddings.
    Its teacher learned from Drive&Act and YawDD; YawDD is not cleared for a weights release."""
    root = tmp_path_factory.mktemp("release")
    torch.manual_seed(0)
    model = BulwarkEdge(default_taxonomy(), pretrained=False, img_size=64)
    checkpoint = save_checkpoint(
        root / "e1.pt", model,
        {"backbone": "efficientvit_b1.r224_in1k", "img_size": 64,
         "provenance": provenance({"dmd": False}, teacher={"drive_and_act": False, "yawdd": False})},
        full=True,
    )
    records = [
        ClipRecord(clip_id=f"dmd/{i}", dataset="dmd", subject_id=f"dmd:{i}", media_path=f"{i}.mp4", media_type="video",
                   licence="t", commercial_ok=False, split="train", labels_gt={"eyes_closed": float(i % 2)},
                   labels_teacher={"yawn": 0.0})
        for i in range(4)
    ]
    manifest = write_manifest(records, root / "manifest.parquet")
    graph = export_edge(
        checkpoint, root / "e1.onnx", frames=4, input_size=96, embeddings=True, manifest=manifest,
        temperature={"eyes_closed": 1.5},
    )
    return {"root": root, "checkpoint": checkpoint, "manifest": manifest, "graph": graph}


def test_graph_carries_terms_training_record_and_label_status(e1: dict[str, Path]) -> None:
    props = {p.key: p.value for p in onnx.load(str(e1["graph"]), load_external_data=False).metadata_props}
    assert props["license"] == RESEARCH_LICENCE and props["model_release"] == "v0"
    card = "https://huggingface.co/Inferensys/bulwark-e1-driver-monitoring-onnx"  # E1's own repository
    assert props["model_card"] == card and json.loads(props["commercial_only"]) is False
    assert card in onnx.load(str(e1["graph"]), load_external_data=False).doc_string
    assert json.loads(props["training_datasets"])["dmd"] == {
        "licence": "CC BY-NC-ND 4.0, academic use", "commercial_ok": False, "weights_release_ok": True,
        "weights_release_basis": MAINTAINERS_RELEASE_DECISION, "in_registry": True, "via": ["training"],
    }
    assert props["trained_with_commit"] == CLEAN_CODE["commit"] and props["teacher_init_sha256"] == "ab" * 32
    assert json.loads(props["release_cleared"]) is False  # the teacher learned from YawDD
    assert json.loads(props["training_datasets"])["drive_and_act"]["via"] == ["teacher"]  # the teacher's data
    assert json.loads(props["supervised_labels"]) == ["eyes_closed", "yawn"] and json.loads(props["ground_truth"]) == ["eyes_closed"]
    assert json.loads(props["teacher_only"]) == ["yawn"] and "kss" in json.loads(props["untrained"])
    terms = read_contract(e1["graph"])
    assert terms["label_status"]["from"]["matches_training_manifest"] is None  # the checkpoint recorded no hash
    assert terms["temperature"] == {"eyes_closed": 1.5} and terms["release_check"]["cleared"] is False
    assert terms["parity"]["max_abs_state_diff"] <= terms["parity"]["tolerance"]
    assert [o["name"] for o in terms["outputs"]][-2:] == ["tokens", "state"]


def test_graph_metadata_leaves_out_unrecorded_values_and_takes_labels_from_the_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = BulwarkEdge(default_taxonomy(), pretrained=False, img_size=64)
    extra = {"img_size": 64, "supervised_labels": ["eyes_closed"], "temperature": {"eyes_closed": 1.25}}  # no provenance
    checkpoint = save_checkpoint(tmp_path / "e1.pt", model, extra, full=True)
    graph = export_edge(checkpoint, tmp_path / "e1.onnx", frames=1, input_size=96)
    props = {p.key: p.value for p in onnx.load(str(graph), load_external_data=False).metadata_props}
    assert not {"commercial_only", "trained_with_commit", "teacher_init_sha256"} & set(props)
    assert json.loads(props["supervised_labels"]) == ["eyes_closed"] and json.loads(props["temperature"]) == {"eyes_closed": 1.25}
    assert read_contract(graph)["supervised_labels"] == ["eyes_closed"]
    onnx_model, torch_model = load(graph), load(checkpoint, device="cpu")
    assert onnx_model.labels == torch_model.labels and onnx_model.temperature == torch_model.temperature

    image = tmp_path / "face.jpg"
    import cv2

    cv2.imwrite(str(image), np.full((48, 64, 3), 90, dtype=np.uint8))
    for model_path in (checkpoint, graph):  # an image is one 1-frame clip, through PyTorch or a frames=1 graph
        out = tmp_path / f"{model_path.suffix[1:]}.json"
        result = runner.invoke(app, ["predict", str(model_path), str(image), "--out", str(out)])
        assert result.exit_code == 0, result.output
        windows = json.loads(out.read_text())["windows"]
        assert len(windows) == 1 and windows[0]["start"] == windows[0]["end"] == 0.0


def test_onnx_embeddings_match_pytorch(e1: dict[str, Path]) -> None:
    graph = load(e1["graph"])
    torch_model = load(e1["checkpoint"], device="cpu")
    frames = np.random.default_rng(1).integers(0, 256, (2, 4, 96, 96, 3), dtype=np.uint8)
    expected = torch_model.run(frames)
    assert graph.labels == ("eyes_closed", "yawn") and graph.labels_known and graph.temperature == {"eyes_closed": 1.5}
    np.testing.assert_allclose(graph.embed(frames, "state"), expected.state, atol=1e-4)
    np.testing.assert_allclose(graph.embed(frames, "pooled"), expected.tokens.mean(1), rtol=1e-4, atol=1e-3)


def test_export_refuses_a_licence_the_checkpoint_is_not_cleared_for(e1: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cleared for commercial release"):
        export_edge(e1["checkpoint"], tmp_path / "x.onnx", frames=4, input_size=96, licence="Apache-2.0")
    assert not (tmp_path / "x.onnx").exists()


def test_training_record_reads_checkpoints_graphs_and_releases(e1: dict[str, Path], tmp_path: Path) -> None:
    from_checkpoint = training_record(e1["checkpoint"])
    from_graph = training_record(e1["graph"])
    assert from_checkpoint.datasets.keys() == from_graph.datasets.keys() == {"dmd"}
    assert from_graph.commit == CLEAN_CODE["commit"] and from_graph.dirty is False

    result = runner.invoke(app, ["release", str(e1["checkpoint"]), str(tmp_path / "rel"), "--labels", "eyes_closed"])
    assert result.exit_code == 0, result.output
    assert "Not cleared for publication" in result.output
    config = json.loads((tmp_path / "rel" / "config.json").read_text())
    assert config["weights_licence"] == RESEARCH_LICENCE and config["release_check"]["cleared"] is False
    released = training_record(tmp_path / "rel")
    assert released.datasets.keys() == {"dmd"} and released.upstream["drive_and_act"]["via"] == ["teacher"]


def test_release_check_command_exits_1_and_prints_the_terms(e1: dict[str, Path]) -> None:
    result = runner.invoke(app, ["release-check", str(e1["graph"])])
    assert result.exit_code == 1 and "not cleared" in result.output and "CC-BY-NC-4.0" in result.output
    assert "drive_and_act" in result.output and "teacher" in result.output
    assert "yawdd (not determined)" in " ".join(result.output.split())
    assert "Traceback" not in result.output


def test_release_check_command_prints_the_basis_of_a_maintainers_decision(tmp_path: Path) -> None:
    """Weights trained only on datasets the maintainers cleared themselves pass, and the
    verdict says on what grounds: their decision, not a statement by the datasets' owners."""
    torch.manual_seed(0)
    model = BulwarkEdge(default_taxonomy(), pretrained=False, img_size=64)
    extra = {"img_size": 64, "provenance": provenance({"dmd": False}, teacher={"drive_and_act": False})}
    checkpoint = save_checkpoint(tmp_path / "e1.pt", model, extra, full=True)
    result = runner.invoke(app, ["release-check", str(checkpoint)])
    assert result.exit_code == 0, result.output
    printed = " ".join(result.output.split())  # the console wraps long lines
    assert f"dmd, drive_and_act: {MAINTAINERS_RELEASE_DECISION}" in printed
    assert printed.endswith("Cleared for release, on the grounds above.")


def test_bad_inputs_are_reported_without_a_traceback(e1: dict[str, Path], tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    unsafe = tmp_path / "unsafe.pt"
    torch.save({"model": {}, "where": Path("x")}, unsafe)  # not loadable with weights_only=True
    for command in (["release-check", str(unsafe)], ["release", str(unsafe), str(tmp_path / "rel")]):
        result = runner.invoke(app, command)
        assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.output  # no uncaught error

    image = make_image(tmp_path / "face.jpg")
    result = runner.invoke(app, ["predict", str(e1["checkpoint"]), str(image), "--adapter", "adapters/x.pt"])
    assert result.exit_code == 2 and "name=path" in result.output
    result = runner.invoke(app, ["predict", str(e1["graph"]), str(image)])  # a 4-frame graph
    assert result.exit_code == 1 and "--frames 1" in result.output


def test_release_and_predict_refuse_temperatures_of_other_weights(e1: dict[str, Path], tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "meta": {"checkpoint": {"path": "other.pt", "sha256": "cd" * 32}},
        "labels": {"eyes_closed": {"threshold": 0.5, "temperature": 2.0}},
    }))
    result = runner.invoke(app, ["release", str(e1["checkpoint"]), str(tmp_path / "rel"), "--labels", "eyes_closed", "--temperature", str(calibration)])
    assert result.exit_code == 1 and "fitted on weights" in plain(result.output) and not (tmp_path / "rel").exists()

    calibration.write_text(json.dumps({
        "meta": {"checkpoint": {"path": "e1.pt", "sha256": sha256_file(e1["checkpoint"])}},
        "labels": {"eyes_closed": {"threshold": 0.5, "temperature": 2.0}, "yawn": {"temperature": None}},
    }))
    video = make_video(tmp_path / "clip.mp4", frames=100)
    out = tmp_path / "timeline.json"
    result = runner.invoke(app, [
        "predict", str(e1["graph"]), str(video), "--out", str(out), "--temperature", str(calibration),
    ])
    assert result.exit_code == 0, result.output
    timeline = json.loads(out.read_text())
    assert timeline["kind"] == "e1-onnx" and timeline["temperature"] == {"eyes_closed": 2.0} and timeline["media"] == "video"
    assert timeline["labels"] == ["eyes_closed", "yawn"] and len(timeline["windows"]) == 2
    assert timeline["video"] == "clip.mp4"  # no local directories in the output
