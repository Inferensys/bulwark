"""bulwark.inference with a stand-in encoder: no pretrained weights or real checkpoints needed."""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from bulwark.adapters import fit_adapter
from bulwark.data.taxonomy import default_taxonomy
from bulwark.inference import (
    FRAME_SIZE,
    ModelOutput,
    OnnxModel,
    RollingMean,
    as_clip_batch,
    load,
    outputs_to_dicts,
    predict_video,
    read_calibration,
    read_clip,
    read_temperatures,
    save_release,
    video_windows,
)
from bulwark.model.backbone import BackboneConfig
from bulwark.model.bulwark import BulwarkBase, BulwarkEdge, save_checkpoint
from bulwark.model.heads import BulwarkHeads, activate
from bulwark.model.lora import LoRALinear

DIM = 32
EDGE = {"backbone": "test_efficientnet", "img_size": 64}
RESEARCH = "CC BY-NC 4.0 (research use only)"


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(DIM, 3 * DIM)
        self.attn.proj = nn.Linear(DIM, DIM)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(DIM, 2 * DIM)
        self.mlp.fc2 = nn.Linear(2 * DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.proj(self.attn.qkv(x)[..., :DIM])
        return x + self.mlp.fc2(nn.functional.gelu(self.mlp.fc1(x)))


class FakeEncoder(nn.Module):
    """Stands in for VJepaEncoder: same interface, a 4x4 token grid, tubelet of 2 frames.

    `pretrained=True` always gives the same weights, as loading the V-JEPA file would.
    """

    def __init__(self, config: BackboneConfig | None = None, *, pretrained: bool = True) -> None:
        super().__init__()
        self.config = config or BackboneConfig()
        self.embed_dim = DIM
        with torch.random.fork_rng(devices=[]):
            if pretrained:
                torch.manual_seed(1234)
            self.encoder = nn.Module()
            self.encoder.patch = nn.Linear(3, DIM)
            self.encoder.blocks = nn.ModuleList([_Block() for _ in range(3)])
            # V-JEPA 2.1 keeps one output norm per hierarchical level; the encoder output uses the last.
            self.encoder.norms_block = nn.ModuleList([nn.LayerNorm(DIM) for _ in range(2)])

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time = frames.shape[:2]
        x = frames.float().div(255).flatten(0, 1).permute(0, 3, 1, 2)
        x = nn.functional.adaptive_avg_pool2d(x, 4).permute(0, 2, 3, 1).reshape(batch, time, 4, 4, 3)
        if time > 1:
            x = x.reshape(batch, time // 2, 2, 4, 4, 3).mean(2)
        steps = x.shape[1]
        tokens = self.encoder.patch(x).reshape(batch, -1, DIM)
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norms_block[-1](tokens)
        return tokens.reshape(batch, steps, 4, 4, DIM)


@pytest.fixture
def fake_vjepa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bulwark.model.bulwark.VJepaEncoder", FakeEncoder)


def clips(batch: int = 2, frames: int = 4, size: int = 40, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (batch, frames, size, size, 3), dtype=np.uint8)


def reference(model: nn.Module, frames: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(frames))
    return {k: v.numpy() for k, v in activate(logits, model.taxonomy).items()}


def assert_outputs_close(a: dict[str, np.ndarray], b: dict[str, np.ndarray], atol: float = 1e-5) -> None:
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_allclose(a[name], b[name], atol=atol, err_msg=name)


def b1_checkpoint(path: Path) -> BulwarkBase:
    """A B1 LoRA checkpoint whose adapters, final encoder norm and heads are not at their initial values."""
    torch.manual_seed(0)
    model = BulwarkBase(lora_blocks=2, lora_rank=4)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.normal_(module.up.weight, std=0.2)
    nn.init.normal_(model.encoder.encoder.norms_block[-1].weight, mean=1.0, std=0.3)
    model.heads.fit_input_stats(torch.randn(8, 25, DIM) * 2 + 1)
    save_checkpoint(path, model, {"step": 3, "lora_blocks": 2, "lora_rank": 4})
    return model


def test_b1_checkpoint_merges_lora_and_matches_the_training_model(tmp_path: Path, fake_vjepa: None) -> None:
    trained = b1_checkpoint(tmp_path / "b1.pt")
    model = load(tmp_path / "b1.pt", device="cpu")
    assert model.kind == "b1" and model.input_size == 256
    assert not any(isinstance(m, LoRALinear) for m in model.module.modules())
    assert model.metadata["step"] == 3 and model.metadata["path"].endswith("b1.pt")

    frames = clips()
    result = model.run(frames)
    batch = as_clip_batch(frames, 256)
    assert_outputs_close(result.outputs, reference(trained, batch))
    with torch.no_grad():
        tokens = trained.tokens(torch.from_numpy(batch))
        state = trained.heads.embed(tokens)
    # Merged LoRA weights round differently from base + low-rank update.
    np.testing.assert_allclose(result.tokens, tokens.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(result.state, state.numpy(), rtol=1e-4, atol=1e-5)
    assert model.embed(frames).shape == (2, 25, DIM)
    assert model.embed(frames, "pooled").shape == (2, DIM)
    assert model.embed(frames, "state").shape == (2, trained.heads.width)
    assert model.predict(frames[0])["activity"].shape == (1, len(default_taxonomy()["activity"].classes))


def test_heads_checkpoint_runs_on_the_frozen_encoder_with_recorded_labels(tmp_path: Path, fake_vjepa: None) -> None:
    heads = BulwarkHeads(DIM, input_tokens=25)
    payload = {"model": heads.state_dict(), "supervised_labels": ["yawn", "activity"], "temperature": {"yawn": 2.0, "activity": 0.5}}
    torch.save(payload, tmp_path / "heads.pt")
    model = load(tmp_path / "heads.pt", device="cpu")
    assert model.kind == "b1-frozen" and model.labels == ("yawn", "activity")

    frozen = BulwarkBase()
    frozen.heads.load_state_dict(heads.state_dict())
    frozen.eval()
    batch = as_clip_batch(clips(frames=1), 256)
    with torch.no_grad():
        logits = frozen(torch.from_numpy(batch))
    outputs = model.predict(batch)
    assert set(outputs) == {"yawn", "activity"}
    np.testing.assert_allclose(outputs["yawn"], torch.sigmoid(logits["yawn"] / 2.0).numpy(), atol=1e-5)
    np.testing.assert_allclose(outputs["activity"], torch.softmax(logits["activity"] / 0.5, -1).numpy(), atol=1e-5)
    assert set(load(tmp_path / "heads.pt", device="cpu", labels=["phone"]).predict(batch)) == {"phone"}


def test_load_rejects_incomplete_unknown_and_unsafe_checkpoints(tmp_path: Path, fake_vjepa: None) -> None:
    trained = b1_checkpoint(tmp_path / "b1.pt")
    no_heads = {k: v for k, v in trained.trainable_state().items() if not k.startswith("heads.")}
    torch.save({"model": no_heads}, tmp_path / "no_heads.pt")
    with pytest.raises(KeyError, match="missing"):
        load(tmp_path / "no_heads.pt", device="cpu")
    torch.save({"model": {"something.weight": torch.zeros(2)}}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a Bulwark"):
        load(tmp_path / "other.pt", device="cpu")
    torch.save({"model": {}, "config": Path("x")}, tmp_path / "pickled.pt")  # arbitrary objects
    with pytest.raises(pickle.UnpicklingError):
        load(tmp_path / "pickled.pt", device="cpu")
    with pytest.raises(KeyError, match="Unknown labels"):
        load(tmp_path / "b1.pt", device="cpu", labels=["not_a_label"])


def edge_checkpoint(path: Path) -> BulwarkEdge:
    torch.manual_seed(0)
    edge = BulwarkEdge(pretrained=False, **EDGE)
    save_checkpoint(path, edge, {"step": 2, **EDGE})
    return edge.eval()


def test_e1_checkpoint_reads_its_architecture_and_exposes_frame_tokens(tmp_path: Path) -> None:
    edge = edge_checkpoint(tmp_path / "e1.pt")
    model = load(tmp_path / "e1.pt", device="cpu")
    # E1 was distilled on shard frames (letterboxed to 256) and resizes them to its own
    # img_size inside forward, so the API letterboxes to 256 as well, not to img_size.
    assert model.kind == "e1" and model.architecture["img_size"] == 64 and model.input_size == FRAME_SIZE
    frames = clips(size=80)
    result = model.run(frames)
    assert_outputs_close(result.outputs, reference(edge, as_clip_batch(frames, FRAME_SIZE)))
    assert result.tokens.shape[:2] == (2, 4) and result.state.shape == (2, edge.heads.width)


class _Graph(nn.Module):
    """E1 as an ONNX graph: logits per label, optionally the state embedding as `state`."""

    def __init__(self, edge: BulwarkEdge, state: bool) -> None:
        super().__init__()
        self.edge, self.state = edge, state

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        captured: list[torch.Tensor] = []
        handle = self.edge.heads.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
        outputs = self.edge(frames)
        handle.remove()
        logits = tuple(outputs[name] for name in self.edge.heads.label_names)
        return (*logits, self.edge.heads.embed(captured[0])) if self.state else logits


def export(edge: BulwarkEdge, path: Path, *, state: bool, frames: int = 4) -> Path:
    names = [*edge.heads.label_names, *(["state"] if state else [])]
    torch.onnx.export(
        _Graph(edge, state), (torch.zeros(1, frames, FRAME_SIZE, FRAME_SIZE, 3, dtype=torch.uint8),), str(path),
        input_names=["frames"], output_names=names, dynamic_axes={"frames": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    return path


def test_onnx_model_matches_pytorch_e1_and_reads_optional_outputs(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    edge_checkpoint(tmp_path / "e1.pt")
    torch_model = load(tmp_path / "e1.pt", device="cpu")
    frames = clips(size=80)
    expected = torch_model.run(frames)

    plain = load(export(torch_model.module, tmp_path / "plain.onnx", state=False))
    assert isinstance(plain, OnnxModel) and plain.clip_frames == 4 and plain.input_size == FRAME_SIZE
    assert_outputs_close(plain.predict(frames), expected.outputs, atol=1e-4)
    with pytest.raises(NotImplementedError, match="state"):
        plain.embed(frames, "state")
    with pytest.raises(ValueError, match="exactly 4 frames"):
        plain.predict(clips(frames=2))

    path = export(torch_model.module, tmp_path / "state.onnx", state=True)
    graph = onnx.load(str(path))
    entry = graph.metadata_props.add()
    entry.key, entry.value = "supervised_labels", json.dumps(["yawn", "phone"])
    onnx.save(graph, str(path))
    with_state = load(path)
    assert with_state.labels == ("yawn", "phone")
    np.testing.assert_allclose(with_state.embed(frames, "state"), expected.state, atol=1e-4)
    with pytest.raises(NotImplementedError, match="tokens"):
        with_state.embed(frames, "tokens")


def test_onnx_model_reads_labels_temperatures_and_checkpoint_from_the_export_contract(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    edge = edge_checkpoint(tmp_path / "e1.pt")
    path = export(edge, tmp_path / "e1.onnx", state=False)
    graph = onnx.load(str(path))
    contract = {"supervised_labels": ["yawn", "activity"], "temperature": {"yawn": 2.0}, "checkpoint": {"sha256": "ab" * 32}}
    entry = graph.metadata_props.add()
    entry.key, entry.value = "bulwark_contract", json.dumps(contract)
    onnx.save(graph, str(path))

    model = load(path)
    assert model.labels == ("yawn", "activity") and model.labels_known and model.temperature == {"yawn": 2.0}
    assert model.embedding_source() == {"family": "e1", "weights": "ab" * 32}
    assert model.metadata["contract"]["checkpoint"]["sha256"] == "ab" * 32
    frames = clips(size=80)
    plain = load(tmp_path / "e1.pt", device="cpu", labels=["yawn"]).predict(frames)["yawn"]
    logit = np.log(plain) - np.log1p(-plain)
    np.testing.assert_allclose(model.predict(frames)["yawn"], 1 / (1 + np.exp(-logit / 2.0)), atol=1e-4)
    assert load(path, temperature={}).temperature == {}  # an explicit value replaces the recorded one


def test_onnx_model_refuses_a_contract_of_another_taxonomy(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from bulwark.export.onnx import contract

    edge = edge_checkpoint(tmp_path / "e1.pt")
    path = export(edge, tmp_path / "e1.onnx", state=False)
    taxonomy = default_taxonomy()
    categorical = next(name for name, spec in taxonomy.labels.items() if spec.kind == "categorical")
    regression = next(name for name, spec in taxonomy.labels.items() if spec.kind == "regression")

    def with_contract(name: str, edit) -> Path:  # type: ignore[no-untyped-def]
        terms = contract(edge, 4, FRAME_SIZE)
        edit(terms, {entry["name"]: entry for entry in terms["outputs"]})
        graph = onnx.load(str(path))
        entry = graph.metadata_props.add()
        entry.key, entry.value = "bulwark_contract", json.dumps(terms)
        onnx.save(graph, str(tmp_path / name))
        return tmp_path / name

    assert load(with_contract("same.onnx", lambda terms, outputs: None)).labels == tuple(taxonomy.labels)
    # Same names and widths, but classes in another order: the probabilities would get the wrong names.
    reordered = with_contract("classes.onnx", lambda terms, outputs: outputs[categorical]["classes"].reverse())
    with pytest.raises(ValueError, match=rf"built for taxonomy v{taxonomy.version}, .*differ: \['{categorical}'\]"):
        load(reordered)
    ranged = with_contract("range.onnx", lambda terms, outputs: outputs[regression].update(range=[0.0, 2.0]))
    with pytest.raises(ValueError, match=rf"differ: \['{regression}'\]"):
        load(ranged)
    older = with_contract("version.onnx", lambda terms, outputs: terms.update(taxonomy_version=taxonomy.version - 1))
    with pytest.raises(ValueError, match=rf"built for taxonomy v{taxonomy.version - 1}, this code has v{taxonomy.version}"):
        load(older)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_file(path: Path, labels: dict, *, model: str = "e1", **checkpoint: str | None) -> Path:
    """calibration.json in the layout `bulwark evaluate` writes (bulwark.eval.report.calibration_file)."""
    meta = {"model": model, "checkpoint": {"path": "e1.pt", "source": "file", **checkpoint}, "fitted_on": "val"}
    path.write_text(json.dumps({"meta": meta, "labels": labels}))
    return path


CALIBRATION = {
    "yawn": {"threshold": 0.4, "threshold_from": "val", "temperature": 2.0, "source": "val", "n_val": 40},
    "activity": {"threshold": 0.5, "threshold_from": "default", "temperature": 0.5, "source": "val", "n_val": 40},
    "phone": {"threshold": 0.5, "threshold_from": "default", "temperature": None, "source": "default"},
}


def test_temperatures_and_thresholds_from_a_calibration_file(tmp_path: Path) -> None:
    edge_checkpoint(tmp_path / "e1.pt")
    path = calibration_file(tmp_path / "calibration.json", CALIBRATION, sha256=sha256(tmp_path / "e1.pt"))
    plain = load(tmp_path / "e1.pt", device="cpu")
    model = load(tmp_path / "e1.pt", device="cpu", temperature=path)
    assert model.temperature == {"yawn": 2.0, "activity": 0.5} and model.raw_thresholds == {"yawn": 0.4}
    frames = clips(size=80)
    with torch.no_grad():
        logits = plain.module(torch.from_numpy(as_clip_batch(frames, FRAME_SIZE)))
    outputs = model.predict(frames)
    np.testing.assert_allclose(outputs["yawn"], torch.sigmoid(logits["yawn"] / 2.0).numpy(), atol=1e-5)
    np.testing.assert_allclose(outputs["activity"], torch.softmax(logits["activity"] / 0.5, -1).numpy(), atol=1e-5)
    np.testing.assert_allclose(outputs["phone"], plain.predict(frames)["phone"], atol=1e-6)

    # The threshold was tuned on sigmoid(logit); on the calibrated output it flags the same clips.
    threshold = model.thresholds["yawn"]
    assert threshold == pytest.approx(1 / (1 + np.exp(-np.log(0.4 / 0.6) / 2.0)))
    raw = torch.sigmoid(logits["yawn"]).numpy()
    np.testing.assert_array_equal(outputs["yawn"] >= threshold, raw >= 0.4)
    assert load(tmp_path / "e1.pt", device="cpu", temperature=path, labels=["phone"]).thresholds == {}

    with pytest.raises(ValueError, match="positive"):
        load(tmp_path / "e1.pt", device="cpu", temperature={"yawn": 0.0})
    for value in (1.5, -0.1, float("nan")):
        bad = calibration_file(tmp_path / "bad.json", {"yawn": {**CALIBRATION["yawn"], "threshold": value}}, sha256=sha256(tmp_path / "e1.pt"))
        with pytest.raises(ValueError, match="between 0 and 1"):
            load(tmp_path / "e1.pt", device="cpu", temperature=bad)

    # `bulwark evaluate` picks a threshold among val probabilities, which saturate to exactly 0 or 1
    # on a confident head; those load and are not moved by the temperature.
    saturated = {
        "yawn": {**CALIBRATION["yawn"], "threshold": 1.0},
        "eyes_closed": {**CALIBRATION["yawn"], "threshold": 0.0},
    }
    edge = load(tmp_path / "e1.pt", device="cpu", temperature=calibration_file(tmp_path / "sat.json", saturated, sha256=sha256(tmp_path / "e1.pt")))
    assert edge.raw_thresholds == {"yawn": 1.0, "eyes_closed": 0.0} and edge.thresholds == {"yawn": 1.0, "eyes_closed": 0.0}


def test_a_calibration_file_must_belong_to_the_weights(tmp_path: Path) -> None:
    edge_checkpoint(tmp_path / "e1.pt")
    other = calibration_file(tmp_path / "other.json", CALIBRATION, sha256="ab" * 32)
    with pytest.raises(ValueError, match="fitted on weights with sha256 abababababab"):
        load(tmp_path / "e1.pt", device="cpu", temperature=other)
    # The same temperatures as a mapping are the caller's explicit choice and are not checked.
    assert load(tmp_path / "e1.pt", device="cpu", temperature={"yawn": 2.0}).temperature == {"yawn": 2.0}

    unrecorded = calibration_file(tmp_path / "unrecorded.json", CALIBRATION, sha256=None)
    with pytest.warns(UserWarning, match="it records no checkpoint"):
        model = load(tmp_path / "e1.pt", device="cpu", temperature=unrecorded)
    assert model.temperature == {"yawn": 2.0, "activity": 0.5}

    # Files written before calibration.json had a "meta" section map labels directly.
    (tmp_path / "flat.json").write_text(json.dumps(CALIBRATION))
    with pytest.warns(UserWarning, match="it records no checkpoint"):
        model = load(tmp_path / "e1.pt", device="cpu", temperature=tmp_path / "flat.json")
    assert model.temperature == {"yawn": 2.0, "activity": 0.5} and model.raw_thresholds == {"yawn": 0.4}


def test_onnx_calibration_matches_the_exported_checkpoint_or_the_graph(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    edge = edge_checkpoint(tmp_path / "e1.pt")
    path = export(edge, tmp_path / "e1.onnx", state=False)
    graph = onnx.load(str(path))
    entry = graph.metadata_props.add()
    # `bulwark export` copies the checkpoint's own record (step, provenance, ...) under "checkpoint".
    record = {"sha256": sha256(tmp_path / "e1.pt"), "supervised_labels": ["yawn", "phone"], "thresholds": {"yawn": 0.3}}
    entry.key, entry.value = "bulwark_contract", json.dumps({"checkpoint": record})
    onnx.save(graph, str(path))
    assert load(path).labels == ("yawn", "phone") and load(path).raw_thresholds == {"yawn": 0.3}

    # calibration.json of `bulwark evaluate --model onnx`: sha256 of the graph and of its source checkpoint.
    fitted = calibration_file(
        tmp_path / "onnx.json", CALIBRATION, model="onnx", sha256=sha256(path), exported_from_sha256=sha256(tmp_path / "e1.pt")
    )
    assert load(path, temperature=fitted).temperature == {"yawn": 2.0, "activity": 0.5}
    assert load(tmp_path / "e1.pt", device="cpu", temperature=fitted).raw_thresholds == {"yawn": 0.4}
    graph_only = calibration_file(tmp_path / "graph.json", CALIBRATION, model="onnx", sha256=sha256(path))
    assert load(path, temperature=graph_only).temperature == {"yawn": 2.0, "activity": 0.5}
    with pytest.raises(ValueError, match="fitted on weights"):
        load(path, temperature=calibration_file(tmp_path / "x.json", CALIBRATION, model="onnx", sha256="cd" * 32))


class NoPretrainedEncoder(FakeEncoder):
    """FakeEncoder without the pretrained file: asking for pretrained weights fails, as VJepaEncoder does."""

    def __init__(self, config: BackboneConfig | None = None, *, pretrained: bool = True) -> None:
        if pretrained:
            raise FileNotFoundError("pretrained V-JEPA weights are not available")
        super().__init__(config, pretrained=False)


@pytest.fixture
def vjepa_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    notice = tmp_path / "LICENSE-upstream"
    notice.write_text("MIT License\nCopyright (c) Meta Platforms, Inc. and affiliates.\n")
    monkeypatch.setattr("bulwark.inference.VJEPA_LICENSE", notice)
    return notice


def test_release_round_trip_needs_no_pretrained_weights(
    tmp_path: Path, fake_vjepa: None, vjepa_notice: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b1_checkpoint(tmp_path / "b1.pt")
    model = load(tmp_path / "b1.pt", device="cpu", labels=["yawn", "activity"])
    release = save_release(model, tmp_path / "release", weights_licence=RESEARCH, extra={"note": "unit test"})
    config = json.loads((release / "config.json").read_text())
    assert config["kind"] == "b1" and config["weights_licence"] == RESEARCH and config["note"] == "unit test"
    assert config["labels"] == ["yawn", "activity"] and config["input"]["size"] == FRAME_SIZE
    assert "checkpoint" not in config["architecture"]["backbone"]
    assert config["source"]["step"] == 3 and (release / "model.safetensors").exists()
    assert (release / "LICENSE-vjepa2").read_text() == vjepa_notice.read_text()

    # From here on, building the encoder with pretrained weights raises.
    monkeypatch.setattr("bulwark.model.bulwark.VJepaEncoder", NoPretrainedEncoder)
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "b1.pt", device="cpu")
    reloaded = load(release, device="cpu")
    assert reloaded.labels == ("yawn", "activity") and reloaded.checkpoint_sha256 == model.checkpoint_sha256
    frames = clips()
    assert_outputs_close(reloaded.predict(frames), model.predict(frames), atol=1e-6)

    edge_checkpoint(tmp_path / "e1.pt")
    edge = load(tmp_path / "e1.pt", device="cpu", labels=["yawn"])
    edge_release = save_release(edge, tmp_path / "e1", weights_licence=RESEARCH)
    assert not (edge_release / "LICENSE-vjepa2").exists()
    assert_outputs_close(load(edge_release, device="cpu").predict(frames), edge.predict(frames), atol=1e-6)


def test_release_config_has_provenance_but_no_local_paths(tmp_path: Path) -> None:
    provenance = {
        "job": "distill",
        "config": {"manifest": str(tmp_path / "data" / "manifest.parquet"), "out": str(tmp_path / "runs")},
        "commercial_only": False,
        "datasets": {"drive_and_act": {"licence": "research only", "commercial_ok": False, "splits": {"train": 10}}},
        "manifest_sha256": "ab" * 32,
        "inputs": {
            "shards": str(tmp_path / "shards"),
            # The teacher's features dir, cached with a B1 trained on DMD (bulwark.train.provenance.features_identity).
            "teacher": {
                "path": str(tmp_path / "features"), "has_provenance": True, "with_logits": True, "init_sha256": "cd" * 32,
                "datasets": {"dmd": {"licence": "CC BY-NC-ND 4.0", "commercial_ok": False, "splits": {"train": 5}}},
            },
        },
        "code": {"commit": "0123abc", "dirty": False, "changed": []},
    }
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "ckpt" / "final.pt", BulwarkEdge(pretrained=False, **EDGE), {"step": 7, **EDGE, "provenance": provenance})
    model = load(tmp_path / "ckpt" / "final.pt", device="cpu", labels=["phone"])
    release = save_release(model, tmp_path / "release", weights_licence=RESEARCH)
    text = (release / "config.json").read_text()
    assert str(tmp_path) not in text and "/private/" not in text
    config = json.loads(text)
    assert config["commercial_only"] is False
    assert config["source"] == {
        "checkpoint": {"file": "final.pt", "sha256": hashlib.sha256((tmp_path / "ckpt" / "final.pt").read_bytes()).hexdigest()},
        "step": 7,
        "job": "distill",
        "commercial_only": False,
        "datasets": provenance["datasets"],
        "manifest_sha256": "ab" * 32,
        "code_commit": "0123abc",
        "code_dirty": False,
        # The teacher's training data: check_weights_licence counts it here and in re-releases.
        "upstream_datasets": {"dmd": {"via": ["teacher"], "licence": "CC BY-NC-ND 4.0", "commercial_ok": False}},
        "unrecorded_inputs": [],
    }
    # Re-releasing a release keeps the original checkpoint record.
    again = save_release(load(release, device="cpu"), tmp_path / "again", weights_licence=RESEARCH)
    assert json.loads((again / "config.json").read_text())["source"] == config["source"]


def test_release_requires_known_labels(tmp_path: Path) -> None:
    edge_checkpoint(tmp_path / "e1.pt")
    model = load(tmp_path / "e1.pt", device="cpu")
    assert not model.labels_known and len(model.labels) == len(default_taxonomy().labels)
    with pytest.raises(ValueError, match="no supervised labels"):
        save_release(model, tmp_path / "release", weights_licence=RESEARCH)
    assert not (tmp_path / "release").exists()
    release = save_release(model, tmp_path / "release", weights_licence=RESEARCH, labels=["phone", "yawn"])
    config = json.loads((release / "config.json").read_text())
    assert config["labels"] == ["phone", "yawn"] and config["commercial_only"] is None


def provenance_checkpoint(path: Path, *, commercial_only: bool | None, datasets: dict[str, bool]) -> Path:
    """An E1 checkpoint whose provenance records `commercial_only` and datasets -> commercial_ok."""
    provenance = {
        "job": "distill",
        "commercial_only": commercial_only,
        "datasets": {name: {"licence": "test", "commercial_ok": ok, "splits": {"train": 4}} for name, ok in datasets.items()},
    }
    torch.manual_seed(0)
    return save_checkpoint(path, BulwarkEdge(pretrained=False, **EDGE), {"step": 1, **EDGE, "provenance": provenance})


@pytest.mark.parametrize(
    ("commercial_only", "datasets", "reason"),
    [
        (False, {"dmd": True}, "commercial_only=false"),
        (None, {"dmd": True}, "records no commercial_only"),
        (True, {"dmd": True, "drive_and_act": False}, "not commercially usable: drive_and_act"),
    ],
)
def test_release_licence_must_be_non_commercial_unless_the_data_is_clean(
    tmp_path: Path, commercial_only: bool | None, datasets: dict[str, bool], reason: str
) -> None:
    path = provenance_checkpoint(tmp_path / "e1.pt", commercial_only=commercial_only, datasets=datasets)
    model = load(path, device="cpu", labels=["yawn"])
    with pytest.raises(ValueError, match=reason):
        save_release(model, tmp_path / "release", weights_licence="Apache-2.0")
    assert not (tmp_path / "release").exists()
    for licence in ("CC BY-NC 4.0", "CC-BY-NC-SA-4.0", "Non-commercial research licence", "research use only", RESEARCH):
        save_release(model, tmp_path / licence.replace(" ", "_"), weights_licence=licence)


@pytest.mark.parametrize(
    "licence",
    [
        "Apache-2.0 (research use and commercial use permitted)",
        "MIT; commercial and non-commercial use",
        "Research-use licence; commercial licences available",
        "MIT (commercial and research use permitted)",
        "Apache-2.0; free for research purposes and commercial use",
        "MIT (research use only)",
        "CC BY 4.0 (research use only)",
        "research use",
        "Bulwark research licence",
    ],
)
def test_release_licence_must_not_also_allow_commercial_use(tmp_path: Path, licence: str) -> None:
    path = provenance_checkpoint(tmp_path / "e1.pt", commercial_only=False, datasets={"dmd": False})
    model = load(path, device="cpu", labels=["yawn"])
    with pytest.raises(ValueError, match="does not restrict use to non-commercial research"):
        save_release(model, tmp_path / "release", weights_licence=licence)
    assert not (tmp_path / "release").exists()


def test_release_of_clean_weights_takes_any_licence_and_keeps_its_calibration(tmp_path: Path) -> None:
    path = provenance_checkpoint(tmp_path / "e1.pt", commercial_only=True, datasets={"dmd": True})
    calibration = calibration_file(tmp_path / "calibration.json", CALIBRATION, sha256=sha256(path))
    model = load(path, device="cpu", labels=["yawn", "phone"], temperature=calibration)
    with pytest.raises(ValueError, match="required"):
        save_release(model, tmp_path / "release", weights_licence=" ")
    with pytest.raises(ValueError, match=r"cannot replace the release fields \['commercial_only', 'weights_licence'\]"):
        save_release(model, tmp_path / "release", weights_licence=RESEARCH, extra={"commercial_only": True, "weights_licence": "MIT"})
    assert not (tmp_path / "release").exists()

    release = save_release(model, tmp_path / "release", weights_licence="Apache-2.0")
    config = json.loads((release / "config.json").read_text())
    assert config["commercial_only"] is True and config["thresholds"] == {"yawn": 0.4}
    reloaded = load(release, device="cpu")
    assert reloaded.temperature == model.temperature and reloaded.thresholds == pytest.approx(model.thresholds)
    # The release's calibration was fitted on the checkpoint it was built from, so the file still applies.
    assert load(release, device="cpu", temperature=calibration).raw_thresholds == {"yawn": 0.4}
    # A mapping replaces the temperatures only; the recorded thresholds stay on the raw logit.
    retuned = load(release, device="cpu", temperature={"yawn": 1.0})
    assert retuned.raw_thresholds == {"yawn": 0.4} and retuned.thresholds == pytest.approx({"yawn": 0.4})


def test_b1_checkpoint_needs_its_final_norm_and_matching_lora_records(tmp_path: Path, fake_vjepa: None) -> None:
    trained = b1_checkpoint(tmp_path / "b1.pt")
    payload = torch.load(tmp_path / "b1.pt", weights_only=True)
    state = payload["model"]
    assert {k for k in state if "norms_block" in k} == {"encoder.encoder.norms_block.1.weight", "encoder.encoder.norms_block.1.bias"}
    frames = clips()
    expected = reference(trained, as_clip_batch(frames, 256))

    # Older LoRA checkpoints also saved the untrained hierarchical norms; they still load.
    old = {**state, **{f"encoder.encoder.norms_block.0.{k}": v for k, v in nn.LayerNorm(DIM).state_dict().items()}}
    torch.save({**payload, "model": old}, tmp_path / "old.pt")
    assert_outputs_close(load(tmp_path / "old.pt", device="cpu").predict(frames), expected)

    no_norm = {k: v for k, v in state.items() if "norms_block" not in k}
    torch.save({**payload, "model": no_norm}, tmp_path / "no_norm.pt")
    with pytest.raises(KeyError, match="missing.*norms_block.1"):
        load(tmp_path / "no_norm.pt", device="cpu")
    torch.save({**payload, "lora_blocks": 4}, tmp_path / "blocks.pt")
    with pytest.raises(ValueError, match="lora_blocks=4, but holds LoRA weights for 2 blocks"):
        load(tmp_path / "blocks.pt", device="cpu")
    torch.save({**payload, "lora_rank": 16}, tmp_path / "rank.pt")
    with pytest.raises(ValueError, match="lora_rank=16, but its LoRA weights have rank 4"):
        load(tmp_path / "rank.pt", device="cpu")


def timeline(starts: list[float], values: list[float], seconds: float = 2.0) -> list[dict]:
    return [{"start": s, "end": s + seconds, "outputs": {"perclos": v}} for s, v in zip(starts, values, strict=True)]


def test_rolling_mean_gives_perclos_over_a_minute_of_consecutive_windows() -> None:
    values = [0.0] * 30 + [1.0] * 40  # windows every 1 s; the eyes close from the 30th
    perclos = RollingMean("perclos", seconds=60)
    rolled = [perclos.update(w) for w in timeline([float(s) for s in range(70)], values)]
    assert rolled[:58] == [None] * 58  # windows end at 2 ... 59 s: less than a minute so far
    assert rolled[58] == pytest.approx(np.mean(values[:59]))  # (58, 60): windows starting at 0 ... 58
    assert rolled[69] == pytest.approx(np.mean(values[11:70]))  # (69, 71): windows starting at 11 ... 69

    gap = RollingMean("perclos", seconds=4)
    windows = timeline([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.5])
    assert [gap.update(w) for w in windows] == [None, None, 1.0, None, None, pytest.approx(0.5 / 3)]
    sparse = RollingMean("perclos", seconds=4)  # hop longer than the window: no consecutive span
    assert [sparse.update(w) for w in timeline([0.0, 3.0, 6.0, 9.0], [1.0] * 4)] == [None] * 4
    # A span shorter than one window (or equal to it, with windows of 2.002 s at 29.97 fps) keeps the current window.
    short = RollingMean("perclos", seconds=1)
    assert [short.update(w) for w in timeline([0.0, 1.0, 2.0], [0.2, 0.4, 0.6])] == [0.2, 0.4, 0.6]
    ntsc = RollingMean("perclos", seconds=2)
    assert [ntsc.update(w) for w in timeline([0.0, 1.001, 2.002], [0.2, 0.4, 0.6], seconds=2.002)] == [0.2, 0.4, 0.6]
    with pytest.raises(TypeError, match="scalar"):
        RollingMean("activity").update({"start": 0.0, "end": 2.0, "outputs": {"activity": {"a": 1.0}}})
    with pytest.raises(ValueError):
        RollingMean(seconds=0)


def test_as_clip_batch_shapes_letterbox_and_dtype() -> None:
    image = np.full((30, 60, 3), 200, dtype=np.uint8)
    batch = as_clip_batch(image, 32)
    assert batch.shape == (1, 1, 32, 32, 3) and batch[0, 0, 0, 0].sum() == 0 and batch[0, 0, 16, 16, 0] == 200
    assert as_clip_batch(np.zeros((4, 32, 32, 3), np.uint8), 32).shape == (1, 4, 32, 32, 3)
    assert as_clip_batch(torch.zeros(2, 4, 32, 32, 3, dtype=torch.uint8), 32).shape == (2, 4, 32, 32, 3)
    with pytest.raises(TypeError):
        as_clip_batch(np.zeros((4, 32, 32, 3), np.float32), 32)
    with pytest.raises(ValueError):
        as_clip_batch(np.zeros((4, 32, 32), np.uint8), 32)


def gray(frame: np.ndarray) -> float:
    return float(frame[frame.shape[0] // 2, frame.shape[1] // 2].mean())


def test_video_windows_sample_like_training_clips(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    path = make_video(tmp_path / "v.avi", frames=90, fps=30.0)  # frame i is gray level i
    windows = list(video_windows(path, size=32))
    assert [(w.start, w.end) for w in windows] == [(0.0, 2.0), (1.0, 3.0)]  # 2 s windows, 1 s hop
    assert windows[1].frames.shape == (16, 32, 32, 3)
    levels = [gray(f) for f in windows[1].frames]
    assert np.allclose(levels, [30 + 3 * k for k in range(16)], atol=4)  # 60-frame span, stride 3

    clip = read_clip(path, start=1.0, size=32)
    assert np.allclose([gray(f) for f in clip], levels, atol=1)

    short = list(video_windows(make_video(tmp_path / "short.avi", frames=20, fps=30.0), size=32))
    assert len(short) == 1 and short[0].frames.shape[0] == 16
    assert np.allclose([gray(f) for f in short[0].frames[6:]], 18, atol=4)  # padded with the last sample


def test_video_windows_never_end_after_the_video(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    # 200 frames at 30 fps is 6.67 s: not a whole number of 1 s hops. The dataset adapters
    # cut a 2 s window only while start + span <= frames, so the last one is (4.0, 6.0).
    path = make_video(tmp_path / "v.avi", frames=200, fps=30.0)
    windows = list(video_windows(path, size=32))
    assert [(w.start, w.end) for w in windows] == [(float(s), s + 2.0) for s in range(5)]
    assert np.allclose([gray(f) for f in windows[-1].frames], [120 + 3 * k for k in range(16)], atol=4)

    # At 5 fps a 2 s window spans 10 frames, fewer than 16 samples at stride 1. Windows are
    # still cut while their span fits, and samples past the end repeat the last frame.
    low = make_video(tmp_path / "low.avi", frames=25, fps=5.0)
    windows = list(video_windows(low, size=32))
    assert [(w.start, w.end) for w in windows] == [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0)]
    assert np.allclose([gray(f) for f in windows[-1].frames], [min(15 + k, 24) for k in range(16)], atol=4)


def test_predict_video_timeline_with_an_adapter(tmp_path: Path, make_video, fake_vjepa: None) -> None:  # type: ignore[no-untyped-def]
    torch.save({"model": BulwarkHeads(DIM, input_tokens=25).state_dict()}, tmp_path / "heads.pt")
    model = load(tmp_path / "heads.pt", device="cpu", labels=["activity", "yawn"])
    rng = np.random.default_rng(0)
    adapter = fit_adapter(rng.normal(size=(12, 25, DIM)), ["a", "b"] * 6, steps=5)
    video = make_video(tmp_path / "v.avi", frames=120, fps=30.0)
    timeline = list(predict_video(model, video, batch_size=2, adapters={"custom": adapter}))
    assert [w["start"] for w in timeline] == [0.0, 1.0, 2.0]
    first = timeline[0]
    assert set(first["outputs"]) == {"activity", "yawn"}
    assert abs(sum(first["outputs"]["activity"].values()) - 1) < 1e-5 and 0 <= first["outputs"]["yawn"] <= 1
    assert set(first["adapters"]["custom"]) == {"a", "b"}
    json.dumps(timeline)


def test_predict_video_refuses_adapters_trained_on_other_embeddings(tmp_path: Path, make_video, fake_vjepa: None) -> None:  # type: ignore[no-untyped-def]
    torch.save({"model": BulwarkHeads(DIM, input_tokens=25).state_dict()}, tmp_path / "heads.pt")
    frozen = load(tmp_path / "heads.pt", device="cpu", labels=["yawn"])
    trained = b1_checkpoint(tmp_path / "b1.pt")
    del trained
    tuned = load(tmp_path / "b1.pt", device="cpu", labels=["yawn"])
    assert frozen.embedding_source("tokens") == {"family": "b1", "weights": "frozen"}
    assert frozen.embedding_source("state")["weights"] == frozen.checkpoint_sha256 != tuned.checkpoint_sha256
    assert tuned.embedding_source("pooled") == {"family": "b1", "weights": tuned.checkpoint_sha256}

    rng = np.random.default_rng(0)
    adapter = fit_adapter(rng.normal(size=(12, 25, DIM)), ["a", "b"] * 6, steps=5, model=frozen.embedding_source("tokens"))
    video = make_video(tmp_path / "v.avi", frames=60, fps=30.0)
    assert len(list(predict_video(frozen, video, adapters={"custom": adapter}))) == 1
    with pytest.raises(ValueError, match="custom.*weights='frozen'"):
        list(predict_video(tuned, video, adapters={"custom": adapter}))
    assert len(list(predict_video(tuned, video, adapters={"custom": adapter}, check_adapters=False))) == 1


def test_model_output_embeddings_and_dicts() -> None:
    tokens = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    result = ModelOutput({}, tokens, None)
    np.testing.assert_allclose(result.embedding("pooled"), tokens.mean(1))
    with pytest.raises(NotImplementedError):
        result.embedding("state")
    with pytest.raises(ValueError):
        result.embedding("pixels")
    taxonomy = default_taxonomy()
    rows = outputs_to_dicts({"yawn": np.array([0.25, 0.5]), "modality": np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])}, taxonomy)
    assert rows[0] == {"yawn": 0.25, "modality": {"rgb": pytest.approx(0.7), "nir": pytest.approx(0.2), "thermal": pytest.approx(0.1)}}


# ---- calibration per media type ------------------------------------------------------------


def fit(threshold: float, temperature: float | None, threshold_from: str = "val") -> dict:
    return {"threshold": threshold, "threshold_from": threshold_from, "temperature": temperature, "source": "val", "n_val": 40}


# calibration.json as `bulwark evaluate` writes it now: the fit on all val clips at the top
# level and one per media type under "media". eyes_closed has no video val clips; phone is
# an entry written before media entries; perclos is a regression label (nothing fitted).
MEDIA_CALIBRATION = {
    "yawn": {**fit(0.4, 2.0), "media": {"video": fit(0.3, 1.5), "image": fit(0.5, None, "default")}},
    "eyes_closed": {**fit(0.6, 1.2), "media": {"image": fit(0.7, 0.8)}},
    "activity": {**fit(0.5, 0.5, "default"), "media": {"video": fit(0.5, 0.7, "default"), "image": fit(0.5, 0.4, "default")}},
    "phone": fit(0.35, 1.1),
    "perclos": {"threshold": 0.5, "threshold_from": "default", "temperature": None, "source": "default"},
}


def moved(p: float, t: float) -> float:
    """A threshold on sigmoid(logit) moved to sigmoid(logit / t)."""
    return float(1 / (1 + np.exp(-np.log(p / (1 - p)) / t)))


def test_a_calibration_file_is_read_per_media_type(tmp_path: Path) -> None:
    path = calibration_file(tmp_path / "calibration.json", MEDIA_CALIBRATION, sha256="ab" * 32)
    calibration = read_calibration(path)
    # The top level stays the fit on all val clips, for readers that predate media entries.
    assert calibration.temperature == {"yawn": 2.0, "eyes_closed": 1.2, "activity": 0.5, "phone": 1.1}
    assert calibration.thresholds == {"yawn": 0.4, "eyes_closed": 0.6, "phone": 0.35}
    assert read_temperatures(path) == calibration.temperature
    # Only phone's entry applies to every media type; the others' top level is not applied.
    assert calibration.pooled == {"phone"} and not calibration.no_val_clips

    video = calibration.for_media("video")
    assert video.temperature == {"yawn": 1.5, "activity": 0.7, "phone": 1.1} and video.thresholds == {"yawn": 0.3, "phone": 0.35}
    # No video val clips for eyes_closed: nothing applies on video, never the fit on all val clips (images).
    assert video.no_val_clips == {"eyes_closed"} and video.pooled == {"phone"}
    image = calibration.for_media("image")
    # yawn's image fit failed (one class): no image threshold or temperature, not the fit on all val clips.
    assert image.temperature == {"eyes_closed": 0.8, "activity": 0.4, "phone": 1.1}
    assert image.thresholds == {"eyes_closed": 0.7, "phone": 0.35} and image.pooled == {"phone"} and not image.no_val_clips
    with pytest.raises(ValueError, match="media_type must be one of"):
        calibration.for_media("audio")

    # A file written before media entries applies to every media type; `pooled` names the
    # labels it gives a value (phone has neither a temperature nor a val threshold).
    old = read_calibration(calibration_file(tmp_path / "old.json", CALIBRATION, sha256="ab" * 32))
    assert old.media == {} and old.for_media("image").temperature == old.temperature == {"yawn": 2.0, "activity": 0.5}
    assert old.for_media("video").thresholds == {"yawn": 0.4} and old.for_media("video").pooled == old.pooled == {"yawn", "activity"}

    misspelt = {"yawn": {**fit(0.4, 2.0), "media": {"vidoe": fit(0.3, 1.5)}}}
    with pytest.raises(ValueError, match=r"The media entry of yawn names unknown media types \['vidoe'\]"):
        read_calibration(calibration_file(tmp_path / "bad.json", misspelt, sha256="ab" * 32))


def test_predict_applies_the_calibration_of_the_clips_media_type(tmp_path: Path) -> None:
    torch.manual_seed(0)
    checkpoint = save_checkpoint(tmp_path / "e1.pt", BulwarkEdge(pretrained=False, **EDGE), {"step": 2, **EDGE, "masked_labels": {"phone": ["image"]}})
    path = calibration_file(tmp_path / "calibration.json", MEDIA_CALIBRATION, sha256=sha256(checkpoint))
    plain = load(checkpoint, device="cpu")
    model = load(checkpoint, device="cpu", temperature=path)
    # The values without regard to media type stay the fit on all val clips.
    assert model.temperature == {"yawn": 2.0, "eyes_closed": 1.2, "activity": 0.5, "phone": 1.1}
    assert model.raw_thresholds == {"yawn": 0.4, "eyes_closed": 0.6, "phone": 0.35}
    # model.thresholds go with model.temperature, which predict applies to no media type
    # here, so reading them warns.
    with pytest.warns(UserWarning, match=r"predict applies the calibration of its input's media type"):
        thresholds = model.thresholds
    assert thresholds == pytest.approx({"yawn": moved(0.4, 2.0), "eyes_closed": moved(0.6, 1.2), "phone": moved(0.35, 1.1)})
    assert model.temperature_for("video") == {"yawn": 1.5, "activity": 0.7, "phone": 1.1}
    assert model.thresholds_for("video") == pytest.approx({"yawn": moved(0.3, 1.5), "phone": moved(0.35, 1.1)})
    # phone is masked on images, so it has no image temperature or threshold to report
    # (the file has its image temperature), and perclos (regression) never takes one.
    assert model.temperature_for("image") == {"eyes_closed": 0.8, "activity": 0.4}
    assert model.thresholds_for("image") == pytest.approx({"eyes_closed": moved(0.7, 0.8)})
    assert load(checkpoint, device="cpu", temperature={"yawn": 2.0, "perclos": 3.0}).temperature_for("video") == {"yawn": 2.0}
    with pytest.raises(ValueError, match="media_type must be one of"):
        model.temperature_for("audio")

    for frames, temperature in ((4, model.temperature_for("video")), (1, model.temperature_for("image"))):
        batch = clips(frames=frames, size=80)
        with torch.no_grad():
            logits = plain.module(torch.from_numpy(as_clip_batch(batch, FRAME_SIZE)))
        outputs = model.predict(batch)
        for name in ("yawn", "eyes_closed"):
            expected = torch.sigmoid(logits[name] / temperature.get(name, 1.0)).numpy()
            np.testing.assert_allclose(outputs[name], expected, atol=1e-5, err_msg=f"{name}, {frames} frames")
        np.testing.assert_allclose(outputs["activity"], torch.softmax(logits["activity"] / temperature["activity"], -1).numpy(), atol=1e-5)
    # The flagged clips are the same on the calibrated output as on sigmoid(logit) at the raw video threshold.
    batch = clips(frames=4, size=80)
    with torch.no_grad():
        raw = torch.sigmoid(plain.module(torch.from_numpy(as_clip_batch(batch, FRAME_SIZE)))["yawn"]).numpy()
    np.testing.assert_array_equal(model.predict(batch)["yawn"] >= model.thresholds_for("video")["yawn"], raw >= 0.3)


def test_a_release_keeps_the_calibration_of_each_media_type(tmp_path: Path) -> None:
    path = provenance_checkpoint(tmp_path / "e1.pt", commercial_only=True, datasets={"dmd": True})
    calibration = calibration_file(tmp_path / "calibration.json", MEDIA_CALIBRATION, sha256=sha256(path))
    model = load(path, device="cpu", labels=["yawn", "eyes_closed", "activity", "phone"], temperature=calibration)
    release = save_release(model, tmp_path / "release", weights_licence=RESEARCH)
    config = json.loads((release / "config.json").read_text())
    assert config["temperature"] == model.temperature and config["thresholds"] == model.raw_thresholds  # for older loaders
    assert list(config["media_calibration"]) == ["image", "video"]
    assert config["media_calibration"]["video"] == {
        "temperature": {"yawn": 1.5, "activity": 0.7, "phone": 1.1}, "thresholds": {"yawn": 0.3, "phone": 0.35},
    }
    reloaded = load(release, device="cpu")
    for media in ("image", "video"):
        assert reloaded.temperature_for(media) == model.temperature_for(media)
        assert reloaded.thresholds_for(media) == pytest.approx(model.thresholds_for(media))
    assert "eyes_closed" not in reloaded.temperature_for("video")  # val had no video clips for it

    # A `calibration` field that `extra` adds to config.json is not an exported graph's record.
    noted = save_release(model, tmp_path / "noted", weights_licence=RESEARCH, extra={"calibration": {"media": "image", "note": "x"}})
    for media in ("image", "video"):
        assert load(noted, device="cpu").temperature_for(media) == model.temperature_for(media)
    noted = save_release(model, tmp_path / "noted2", weights_licence=RESEARCH, extra={"calibration": "see the report"})
    assert load(noted, device="cpu").temperature_for("video") == model.temperature_for("video")

    # A plain mapping replaces the temperatures on every media type and keeps each one's thresholds.
    retuned = load(release, device="cpu", temperature={"yawn": 1.0})
    assert retuned.temperature_for("video") == retuned.temperature_for("image") == {"yawn": 1.0}
    assert retuned.thresholds_for("video") == pytest.approx({"yawn": 0.3, "phone": 0.35})
    assert retuned.thresholds_for("image") == pytest.approx({"eyes_closed": 0.7, "phone": 0.35})

    # A release written before media entries applies its values to images and video alike.
    del config["media_calibration"]
    (release / "config.json").write_text(json.dumps(config))
    older = load(release, device="cpu")
    assert older.calibration.media == {}
    assert older.temperature_for("image") == older.temperature_for("video") == model.temperature
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the same calibration on every media type: no warning
        assert older.thresholds_for("video") == older.thresholds == pytest.approx(
            {"yawn": moved(0.4, 2.0), "eyes_closed": moved(0.6, 1.2), "phone": moved(0.35, 1.1)}
        )
    config["media_calibration"] = {"vid": {"temperature": {"yawn": 1.5}}}
    (release / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=r"media_calibration names unknown media types \['vid'\]"):
        load(release, device="cpu")


def with_contract(path: Path, contract: dict) -> Path:
    onnx = pytest.importorskip("onnx")
    graph = onnx.load(str(path))
    entry = graph.metadata_props.add()
    entry.key, entry.value = "bulwark_contract", json.dumps(contract)
    onnx.save(graph, str(path))
    return path


def test_an_exported_graph_records_the_media_type_its_calibration_is_for(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    edge = edge_checkpoint(tmp_path / "e1.pt")
    record = {"sha256": "ab" * 32, "temperature": {"yawn": 9.0}}  # the checkpoint's own record
    contract = {
        "temperature": {"yawn": 1.5}, "thresholds": {"yawn": 0.3},
        "calibration": {"media": "video", "pooled": [], "no_val_clips": ["eyes_closed"]}, "checkpoint": record,
    }
    model = load(with_contract(export(edge, tmp_path / "e1.onnx", state=False), contract))
    # The graph's flat values are those of its media type, so model.thresholds go with
    # what it returns; the contract's record stays in its metadata.
    assert model.temperature == model.temperature_for("video") == {"yawn": 1.5} and model.raw_thresholds == {"yawn": 0.3}
    assert list(model.calibration.media) == ["video"] and model.metadata["contract"]["calibration"] == contract["calibration"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert model.thresholds == model.thresholds_for("video") == pytest.approx({"yawn": moved(0.3, 1.5)})

    # Exported with a calibration.json that had no values for its media type: nothing, not
    # the temperatures the source checkpoint records.
    empty = {"calibration": {"media": "video", "pooled": [], "no_val_clips": ["yawn"]}, "checkpoint": record}
    assert load(with_contract(export(edge, tmp_path / "empty.onnx", state=False), empty)).temperature == {}
    # A graph exported before the record: its values were fitted on all val clips.
    older = load(with_contract(export(edge, tmp_path / "older.onnx", state=False), {"temperature": {"yawn": 1.5}}))
    assert older.calibration.media == {} and older.temperature_for("image") == older.temperature_for("video") == {"yawn": 1.5}
    wrong = {"temperature": {"yawn": 1.5}, "calibration": {"media": "clip", "pooled": []}}
    with pytest.raises(ValueError, match=r"calibration record names unknown media types \['clip'\]"):
        load(with_contract(export(edge, tmp_path / "wrong.onnx", state=False), wrong))


def test_a_one_frame_graph_applies_the_image_calibration(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    edge = edge_checkpoint(tmp_path / "e1.pt")
    graph = export(edge, tmp_path / "e1-image.onnx", state=False, frames=1)
    fitted = calibration_file(tmp_path / "calibration.json", MEDIA_CALIBRATION, model="onnx", sha256=sha256(graph))
    model = load(graph, temperature=fitted)
    assert model.clip_frames == 1
    frames = clips(frames=1, size=80)
    with torch.no_grad():  # export left `edge` in training mode; the loaded checkpoint is in eval mode
        logits = load(tmp_path / "e1.pt", device="cpu").module(torch.from_numpy(as_clip_batch(frames, FRAME_SIZE)))
    outputs = model.predict(frames)
    np.testing.assert_allclose(outputs["eyes_closed"], torch.sigmoid(logits["eyes_closed"] / 0.8).numpy(), atol=1e-4)
    np.testing.assert_allclose(outputs["yawn"], torch.sigmoid(logits["yawn"]).numpy(), atol=1e-4)  # no image fit for yawn
    np.testing.assert_allclose(outputs["activity"], torch.softmax(logits["activity"] / 0.4, -1).numpy(), atol=1e-4)
