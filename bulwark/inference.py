"""Run Bulwark on clips or videos: activated predictions and embeddings.

    from bulwark.inference import load, predict_video

    model = load("checkpoints/b1-lora-v0/best.pt")  # B1, heads-only, E1, ONNX or a release dir
    outputs = model.predict(frames)              # (T, H, W, 3) or (B, T, H, W, 3) uint8 RGB
    tokens = model.embed(frames)                 # B1: (B, 25, 768) summary tokens
    for window in predict_video(model, "cabin.mp4"):
        print(window["start"], window["outputs"]["eyes_closed"])

Outputs follow each label's kind in the taxonomy: a probability for binary labels, one
probability per class for categorical labels, a value in the label's range for regression
labels. They are not calibrated unless per-label temperatures are given to `load` (a
mapping, or the calibration.json of `bulwark evaluate`) or recorded with the model. A
1-frame clip counts as an image and a longer one as video, as in training: a label whose
targets were masked on that media type is left out (`masked_labels`), and a calibration
fitted per media type applies the temperatures fitted on that media type's val clips
(`BulwarkModel.temperature_for`, `thresholds_for`). The input and
output contract is described in docs/inference-and-adapters.md; `bulwark predict` and
`bulwark release` are the command-line equivalents of `predict_video` and `save_release`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
import warnings
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from torch import nn

from bulwark.data.clips import letterbox
from bulwark.data.manifest import MEDIA_TYPES
from bulwark.data.taxonomy import LabelSpec, Taxonomy, default_taxonomy
from bulwark.export.release import (
    COMMERCIAL_LICENCE,
    NON_COMMERCIAL_LICENCE,
    RESEARCH_LICENCE,
    upstream_datasets,
)
from bulwark.model.backbone import VJEPA_LICENSE, BackboneConfig
from bulwark.model.bulwark import BulwarkBase, BulwarkEdge, load_payload
from bulwark.model.heads import activate
from bulwark.model.lora import LoRALinear, apply_lora
from bulwark.train.common import autocast_dtype, pick_device
from bulwark.train.provenance import file_sha256

if TYPE_CHECKING:
    from bulwark.adapters import Adapter

CLIP_FRAMES = 16          # frames per video clip in training (images are 1-frame clips)
CLIP_SECONDS = 2.0        # default time span of a clip, as cut by the dataset adapters
FRAME_SIZE = 256          # shards store frames letterboxed to this size
PERCLOS_SECONDS = 60.0    # PERCLOS in the drowsiness literature averages closure over a minute or more
EDGE_BACKBONE = "efficientvit_b1.r224_in1k"
EMBEDDING_KINDS = ("tokens", "pooled", "state")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
RELEASE_FORMAT = "bulwark-release"
RELEASE_VERSION = 1
ONNX_CONTRACT_KEY = "bulwark_contract"  # JSON contract that `bulwark export` stores in the graph
# E1's Hugging Face repository (scripts/hf_release.py): its two graphs by the frames they
# take, and the format of its config.json, which holds both graphs' contracts.
E1_GRAPHS = {16: "bulwark-e1-video.onnx", 1: "bulwark-e1-image.onnx"}
ONNX_RELEASE_FORMAT = "bulwark-onnx-release"
# A Hugging Face model repository id ("owner/name"), which `load` takes in place of a path.
HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")
# Model files `load` reads from a local path; an "owner/name" ending in one is a path.
LOCAL_MODEL_SUFFIXES = (".pt", ".pth", ".ckpt", ".onnx", ".safetensors")
# Provenance fields a release may publish: names, flags and hashes, never local paths.
_RELEASE_PROVENANCE = ("job", "commercial_only", "datasets", "manifest_sha256")
# What a model records about its calibration (checkpoint, release config.json, ONNX metadata).
_CALIBRATION_KEYS = ("temperature", "thresholds")
_LABEL_KINDS = ("binary", "categorical", "regression")

# Checkpoint entries that are training state rather than model description.
_TRAINING_STATE = {"model", "optimizer", "loss", "scaler"}
_LORA_KEY = re.compile(r"^encoder\.encoder\.blocks\.(\d+)\..*\.(?:down|up)\.weight$")


@dataclass(frozen=True)
class ModelOutput:
    """One forward pass over a batch of clips.

    `outputs` maps label -> activated array: (B,) for binary and regression labels,
    (B, C) for categorical labels, for the labels `BulwarkModel.labels_for` gives the
    batch's media type. `tokens` is the heads' input (B1: 25 summary tokens of width 768;
    E1: one feature vector per frame) and `state` the heads' pooled state embedding. ONNX
    graphs exported with `--no-embeddings` (logits only) leave both as None.
    """

    outputs: dict[str, np.ndarray]
    tokens: np.ndarray | None = None
    state: np.ndarray | None = None

    def embedding(self, kind: str = "tokens") -> np.ndarray:
        """`tokens` (B, P, D), `pooled` (B, D): mean over tokens, or `state` (B, W)."""
        if kind not in EMBEDDING_KINDS:
            raise ValueError(f"kind must be one of {EMBEDDING_KINDS}, got {kind!r}")
        value = self.state if kind == "state" else self.tokens
        if value is None:
            raise NotImplementedError(
                f"This model does not expose the {kind!r} embedding. ONNX graphs need a "
                f"{'state' if kind == 'state' else 'tokens'!r} output; see docs/inference-and-adapters.md."
            )
        return value.mean(axis=1) if kind == "pooled" else value


@dataclass(frozen=True)
class Calibration:
    """Per-label temperatures and decision thresholds, and the weights they were fitted on.

    `thresholds` are val-tuned F1 thresholds of binary labels on the uncalibrated
    probability sigmoid(logit), as `bulwark evaluate` writes them; None when the source
    says nothing about thresholds (a plain mapping of temperatures). `fitted_on` holds the
    sha256 values calibration.json records for the weights it was fitted on (the checkpoint
    and, for an ONNX graph, the checkpoint it was exported from); None when the source is
    not a calibration file, so there is nothing to check.

    `temperature` and `thresholds` are the values without regard to media type: in a
    calibration.json, the fit on all val clips; in an exported graph's record, the values
    of the graph's own media type. `media` holds the values resolved for each media type
    ("image", "video") when the source has them per media type (see `for_media`).

    `pooled` and `no_val_clips` describe what `read_calibration` read, for `bulwark export`
    to record in a graph's contract; a model's own record (checkpoint, release, graph)
    leaves them empty. `pooled` names the labels whose values here come from an entry that
    applies to every media type (a calibration.json entry without `media`, as in files
    written before media entries, or a mapping of temperatures) and give a temperature or
    a val threshold. `no_val_clips` names the labels of a calibration.json fitted per media
    type whose val clips held none of this media type, so nothing applies to them here.
    """

    temperature: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] | None = None
    fitted_on: frozenset[str] | None = None
    media: dict[str, Calibration] = field(default_factory=dict)
    pooled: frozenset[str] = frozenset()
    no_val_clips: frozenset[str] = frozenset()

    def for_media(self, media_type: str) -> Calibration:
        """The temperatures and thresholds for clips of `media_type`: its entry in `media`,
        else these values, which apply to every media type."""
        if media_type not in MEDIA_TYPES:
            raise ValueError(f"media_type must be one of {sorted(MEDIA_TYPES)}, got {media_type!r}")
        if media_type in self.media:
            return self.media[media_type]
        return Calibration(self.temperature, self.thresholds, self.fitted_on, pooled=self.pooled)


class BulwarkModel:
    """Common interface of the PyTorch models and the ONNX graph."""

    kind: str               # "b1", "b1-frozen", "e1" or "e1-onnx"
    taxonomy: Taxonomy
    labels: tuple[str, ...]  # outputs returned by `predict`, less the masked ones (`labels_for`)
    labels_known: bool       # False: no caller or record named the supervised labels, so `labels` is every head
    masked_labels: dict[str, tuple[str, ...]]  # label -> media types its training targets were masked on
    input_size: int          # frames are letterboxed to (input_size, input_size)
    clip_frames: int | None  # frames per clip the model requires (None: any length)
    calibration: Calibration  # what `temperature_for` and `thresholds_for` read, per media type
    temperature: dict[str, float]  # = calibration.temperature: from a calibration.json the fit on all val clips
    raw_thresholds: dict[str, float]  # val-tuned thresholds on sigmoid(logit), before any temperature, as `temperature`
    checkpoint_sha256: str | None  # the Bulwark checkpoint the weights come from, if known
    metadata: dict[str, Any]

    def run(self, frames: np.ndarray | torch.Tensor) -> ModelOutput:
        raise NotImplementedError

    def labels_for(self, media_type: str) -> tuple[str, ...]:
        """The outputs `predict` returns for "video" or "image" input.

        Training masks a label on a media type when its training targets there cover one
        class only, and on every media type when the run's `mask_labels` names it
        (`bulwark.train.labels`); checkpoints record the result as `masked_labels`. A head
        learned nothing on a media type its label was masked on, so that output is left out.
        """
        if media_type not in MEDIA_TYPES:
            raise ValueError(f"media_type must be one of {sorted(MEDIA_TYPES)}, got {media_type!r}")
        return tuple(name for name in self.labels if media_type not in self.masked_labels.get(name, ()))

    def temperature_for(self, media_type: str) -> dict[str, float]:
        """The temperatures `predict` applies to "video" or "image" input, for the binary
        and categorical labels it returns (`labels_for`).

        A calibration fitted per media type (a calibration.json from `bulwark evaluate`, or a
        release built with one) gives each media type the temperatures fitted on its own val
        clips; a label whose val clips held none of that media type gets none there. Without
        per-media-type values, the temperatures that apply to every media type
        (`temperature`). An ONNX graph's recorded values are those of its own media type.
        """
        returned = self.labels_for(media_type)
        return {
            name: t
            for name, t in self.calibration.for_media(media_type).temperature.items()
            if name in returned and self.taxonomy[name].kind != "regression"
        }

    def thresholds_for(self, media_type: str) -> dict[str, float]:
        """Val-tuned decision thresholds of the binary labels `predict` returns for "video"
        or "image" input (`labels_for`), on the probabilities it returns: those fitted on
        that media type's val clips, converted with its temperatures (see `thresholds`)."""
        chosen = self.calibration.for_media(media_type)
        return _calibrated_thresholds(chosen.thresholds or {}, chosen.temperature, self.labels_for(media_type), self.taxonomy)

    @property
    def thresholds(self) -> dict[str, float]:
        """Val-tuned decision thresholds of the binary labels, on the probabilities `predict`
        returns when its calibration is the same on every media type.

        `bulwark evaluate` tunes them on the uncalibrated probability (`raw_thresholds`). A
        temperature t moves a threshold p to sigmoid(logit(p) / t), so a clip is flagged at
        the same logit with or without calibration. A threshold of exactly 0 or 1 (tuned on
        a head whose probabilities saturate) stays as it is, since sigmoid(+-inf / t) does
        not depend on t. Empty unless a calibration.json was given to `load` or the model
        records thresholds.

        These are `raw_thresholds` converted with `temperature`. A calibration fitted per
        media type (a calibration.json with per-media-type entries, or a release built with
        one) makes `predict` apply the values of its input's media type, which these (from
        a calibration.json, the fit on all val clips) are not: when they differ from
        `thresholds_for(media_type)` for either media type, reading them warns, and
        `thresholds_for` gives the thresholds that go with what `predict` returns. An ONNX
        graph's recorded values are those of its own media type, so they go with what it
        returns.
        """
        thresholds = _calibrated_thresholds(self.raw_thresholds, self.temperature, self.labels, self.taxonomy)
        if any(
            self.thresholds_for(media) != {name: p for name, p in thresholds.items() if name in self.labels_for(media)}
            for media in sorted(MEDIA_TYPES)
        ):
            warnings.warn(
                "model.thresholds go with model.temperature (from a calibration.json, the fit on all val clips), but "
                "predict applies the calibration of its input's media type, which differs here; compare its outputs "
                'with model.thresholds_for("video") or model.thresholds_for("image").',
                stacklevel=2,
            )
        return thresholds

    def weights_sha256(self) -> set[str]:
        """sha256 values that identify these weights: the source checkpoint's, if known."""
        return {self.checkpoint_sha256} if self.checkpoint_sha256 else set()

    def _set_calibration(
        self, given: Mapping[str, Any] | str | Path | None, recorded: Mapping[str, Any], graph: Mapping[str, Any] | None = None
    ) -> None:
        """Temperatures and thresholds from `given` (the caller's), else from what the model records.

        A calibration.json must have been fitted on these weights: a file that names other
        weights is refused, and one that cannot be checked gives a warning. A plain mapping
        of temperatures is the caller's choice and is not checked; it replaces the
        temperatures on every media type and keeps the recorded thresholds of each, which
        do not depend on the temperature. `graph` is an exported graph's contract
        `calibration` record (see `_recorded_calibration`).
        """
        record = _recorded_calibration(recorded, graph)
        calibration = record if given is None else read_calibration(given)
        if calibration.fitted_on is not None:
            self._check_fitted_on(calibration.fitted_on, given)
        if calibration.thresholds is None:  # a plain mapping of temperatures
            calibration = _with_temperatures(record, calibration.temperature)
        self.calibration = calibration
        self.temperature = calibration.temperature
        self.raw_thresholds = dict(calibration.thresholds or {})

    def _check_fitted_on(self, fitted_on: frozenset[str], source: Any) -> None:
        name = source if isinstance(source, (str, Path)) else "the calibration"
        mine = self.weights_sha256()
        if fitted_on and mine:
            if not fitted_on & mine:
                raise ValueError(
                    f"{name} was fitted on weights with sha256 {', '.join(sorted(s[:12] for s in fitted_on))}, "
                    f"not on this model ({', '.join(sorted(s[:12] for s in mine))}). Temperatures and thresholds "
                    "belong to one set of weights: run `bulwark evaluate` on this model, or pass the temperatures "
                    "as a mapping to apply them anyway."
                )
            return
        missing = "it records no checkpoint" if not fitted_on else "this model does not record which checkpoint it comes from"
        warnings.warn(f"Cannot check that {name} was fitted on this model: {missing}.", stacklevel=2)

    def embedding_source(self, kind: str = "tokens") -> dict[str, Any]:
        """What produces this model's `kind` embeddings, as `Adapter.check` compares it.

        {"family": "b1" or "e1", "weights": ...}: "frozen" for tokens of the frozen V-JEPA
        encoder (they do not depend on any Bulwark checkpoint), otherwise the sha256 of the
        checkpoint, or None when it is unknown.
        """
        family = "e1" if self.kind.startswith("e1") else "b1"
        if self.kind == "b1-frozen" and kind != "state":
            return {"family": family, "weights": "frozen"}
        return {"family": family, "weights": self.checkpoint_sha256}

    def predict(self, frames: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        """Activated outputs per label for one clip (T, H, W, 3) or a batch (B, T, H, W, 3).

        The labels are `labels_for("image")` for 1-frame clips and `labels_for("video")`
        otherwise.
        """
        return self.run(frames).outputs

    def embed(self, frames: np.ndarray | torch.Tensor, kind: str = "tokens") -> np.ndarray:
        """Embeddings for adapters and retrieval; see `ModelOutput.embedding`."""
        return self.run(frames).embedding(kind)

    def to_dicts(self, outputs: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
        """Per-clip JSON-ready dicts: floats, and {class: probability} for categorical labels."""
        return outputs_to_dicts(outputs, self.taxonomy)

    def _activate(self, logits: Mapping[str, torch.Tensor], media_type: str) -> dict[str, np.ndarray]:
        temperature = self.temperature_for(media_type)
        scaled = {}
        for name in self.labels_for(media_type):
            value = logits[name].float()
            if name in temperature and self.taxonomy[name].kind != "regression":
                value = value / temperature[name]
            scaled[name] = value
        return {k: v.cpu().numpy() for k, v in activate(scaled, self.taxonomy).items()}

    def _prepare(self, frames: np.ndarray | torch.Tensor) -> np.ndarray:
        clips = as_clip_batch(frames, self.input_size)
        if self.clip_frames is not None and clips.shape[1] != self.clip_frames:
            raise ValueError(f"This model takes clips of exactly {self.clip_frames} frames, got {clips.shape[1]}")
        return clips


class TorchModel(BulwarkModel):
    """Bulwark-B1 or Bulwark-E1 in PyTorch.

    `temperature` is the caller's calibration (see `load`); `recorded` holds what the
    checkpoint or release records (`temperature`, `thresholds` and a release's
    `media_calibration`, used when it is None, and `masked_labels`).
    """

    def __init__(
        self,
        module: BulwarkBase | BulwarkEdge,
        kind: str,
        *,
        architecture: dict[str, Any],
        labels: Sequence[str] | None = None,
        temperature: Mapping[str, Any] | str | Path | None = None,
        recorded: Mapping[str, Any] | None = None,
        checkpoint_sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        device: str = "auto",
        precision: str = "fp32",
        batch_size: int = 8,
    ) -> None:
        self.device = pick_device(device)
        self.module = module.eval().to(self.device)
        self.kind = kind
        self.architecture = dict(architecture)
        self.taxonomy = module.taxonomy
        self.labels = _check_labels(labels, self.taxonomy)
        self.labels_known = bool(labels)
        self.masked_labels = _masked_labels((recorded or {}).get("masked_labels"), self.taxonomy)
        self.checkpoint_sha256 = checkpoint_sha256
        self._set_calibration(temperature, recorded or {})
        self.metadata = dict(metadata or {})
        self.dtype = autocast_dtype(self.device, precision)
        self.batch_size = batch_size
        # E1 was distilled on shard frames (letterboxed to FRAME_SIZE) and resizes them to its
        # own img_size in forward; the exported graph does the same, so both see equal pixels.
        self.input_size = FRAME_SIZE if isinstance(module, BulwarkEdge) else module.encoder.config.img_size
        self.clip_frames = None

    @torch.no_grad()
    def run(self, frames: np.ndarray | torch.Tensor) -> ModelOutput:
        clips = self._prepare(frames)
        tokens, states, logits = [], [], []
        for start in range(0, len(clips), self.batch_size):
            batch = torch.from_numpy(clips[start : start + self.batch_size]).to(self.device)
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
                token = self._tokens(batch)
                heads = self.module.heads
                state = heads.embed(token)
                logit = heads.split(heads.out(state))
            tokens.append(token.float().cpu())
            states.append(state.float().cpu())
            logits.append({k: v.float().cpu() for k, v in logit.items()})
        merged = {name: torch.cat([chunk[name] for chunk in logits]) for name in self.labels}
        outputs = self._activate(merged, media_type_of(clips))
        return ModelOutput(outputs, torch.cat(tokens).numpy(), torch.cat(states).numpy())

    def _tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """The heads' input tokens: `tokens(frames)`, which B1 and E1 both provide.

        A module without that method is run whole and its tokens read off the heads' input.
        """
        tokens = getattr(self.module, "tokens", None)
        if callable(tokens):
            return tokens(frames)
        captured: list[torch.Tensor] = []
        handle = self.module.heads.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
        try:
            self.module(frames)
        finally:
            handle.remove()
        return captured[0]


class OnnxModel(BulwarkModel):
    """An exported Bulwark-E1 graph run with onnxruntime.

    The graph takes `frames` (B, T, S, S, 3) uint8 and returns one logit output per label.
    Optional `tokens` and `state` outputs (`bulwark export` adds them unless
    `--no-embeddings`) are exposed as embeddings when present. `temperature` and
    `thresholds` are read from the graph's metadata: at the top level, inside the
    `bulwark_contract` JSON that `bulwark export` writes, or in that contract's record of
    the source checkpoint (`checkpoint`). A graph exported with a calibration.json holds
    the values for its own media type (the video entries for T > 1, the image entries for
    T = 1), and its contract's `calibration` record says which and names the labels that
    took values for every media type (`pooled`) or had no val clips of it (`no_val_clips`;
    both stay in `metadata["contract"]`); `temperature` and `thresholds` then come from the
    graph alone.
    `supervised_labels` and `masked_labels` are read from that checkpoint record first, so
    the graph returns what the PyTorch checkpoint returns; the graph's own
    `supervised_labels` (from `bulwark export --manifest`) are used only when the
    checkpoint records none. `masked_labels` falls back the same way, to the top-level
    `masked_labels` key and then the contract's, so a graph that keeps any one of them
    still withholds a label on the media type it was masked on (`labels_for`).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        labels: Sequence[str] | None = None,
        temperature: Mapping[str, Any] | str | Path | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.path = Path(path)
        self.session = ort.InferenceSession(str(self.path), providers=list(providers or ["CPUExecutionProvider"]))
        self.kind = "e1-onnx"
        self.taxonomy = default_taxonomy()
        meta = dict(self.session.get_modelmeta().custom_metadata_map)
        contract = _json_meta(meta, ONNX_CONTRACT_KEY) or {}
        meta.pop(ONNX_CONTRACT_KEY, None)
        self.metadata = {"path": str(self.path), **meta, **({"contract": contract} if contract else {})}
        self.outputs = [o.name for o in self.session.get_outputs()]
        missing = [name for name in self.taxonomy.labels if name not in self.outputs]
        if missing:
            raise ValueError(f"{self.path} lacks outputs for {missing[:5]}; was it exported with this taxonomy?")
        _check_contract(self.path, contract, self.taxonomy)
        source = contract.get("checkpoint") or {}
        # The labels come from the training job's own record first. The supervised_labels
        # that `bulwark export --manifest` stores count every label with a training target,
        # including those the job masked, so they serve only when the checkpoint records none.
        recorded = {
            key: source.get(key) or _json_meta(meta, key) or contract.get(key)
            for key in ("supervised_labels", "masked_labels")
        }
        calibrated = contract.get("calibration")  # written by `bulwark export --calibration`
        recorded.update({
            key: _json_meta(meta, key) or contract.get(key) or (None if calibrated else source.get(key)) for key in _CALIBRATION_KEYS
        })
        labels = labels or recorded["supervised_labels"]
        self.labels = _check_labels(labels, self.taxonomy)
        self.labels_known = bool(labels)
        self.masked_labels = _masked_labels(recorded["masked_labels"], self.taxonomy)
        self.checkpoint_sha256 = source.get("sha256")
        self._set_calibration(temperature, recorded, calibrated)
        graph_input = self.session.get_inputs()[0]
        self.input_name = graph_input.name
        shape = graph_input.shape  # [batch, T, S, S, 3]; dims may be symbolic
        self.clip_frames = shape[1] if isinstance(shape[1], int) else None
        self.input_size = shape[2] if isinstance(shape[2], int) else FRAME_SIZE

    def weights_sha256(self) -> set[str]:
        """The checkpoint the graph was exported from, and the graph file itself (`bulwark
        evaluate` on an ONNX graph records both)."""
        return {*super().weights_sha256(), file_sha256(self.path)}

    def run(self, frames: np.ndarray | torch.Tensor) -> ModelOutput:
        clips = self._prepare(frames)
        media_type = media_type_of(clips)
        labels = self.labels_for(media_type)
        wanted = [*labels, *(name for name in ("tokens", "state") if name in self.outputs)]
        results = self.session.run(wanted, {self.input_name: clips}) if wanted else []  # [] would ask for every output
        values = dict(zip(wanted, results, strict=True))
        logits = {name: torch.from_numpy(np.asarray(values[name])) for name in labels}
        return ModelOutput(self._activate(logits, media_type), values.get("tokens"), values.get("state"))


# ---- loading -------------------------------------------------------------------------------


def load(
    path: str | Path,
    *,
    device: str = "auto",
    precision: str = "fp32",
    labels: Sequence[str] | None = None,
    temperature: Mapping[str, Any] | str | Path | None = None,
    backbone_checkpoint: str | Path | None = None,
    providers: Sequence[str] | None = None,
) -> BulwarkModel:
    """Load a model for inference; the format is detected from the path and its contents.

    - a Hugging Face model repository id, when no local path has that name
      (`resolve_model`): B1's repository loads as a release directory, E1's as its
      16-frame graph; "<id>/bulwark-e1-image.onnx" loads E1's graph for still images.
    - `*.onnx`: an exported E1 graph (onnxruntime, `providers`).
    - a directory: a release written by `save_release` or `bulwark release` (full
      weights, LoRA merged).
    - a B1 checkpoint from `bulwark train lora` (LoRA weights + heads). The frozen V-JEPA
      weights come from `backbone_checkpoint` (default: checkpoints/pretrained/).
    - a heads checkpoint from `bulwark train heads`: B1 with the frozen encoder.
    - an E1 checkpoint from `bulwark train distill`.

    `labels` selects the outputs, in place of the labels the model records under
    `supervised_labels` (the default; every taxonomy label when it records none). A label
    outside the recorded `supervised_labels` comes from a head that may have had no
    training targets. Either way a label is left out for the media type its training
    targets were masked on (`masked_labels`, see `BulwarkModel.labels_for`).

    `temperature` calibrates binary and categorical outputs (logits are divided by it
    before activation). It is either a mapping label -> temperature, or the path of the
    `calibration.json` that `bulwark evaluate` writes ({"meta": {...}, "labels": {label:
    {"temperature": t, "threshold": p, ..., "media": {media type: {...}}}}}; labels
    without a temperature are left as they are). A calibration.json also sets the val-tuned
    thresholds, and it must have been fitted on these weights: one whose `meta.checkpoint`
    names another checkpoint is refused, and one that cannot be checked gives a warning.
    Its entries per media type apply to clips of that media type (`model.temperature_for`,
    `model.thresholds_for`); see `read_calibration`. Either form replaces the temperatures
    the model records; a mapping applies to every media type.
    """
    path = resolve_model(path)
    if path.suffix == ".onnx":
        return OnnxModel(path, labels=labels, temperature=temperature, providers=providers)
    if path.is_dir():
        return _load_release(path, device=device, precision=precision, labels=labels, temperature=temperature)

    payload = load_payload(path)
    state = {k.removeprefix("module."): v for k, v in payload["model"].items()}  # DataParallel saves
    common: dict[str, Any] = {
        "labels": labels or payload.get("supervised_labels"),
        "temperature": temperature,
        "recorded": payload,
        "checkpoint_sha256": file_sha256(path),
        "metadata": {"path": str(path), **_plain_metadata(payload)},
        "device": device,
        "precision": precision,
    }
    if any(k.startswith("frame_encoder.") for k in state):
        architecture = {
            "backbone": payload.get("backbone", EDGE_BACKBONE),
            "img_size": int(payload.get("img_size", 224)),
        }
        edge = BulwarkEdge(pretrained=False, **architecture)
        edge.load_state_dict(state, strict=True)
        return TorchModel(edge, "e1", architecture=architecture, **common)

    kind = "b1"
    if not any(k.startswith(("encoder.", "heads.")) for k in state):
        if not {"cls", "out.weight"} <= set(state):
            raise ValueError(f"{path} is not a Bulwark B1, heads or E1 checkpoint")
        state = {f"heads.{k}": v for k, v in state.items()}
        kind = "b1-frozen"
    config = BackboneConfig(checkpoint=str(backbone_checkpoint)) if backbone_checkpoint else BackboneConfig()
    base = BulwarkBase(backbone=config, pretrained=True)
    _add_lora(base, state, payload)
    # strict=False: the frozen encoder weights come from the pretrained file. Older LoRA
    # checkpoints also hold the untrained hierarchical norms; those keys exist and load.
    missing, unexpected = base.load_state_dict(state, strict=False)
    if unexpected:
        raise KeyError(f"Unexpected keys in {path}: {unexpected[:5]}")
    # Everything training changed (LoRA, the final encoder norm, the heads) must be here.
    absent = sorted(set(base.trainable_state()) & set(missing))
    if absent:
        raise KeyError(f"{path} is missing {absent[:5]}")
    merge_lora(base)
    architecture = {"backbone": _backbone_fields(config)}
    return TorchModel(base, kind, architecture=architecture, **common)


def resolve_model(path: str | Path) -> Path:
    """The local path of a model: `path` itself when it exists or is not a Hugging Face
    repository id (a missing path whose first part is a local folder, or that names a
    model file, stays a path), else the files downloaded from that repository with huggingface_hub
    (an optional dependency; a gated repository needs an approved access request and `hf
    auth login`). B1's repository gives the folder of its config.json and
    model.safetensors, E1's its 16-frame graph; "<owner>/<name>/<graph>.onnx" gives that
    graph. Downloads go to the huggingface_hub cache."""
    local = Path(path)
    if local.exists() or not isinstance(path, str):
        return local
    repo, filename = path.strip("/"), ""
    parts = repo.split("/")
    if len(parts) == 3 and parts[2].endswith(".onnx"):
        repo, filename = "/".join(parts[:2]), parts[2]
    # A missing local path is not sent to the Hub as a repository id: one under a folder
    # that exists here (runs/b1-release), or one naming a model file (export/e1.onnx).
    if not HUB_ID.fullmatch(repo) or Path(parts[0]).is_dir() or repo.lower().endswith(LOCAL_MODEL_SUFFIXES):
        return local
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    except ImportError as error:
        raise ImportError(
            f"{path} is not a local path. To download it from the Hugging Face Hub, install huggingface_hub "
            "(pip install huggingface_hub) and log in with `hf auth login`."
        ) from error
    try:
        if filename:
            return Path(hf_hub_download(repo, filename))
        config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text(encoding="utf-8"))
        if config.get("format") == ONNX_RELEASE_FORMAT:
            return Path(hf_hub_download(repo, E1_GRAPHS[16]))
        return Path(snapshot_download(repo, allow_patterns=["config.json", "model.safetensors", "LICENSE*", "NOTICE"]))
    except (HfHubHTTPError, LocalEntryNotFoundError) as error:
        raise FileNotFoundError(
            f"{path} is neither a local path nor a Hugging Face repository this login can read ({type(error).__name__}: "
            f"{error}). For a gated Bulwark repository, request access on its page, then run `hf auth login`."
        ) from error


def merge_lora(module: nn.Module) -> nn.Module:
    """Fold every LoRA adapter into its base Linear: same outputs, plain layers."""
    for name, child in list(module.named_modules()):
        if isinstance(child, LoRALinear):
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            setattr(parent, leaf, child.merged().to(child.base.weight.device))
    return module


def save_release(
    model: TorchModel,
    out_dir: str | Path,
    *,
    weights_licence: str,
    labels: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a self-contained release: `model.safetensors` (all weights, LoRA merged) and `config.json`.

    Loading it needs this package (which vendors the V-JEPA model code), but not the
    pretrained V-JEPA file. A B1 release also gets `LICENSE-vjepa2`, Meta's notice for the
    encoder weights.

    `weights_licence` states the terms the weights are released under. Unless the training
    provenance shows commercially usable data only (`commercial_only` true, and every
    recorded dataset, including those of upstream models, `commercial_ok`), it must
    restrict use to non-commercial research (`bulwark.export.release.research_only`; the
    research licence `bulwark release` writes, `RESEARCH_LICENCE`, does); v0 weights
    trained on research-only data cannot be released for commercial use. `labels` are the outputs the release advertises: the labels its training data
    supervised. They default to the model's labels when the caller or the checkpoint named
    them; a release is refused otherwise, so it never lists heads that were never trained.
    config.json also records the model's `masked_labels`, its calibration for every media
    type (`temperature`, `thresholds`; from a calibration.json, the fit on all val clips,
    for loaders that predate media entries) and, under `media_calibration`, the values
    resolved per media type ({media type: {"temperature", "thresholds"}}; {} without
    per-media-type values). `extra` adds other fields to
    config.json but cannot replace the ones written here; `bulwark release` and
    examples/make_release.py pass the release check there
    (`bulwark.export.release.prepare_release`). config.json holds no local paths: see
    `release_source`. Nothing is written when a check fails.
    """
    from safetensors.torch import save_file

    if labels:
        names = _check_labels(labels, model.taxonomy)
    elif model.labels_known:
        names = model.labels
    else:
        raise ValueError(
            "The model records no supervised labels. Pass labels=[...] with the labels its training "
            "data covered (keys of labels_gt or labels_teacher in its training clips)."
        )
    source = release_source(model)
    check_weights_licence(weights_licence, source)
    config = {
        "format": RELEASE_FORMAT,
        "format_version": RELEASE_VERSION,
        "kind": model.kind,
        "architecture": model.architecture,
        "taxonomy_version": model.taxonomy.version,
        "outputs": list(model.module.heads.label_names),
        "labels": list(names),
        "masked_labels": {name: list(media) for name, media in model.masked_labels.items()},
        "temperature": model.temperature,
        "thresholds": model.raw_thresholds,  # on sigmoid(logit), before any temperature, as calibration.json
        "media_calibration": {
            media: {"temperature": values.temperature, "thresholds": values.thresholds or {}}
            for media, values in sorted(model.calibration.media.items())
        },
        "input": {
            "layout": "frames (B, T, H, W, 3) uint8 RGB",
            "size": model.input_size,
            "clip_frames": CLIP_FRAMES,
            "clip_seconds": CLIP_SECONDS,
        },
        "weights_licence": weights_licence,
        "commercial_only": source["commercial_only"],
        "source": source,
    }
    clash = sorted(set(extra or {}) & set(config))
    if clash:
        raise ValueError(f"extra cannot replace the release fields {clash}")
    config.update(extra or {})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.module.state_dict().items()}
    save_file(state, str(out / "model.safetensors"), metadata={"format": RELEASE_FORMAT})
    if model.kind.startswith("b1"):
        shutil.copyfile(VJEPA_LICENSE, out / "LICENSE-vjepa2")
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def check_weights_licence(weights_licence: str, source: Mapping[str, Any]) -> None:
    """Refuse a licence that allows commercial use of weights not shown to be commercially clean.

    `source` is `release_source(model)`. Weights count as clean only when the training
    provenance records `commercial_only: true`, no dataset without `commercial_ok` (its
    own, or one an upstream model was trained on), and no upstream input whose training
    data is unrecorded. Any other weights need a licence that restricts use to
    non-commercial research: it must say so (`NON_COMMERCIAL_LICENCE`: "non-commercial",
    CC BY-NC, "research use only"), and it must not mention commercial use other than as
    non-commercial or name a licence that allows commercial use (`COMMERCIAL_LICENCE`:
    MIT, Apache, BSD, CC BY without NC, ...), so "Apache-2.0 (research and commercial
    use)" is refused.
    """
    if not weights_licence.strip():
        raise ValueError("weights_licence is required: state the terms the weights are released under")
    datasets = source.get("datasets") or {}
    if not isinstance(datasets, Mapping):  # names only: commercial use is not recorded
        datasets = dict.fromkeys(datasets)
    upstream = source.get("upstream_datasets") or {}
    unclean = sorted(
        {name for name, info in datasets.items() if not (info or {}).get("commercial_ok")}
        | {name for name, info in upstream.items() if not (info or {}).get("commercial_ok")}
    )
    unrecorded = list(source.get("unrecorded_inputs") or ())
    if source.get("commercial_only") is True and not unclean and not unrecorded:
        return
    grant = COMMERCIAL_LICENCE.search(weights_licence)
    if grant is None and NON_COMMERCIAL_LICENCE.search(weights_licence):
        return
    if unclean:
        reason = f"was trained on datasets that are not commercially usable: {', '.join(unclean)}"
    elif unrecorded:
        reason = f"learned from upstream inputs whose training data is not recorded: {', '.join(unrecorded)}"
    elif source.get("commercial_only") is False:
        reason = "was trained with commercial_only=false"
    else:
        reason = "records no commercial_only provenance, so its training data cannot be shown to be commercially usable"
    problem = f" (it mentions {grant.group(0)!r})" if grant else ""
    raise ValueError(
        f"weights_licence {weights_licence!r} does not restrict use to non-commercial research{problem}, but the "
        f"model {reason}. Release it under a non-commercial licence that names no commercial terms (for example "
        f"the research licence {RESEARCH_LICENCE!r}), or release weights from the commercially clean retrain."
    )


def release_source(model: BulwarkModel) -> dict[str, Any]:
    """Where a model's weights came from, as a release publishes it: names and hashes only.

    The checkpoint's file name and sha256 and its training step; from the provenance the
    training jobs record, the job, `commercial_only` (None when not recorded), the datasets
    (licence, commercial use and clips per split), the manifest's sha256 and the code
    commit; and the datasets of the upstream models the run learned from (distillation
    teacher, init checkpoint, the model behind cached features: name -> `via`, `licence`,
    `commercial_ok`) with the upstream inputs whose training data is not recorded, as
    `bulwark.export.release.upstream_datasets` reads them. Both are None when the model
    records no provenance. Local paths (checkpoint, manifest, shards, features, init) are
    left out.
    """
    meta = model.metadata
    provenance = meta.get("provenance") or {}
    code = provenance.get("code") or {}
    checkpoint = meta.get("checkpoint")  # a release being re-released keeps its original record
    origin = Path(str(meta.get("path", "")))
    if origin.is_file():
        checkpoint = {"file": origin.name, "sha256": model.checkpoint_sha256}
    upstream: dict[str, dict[str, Any]] | None
    unrecorded: list[str] | None
    if provenance:
        upstream, unrecorded = upstream_datasets(dict(provenance.get("inputs") or {}))
    else:  # a release keeps what its source recorded
        upstream, unrecorded = meta.get("upstream_datasets"), meta.get("unrecorded_inputs")
    return {
        "checkpoint": checkpoint,
        "step": meta.get("step"),
        **{key: provenance.get(key, meta.get(key)) for key in _RELEASE_PROVENANCE},
        "code_commit": code.get("commit", meta.get("code_commit")),
        "code_dirty": code.get("dirty", meta.get("code_dirty")),
        "upstream_datasets": upstream,
        "unrecorded_inputs": unrecorded,
    }


def _load_release(
    path: Path,
    *,
    device: str,
    precision: str,
    labels: Sequence[str] | None,
    temperature: Mapping[str, Any] | str | Path | None = None,
) -> TorchModel:
    from safetensors.torch import load_file

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != RELEASE_FORMAT or config.get("format_version") != RELEASE_VERSION:
        raise ValueError(f"{path} is not a {RELEASE_FORMAT} v{RELEASE_VERSION} directory")
    taxonomy = default_taxonomy()
    if config["taxonomy_version"] != taxonomy.version or config["outputs"] != list(taxonomy.labels):
        raise ValueError(f"{path} was built for taxonomy v{config['taxonomy_version']}, this code has v{taxonomy.version}")
    architecture = config["architecture"]
    module: BulwarkBase | BulwarkEdge
    if config["kind"] == "e1":
        module = BulwarkEdge(taxonomy, pretrained=False, **architecture)
    else:
        module = BulwarkBase(taxonomy, backbone=BackboneConfig(**architecture["backbone"]), pretrained=False)
    module.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    source = config.get("source") or {}
    return TorchModel(
        module,
        config["kind"],
        architecture=architecture,
        labels=labels or config.get("labels"),
        temperature=temperature,
        recorded=config,
        checkpoint_sha256=(source.get("checkpoint") or {}).get("sha256"),
        metadata={"path": str(path), **source},
        device=device,
        precision=precision,
    )


def _add_lora(model: BulwarkBase, state: Mapping[str, torch.Tensor], payload: Mapping[str, Any]) -> None:
    """Recreate the LoRA layers a checkpoint was trained with, from its own keys.

    The `lora_blocks` and `lora_rank` the training job records must agree with the keys.
    """
    blocks = sorted({int(m.group(1)) for k in state if (m := _LORA_KEY.match(k))})
    recorded_blocks = payload.get("lora_blocks")
    if recorded_blocks is not None and int(recorded_blocks) != len(blocks):
        raise ValueError(f"The checkpoint records lora_blocks={recorded_blocks}, but holds LoRA weights for {len(blocks)} blocks")
    if not blocks:
        return
    rank = next(v.shape[0] for k, v in state.items() if _LORA_KEY.match(k) and k.endswith(".down.weight"))
    recorded_rank = payload.get("lora_rank")
    if recorded_rank is not None and int(recorded_rank) != rank:
        raise ValueError(f"The checkpoint records lora_rank={recorded_rank}, but its LoRA weights have rank {rank}")
    depth = len(model.encoder.encoder.blocks)
    if blocks != list(range(blocks[0], depth)):
        raise ValueError(f"LoRA on blocks {blocks}; expected the top blocks of {depth}")
    model.lora_layers = apply_lora(model.encoder.encoder, last_blocks=depth - blocks[0], rank=rank)


def _backbone_fields(config: BackboneConfig) -> dict[str, Any]:
    fields = asdict(config)
    fields.pop("checkpoint")  # a local path; releases carry the weights themselves
    return fields


def _plain_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Checkpoint entries that describe the model (step, provenance), not training state."""
    out = {}
    for key, value in payload.items():
        if key in _TRAINING_STATE:
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        out[key] = value
    return out


def _json_meta(meta: Mapping[str, str], key: str) -> Any:
    return json.loads(meta[key]) if key in meta else None


def _check_contract(path: Path, contract: Mapping[str, Any], taxonomy: Taxonomy) -> None:
    """Refuse a graph whose `bulwark export` contract records another taxonomy.

    Output names alone do not show it: a categorical label whose classes were renamed or
    reordered keeps its name and width, and `to_dicts` would put the wrong class names on
    its probabilities. A graph without a contract is checked by its output names only.
    """
    version = contract.get("taxonomy_version")
    differ = sorted(
        str(entry.get("name"))
        for entry in contract.get("outputs") or []
        if isinstance(entry, Mapping)
        and entry.get("kind") in _LABEL_KINDS
        and (
            entry.get("name") not in taxonomy.labels
            or _label_terms(entry.get("kind"), entry.get("classes"), entry.get("range")) != _spec_terms(taxonomy[entry["name"]])
        )
    )
    if (version is not None and version != taxonomy.version) or differ:
        built = f"v{version}" if version is not None else "an unrecorded version"
        detail = f" (outputs that differ: {differ[:5]})" if differ else ""
        raise ValueError(f"{path} was built for taxonomy {built}, this code has v{taxonomy.version}{detail}")


def _label_terms(kind: Any, classes: Any, value_range: Any) -> tuple[Any, ...]:
    """What fixes the meaning of a label's output: its kind, its classes and its range."""
    return (
        kind,
        tuple(classes or ()) if kind == "categorical" else (),
        tuple(float(v) for v in value_range or ()) if kind == "regression" else (),
    )


def _spec_terms(spec: LabelSpec) -> tuple[Any, ...]:
    return _label_terms(spec.kind, spec.classes, spec.value_range)


def read_calibration(source: Mapping[str, Any] | str | Path) -> Calibration:
    """Temperatures and thresholds from a mapping label -> temperature, or from a calibration.json.

    calibration.json, as `bulwark evaluate` writes it, is {"meta": {...}, "labels": {label:
    {"temperature": t, "threshold": p, "threshold_from": "val" | "default", ..., "media":
    {media type: {the same keys}}}}}; files from before it had a "meta" section map labels
    directly. Null temperatures are skipped and only thresholds tuned on val are kept.
    `meta.checkpoint` names the weights the file was fitted on (`sha256`, and for an ONNX
    graph `exported_from_sha256`): `fitted_on`. A mapping of temperatures (label -> t)
    gives temperatures only, with nothing to check.

    A label's top-level entry is its fit on all val clips: `temperature` and `thresholds`.
    When any entry has `media`, `media` gets the values for each media type: a label with
    `media` takes its entry for that media type, which applies even when its fit failed (a
    null temperature or a default threshold gives none, never the fit on all val clips),
    and a label whose `media` has no entry for it (val held none of its clips of that
    media type) gets nothing there and is listed in `no_val_clips`. A label without
    `media` keeps its top-level values and is listed in `pooled`.
    """
    data: Mapping[str, Any] = (
        json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else source
    )
    if isinstance(data.get("labels"), Mapping):
        checkpoint = (data.get("meta") or {}).get("checkpoint") or {}
        fitted_on = frozenset(v for v in (checkpoint.get("sha256"), checkpoint.get("exported_from_sha256")) if v)
        entries: Mapping[str, Any] = data["labels"]
    elif isinstance(source, (str, Path)):
        fitted_on, entries = frozenset(), data  # a calibration.json without its meta section
    else:
        temperature = _temperatures(data)
        return Calibration(temperature, pooled=frozenset(temperature))
    with_media = {name for name, value in entries.items() if isinstance(value, Mapping) and "media" in value}
    values = _fitted(entries, fitted_on, set(entries) - with_media)
    return dataclasses.replace(values, media=_media_calibration(entries, fitted_on))


def _fitted(entries: Mapping[str, Any], fitted_on: frozenset[str] | None, pooled: set[str]) -> Calibration:
    """Temperatures and val thresholds of calibration.json entries; `pooled` are the labels
    whose entries apply to every media type (listed when they give a value)."""
    temperature = _temperatures(entries)
    thresholds = _thresholds({
        name: value["threshold"]
        for name, value in entries.items()
        if isinstance(value, Mapping) and value.get("threshold_from") == "val" and value.get("threshold") is not None
    })
    supplied = frozenset(name for name in pooled if name in temperature or name in thresholds)
    return Calibration(temperature, thresholds, fitted_on, pooled=supplied)


def _media_calibration(entries: Mapping[str, Any], fitted_on: frozenset[str] | None) -> dict[str, Calibration]:
    """{media type: Calibration} from the `media` sections of calibration.json entries;
    {} when no entry has one (a file written before media entries)."""
    sections = {name: value["media"] for name, value in entries.items() if isinstance(value, Mapping) and "media" in value}
    if not sections:
        return {}
    for name, section in sections.items():
        if not isinstance(section, Mapping):
            raise ValueError(f"The media entry of {name} must map media types to entries, got {section!r}")
        _check_media_names(section, f"The media entry of {name}")
    out = {}
    for media in sorted(MEDIA_TYPES):
        chosen = {
            name: (sections[name][media] if name in sections else value)
            for name, value in entries.items()
            if name not in sections or media in sections[name]
        }
        missing = frozenset(name for name, section in sections.items() if media not in section)
        out[media] = dataclasses.replace(_fitted(chosen, fitted_on, set(entries) - set(sections)), no_val_clips=missing)
    return out


def _recorded_calibration(recorded: Mapping[str, Any], graph: Mapping[str, Any] | None = None) -> Calibration:
    """The calibration a model records: `temperature` and `thresholds` (in a checkpoint,
    release config.json or ONNX metadata), with the values per media type from a release's
    `media_calibration` or, for an exported graph, from `graph`, its contract's
    `calibration` record ({"media", ...}), which `OnnxModel` passes on its own so that no
    checkpoint or config.json field is ever taken for one."""
    values = Calibration(_temperatures(recorded.get("temperature") or {}), _thresholds(recorded.get("thresholds") or {}))
    if graph:  # an exported graph's flat values are those of its own media type
        _check_media_names([graph.get("media")], "The graph's calibration record")
        return dataclasses.replace(values, media={graph["media"]: values})
    media = recorded.get("media_calibration") or {}
    _check_media_names(media, "media_calibration")
    return dataclasses.replace(values, media={
        name: Calibration(_temperatures(entry.get("temperature") or {}), _thresholds(entry.get("thresholds") or {}))
        for name, entry in sorted(media.items())
    })


def _with_temperatures(record: Calibration, temperature: dict[str, float]) -> Calibration:
    """`record` with a plain mapping's temperatures on every media type; the recorded
    thresholds stay."""
    return dataclasses.replace(
        record, temperature=temperature,
        media={media: dataclasses.replace(values, temperature=temperature) for media, values in record.media.items()},
    )


def _check_media_names(names: Any, what: str) -> None:
    unknown = sorted(str(name) for name in names if name not in MEDIA_TYPES)
    if unknown:
        raise ValueError(f"{what} names unknown media types {unknown}; expected {sorted(MEDIA_TYPES)}")


def _calibrated_thresholds(
    raw: Mapping[str, float], temperature: Mapping[str, float], labels: Sequence[str], taxonomy: Taxonomy
) -> dict[str, float]:
    """Thresholds on sigmoid(logit) moved to the calibrated probability sigmoid(logit / t),
    for the binary labels among `labels` (see `BulwarkModel.thresholds`)."""
    out = {}
    for name, p in raw.items():
        if name in labels and taxonomy[name].kind == "binary":
            if 0 < p < 1:
                p = 1 / (1 + math.exp(-math.log(p / (1 - p)) / temperature.get(name, 1.0)))
            out[name] = p
    return out


def read_temperatures(source: Mapping[str, Any] | str | Path | None) -> dict[str, float]:
    """Per-label temperatures from a mapping label -> t, or from a calibration.json path.

    calibration.json is read as `read_calibration` reads it ({"meta": ..., "labels": {label:
    {"temperature": t, ...}}}, or the older flat layout), and these are the temperatures
    fitted on all val clips; `read_calibration(path).for_media(media_type).temperature`
    gives those of one media type. This does not check which weights the file was fitted
    on; `load(..., temperature=path)` does.
    """
    return {} if source is None else read_calibration(source).temperature


def _temperatures(entries: Mapping[str, Any]) -> dict[str, float]:
    """label -> t or label -> {"temperature": t, ...}; null temperatures are skipped."""
    temperatures = {}
    for name, value in entries.items():
        t = value.get("temperature") if isinstance(value, Mapping) else value
        if t is None:
            continue
        if not (math.isfinite(float(t)) and float(t) > 0):
            raise ValueError(f"Temperature for {name} must be a positive number, got {t!r}")
        temperatures[name] = float(t)
    return temperatures


def _thresholds(entries: Mapping[str, Any]) -> dict[str, float]:
    """label -> probability threshold. 0 and 1 are valid: `bulwark evaluate` picks a threshold
    among the val probabilities, which saturate to exactly 0 or 1 for logits beyond about 37."""
    thresholds = {}
    for name, value in entries.items():
        if not 0 <= float(value) <= 1:  # also false for NaN
            raise ValueError(f"Threshold for {name} must be a probability between 0 and 1, got {value!r}")
        thresholds[name] = float(value)
    return thresholds


def _check_labels(labels: Sequence[str] | None, taxonomy: Taxonomy) -> tuple[str, ...]:
    if not labels:
        return tuple(taxonomy.labels)
    unknown = [name for name in labels if name not in taxonomy.labels]
    if unknown:
        raise KeyError(f"Unknown labels {unknown}")
    return tuple(labels)


def _masked_labels(value: Any, taxonomy: Taxonomy) -> dict[str, tuple[str, ...]]:
    """{label: media types} as the training jobs record `masked_labels`; {} when not recorded."""
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"masked_labels must map labels to media types, got {value!r}")
    _check_labels(list(value), taxonomy)
    masked = {}
    for name, media in value.items():
        media = (media,) if isinstance(media, str) else tuple(media)
        unknown = sorted(set(media) - MEDIA_TYPES)
        if unknown:
            raise ValueError(f"masked_labels[{name!r}] names unknown media types {unknown}; expected {sorted(MEDIA_TYPES)}")
        masked[str(name)] = tuple(sorted(media))
    return masked


# ---- inputs and outputs --------------------------------------------------------------------


def as_clip_batch(frames: np.ndarray | torch.Tensor, size: int = FRAME_SIZE) -> np.ndarray:
    """(H, W, 3) image, (T, H, W, 3) clip or (B, T, H, W, 3) batch -> (B, T, size, size, 3) uint8.

    Frames must be RGB uint8. Frames of another size are letterboxed (aspect kept, black
    padding), which is how the training shards were made.
    """
    array = frames.detach().cpu().numpy() if isinstance(frames, torch.Tensor) else np.asarray(frames)
    if array.dtype != np.uint8:
        raise TypeError(f"Frames must be uint8 RGB, got {array.dtype}")
    if array.ndim < 3 or array.ndim > 5 or array.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3), (T, H, W, 3) or (B, T, H, W, 3), got {array.shape}")
    array = array.reshape((1,) * (5 - array.ndim) + array.shape)
    if array.shape[2:4] != (size, size):
        flat = array.reshape(-1, *array.shape[2:])
        array = np.stack([letterbox(frame, size) for frame in flat]).reshape(*array.shape[:2], size, size, 3)
    return np.ascontiguousarray(array)


def media_type_for(frames: int) -> str:
    """The media type of clips of `frames` frames: "image" for 1, as in training, else "video"."""
    return "image" if frames == 1 else "video"


def media_type_of(clips: np.ndarray) -> str:
    """The media type of a clip batch (B, T, H, W, 3); see `media_type_for`."""
    return media_type_for(clips.shape[1])


def read_image(path: str | Path, size: int = FRAME_SIZE) -> np.ndarray:
    """An image file as a 1-frame clip (1, size, size, 3) uint8 RGB."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return letterbox(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size)[None]


def read_clip(
    path: str | Path,
    *,
    start: float = 0.0,
    window_seconds: float = CLIP_SECONDS,
    clip_frames: int = CLIP_FRAMES,
    size: int = FRAME_SIZE,
    fps: float | None = None,
) -> np.ndarray:
    """One clip (T, size, size, 3) uint8 RGB from a video at `start` seconds; images give T=1.

    Frames are sampled as in `video_windows`; a clip cut short by the end of the video is
    padded with its last frame.
    """
    if Path(path).suffix.lower() in IMAGE_SUFFIXES:
        return read_image(path, size)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video {path}")
    try:
        rate = fps or float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if rate <= 0:
            raise ValueError(f"{path} reports no frame rate; pass fps")
        stride = max(1, max(1, round(window_seconds * rate)) // clip_frames)
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(start * rate))
        frames: list[np.ndarray] = []
        index = 0
        while len(frames) < clip_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append(letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size))
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path} at {start} s")
    frames += [frames[-1]] * (clip_frames - len(frames))
    return np.stack(frames)


@dataclass(frozen=True)
class Window:
    start: float          # seconds
    end: float            # seconds
    frames: np.ndarray    # (T, size, size, 3) uint8 RGB


def video_windows(
    path: str | Path,
    *,
    window_seconds: float = CLIP_SECONDS,
    hop_seconds: float = 1.0,
    clip_frames: int = CLIP_FRAMES,
    size: int = FRAME_SIZE,
    fps: float | None = None,
) -> Iterator[Window]:
    """Sliding clips over a video, cut and sampled the way the dataset adapters cut training clips.

    A window covers `window_seconds` and takes `clip_frames` frames at a fixed stride from
    its start; windows start every `hop_seconds`. As in the dataset adapters, a window is
    cut only when its whole span lies inside the video, so no window ends after the video
    does; samples that would fall past the end (at frame rates too low for `clip_frames`
    samples per span) repeat the last frame, as `load_clip` pads a short tail. A video
    shorter than one window gives one window, padded the same way and ending at the end
    of the video. Decoding streams, so long videos do not need to fit in memory.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video {path}")
    rate = fps or float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if rate <= 0:
        capture.release()
        raise ValueError(f"{path} reports no frame rate; pass fps")
    span = max(1, round(window_seconds * rate))
    stride = max(1, span // clip_frames)
    hop = max(1, round(hop_seconds * rate))
    reach = (clip_frames - 1) * stride  # offset of a window's last sampled frame
    ready = max(reach, span - 1)  # offset after which a window has its span and its samples

    buffer: dict[int, np.ndarray] = {}

    def cut(start: int, end: int) -> Window:
        samples = [buffer[i] for i in range(start, start + clip_frames * stride, stride) if i in buffer]
        samples += [samples[-1]] * (clip_frames - len(samples))
        return Window(start / rate, end / rate, np.stack(samples))

    window, index = 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            first = max(window, math.ceil((index - reach) / hop))
            if any((index - w * hop) % stride == 0 for w in range(first, index // hop + 1)):
                buffer[index] = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size)
            while window * hop + ready <= index:
                yield cut(window * hop, window * hop + span)
                window += 1
                buffer = {i: f for i, f in buffer.items() if i >= window * hop}
            index += 1
    finally:
        capture.release()
    # `index` is now the number of decoded frames. Windows whose span fits but whose last
    # samples do not exist (low frame rates) are still cut, as the dataset adapters cut them.
    while window * hop + span <= index:
        yield cut(window * hop, window * hop + span)
        window += 1
    if window == 0 and buffer:
        yield cut(0, index)


def predict_video(
    model: BulwarkModel,
    path: str | Path,
    *,
    window_seconds: float = CLIP_SECONDS,
    hop_seconds: float = 1.0,
    batch_size: int = 8,
    fps: float | None = None,
    adapters: Mapping[str, Adapter] | None = None,
    check_adapters: bool = True,
) -> Iterator[dict[str, Any]]:
    """Per-window predictions over a video: {"start", "end", "outputs", "adapters"?}.

    With the default 1 s hop this is a per-second timeline. `outputs` holds
    `model.labels_for("video")`; for an ONNX graph exported with `--frames 1`, whose
    windows are single frames, `model.labels_for("image")`. `adapters` maps a name to an
    `Adapter` trained on this model's embeddings; each adds {class: probability}. An
    adapter that records other embeddings than this model produces (see `Adapter.check`)
    is refused unless `check_adapters` is False. For an image file, use `predict_image`.
    """
    if check_adapters:
        _check_adapters(model, adapters)
    frames = model.clip_frames or CLIP_FRAMES
    windows = video_windows(
        path, window_seconds=window_seconds, hop_seconds=hop_seconds, clip_frames=frames, size=model.input_size, fps=fps
    )
    batch: list[Window] = []
    for item in windows:
        batch.append(item)
        if len(batch) == batch_size:
            yield from _predict_windows(model, batch, adapters)
            batch = []
    if batch:
        yield from _predict_windows(model, batch, adapters)


def predict_image(
    model: BulwarkModel,
    path: str | Path,
    *,
    adapters: Mapping[str, Adapter] | None = None,
    check_adapters: bool = True,
) -> dict[str, Any]:
    """Predictions for an image file, in the form of one `predict_video` window (start = end = 0).

    The image is one 1-frame clip, as the image datasets were trained, so `outputs` holds
    `model.labels_for("image")`. An ONNX graph takes images only when it was exported
    with `bulwark export --frames 1`.
    """
    if model.clip_frames not in (None, 1):
        raise ValueError(
            f"This graph takes {model.clip_frames}-frame clips, and images were trained as 1-frame clips; "
            "export a graph with --frames 1 to run it on images."
        )
    if check_adapters:
        _check_adapters(model, adapters)
    return next(_predict_windows(model, [Window(0.0, 0.0, read_image(path, model.input_size))], adapters))


def _check_adapters(model: BulwarkModel, adapters: Mapping[str, Adapter] | None) -> None:
    for name, adapter in (adapters or {}).items():
        try:
            adapter.check(model.embedding_source(adapter.config.embedding))
        except ValueError as error:
            raise ValueError(f"Adapter {name!r}: {error}") from None


def _predict_windows(
    model: BulwarkModel, windows: list[Window], adapters: Mapping[str, Adapter] | None
) -> Iterator[dict[str, Any]]:
    result = model.run(np.stack([w.frames for w in windows]))
    rows = model.to_dicts(result.outputs) if result.outputs else [{} for _ in windows]  # every label masked
    extra: dict[str, list[dict[str, float]]] = {}
    for name, adapter in (adapters or {}).items():
        probs = adapter.predict_proba(result.embedding(adapter.config.embedding))
        extra[name] = [dict(zip(adapter.config.classes, map(float, row), strict=True)) for row in probs]
    for i, (window, row) in enumerate(zip(windows, rows, strict=True)):
        entry: dict[str, Any] = {"start": round(window.start, 3), "end": round(window.end, 3), "outputs": row}
        if extra:
            entry["adapters"] = {name: values[i] for name, values in extra.items()}
        yield entry


class RollingMean:
    """Mean of one scalar output over the last `seconds` of a timeline, such as PERCLOS.

    The taxonomy's `perclos` is a per-clip fraction of frames with the eyes closed. PERCLOS
    as the drowsiness literature uses it averages that fraction over a minute or more,
    which is what this computes from consecutive clips:

        perclos = RollingMean("perclos", seconds=60)
        for window in predict_video(model, "cabin.mp4"):
            value = perclos.update(window)   # None until 60 s of consecutive windows

    `update` takes the windows of `predict_video` in time order and returns the mean of
    `outputs[label]` over the windows that start at most `seconds` before the current one
    ends. It returns None until consecutive windows cover `seconds`. A window that starts
    after the previous one ended is a gap (missing video, or a hop longer than the
    window), and the average starts again from it, so it never spans unseen time.
    Overlapping windows (the default 2 s windows every 1 s) weight every moment equally
    except the first and last second of the span. The current window always counts, so a
    span shorter than one window gives that window's own value; choose `seconds` of at
    least one window.
    """

    def __init__(self, label: str = "perclos", seconds: float = PERCLOS_SECONDS) -> None:
        if not seconds > 0:
            raise ValueError(f"seconds must be positive, got {seconds}")
        self.label, self.seconds = label, float(seconds)
        self._recent: deque[tuple[float, float]] = deque()  # (start, value) of the windows in the span
        self._since = 0.0  # start of the current run of consecutive windows
        self._end: float | None = None

    def update(self, window: Mapping[str, Any]) -> float | None:
        start, end = float(window["start"]), float(window["end"])
        if self.label not in window["outputs"]:
            raise KeyError(f"{self.label} is not among the window's outputs; check model.labels_for('video')")
        value = window["outputs"][self.label]
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.label} is not a scalar output; RollingMean averages binary or regression labels")
        if self._end is None or start > self._end + 1e-3 or start < self._recent[-1][0]:
            self._recent.clear()  # first window, a gap, or windows out of order
            self._since = start
        self._recent.append((start, float(value)))
        self._end = end
        while len(self._recent) > 1 and self._recent[0][0] < end - self.seconds - 1e-3:
            self._recent.popleft()  # the current window always stays, even when longer than `seconds`
        if end - self._since < self.seconds - 1e-3:
            return None
        return float(np.mean([v for _, v in self._recent]))


def outputs_to_dicts(outputs: Mapping[str, np.ndarray], taxonomy: Taxonomy | None = None) -> list[dict[str, Any]]:
    """Batched activated outputs -> one dict per clip, with class names for categorical labels."""
    taxonomy = taxonomy or default_taxonomy()
    count = len(next(iter(outputs.values()))) if outputs else 0
    rows: list[dict[str, Any]] = [{} for _ in range(count)]
    for name, values in outputs.items():
        spec = taxonomy[name]
        for row, value in zip(rows, values, strict=True):
            if spec.kind == "categorical":
                row[name] = {cls: float(p) for cls, p in zip(spec.classes, value, strict=True)}
            else:
                row[name] = float(value)
    return rows
