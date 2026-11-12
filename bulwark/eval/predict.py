"""Model outputs per split, kept as `<split>.npz` files of clip ids and raw logits.

Every model is scored through these files, so frozen heads, Bulwark-B1 (LoRA), Bulwark-E1
and the exported ONNX graph all go through one report (`bulwark.eval.report`):

    heads   heads checkpoint run on cached summary tokens (`bulwark train cache`)
    logits  `logit__*` arrays already in a features dir (cached with `init` = a B1 checkpoint)
    b1      B1 checkpoint (LoRA + heads) streamed over the tar shards
    e1      E1 checkpoint (distilled student) streamed over the tar shards
    onnx    exported E1 graph streamed over the tar shards with onnxruntime
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

from bulwark.data.shards import read_shard
from bulwark.data.taxonomy import Taxonomy, default_taxonomy
from bulwark.eval.provenance import features_init, features_meta, sha256_file
from bulwark.model.bulwark import load_payload

console = Console()
MODEL_KINDS = ("auto", "heads", "logits", "b1", "e1", "onnx")
SHARD_KINDS = ("b1", "e1", "onnx")


@dataclass(slots=True)
class Predictions:
    """Raw logits per label for a list of clips (rows aligned with `clip_ids`)."""

    clip_ids: list[str]
    logits: dict[str, np.ndarray]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        # Any: see cache_features._write_shard (savez's `allow_pickle` in numpy's stubs).
        logits: dict[str, Any] = {f"logit__{k}": v for k, v in self.logits.items()}
        np.savez(tmp, clip_ids=np.array(self.clip_ids), **logits)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> Predictions:
        with np.load(path, allow_pickle=False) as data:
            logits = {k.removeprefix("logit__"): data[k] for k in data.files if k.startswith("logit__")}
            return cls([str(c) for c in data["clip_ids"]], logits)

    def select(self, clip_ids: Iterable[str]) -> Predictions:
        row = {c: i for i, c in enumerate(self.clip_ids)}
        keep = [c for c in clip_ids if c in row]
        index = np.array([row[c] for c in keep], dtype=np.int64)
        return Predictions(keep, {k: v[index] for k, v in self.logits.items()})


def read_prediction_meta(directory: Path) -> dict[str, Any]:
    path = directory / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{directory} has no meta.json; write predictions with `bulwark evaluate`")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def load_predictions(directory: Path) -> dict[str, Predictions]:
    """{split: Predictions} for exactly the files meta.json lists, each checked by sha256.

    A file that is not listed, or whose hash differs, belongs to another run; it is refused
    rather than scored under this run's meta.
    """
    files: dict[str, dict[str, Any]] = read_prediction_meta(directory).get("files") or {}
    if not files:
        raise FileNotFoundError(f"{directory}/meta.json lists no prediction files")
    present = {p.name for p in directory.glob("*.npz") if ".tmp" not in p.name}
    extra = present - {entry["file"] for entry in files.values()}
    if extra:
        raise ValueError(f"{directory} holds predictions meta.json does not list: {sorted(extra)}; they come from another run")
    found = {}
    for split, entry in files.items():
        path = directory / entry["file"]
        if not path.exists() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path} is missing or differs from the file meta.json records")
        found[split] = Predictions.load(path)
    return found


def write_predictions(predictions: dict[str, Predictions], directory: Path, meta: dict[str, Any]) -> Path:
    """Replace the directory's predictions with these splits; meta.json lists each file and its hash."""
    directory.mkdir(parents=True, exist_ok=True)
    for old in [*directory.glob("*.npz"), directory / "meta.json"]:
        old.unlink(missing_ok=True)  # files of an earlier run must not be read with this run's meta
    files = {}
    for split, preds in predictions.items():
        path = preds.save(directory / f"{split}.npz")
        files[split] = {"file": path.name, "sha256": sha256_file(path), "clips": len(preds.clip_ids)}
    (directory / "meta.json").write_text(json.dumps({**meta, "files": files}, indent=2, default=str), encoding="utf-8")
    return directory


def detect_kind(checkpoint: Path | None) -> str:
    """Model kind from the checkpoint file: ONNX graph, E1, B1 (LoRA) or heads-only."""
    if checkpoint is None:
        return "logits"
    if checkpoint.suffix == ".onnx":
        return "onnx"
    keys = list(load_payload(checkpoint)["model"])
    if any(k.startswith("frame_encoder.") for k in keys):
        return "e1"
    if any(k.startswith(("heads.", "encoder.")) for k in keys):
        return "b1"
    return "heads"


# ---- cached features ----------------------------------------------------------------------


def check_features_source(kind: str, checkpoint: Path | None, features: Path) -> None:
    """Refuse a features dir whose recorded provenance contradicts the model being scored.

    logits: the cached logits must come from `checkpoint` (when one is given). heads: the
    tokens must come from the frozen encoder, with no fine-tuned `init` behind them.
    """
    meta = features_meta(features)
    if meta is None:
        return  # cached before features recorded provenance; the report says so
    init = features_init(meta)
    if kind == "heads" and (init or (meta.get("run") or {}).get("with_logits")):
        raise ValueError(f"{features} was cached with a fine-tuned init; score frozen heads on frozen-encoder features.")
    layout = (meta.get("run") or {}).get("logit_layout_sha256")
    if kind == "logits" and layout:
        from bulwark.train.cache_features import logit_layout_sha256

        if layout != logit_layout_sha256():
            raise ValueError(
                f"The logits in {features} follow another taxonomy (labels, heads, classes or ranges differ "
                "from taxonomy.yaml); re-cache them with this taxonomy."
            )
    if kind == "logits" and checkpoint is not None:
        recorded = (init or {}).get("sha256")
        actual = sha256_file(checkpoint)
        if recorded != actual:
            raise ValueError(
                f"The logits in {features} were cached from a checkpoint with sha256 {str(recorded)[:12]}, "
                f"not {checkpoint} ({actual[:12]}); pass the checkpoint they were cached with, or none."
            )


def predict_features(kind: str, checkpoint: Path | None, features: Path, wanted: set[str], taxonomy: Taxonomy | None = None) -> Predictions:
    """Heads on cached tokens, or the cached `logit__*` arrays themselves."""
    from bulwark.train.common import load_feature_dir

    taxonomy = taxonomy or default_taxonomy()
    check_features_source(kind, checkpoint, features)
    clip_ids, tokens, cached = load_feature_dir(features)
    rows = np.array([i for i, c in enumerate(clip_ids) if c in wanted], dtype=np.int64)
    ids = [clip_ids[i] for i in rows]
    if kind == "logits":
        if not cached:
            raise ValueError(f"{features} has no logit__* arrays; cache it with init=<B1 checkpoint> first.")
        return Predictions(ids, {k: v[rows].astype(np.float32) for k, v in cached.items()})

    from bulwark.model.heads import BulwarkHeads

    if checkpoint is None:
        raise ValueError("A heads checkpoint is required")
    state = load_payload(checkpoint)["model"]
    if any(k.startswith("heads.") for k in state):
        raise ValueError(f"{checkpoint} is a B1 checkpoint; evaluate it with --model b1 (or --model logits on its cached features).")
    if cached:
        # Tokens in a dir with logits came from a fine-tuned encoder; frozen heads never saw them.
        raise ValueError(f"{features} was cached with a fine-tuned encoder; score frozen heads on frozen-encoder features.")
    heads = BulwarkHeads(tokens.shape[-1], taxonomy, input_tokens=tokens.shape[1])
    heads.load_state_dict(state)
    heads.eval()
    outputs: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(rows), 512):
            batch = torch.from_numpy(tokens[rows[start : start + 512]].astype(np.float32))
            for name, value in heads(batch).items():
                outputs.setdefault(name, []).append(value.numpy())
    return Predictions(ids, {k: np.concatenate(v) for k, v in outputs.items()})


# ---- shards -------------------------------------------------------------------------------


def shard_batches(shards: Iterable[Path], wanted: set[str], batch_size: int) -> Iterator[tuple[list[str], np.ndarray]]:
    """Batches of equal clip length in shard order (images and videos take different paths)."""
    pending: dict[int, list[tuple[str, np.ndarray]]] = {}
    for path in shards:
        for clip_id, frames in read_shard(path, only=wanted):
            bucket = pending.setdefault(frames.shape[0], [])
            bucket.append((clip_id, frames))
            if len(bucket) >= batch_size:
                yield [c for c, _ in bucket], np.stack([f for _, f in bucket])
                bucket.clear()
    for bucket in pending.values():
        if bucket:
            yield [c for c, _ in bucket], np.stack([f for _, f in bucket])


class _OnnxModel:
    """onnxruntime session with the same call shape as the PyTorch models (raw logits)."""

    def __init__(self, path: Path, label_names: list[str]) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.outputs = [o.name for o in self.session.get_outputs()]
        missing = set(label_names) - set(self.outputs)
        if missing:
            raise ValueError(f"{path} lacks outputs for {sorted(missing)}; re-export it with this taxonomy.")
        frames = self.input.shape[1]
        # `bulwark export` fixes the clip length; a graph with a symbolic time axis takes any.
        self.frames = frames if isinstance(frames, int) else None
        self.size = int(self.input.shape[2])

    def accepts(self, frames: np.ndarray) -> bool:
        return (self.frames is None or frames.shape[1] == self.frames) and frames.shape[2] == self.size

    def __call__(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        return dict(zip(self.outputs, self.session.run(self.outputs, {self.input.name: frames}), strict=True))


def build_model(kind: str, checkpoint: Path, taxonomy: Taxonomy, device: torch.device) -> Any:
    """B1, E1 or ONNX model returning raw logits per label.

    B1 and E1 checkpoints are read by `bulwark.inference.load`, the loader the public API
    uses (checkpoint detection, LoRA layers rebuilt from the checkpoint's own keys and
    merged); evaluation calls the underlying module, since it scores raw logits rather
    than the API's activated outputs.
    """
    if kind == "onnx":
        return _OnnxModel(checkpoint, list(taxonomy.labels))
    if kind not in ("b1", "e1"):
        raise ValueError(f"Not a shard model kind: {kind}")
    from bulwark.inference import TorchModel, load

    model = load(checkpoint, device=str(device))
    family = "e1" if model.kind == "e1" else "b1"  # b1-frozen: heads on the frozen encoder
    if family != kind or not isinstance(model, TorchModel):
        raise ValueError(f"{checkpoint} is a {model.kind} checkpoint, not {kind}; pass --model {family} or --model auto.")
    return model.module


def predict_shards(
    kind: str,
    checkpoint: Path,
    shards: Path,
    wanted: set[str],
    *,
    batch_size: int = 16,
    device: str = "auto",
    precision: str = "auto",
    taxonomy: Taxonomy | None = None,
) -> tuple[Predictions, dict[str, int]]:
    """Stream the wanted clips through a model. Returns predictions and skip counts."""
    from bulwark.train.common import autocast_dtype, pick_device

    taxonomy = taxonomy or default_taxonomy()
    torch_device = pick_device(device)
    model = build_model(kind, checkpoint, taxonomy, torch_device)
    dtype = autocast_dtype(torch_device, precision) if kind == "b1" else None
    paths = sorted(shards.glob("*.tar"))
    if not paths:
        raise FileNotFoundError(f"No *.tar shards in {shards}")
    clip_ids: list[str] = []
    outputs: dict[str, list[np.ndarray]] = {}
    skipped = {"frame_shape": 0}
    with torch.no_grad():
        for ids, frames in shard_batches(paths, wanted, batch_size):
            if kind == "onnx":
                if not model.accepts(frames):
                    skipped["frame_shape"] += len(ids)
                    continue
                result = model(frames)
            else:
                with torch.autocast(torch_device.type, dtype=dtype, enabled=dtype is not None):
                    result = {k: v.float().cpu().numpy() for k, v in model(torch.from_numpy(frames).to(torch_device)).items()}
            clip_ids.extend(ids)
            for name in taxonomy.labels:
                outputs.setdefault(name, []).append(np.asarray(result[name], dtype=np.float32))
            if len(clip_ids) % (batch_size * 50) < len(ids):
                console.print(f"{kind}: {len(clip_ids)}/{len(wanted)} clips")
    skipped["not_in_shards"] = len(wanted) - len(clip_ids) - skipped["frame_shape"]
    logits = {k: np.concatenate(v) for k, v in outputs.items()} if outputs else {}
    return Predictions(clip_ids, logits), skipped


def sample_ids(clip_ids: Iterable[str], limit: int) -> list[str]:
    """A fixed pseudo-random subset (by hash of the clip id), for quick smoke checks."""
    ordered = sorted(clip_ids, key=lambda c: hashlib.sha1(c.encode()).hexdigest())
    return ordered[:limit] if limit else ordered
