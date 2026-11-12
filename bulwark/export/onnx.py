"""Export Bulwark-E1 to ONNX, check the graph against PyTorch, and benchmark it.

Contract of the exported graph (also stored in its metadata; see `read_contract`):

    input    `frames`: uint8 RGB, shape (batch, T, S, S, 3). T (default 16) and S (default 256)
             are fixed at export; batch is dynamic. Each frame is letterboxed to S x S: scaled
             to fit keeping its aspect ratio (cv2.INTER_AREA when shrinking, INTER_LINEAR when
             enlarging) and centred on black padding, exactly as `bulwark shard` stores frames.
             Resizing to the student's own resolution and ImageNet normalisation run inside
             the graph.
    outputs  one tensor per taxonomy label, in taxonomy order, all raw logits:
             binary (batch,) -> sigmoid; categorical (batch, C) -> softmax over the label's
             classes; regression (batch,) -> low + (high - low) * sigmoid, with (low, high)
             the label's range. With `embeddings=True` two more follow: `tokens`, the
             per-frame features the heads read (batch, T, D), and `state`, the heads' state
             embedding (batch, 256), which `bulwark.inference.OnnxModel.embed` returns.
    time     T stays fixed: the legacy TorchScript exporter records the attention's sequence
             length in the heads as a constant, so a graph traced at one T fails at another.
             Still images (1-frame clips in training) need a graph exported with frames=1.
    window   every call encodes all T frames. Per-frame features are not reused between
             overlapping windows, so a sliding window costs one full call per hop.
    temporal training clips are 16 frames taken every k-th source frame, with
             k = max(1, round(2.0 * fps) // 16) (`bulwark.data.sources.base`): at 30 fps every
             3rd frame, so a clip spans about 1.5 s; at 15 fps every frame, about 1.07 s.
             16 consecutive frames of a 30 fps camera span about 0.5 s, a third of a training
             clip at that rate.
    domain   training frames were letterboxed as above and stored as JPEG at quality 90 in
             the shards (`bulwark shard`), so the model saw JPEG-compressed frames.

The metadata also carries the weights licence, the model card, the model release, the
package version and the checkpoint's training record (datasets with their terms and how
they reached the weights: own training clips, or an upstream teacher or init checkpoint;
`commercial_only`, code commit, teacher), so the terms travel with the file.
`supervised_labels`, the outputs `bulwark.inference.OnnxModel` returns by default, are the
checkpoint's own record of the labels it trained (training leaves out labels it masks on
every media type, such as the v0 configs' `mask_labels`). Given the training manifest,
`label_status` adds what the train split held for each output: `ground_truth` (trained,
with ground truth), `teacher_only` (trained, teacher pseudo-labels only), `untrained` (no
training target) and `not_trained_by_checkpoint` (targets in the manifest, but not in the
checkpoint's `supervised_labels`). Only a checkpoint that records no `supervised_labels`
takes them from the manifest: every output with a training target, masked or not
(`supervised_labels_from` says which). `masked_labels` ({label: media types}) repeats the
checkpoint's record of the labels training masked on a media type, so a reader can
withhold those outputs on that media. Given the
calibration.json of `bulwark evaluate` for these weights, it holds the val-fitted
`temperature`s and the val-tuned `thresholds` of binary labels (on the uncalibrated
probability, as in calibration.json), one per label: those fitted for the graph's media
type (video for T > 1, image for T = 1), and `calibration` records which media type and
the labels that had no entry of their own for it; without one, the temperatures the
checkpoint records.

`bulwark_contract` holds all of this as one JSON record. The other metadata_props entries
repeat its main fields for tools that read single keys: text values are stored as they
are, other values JSON-encoded, and entries without a value are left out.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from bulwark.data.taxonomy import Taxonomy, default_taxonomy
from bulwark.eval.provenance import (
    checkpoint_meta,
    git_state,
    public_path,
    sha256_file,
    training_scope,
)
from bulwark.export.release import (
    MODEL_RELEASE,
    RESEARCH_LICENCE,
    check_release,
    model_card_url,
    record_from_provenance,
    research_only,
)
from bulwark.model.bulwark import BulwarkEdge, load_into, load_payload

CONTRACT_KEY = "bulwark_contract"
MODEL_CARD = model_card_url("e1")  # the E1 model repository, which hosts the graphs
PARITY_TOLERANCE = 1e-3
OPSET = 17
EMBEDDING_OUTPUTS = ("tokens", "state")
# Sampling and storage of the training clips (bulwark.data.sources.base, bulwark shard).
TRAIN_CLIP_FRAMES = 16
TRAIN_CLIP_SECONDS = 2.0
SHARD_JPEG_QUALITY = 90


class _Flat(nn.Module):
    """ONNX wants a fixed output list: logits in taxonomy order, then optionally the heads'
    input tokens and state embedding (read with hooks, so the model's own forward runs once)."""

    def __init__(self, model: BulwarkEdge, embeddings: bool = False) -> None:
        super().__init__()
        self.model = model
        self.embeddings = embeddings

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not self.embeddings:
            outputs = self.model(frames)
            return tuple(outputs[name] for name in self.model.heads.label_names)
        heads = self.model.heads
        captured: dict[str, torch.Tensor] = {}
        hooks = [
            heads.register_forward_pre_hook(lambda _m, args: captured.__setitem__("tokens", args[0])),
            heads.out.register_forward_pre_hook(lambda _m, args: captured.__setitem__("state", args[0])),
        ]
        try:
            outputs = self.model(frames)
        finally:
            for hook in hooks:
                hook.remove()
        return (*(outputs[name] for name in heads.label_names), captured["tokens"], captured["state"])

    @property
    def output_names(self) -> list[str]:
        return [*self.model.heads.label_names, *(EMBEDDING_OUTPUTS if self.embeddings else ())]


def load_edge(checkpoint: Path, *, size: int | None = None, backbone: str | None = None) -> tuple[BulwarkEdge, dict[str, Any]]:
    """E1 from a distillation checkpoint, at the image size it was trained at unless overridden."""
    payload = load_payload(checkpoint)
    model = BulwarkEdge(
        default_taxonomy(),
        backbone=backbone or payload.get("backbone", "efficientvit_b1.r224_in1k"),
        pretrained=False,
        img_size=size or int(payload.get("img_size", 224)),
    )
    load_into(model, checkpoint, strict=True, payload=payload)
    return model.eval(), payload


def package_version() -> str | None:
    try:
        return metadata.version("inferensys-bulwark")
    except metadata.PackageNotFoundError:
        return None


def contract(model: BulwarkEdge, frames: int, input_size: int, *, embeddings: bool = False) -> dict[str, Any]:
    taxonomy: Taxonomy = model.taxonomy
    outputs: list[dict[str, Any]] = [
        {
            "name": name,
            "kind": taxonomy[name].kind,
            "shape": ["batch", taxonomy[name].width] if taxonomy[name].kind == "categorical" else ["batch"],
            **({"classes": list(taxonomy[name].classes)} if taxonomy[name].kind == "categorical" else {}),
            **({"range": list(taxonomy[name].value_range)} if taxonomy[name].kind == "regression" else {}),
        }
        for name in model.heads.label_names
    ]
    if embeddings:
        outputs += [
            {"name": "tokens", "kind": "embedding", "shape": ["batch", frames, model.heads.in_dim],
             "description": "per-frame features the heads read (before their input standardisation)"},
            {"name": "state", "kind": "embedding", "shape": ["batch", model.heads.width],
             "description": "the heads' state embedding (CLS token after the last layer norm)"},
        ]
    return {
        "input": {
            "name": "frames",
            "dtype": "uint8",
            "layout": "batch, time, height, width, channel (RGB)",
            "shape": ["batch", frames, input_size, input_size, 3],
            "preprocessing": (
                f"letterbox each frame to {input_size}x{input_size}: keep aspect ratio, cv2.INTER_AREA "
                "when shrinking (INTER_LINEAR when enlarging), centre on black padding"
            ),
            "time_axis": (
                f"fixed at {frames} frames: the exporter records the attention's sequence length as a "
                "constant; export with frames=1 for still images"
            ),
        },
        "model_img_size": model.img_size,
        "outputs": outputs,
        "activation": {
            "binary": "sigmoid(logit)",
            "categorical": "softmax(logits) over classes",
            "regression": "low + (high - low) * sigmoid(logit)",
        },
        "window": f"each call encodes all {frames} frames; overlapping windows are not cached",
        "temporal": {
            "training_clip": (
                f"{TRAIN_CLIP_FRAMES} frames taken every k-th source frame, "
                f"k = max(1, round({TRAIN_CLIP_SECONDS} * fps) // {TRAIN_CLIP_FRAMES})"
            ),
            "examples": "30 fps: every 3rd frame, a clip spans about 1.5 s; 15 fps: every frame, about 1.07 s",
            "note": (
                "feed frames sampled the same way; 16 consecutive frames of a 30 fps camera span about "
                "0.5 s, a third of a training clip at that rate. Image datasets were trained as single frames."
            ),
        },
        "training_domain": (
            f"frames letterboxed as in `preprocessing` and stored as JPEG quality {SHARD_JPEG_QUALITY} "
            "in the training shards, so the model saw JPEG-compressed frames"
        ),
        "taxonomy_version": taxonomy.version,
    }


def label_status(payload: dict[str, Any], manifest: Path, taxonomy: Taxonomy | None = None) -> dict[str, Any]:
    """Which outputs had ground truth, only teacher labels, or no targets in the checkpoint's
    training split (the manifest's train split, narrowed by the scope the checkpoint records).

    When the checkpoint records `supervised_labels`, `ground_truth` and `teacher_only` hold
    only those labels, and labels with targets that it did not train (masked on every media
    type) go to `not_trained_by_checkpoint`.
    """
    from bulwark.data.manifest import read_manifest
    from bulwark.eval.report import training_records, training_support

    taxonomy = taxonomy or default_taxonomy()
    scope = training_scope(checkpoint_meta(payload))
    support = training_support(training_records(read_manifest(manifest), scope), taxonomy)
    trained = set(payload.get("supervised_labels") or taxonomy.labels)
    digest = sha256_file(manifest)
    recorded = (payload.get("provenance") or {}).get("manifest_sha256")
    status: dict[str, Any] = {
        "ground_truth": [n for n, e in support.items() if e["status"] == "gt" and n in trained],
        "teacher_only": [n for n, e in support.items() if e["status"] == "teacher_only" and n in trained],
        "untrained": [n for n, e in support.items() if e["status"] == "none"],
    }
    if payload.get("supervised_labels"):
        status["not_trained_by_checkpoint"] = [n for n, e in support.items() if e["status"] != "none" and n not in trained]
    status["from"] = {
        "manifest_sha256": digest,
        "matches_training_manifest": None if recorded is None else recorded == digest,
        "scope": scope or "not recorded; the whole train split",
    }
    return status


def export_edge(
    checkpoint: Path,
    out: Path,
    *,
    frames: int = 16,
    size: int | None = None,
    input_size: int = 256,
    backbone: str | None = None,
    parity_frames: np.ndarray | None = None,
    tolerance: float = PARITY_TOLERANCE,
    embeddings: bool = False,
    manifest: Path | None = None,
    temperature: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    calibration: Mapping[str, Any] | None = None,
    licence: str = RESEARCH_LICENCE,
) -> Path:
    """Export E1, check ONNX Runtime against PyTorch on a fixed batch, and store the contract.

    `size` is the student's resolution (default: the `img_size` saved at distillation);
    `input_size` is the frame size the graph takes (default 256, the size of the shards).
    `parity_frames` (B, frames, input_size, input_size, 3) uint8 should be real clips; a
    seeded random batch is used without them. Raises ValueError if they have another shape
    and AssertionError if outputs differ by more than `tolerance`. The graph is written to a
    temporary file and moved to `out` only once it passes, so a failed export leaves nothing
    at `out` that a later evaluation or release could pick up.

    `embeddings` adds the `tokens` and `state` outputs. It defaults to False here (label
    logits only), while `bulwark export` adds them unless given `--no-embeddings`.
    `supervised_labels` and
    `masked_labels` are stored as the checkpoint records them. `manifest` (the training
    manifest) adds the label status lists (see `label_status`); for a checkpoint that
    records no `supervised_labels`, it also gives them: every output with a training
    target in the manifest. `temperature`
    ({label: t}) adds the val-fitted temperatures; without it, those the checkpoint
    records. `thresholds` ({label: p}) adds val-tuned thresholds of binary labels. Both
    must come from a calibration.json fitted on this checkpoint, and for the graph's media
    type (`bulwark.inference.media_type_for(frames)`):
    `bulwark.inference.read_calibration(path).for_media(media_type)` gives them, and the
    caller checks the weights (`bulwark export` does both). `calibration` records that
    choice in the contract: {"media": the media type, "pooled": labels whose values were
    fitted on all val clips (a file without per-media-type entries), "no_val_clips": labels
    whose val clips held none of that media type}; its media type must be the graph's.
    `licence` is the weights licence; one that does not restrict use to non-commercial
    research (`bulwark.export.release.research_only`) is refused unless the checkpoint
    passes `check_release(..., commercial=True)`.
    """
    import onnx

    expected = (frames, input_size, input_size, 3)
    if parity_frames is not None and tuple(parity_frames.shape[1:]) != expected:
        raise ValueError(f"Parity clips have shape {tuple(parity_frames.shape[1:])}; the graph takes {expected}")
    model, payload = load_edge(checkpoint, size=size, backbone=backbone)
    record = record_from_provenance(payload.get("provenance") or {}, checkpoint.name)
    commercial = not research_only(licence)
    release = check_release(record, commercial=commercial)
    if commercial and not release.ok:
        raise ValueError(
            f"Licence {licence!r} does not restrict use to non-commercial research, so it needs weights cleared "
            f"for commercial release: {' '.join(release.problems)}"
        )
    status = label_status(payload, manifest, model.taxonomy) if manifest is not None else None
    supervised, supervised_from = payload.get("supervised_labels"), "checkpoint"
    if not supervised and status is not None:
        with_targets = {*status["ground_truth"], *status["teacher_only"]}
        supervised = [n for n in model.taxonomy.labels if n in with_targets]
        supervised_from = "training manifest (the checkpoint records none)"
    if temperature is None and payload.get("temperature"):
        from bulwark.inference import read_temperatures

        temperature = read_temperatures(payload["temperature"])
    masked = payload.get("masked_labels") or {}
    calibrated: dict[str, Any] | None = None
    if calibration is not None:
        from bulwark.inference import media_type_for

        if calibration.get("media") != media_type_for(frames):
            raise ValueError(
                f"The calibration is for {calibration.get('media')!r} clips, but a {frames}-frame graph takes "
                f"{media_type_for(frames)} clips; pass the values of that media type"
            )
        calibrated = {
            "media": calibration["media"],
            "pooled": sorted(calibration.get("pooled") or ()),
            "no_val_clips": sorted(calibration.get("no_val_clips") or ()),
        }
    named = (
        ("Temperatures", temperature or {}), ("Thresholds", thresholds or {}),
        ("Supervised labels", supervised or ()), ("Masked labels", masked),
        ("Calibration records", [*(calibrated or {}).get("pooled", ()), *(calibrated or {}).get("no_val_clips", ())]),
    )
    for what, names in named:
        unknown = sorted(set(names) - set(model.taxonomy.labels))
        if unknown:
            raise ValueError(f"{what} for labels the taxonomy does not have: {unknown}")

    flat = _Flat(model, embeddings).eval()  # export restores the wrapper's mode, so it must start in eval
    names = flat.output_names
    dummy = torch.zeros(1, *expected, dtype=torch.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        torch.onnx.export(
            flat,
            (dummy,),
            str(tmp),
            input_names=["frames"],
            output_names=names,
            dynamic_axes={"frames": {0: "batch"}, **{name: {0: "batch"} for name in names}},
            opset_version=OPSET,
            dynamo=False,
        )
        model.eval()

        source = "real clips"
        if parity_frames is None:
            source = "seeded random frames"
            parity_frames = np.random.default_rng(0).integers(0, 256, (2, *expected), dtype=np.uint8)
        diffs = parity_diffs(flat, tmp, parity_frames)
        checked = {"max_abs_logit_diff": max(diffs[name] for name in model.heads.label_names)}
        if embeddings:
            # Tokens are raw backbone features (large values), so they are compared relative to their scale.
            checked.update(max_abs_state_diff=diffs["state"], max_rel_token_diff=diffs["tokens_relative"])
        worst = max(checked.values())
        if not worst <= tolerance:
            raise AssertionError(f"ONNX output differs from PyTorch by {worst:.2e} (> {tolerance:.0e}); nothing was written to {out}")

        graph = onnx.load(str(tmp))
        terms = contract(model, frames, input_size, embeddings=embeddings)
        training = {
            "commercial_only": record.commercial_only,
            "datasets": release.summary()["datasets"],
            "code_commit": record.commit,
            "code_dirty": record.dirty,
            "teacher_init_sha256": (((payload.get("provenance") or {}).get("inputs") or {}).get("teacher") or {}).get("init_sha256"),
        }
        terms.update(
            parity={**checked, "tolerance": tolerance, "batch": f"{len(parity_frames)} {source}"},
            checkpoint={"path": public_path(checkpoint), "sha256": sha256_file(checkpoint), **checkpoint_meta(payload)},
            exporter={"torch": torch.__version__, "opset": OPSET, "git": git_state(), "bulwark": package_version()},
            licence=licence,
            model_card=MODEL_CARD,
            model_release=MODEL_RELEASE,
            training=training,
            release_check=release.summary(),
        )
        if status is not None:
            terms["label_status"] = status
        if supervised:
            terms.update(supervised_labels=list(supervised), supervised_labels_from=supervised_from)
        if masked:
            terms["masked_labels"] = dict(masked)
        if temperature:
            terms["temperature"] = dict(temperature)
        if thresholds:
            terms["thresholds"] = dict(thresholds)
        if calibrated is not None:
            terms["calibration"] = calibrated
        entries: dict[str, Any] = {
            CONTRACT_KEY: terms,
            "frames": frames,
            "input_size": input_size,
            "img_size": model.img_size,
            "license": licence,
            "model_card": MODEL_CARD,
            "model_release": MODEL_RELEASE,
            "bulwark_version": package_version(),
            "commercial_only": training["commercial_only"],
            "training_datasets": training["datasets"],
            "trained_with_commit": training["code_commit"],
            "teacher_init_sha256": training["teacher_init_sha256"],
            "release_cleared": release.ok,
        }
        if status is not None:
            entries.update({k: v for k, v in status.items() if k != "from"})
        if supervised:
            entries["supervised_labels"] = list(supervised)
        if masked:
            entries["masked_labels"] = dict(masked)
        if temperature:
            entries["temperature"] = dict(temperature)
        if thresholds:
            entries["thresholds"] = dict(thresholds)
        del graph.metadata_props[:]
        for key, value in entries.items():
            if value is None:
                continue  # not recorded; the contract holds the full record
            prop = graph.metadata_props.add()
            prop.key, prop.value = key, value if isinstance(value, str) else json.dumps(value)
        graph.doc_string = f"Bulwark-E1 {MODEL_RELEASE}. Weights licence: {licence}. Model card: {MODEL_CARD}"
        onnx.save(graph, str(tmp))
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


@torch.no_grad()
def parity_diffs(flat: _Flat, onnx_path: Path, frames: np.ndarray) -> dict[str, float]:
    """Largest absolute difference per output between PyTorch and ONNX Runtime on `frames`
    (plus `tokens_relative`, the token difference over the tokens' largest magnitude)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    reference = dict(zip(flat.output_names, flat.eval()(torch.from_numpy(frames)), strict=True))
    results = dict(zip(flat.output_names, session.run(flat.output_names, {"frames": frames}), strict=True))
    diffs = {name: float(np.abs(results[name] - reference[name].numpy()).max()) for name in flat.output_names}
    if "tokens" in diffs:
        diffs["tokens_relative"] = diffs["tokens"] / max(1.0, float(reference["tokens"].abs().max()))
    return diffs


def parity(model: BulwarkEdge, onnx_path: Path, frames: np.ndarray) -> float:
    """Largest absolute logit difference between PyTorch and ONNX Runtime on `frames`
    (for a graph with logit outputs only)."""
    return max(parity_diffs(_Flat(model), onnx_path, frames).values())


def read_contract(onnx_path: Path) -> dict[str, Any]:
    """The input/output contract and provenance stored in an exported graph."""
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False)
    props = {p.key: p.value for p in graph.metadata_props}
    return json.loads(props[CONTRACT_KEY]) if CONTRACT_KEY in props else {}


def cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


def benchmark(onnx_path: Path, *, warmup: int = 10, runs: int = 100, threads: int = 0, batch: int = 1) -> dict[str, Any]:
    """CPU latency of one call (model only: decoding and letterboxing are not included).

    threads=0 lets ONNX Runtime use every core. Each run is timed on its own; the result
    gives percentiles, the runtime version, the CPU and the input shape.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    shape = [batch, *(int(d) for d in session.get_inputs()[0].shape[1:])]
    frames = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    for _ in range(warmup):
        session.run(None, {"frames": frames})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {"frames": frames})
        times.append((time.perf_counter() - start) * 1000)
    p50, p95 = np.percentile(times, [50, 95])
    return {
        "ms_p50": float(p50),
        "ms_p95": float(p95),
        "ms_mean": float(np.mean(times)),
        "windows_per_s_p50": float(1000 * batch / p50),
        "runs": runs,
        "warmup": warmup,
        "threads": threads or f"onnxruntime default ({os.cpu_count()} logical cores on this machine)",
        "onnxruntime": ort.__version__,
        "cpu": cpu_name(),
        "input": "x".join(str(d) for d in shape) + " uint8",
        "scope": f"model only; one call = {shape[1]} frames, non-overlapping windows; decoding and letterboxing excluded",
    }


def write_benchmark(onnx_path: Path, results: list[dict[str, Any]]) -> Path:
    """Keep benchmark results next to the graph (`<graph>.bench.json`), so a quoted latency
    can be traced to the graph, machine and method that produced it."""
    path = onnx_path.with_name(onnx_path.name + ".bench.json")
    record = {
        "onnx": {"path": public_path(onnx_path), "sha256": sha256_file(onnx_path)},
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "results": results,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
