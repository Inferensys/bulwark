"""Shared training plumbing: config, device, data, checkpoints and resume."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rich.console import Console
from torch import nn
from torch.utils.data import Dataset

from bulwark.data.manifest import ClipRecord, read_manifest
from bulwark.data.splits import SplitPolicy, check_disjoint
from bulwark.model.bulwark import load_payload
from bulwark.train.provenance import (
    checkpoint_identity,
    differences,
    fingerprint,
    jsonable,
    read_features_meta,
    require_clean,
    require_releasable,
    weights_release_ok,
    write_json,
)
from bulwark.train.stream import Batch, ShardStream, index_shards  # noqa: F401 - Batch and ShardStream re-exported for callers

console = Console()


@dataclass(slots=True)
class TrainConfig:
    manifest: str = "data/manifest.parquet"
    shards: str = "data/shards"
    features: str = "data/features"
    out: str = "checkpoints/run"
    init: str = ""                    # checkpoint to start from (heads or B1); for cache, the B1 whose logits to store
    teacher: str = ""                 # distillation: features dir that holds teacher logits
    device: str = "auto"
    seed: int = 7
    epochs: int = 10
    max_steps: int = 0                # 0 = epochs decide
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 200
    loss_weight_lr: float = 1e-3      # fixed LR of the learned loss weights (not on the schedule)
    num_workers: int = 4
    precision: str = "auto"           # auto | fp32 | bf16 | fp16
    multi_gpu: bool = True            # DataParallel over all visible GPUs when there are several
    lora_blocks: int = 4
    lora_rank: int = 16
    checkpoint_minutes: float = 15.0
    time_limit_minutes: float = 0.0   # lora, distill: stop cleanly with last.pt saved; cache: stop after the shard in progress (Kaggle ends sessions at 12 h); 0 = none
    eval_every: int = 1000            # lora, distill: val pass (best.pt) every N optimizer steps and at each epoch end; 0 = epoch end only
    select_folds: int = 0             # heads: pick best.pt's epoch by k-fold cross-validation over train and val subjects; 0 = on the val split (heads*.yaml set 5)
    commercial_only: bool = False     # train on commercially usable data only (see configs/train/*-clean.yaml)
    release_only: bool = False        # train only on datasets cleared for publishing weights (registry weights_release_ok)
    datasets: list[str] = field(default_factory=list)  # empty = all
    mask_labels: list[str] = field(default_factory=list)  # never trained or distilled; one-class labels are masked anyway (train/labels.py)
    dataset_power: float = 0.5        # each dataset's share of an epoch scales with size**power
    augment: bool = True              # light, label-safe clip augmentation in lora; photometric only in distill
    edge_backbone: str = "efficientvit_b1.r224_in1k"
    edge_img_size: int = 224
    distill_weight: float = 1.0
    log_every: int = 50
    fresh: bool = False               # set the previous run in `out` aside (out/stale-<time>/) and start over
    allow_mismatch: bool = False      # resume a last.pt from a different config or data (never across commercial_only or release_only)
    overwrite: bool = False           # cache: clear a features dir from a different encoder, init or precision, or without meta.json


def load_config(path: str | Path | None, **overrides: Any) -> TrainConfig:
    config = TrainConfig()
    data: dict[str, Any] = {}
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key, value in data.items():
        setattr(config, key, _coerce(getattr(config, key), value, key))
    return config


def _coerce(default: Any, value: Any, key: str) -> Any:
    """YAML reads `1e-3` as a string and `--set x=1` as an int; match the field's type."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false, got {value!r}")
        return value
    if isinstance(default, float) and isinstance(value, (int, str)):
        return float(value)
    return value


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if device.type == "cuda":
        # Native bf16 needs Ampere (sm80) or newer. A T4 (sm75) only emulates it, which is
        # slow, so `auto` picks fp16 (with a GradScaler) there.
        return torch.bfloat16 if torch.cuda.get_device_capability(device) >= (8, 0) else torch.float16
    return None


def grad_scaler(device: torch.device, dtype: torch.dtype | None) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)


def data_parallel(model: nn.Module, device: torch.device, enabled: bool = True, dtype: torch.dtype | None = None) -> nn.Module:
    """Wrap in DataParallel when several GPUs are visible (Kaggle's T4 x2), else return as is.

    Each batch is split across the GPUs; checkpoints keep using the unwrapped model.
    """
    count = torch.cuda.device_count() if device.type == "cuda" else 0
    if enabled and count > 1 and device.index in (None, 0):
        console.print(f"DataParallel over {count} GPUs")
        return nn.DataParallel(_Autocast(model, dtype))
    return model


class _Autocast(nn.Module):
    """Autocast inside each replica: DataParallel's worker threads re-enter autocast with
    the default dtype (fp16 on CUDA), which would silently turn a bf16 run into fp16."""

    def __init__(self, module: nn.Module, dtype: torch.dtype | None) -> None:
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, *args: Any) -> Any:
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype is not None):
            return self.module(*args)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict[str, Any]:
    """Python, NumPy, torch, CUDA and MPS RNG states, in a form `weights_only` loading accepts."""
    _, keys, position, has_gauss, gauss = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {"keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)), "pos": int(position),
                  "has_gauss": int(has_gauss), "gauss": float(gauss)},
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():  # dropout on Apple silicon (MPS) draws from the MPS generator
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    version, internal, gauss_next = state["python"]
    random.setstate((version, tuple(internal), gauss_next))
    numpy_state = state["numpy"]
    np.random.set_state(("MT19937", numpy_state["keys"].numpy().astype(np.uint32), numpy_state["pos"],
                         numpy_state["has_gauss"], numpy_state["gauss"]))
    torch.set_rng_state(state["torch"])
    cuda = state.get("cuda")
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda)
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def load_records(config: TrainConfig) -> dict[str, ClipRecord]:
    """The manifest's clips after the data filters, once its splits pass the split policy.

    Trainers select clips by split, so "excluded" and "holdout" rows are never used.
    """
    records = read_manifest(Path(config.manifest))
    try:
        check_disjoint(records, SplitPolicy())
    except AssertionError as exc:
        raise ValueError(
            f"{config.manifest} breaks the split policy: {exc}. Rebuild it with `bulwark split` (re-ingest "
            "datasets whose adapters changed first) before training on it."
        ) from exc
    if config.commercial_only:
        records = [r for r in records if r.commercial_ok]
    if config.release_only:
        records = [r for r in records if weights_release_ok(r.dataset) is True]
    if config.datasets:
        records = [r for r in records if r.dataset in config.datasets]
    return {r.clip_id: r for r in records}


def require_split(records: dict[str, ClipRecord], split: str, config: TrainConfig) -> None:
    """Fail loudly when the data filters leave nothing to train on."""
    if not any(r.split == split for r in records.values()):
        raise ValueError(
            f"No {split} clips in {config.manifest} after filtering (commercial_only={config.commercial_only}, "
            f"release_only={config.release_only}, datasets={config.datasets or 'all'})."
        )


def find_shards(path: str | Path) -> list[Path]:
    """The *.tar shards under `path`, indexed (in threads; see `index_shards`)."""
    shards = sorted(Path(path).glob("*.tar"))
    if not shards:
        raise FileNotFoundError(
            f"No *.tar shards in {path}. Check that the shard datasets are attached (Kaggle) or synced (Azure)."
        )
    started = time.monotonic()
    index_shards(shards)
    console.print(f"Indexed {len(shards)} shards in {time.monotonic() - started:.0f} s")
    return shards


def train_stream(
    shards: Sequence[Path], records: dict[str, ClipRecord], config: TrainConfig, *,
    augment: bool | None = None, policy: dict[str, Any] | None = None,
) -> ShardStream:
    """The training stream for lora and distill, with its data checked and summarised."""
    stream = ShardStream(
        shards, records, "train", config.batch_size, seed=config.seed, dataset_power=config.dataset_power,
        augment=config.augment if augment is None else augment, policy=policy,
    )
    if stream.num_clips == 0:
        raise ValueError(
            f"None of the manifest's train clips are in the {len(shards)} shards under {config.shards}. "
            f"Check the attached shard datasets and the filters (commercial_only={config.commercial_only}, "
            f"release_only={config.release_only}, datasets={config.datasets or 'all'})."
        )
    console.print(f"train: {stream.num_clips} clips in {len(shards)} shards {stream.counts}; per epoch {stream.quotas()}")
    if stream.missing:
        console.print(
            f"[yellow]{stream.missing} manifest train clips have no shard here (not attached, or failed to "
            "decode); epochs and the LR schedule count only the clips present.[/yellow]"
        )
    return stream


def val_stream(shards: Sequence[Path], records: dict[str, ClipRecord], config: TrainConfig) -> ShardStream | None:
    """Every val clip present in the shards, once, in shard order (no balancing, nothing dropped)."""
    stream = ShardStream(shards, records, "val", config.batch_size, shuffle=False, drop_last=False)
    console.print(f"val: {stream.num_clips} clips in the shards {stream.counts}, {stream.missing} manifest val clips without a shard")
    if not stream.num_clips:
        console.print("[yellow]No val clips: best.pt is not written, only last.pt and final.pt.[/yellow]")
        return None
    return stream


def dataset_weights(records: Sequence[ClipRecord], power: float = 0.5) -> list[float]:
    """Per-sample weights so each dataset's share scales with size**power, not size."""
    counts = Counter(r.dataset for r in records)
    return [counts[r.dataset] ** (power - 1.0) for r in records]


# ---- optimisation ------------------------------------------------------------------------


def param_groups(named: Iterable[tuple[str, nn.Parameter]], weight_decay: float) -> list[dict[str, Any]]:
    """Trainable parameters in two AdamW groups: weights decay; norms, biases and tokens do not.

    Decaying a pretrained LayerNorm gain pulls it toward 0 and shifts the features the
    heads were standardised on.
    """
    decay: list[nn.Parameter] = []
    keep: list[nn.Parameter] = []
    for name, parameter in named:
        if not parameter.requires_grad:
            continue
        leaf = name.rsplit(".", 1)[-1]
        (keep if parameter.ndim <= 1 or leaf in ("pos", "cls") else decay).append(parameter)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": keep, "weight_decay": 0.0}]
    return [g for g in groups if g["params"]]


def loss_param_group(loss_fn: nn.Module, lr: float) -> list[dict[str, Any]]:
    """The learned loss weights: no decay and a fixed LR, so the LR schedule (and a small
    fine-tuning LR) does not freeze them."""
    params = [p for p in loss_fn.parameters() if p.requires_grad]
    return [{"params": params, "weight_decay": 0.0, "lr": lr, "scheduled": False}] if params else []


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        if group.get("scheduled", True):
            group["lr"] = lr


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    clip: Iterable[nn.Parameter],
    *,
    has_grads: bool,
    max_norm: float = 1.0,
) -> bool:
    """Clip and apply one gradient-accumulation window; False when the step was skipped.

    Skipped when no micro-batch of the window had a finite loss (there is nothing to apply,
    and an fp16 GradScaler would assert), or when any gradient is non-finite: an fp16
    overflow (the scaler then lowers its scale, as usual) or a bf16/fp32 blow-up, which
    would otherwise write NaN into the weights.
    """
    if has_grads:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(clip, max_norm)
        grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        applied = bool(torch.isfinite(torch.nn.utils.get_total_norm(grads)))
        if applied:
            scaler.step(optimizer)
        scaler.update()
    else:
        applied = False
    optimizer.zero_grad(set_to_none=True)
    return applied


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ---- cached features ---------------------------------------------------------------------


def load_feature_dir(path: Path) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Concatenate a features dir: clip ids, summary tokens and teacher logits (if cached).

    With a meta.json (written by `bulwark train cache`) only files for the shards it lists
    are read, and every file must come from the same encoder and init. Logits must cover
    every clip, and a clip may appear only once.
    """
    clip_ids, tokens, logits = _read_feature_dir(Path(path), with_tokens=True)
    assert tokens is not None
    return clip_ids, tokens, logits


def load_teacher_logits(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Clip ids and teacher logits of a features dir, checked as `load_feature_dir` checks
    them, without reading the summary tokens: distillation needs only the logits, and the
    tokens of the first full run's B1 cache (170 shards, without AUC Distracted Driver v2)
    take 6.5 GB (about 11 GB at peak while they are joined)."""
    clip_ids, _, logits = _read_feature_dir(Path(path), with_tokens=False)
    return clip_ids, logits


def _read_feature_dir(path: Path, *, with_tokens: bool) -> tuple[list[str], np.ndarray | None, dict[str, np.ndarray]]:
    meta = read_features_meta(path)
    files = sorted(path.glob("features-*.npz"))
    expected_init = ""
    if meta is None:
        console.print(f"[yellow]{path} has no meta.json, so the encoder and init behind it cannot be checked.[/yellow]")
    else:
        listed = set(meta.get("shards") or {})
        stale = [f.name for f in files if _shard_name(f) not in listed]
        if stale:
            console.print(f"[yellow]{path}: ignoring {len(stale)} files for shards meta.json does not list: {stale[:3]}[/yellow]")
        files = [f for f in files if _shard_name(f) in listed]
        expected_init = (meta["run"].get("init") or {}).get("sha256") or ""
    clip_ids: list[str] = []
    tokens: list[np.ndarray] = []
    logits: dict[str, list[np.ndarray]] = {}
    logit_keys: list[str] | None = None
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            ids = [str(c) for c in data["clip_ids"]]
            keys = sorted(k for k in data.files if k.startswith("logit__"))
            if logit_keys is None:
                logit_keys = keys
            elif keys != logit_keys:
                raise ValueError(f"{file} has logits {keys}, other files have {logit_keys}: {path} mixes runs")
            recorded = str(data["init_sha256"]) if "init_sha256" in data.files else None
            if meta is not None and recorded != expected_init:
                raise ValueError(f"{file} was written by a different init than {path}/meta.json records")
            if (with_tokens and len(data["tokens"]) != len(ids)) or any(len(data[k]) != len(ids) for k in keys):
                raise ValueError(f"{file}: tokens or logits do not have one row per clip id")
            clip_ids.extend(ids)
            if with_tokens:  # an npz reads each array only when it is indexed
                tokens.append(data["tokens"])
            for key in keys:
                logits.setdefault(key.removeprefix("logit__"), []).append(data[key])
    if not files:
        raise FileNotFoundError(f"No features-*.npz in {path}")
    duplicates = len(clip_ids) - len(set(clip_ids))
    if duplicates:
        raise ValueError(f"{path} holds {duplicates} duplicate clip ids (shards that share clips?); re-cache into a new dir")
    return clip_ids, np.concatenate(tokens) if with_tokens else None, {k: np.concatenate(v) for k, v in logits.items()}


def _shard_name(file: Path) -> str:
    return file.name.removeprefix("features-").removesuffix(".npz") + ".tar"


class FeatureDataset(Dataset[tuple[torch.Tensor, ClipRecord]]):
    """Cached summary tokens of the clips in `records` of one split, or that `split` accepts."""

    def __init__(
        self, clip_ids: list[str], tokens: np.ndarray, records: dict[str, ClipRecord],
        split: str | Callable[[ClipRecord], bool],
    ) -> None:
        wanted = split if callable(split) else lambda record: record.split == split
        keep = [i for i, c in enumerate(clip_ids) if c in records and wanted(records[c])]
        self.tokens = tokens[keep]
        self.records = [records[clip_ids[i]] for i in keep]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ClipRecord]:
        return torch.from_numpy(self.tokens[index].astype(np.float32)), self.records[index]


# ---- checkpoints, resume and time limits --------------------------------------------------


RUN_FILES = ("last.pt", "best.pt", "final.pt", "log.jsonl", "provenance.json")


class ResumeError(RuntimeError):
    """`out` holds a run this job must not continue or overwrite."""


class Checkpointer:
    """Writes `last.pt` every N minutes (evictions, session limits) and named checkpoints.

    Every payload carries the run's provenance and `fields` (plain values every checkpoint
    of the run records at its top level, such as `supervised_labels` and the architecture),
    and `last.pt` also the run's fingerprint. `resume` continues only a `last.pt` whose
    fingerprint matches this run, and refuses to start over in a directory that still holds
    another run's checkpoints. The provenance lists every session's start time and code
    version (`sessions`), so a run whose code changed between Kaggle sessions says so in
    its final.pt.

    A best.pt written before the run's first last.pt makes `due` true at once: a rerun
    resumes only from last.pt, and refuses a directory that holds best.pt alone.
    """

    def __init__(
        self, out: Path, minutes: float, provenance: dict[str, Any] | None = None, fields: dict[str, Any] | None = None
    ) -> None:
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.interval = minutes * 60
        self.last_save = time.monotonic()
        self.best: float | None = None
        self.best_without_last = False
        self.fields = jsonable(fields or {})
        self.provenance = dict(provenance) if provenance is not None else None
        self.fingerprint = fingerprint(provenance) if provenance is not None else None
        if self.provenance is not None:
            self.provenance["sessions"] = [self._session(self.provenance)]

    def due(self) -> bool:
        """Time to write last.pt: every `minutes`, and as soon as best.pt exists without one."""
        return self.best_without_last or time.monotonic() - self.last_save >= self.interval

    def save(self, name: str, payload: dict[str, Any]) -> Path:
        payload = {**self.fields, **payload}
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        if name == "last.pt" and self.fingerprint is not None:
            payload["fingerprint"] = self.fingerprint
        path = self.out / name
        tmp = path.with_name(name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        if name == "last.pt":
            self.last_save = time.monotonic()
            self.best_without_last = False
        elif name == "best.pt" and not (self.out / "last.pt").exists():
            self.best_without_last = True
        return path

    def resume(self, total: int | None = None, *, fresh: bool = False, allow_mismatch: bool = False) -> dict[str, Any] | None:
        """The `last.pt` state to continue from, or None to start a new run in `out`.

        With `total` (or the total saved in `last.pt`), a run that already reached it and
        wrote final.pt is refused instead of being resumed and rewritten.
        """
        if fresh:
            self._set_aside()
        path = self.out / "last.pt"
        if not path.exists():
            leftovers = [name for name in ("best.pt", "final.pt") if (self.out / name).exists()]
            if leftovers:
                raise ResumeError(
                    f"{self.out} holds {', '.join(leftovers)} from an earlier run but no last.pt to resume. "
                    "Use a new `out`, or --set fresh=true to move the old files to a stale-<time>/ folder."
                )
            self._start()
            return None
        payload = load_payload(path)
        if self.provenance is not None and self.fingerprint is not None:
            found = payload.get("fingerprint")
            if found:
                changed = differences(self.fingerprint, _with_new_defaults(found, self.fingerprint))
            else:
                changed = ["(last.pt records no run fingerprint)"]
            differs = f"{path} was written by a different run; it differs in: {changed}."
            fixed = sorted({key for item in changed for key in ("commercial_only", "release_only") if key in item})
            if fixed:
                raise ResumeError(f"{differs} {' and '.join(fixed)} cannot be overridden. Use a new `out`, or --set fresh=true.")
            # No override: a last.pt that does not itself record clean (releasable) data, one
            # written before provenance existed, say, never continues such a run.
            if self.provenance.get("commercial_only"):
                require_clean(checkpoint_identity(path, payload), "last.pt")
            if self.provenance.get("release_only"):
                require_releasable(checkpoint_identity(path, payload), "last.pt")
            if changed:
                if not allow_mismatch:
                    raise ResumeError(
                        f"{differs} Use a new `out`, --set fresh=true to move the old files aside, or "
                        "--set allow_mismatch=true to resume anyway."
                    )
                console.print(f"[yellow]Resuming anyway (allow_mismatch): {differs}[/yellow]")
                log_line(self.out, {"event": "allow_mismatch", "changed": changed})
            self._continue_sessions(payload.get("provenance") or {})
        step, total = payload.get("step", 0), payload.get("total", total)
        if total is not None and step >= total and (self.out / "final.pt").exists():
            raise ResumeError(
                f"The run in {self.out} is already complete (step {step}/{total}, final.pt written). "
                "Use a new `out`, or --set fresh=true to move it aside and train again."
            )
        self.best = payload.get("best")
        self._start()
        return payload

    @staticmethod
    def _session(provenance: dict[str, Any]) -> dict[str, Any]:
        return {"started": provenance.get("created"), "code": provenance.get("code")}

    def _continue_sessions(self, previous: dict[str, Any]) -> None:
        """Carry the earlier sessions' code versions forward and warn when the code changed."""
        assert self.provenance is not None
        current = self.provenance["sessions"][-1]
        earlier = previous.get("sessions") or ([self._session(previous)] if previous else [])
        self.provenance["sessions"] = [*earlier, current]
        before = (earlier[-1].get("code") or {}) if earlier else {}
        now = current.get("code") or {}
        if earlier and (before.get("commit"), before.get("dirty")) != (now.get("commit"), now.get("dirty")):
            console.print(
                f"[yellow]The code changed since the last session (commit {before.get('commit')}, dirty "
                f"{before.get('dirty')} -> commit {now.get('commit')}, dirty {now.get('dirty')}); both are "
                "recorded in the checkpoint's provenance (sessions).[/yellow]"
            )
            log_line(self.out, {"event": "code_changed", "before": before, "now": now})

    def _start(self) -> None:
        if self.provenance is not None:
            write_json(self.out / "provenance.json", self.provenance)
            log_line(self.out, {"event": "start", "provenance": self.provenance})

    def _set_aside(self) -> None:
        present = [name for name in RUN_FILES if (self.out / name).exists()]
        if not present:
            return
        stale = self.out / f"stale-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        stale.mkdir()
        for name in present:
            shutil.move(str(self.out / name), stale / name)
        console.print(f"Moved the previous run's {', '.join(present)} to {stale}")


def _with_new_defaults(found: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """A last.pt fingerprint with the config keys added since it was written set to their
    defaults, which is how the code that wrote it behaved."""
    config, wanted = found.get("config"), expected.get("config")
    if not isinstance(config, dict) or not isinstance(wanted, dict):
        return found
    defaults = jsonable(asdict(TrainConfig()))
    added = {k: defaults[k] for k in wanted if k not in config and k in defaults}
    return {**found, "config": {**added, **config}}


class Deadline:
    """Wall-clock budget for one session (Kaggle ends sessions at 12 h); 0 minutes = none."""

    def __init__(self, minutes: float) -> None:
        self.end = time.monotonic() + minutes * 60 if minutes else None

    def reached(self) -> bool:
        return self.end is not None and time.monotonic() >= self.end


def stop_at_time_limit(
    checkpointer: Checkpointer, snapshot: Callable[[], dict[str, Any]], step: int, total: int, *, nonfinite: int = 0
) -> None:
    """End a session at its time limit: save last.pt, unless the latest losses were non-finite."""
    if nonfinite:
        console.print(f"Time limit reached at step {step}/{total} during non-finite losses; last.pt was not updated.")
        return
    checkpointer.save("last.pt", snapshot())
    console.print(f"Time limit reached at step {step}/{total}; saved last.pt. Run again to resume.")


def log_line(out: Path, entry: dict[str, Any]) -> None:
    with (out / "log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")
