"""Run the frozen (or fine-tuned) encoder once over every shard and store summary tokens.

One `features-<shard>.npz` per input shard, plus a `meta.json` that records the encoder
weights, init checkpoint and precision behind them, a fingerprint of every shard done, and
the code commit, package version (`bulwark_version`), taxonomy version and split salt of
the session that created the dir. With `init` pointing at a fine-tuned Bulwark-B1
checkpoint it also stores head logits
(`logit__<label>`), which the edge student distils from; their layout follows the
taxonomy's labels, so meta.json then records those too, and the init's
`supervised_labels` (the outputs it was trained on) and `masked_labels` (the media types
some of them were not trained on). Frozen summary tokens do not depend on the taxonomy,
so editing taxonomy.yaml never invalidates them.

meta.json also records `commercial_only` and `release_only`: true for frozen features (no
Bulwark training behind the encoder), otherwise whatever the init records, so the clean
and release checks of the stages that read the dir follow the whole chain.

A clip whose tokens or logits are not finite (an fp16 overflow, say) is left out of its
shard's file and listed in meta.json under `nonfinite`, with a warning.

The job resumes by skipping shards whose features exist and whose fingerprint still
matches. A directory written by a different encoder, init or precision is refused rather
than topped up, so a features dir never mixes runs. `--overwrite` (`--set
overwrite=true`) clears such a dir (or one without meta.json); it leaves a dir that
already belongs to this run alone, so passing it again to the next session of a long
cache job does not undo the first. The job writes no last.pt, so the `fresh` and
`allow_mismatch` overrides of the other jobs are refused rather than ignored.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from bulwark.data.shards import read_shard
from bulwark.data.splits import SplitPolicy
from bulwark.data.taxonomy import Taxonomy, default_taxonomy
from bulwark.model.bulwark import BulwarkBase, load_into, load_payload
from bulwark.train.common import (
    Deadline,
    TrainConfig,
    autocast_dtype,
    data_parallel,
    find_shards,
    pick_device,
)
from bulwark.train.provenance import (
    CHAIN_KEYS,
    FEATURES_META,
    SCHEMA,
    ProvenanceError,
    bulwark_version,
    checkpoint_identity,
    code_version,
    differences,
    encoder_identity,
    json_sha256,
    jsonable,
    library_versions,
    read_features_meta,
    require_clean,
    require_releasable,
    write_json,
)
from bulwark.train.stream import shard_fingerprint

console = Console()


def _batches(shard: Path, batch_size: int) -> Iterator[list[tuple[str, np.ndarray]]]:
    """Batches of equal clip length (images and videos use different encoder paths)."""
    pending: dict[int, list[tuple[str, np.ndarray]]] = {}
    for clip_id, frames in read_shard(shard):
        bucket = pending.setdefault(frames.shape[0], [])
        bucket.append((clip_id, frames))
        if len(bucket) >= batch_size:
            yield pending.pop(frames.shape[0])
    yield from (bucket for bucket in pending.values() if bucket)


class _ShardBatches(IterableDataset[tuple[int, list[str], Any]]):
    """Decodes shards in DataLoader workers: (shard number, clip ids, frames), then
    (shard number, [], None) once a shard is complete."""

    def __init__(self, shards: list[Path], batch_size: int) -> None:
        self.shards = shards
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[int, list[str], Any]]:
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        for number in range(worker_id, len(self.shards), workers):
            for batch in _batches(self.shards[number], self.batch_size):
                yield number, [c for c, _ in batch], torch.from_numpy(np.stack([f for _, f in batch]))
            yield number, [], None


class _Encode(nn.Module):
    """Summary tokens and (with a fine-tuned init) head logits, as one forward for DataParallel."""

    def __init__(self, model: BulwarkBase, with_logits: bool) -> None:
        super().__init__()
        self.model = model
        self.with_logits = with_logits

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        summary = self.model.tokens(frames)
        return summary, self.model.heads(summary) if self.with_logits else {}


def check_config(config: TrainConfig) -> None:
    """Refuse the resume overrides of the checkpointing jobs: this job writes no last.pt.

    `fresh` and `allow_mismatch` would otherwise be ignored, so a run meant to start over
    would silently resume the dir. `overwrite` is the cache job's own switch.
    """
    given = [key for key in ("fresh", "allow_mismatch") if getattr(config, key)]
    if given:
        raise ValueError(
            f"The cache job resumes shard by shard and has no last.pt, so {' and '.join(given)} do not apply. "
            "To rebuild a features dir cached by a different run, pass --overwrite (--set overwrite=true), "
            "or cache into a new dir."
        )


@torch.no_grad()
def run(config: TrainConfig) -> Path:
    check_config(config)
    device = pick_device(config.device)
    dtype = autocast_dtype(device, config.precision)
    with_logits = bool(config.init)
    init_source: dict[str, Any] | None = None
    model = BulwarkBase(lora_blocks=config.lora_blocks if with_logits else 0, lora_rank=config.lora_rank)
    if with_logits:
        payload = load_payload(Path(config.init))
        init_source = checkpoint_identity(Path(config.init), payload)
        if config.commercial_only:
            require_clean(init_source, "init")
        if config.release_only:
            require_releasable(init_source, "init")
        load_into(model, Path(config.init), payload=payload)
    model.eval().to(device)

    run_id = jsonable({
        "encoder": encoder_identity(model.encoder),
        "init": {k: v for k, v in init_source.items() if k != "path"} if init_source else None,
        "lora_blocks": config.lora_blocks if with_logits else 0,
        "lora_rank": config.lora_rank if with_logits else 0,
        "precision": str(dtype or torch.float32).removeprefix("torch."),
        "with_logits": with_logits,
        **({"logit_layout_sha256": logit_layout_sha256(model.taxonomy)} if with_logits else {}),
    })
    out = Path(config.features)
    meta = _open_features_dir(out, run_id, overwrite=config.overwrite)
    if meta is None:
        meta = {
            "schema": SCHEMA,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run": run_id,
            "init_path": config.init or None,
            # Clean (releasable) when no Bulwark training is behind the encoder, or the init is.
            "commercial_only": init_source is None or init_source.get("commercial_only") is True,
            "release_only": init_source is None or init_source.get("release_only") is True,
            "bulwark_version": bulwark_version(),
            "taxonomy_version": default_taxonomy().version,
            "split_salt": SplitPolicy().salt,
            "code": code_version(),
            "versions": library_versions(),
            "shards": {},
        }
    if with_logits:
        meta["supervised_labels"] = payload.get("supervised_labels")
        meta["masked_labels"] = payload.get("masked_labels")

    shards = find_shards(config.shards)
    todo: list[tuple[Path, str]] = []
    for shard in shards:
        fingerprint = shard_fingerprint(shard)
        if (out / f"features-{shard.stem}.npz").exists() and meta["shards"].get(shard.name) == fingerprint:
            continue
        meta["shards"].pop(shard.name, None)
        (meta.get("nonfinite") or {}).pop(shard.name, None)
        todo.append((shard, fingerprint))
    write_json(out / FEATURES_META, meta)
    console.print(f"Caching {len(todo)} of {len(shards)} shards on {device} ({run_id['precision']}) -> {out}")
    if not todo:
        return out

    net = data_parallel(_Encode(model, with_logits), device, config.multi_gpu, dtype)
    deadline = Deadline(config.time_limit_minutes)
    init_sha = (init_source or {}).get("sha256") or ""
    loader = DataLoader(
        _ShardBatches([s for s, _ in todo], config.batch_size), batch_size=None, num_workers=min(config.num_workers, len(todo))
    )
    pending: dict[int, dict[str, list[Any]]] = {}
    for number, clip_ids, frames in loader:
        parts = pending.setdefault(number, {"clip_ids": [], "tokens": [], "logits": [], "nonfinite": []})
        if frames is not None:
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                summary, outputs = net(frames.to(device, non_blocking=True))
            tokens = summary.float().cpu().numpy().astype(np.float16)  # stored as fp16, so check after the cast
            logits = {k: v.float().cpu().numpy() for k, v in outputs.items()}
            finite = _finite_rows(tokens, logits)
            parts["nonfinite"].extend(c for c, ok in zip(clip_ids, finite, strict=True) if not ok)
            if finite.any():
                parts["clip_ids"].extend(c for c, ok in zip(clip_ids, finite, strict=True) if ok)
                parts["tokens"].append(tokens[finite])
                parts["logits"].append({k: v[finite] for k, v in logits.items()})
            continue
        shard, fingerprint = todo[number]
        parts = pending.pop(number)
        if parts["nonfinite"]:
            meta.setdefault("nonfinite", {})[shard.name] = parts["nonfinite"]
            console.print(
                f"[yellow]{shard.name}: left out {len(parts['nonfinite'])} clips with non-finite tokens or logits "
                f"(listed in meta.json under nonfinite), e.g. {parts['nonfinite'][:3]}[/yellow]"
            )
        _write_shard(out, shard, parts, init_sha, fingerprint)
        meta["shards"][shard.name] = fingerprint
        write_json(out / FEATURES_META, meta)
        if deadline.reached():
            console.print("Time limit reached; finished shards are saved. Run again to continue.")
            break
    return out


def _finite_rows(tokens: np.ndarray, logits: dict[str, np.ndarray]) -> np.ndarray:
    """Per clip: True when its tokens and every logit are finite."""
    finite = np.isfinite(tokens).reshape(len(tokens), -1).all(1)
    for value in logits.values():
        finite &= np.isfinite(value).reshape(len(value), -1).all(1)
    return finite


def logit_layout_sha256(taxonomy: Taxonomy | None = None) -> str:
    """Hash of what cached logits depend on: each label's name, head, kind, classes and range.

    Doc-string edits to taxonomy.yaml leave it unchanged.
    """
    taxonomy = taxonomy or default_taxonomy()
    return json_sha256([
        [s.name, s.head, s.kind, list(s.classes), list(s.value_range)] for s in taxonomy.labels.values()
    ])


def _open_features_dir(out: Path, run_id: dict[str, Any], *, overwrite: bool) -> dict[str, Any] | None:
    """The dir's existing meta when it belongs to this run, None for a new dir; else refuse.

    `overwrite` clears a dir that belongs to another run or has no meta.json, never one
    that belongs to this run (that one resumes).
    """
    out.mkdir(parents=True, exist_ok=True)
    meta = read_features_meta(out)
    changed = differences(_comparable(meta.get("run")), _comparable(run_id)) if meta is not None else None
    if changed == []:
        if overwrite:
            console.print(f"{out} already belongs to this run; resuming it (overwrite=true clears only another run's dir).")
        return meta
    files = list(out.glob("features-*.npz"))
    if overwrite:
        if meta is not None or files:
            for file in [*files, out / FEATURES_META]:
                file.unlink(missing_ok=True)
            console.print(f"Cleared {out} (overwrite=true): " + (f"it differed in {changed}" if changed else "it had no meta.json"))
        return None
    if meta is None:
        if files:
            raise ProvenanceError(
                f"{out} holds features without meta.json, so their encoder and init are unknown. "
                "Cache into a new dir, or --set overwrite=true to rebuild this one."
            )
        return None
    raise ProvenanceError(
        f"{out} was cached by a different run; it differs in {changed}. Cache into a new dir, "
        "or --set overwrite=true to rebuild this one."
    )


def _comparable(run: Any) -> Any:
    """A run id without the init's `upstream_datasets` and `unrecorded_inputs` (CHAIN_KEYS):
    they are read from the init file, whose sha256 is compared already, and dirs cached
    before they were recorded lack them."""
    if not isinstance(run, dict) or not isinstance(run.get("init"), dict):
        return run
    return {**run, "init": {k: v for k, v in run["init"].items() if k not in CHAIN_KEYS}}


def _write_shard(out: Path, shard: Path, parts: dict[str, list[Any]], init_sha: str, fingerprint: str) -> None:
    if not parts["clip_ids"]:
        console.print(f"[yellow]{shard.name}: no clips, nothing written[/yellow]")
        return
    # Typed Any so mypy accepts `**payload`: NumPy 2.x's savez stub has an `allow_pickle`
    # bool the dict could bind to. Passing allow_pickle=False instead would store an extra
    # array named `allow_pickle` under NumPy 1.x, whose savez has no such parameter.
    payload: dict[str, Any] = {
        "clip_ids": np.array(parts["clip_ids"]),
        "tokens": np.concatenate(parts["tokens"]),
        "init_sha256": np.array(init_sha),
        "shard_fingerprint": np.array(fingerprint),
    }
    for name in parts["logits"][0]:
        payload[f"logit__{name}"] = np.concatenate([batch[name] for batch in parts["logits"]])
    target = out / f"features-{shard.stem}.npz"
    tmp = target.with_name(target.name + ".tmp")  # not matched by features-*.npz while partial
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(target)
    console.print(f"{shard.name}: {len(parts['clip_ids'])} clips")
