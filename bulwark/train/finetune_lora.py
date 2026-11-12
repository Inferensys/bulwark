"""End-to-end fine-tune: LoRA on the top encoder blocks + the heads, streaming from shards.

Saves `last.pt` every `checkpoint_minutes` and at each epoch end, and resumes from it at
the exact batch where it stopped (data order, optimizer, loss scaler and RNG state), so an
eviction or a lost session costs at most that many minutes. With `time_limit_minutes` it
also stops on its own, saving `last.pt`, before a Kaggle session (12 h) runs out;
rerunning resumes. A `last.pt` from a different config or data is refused (see
`Checkpointer.resume`), and so is rerunning a finished run.

Every `eval_every` optimizer steps and at each epoch end, a validation pass over every val
clip in the shards scores the ground-truth labels (`bulwark.train.validation`) and keeps
the best as `best.pt`. Labels the training clips can teach only one class of are masked
per media type (`bulwark.train.labels`) and left out of that score; every checkpoint lists
the labels it trained as `supervised_labels` and the masked pairs as `masked_labels`. The
first best.pt is followed at once by a last.pt, so a run that stops before its first
timed save can still be resumed.

A non-finite loss skips its backward pass, and a window with no finite loss or with
non-finite gradients skips its optimizer step (still counted as a step, as an fp16
GradScaler skip is, so data order and LR schedule stay where an uninterrupted run would
be). last.pt is not written while such losses continue, and MAX_NONFINITE in a row stop
the run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from torch.utils.data import DataLoader

from bulwark.data.manifest import ClipRecord
from bulwark.data.taxonomy import default_taxonomy
from bulwark.model.bulwark import BulwarkBase, load_payload
from bulwark.model.losses import MultiTaskLoss
from bulwark.train.common import (
    Checkpointer,
    Deadline,
    ShardStream,
    TrainConfig,
    autocast_dtype,
    cosine_lr,
    data_parallel,
    find_shards,
    grad_scaler,
    load_records,
    log_line,
    loss_param_group,
    optimizer_step,
    param_groups,
    pick_device,
    require_split,
    rng_state,
    seed_everything,
    set_lr,
    set_rng_state,
    stop_at_time_limit,
    train_stream,
    val_stream,
)
from bulwark.train.labels import label_plan, print_plan
from bulwark.train.provenance import (
    build_provenance,
    checkpoint_identity,
    encoder_identity,
    require_clean,
    require_releasable,
)
from bulwark.train.stream import AUGMENT_POLICY, shards_identity
from bulwark.train.validation import keep_best, predict, report, validate

console = Console()

MAX_NONFINITE = 10  # consecutive non-finite losses before the run is stopped


def run(config: TrainConfig) -> Path:
    seed_everything(config.seed)
    device = pick_device(config.device)
    taxonomy = default_taxonomy()
    records = load_records(config)
    require_split(records, "train", config)
    shards = find_shards(config.shards)
    stream = train_stream(shards, records, config)
    steps_per_epoch = stream.num_batches() // config.grad_accum
    if steps_per_epoch == 0:
        raise ValueError(
            f"{stream.num_clips} train clips give fewer than grad_accum={config.grad_accum} full batches of "
            f"{config.batch_size} per epoch; lower batch_size or grad_accum."
        )
    val = val_stream(shards, records, config)
    plan = label_plan([records[c] for c in stream.clip_ids], taxonomy, config.mask_labels)
    print_plan(plan, taxonomy)
    init_payload = load_payload(Path(config.init)) if config.init else None
    init = checkpoint_identity(Path(config.init), init_payload) if init_payload is not None else None
    if config.commercial_only:
        require_clean(init, "init")
    if config.release_only:
        require_releasable(init, "init")
    model = BulwarkBase(taxonomy, lora_blocks=config.lora_blocks, lora_rank=config.lora_rank)
    model.to(device)
    used = [*stream.clip_ids, *(val.clip_ids if val is not None else ())]  # val picks best.pt
    provenance = build_provenance(
        "lora", config, {c: records[c] for c in used},
        {"shards": shards_identity(shards), "init": init, "encoder": encoder_identity(model.encoder)},
        extra={"augment": AUGMENT_POLICY if config.augment else None, "labels": plan.summary()},
    )
    total = config.max_steps or config.epochs * steps_per_epoch
    out = Path(config.out)
    fields = {
        "supervised_labels": plan.supervised, "masked_labels": plan.masked_media(),
        "lora_blocks": config.lora_blocks, "lora_rank": config.lora_rank,
    }
    checkpointer = Checkpointer(out, config.checkpoint_minutes, provenance, fields)
    state = checkpointer.resume(total, fresh=config.fresh, allow_mismatch=config.allow_mismatch)

    loss_fn = MultiTaskLoss(taxonomy).to(device)
    if init_payload is not None:  # heads trained on cached features (carry the input statistics)
        _load_init(model, loss_fn, init_payload)
    elif state is None:
        _fit_stats(model, shards, records, config, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    console.print(
        f"LoRA layers: {len(model.lora_layers)}, trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f}M, device={device}"
    )
    optimizer = torch.optim.AdamW(
        [*param_groups(model.named_parameters(), config.weight_decay), *loss_param_group(loss_fn, config.loss_weight_lr)],
        lr=config.lr,
    )
    dtype = autocast_dtype(device, config.precision)
    scaler = grad_scaler(device, dtype)
    console.print(f"precision={dtype or torch.float32}, {steps_per_epoch} optimizer steps per epoch, {total} in total")

    step, epoch, batches_done = 0, 0, 0
    if state is not None:
        model.load_state_dict(state["model"], strict=False)
        loss_fn.load_state_dict(state["loss"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        step, epoch, batches_done = state["step"], state["epoch"], state.get("batches_done", 0)
        total = state.get("total", total)
        set_rng_state(state.get("rng"))
        log_line(out, {"event": "resume", "step": step, "epoch": epoch, "batches_done": batches_done})
        console.print(f"Resumed at step {step}/{total} (epoch {epoch}, batch {batches_done})")

    def snapshot() -> dict[str, Any]:
        return _state(
            model, loss_fn, optimizer, scaler, step=step, epoch=epoch, batches_done=batches_done, total=total,
            best=checkpointer.best,
        )

    started = time.monotonic()  # set again when the training loop starts

    def evaluate() -> None:
        nonlocal started
        if val is None:
            return
        begun = time.monotonic()
        outputs, val_records = predict(net, model, val, device, dtype, config.num_workers)
        result = validate(outputs, val_records, taxonomy, plan.supervised, plan)
        report(out, step, result, seconds=time.monotonic() - begun)
        keep_best(checkpointer, result, {"model": model.trainable_state(), "loss": loss_fn.state_dict(), "step": step})
        started += time.monotonic() - begun  # s_per_step counts training time only

    net = data_parallel(model, device, config.multi_gpu, dtype)
    # Its own generator: creating the loader's iterator must not advance the global RNG
    # (dropout), or a resumed run would diverge from an uninterrupted one.
    loader = DataLoader(stream, batch_size=None, num_workers=config.num_workers, generator=torch.Generator())
    model.train()
    deadline = Deadline(config.time_limit_minutes)
    nonfinite = skipped = 0
    started, first_step = time.monotonic(), step
    while step < total:
        stream.epoch, stream.skip = epoch, batches_done
        usable = (stream.num_batches(epoch) // config.grad_accum) * config.grad_accum  # end on a step boundary
        seen, micro, has_grads = batches_done, 0, False
        optimizer.zero_grad(set_to_none=True)
        for frames, batch in loader:
            seen += 1
            targets = plan.targets(batch, taxonomy).to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = net(frames.to(device, non_blocking=True))
            loss, parts = loss_fn(logits, targets)
            if torch.isfinite(loss):
                nonfinite = 0
                scaler.scale(loss / config.grad_accum).backward()
                has_grads = True
            else:
                nonfinite += 1
                log_line(out, {"event": "nonfinite_loss", "step": step, "parts": parts})
                if nonfinite >= MAX_NONFINITE:
                    raise FloatingPointError(
                        f"{nonfinite} non-finite losses in a row at step {step}; last.pt was not written since they began."
                    )
            micro += 1
            if micro < config.grad_accum:
                continue
            micro = 0
            set_lr(optimizer, cosine_lr(step, total, config.lr, config.warmup_steps))
            if not optimizer_step(optimizer, scaler, trainable, has_grads=has_grads):
                skipped += 1
            has_grads = False
            step += 1
            batches_done = seen
            if step % config.log_every == 0:
                seconds = (time.monotonic() - started) / (step - first_step)
                _log(out, step, total, epoch, float(loss.detach()), parts, scaler, device, seconds, skipped)
            if step >= total or seen >= usable:
                break
            if config.eval_every and step % config.eval_every == 0:
                evaluate()
            if checkpointer.due() and not nonfinite:
                checkpointer.save("last.pt", snapshot())
            if deadline.reached():
                stop_at_time_limit(checkpointer, snapshot, step, total, nonfinite=nonfinite)
                return out
        if seen == stream.skip:
            raise RuntimeError(
                f"Epoch {epoch} produced no batches from {len(shards)} shards. Check the attached shard datasets, "
                "the commercial_only and datasets filters, and batch_size against the clips per bucket."
            )
        evaluate()  # epoch end (or the last step)
        if step < total:
            epoch, batches_done = epoch + 1, 0
        if step < total and deadline.reached():
            stop_at_time_limit(checkpointer, snapshot, step, total, nonfinite=nonfinite)
            return out
        if not nonfinite:
            checkpointer.save("last.pt", snapshot())
    checkpointer.save("final.pt", {"model": model.trainable_state(), "loss": loss_fn.state_dict(), "step": step})
    return out


def _load_init(model: BulwarkBase, loss_fn: MultiTaskLoss, payload: dict[str, Any]) -> None:
    """Start from a heads checkpoint (train_heads) or a B1 checkpoint, with its loss weights."""
    weights = payload["model"]
    if any(key.startswith("heads.") for key in weights):
        _, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            raise KeyError(f"Unexpected keys in init: {unexpected[:5]}")
    else:
        model.heads.load_state_dict(weights)
    if "loss" in payload:
        loss_fn.load_state_dict(payload["loss"])


@torch.no_grad()
def _fit_stats(model: BulwarkBase, shards: list[Path], records: dict[str, ClipRecord], config: TrainConfig, device: torch.device, batches: int = 8) -> None:
    """Without a heads checkpoint, estimate the head input statistics from a few batches.

    The stream mixes every dataset, so the sample is not one shard's worth of one dataset.
    """
    stream = ShardStream(shards, records, "train", config.batch_size, seed=config.seed, dataset_power=config.dataset_power)
    tokens = []
    for index, (frames, _) in enumerate(stream):
        tokens.append(model.tokens(frames.to(device)).float().cpu())
        if index + 1 >= batches:
            break
    if tokens:
        model.heads.fit_input_stats(torch.cat(tokens))


def _log(
    out: Path, step: int, total: int, epoch: int, loss: float, parts: dict[str, float],
    scaler: torch.amp.GradScaler, device: torch.device, seconds_per_step: float, skipped: int,
) -> None:
    """One log.jsonl line; speed, loss scale and peak GPU memory size later runs (T4 budget).

    `skipped_steps` counts this session's optimizer steps skipped for non-finite losses or
    gradients, fp16 overflows that make the GradScaler lower `loss_scale` included.
    """
    entry: dict[str, Any] = {
        "step": step, "epoch": epoch, "loss": loss, "s_per_step": seconds_per_step, "skipped_steps": skipped,
    }
    if scaler.is_enabled():
        entry["loss_scale"] = scaler.get_scale()
    if device.type == "cuda":
        entry["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    log_line(out, {**entry, **{f"loss/{k}": v for k, v in parts.items()}})
    console.print(f"step {step}/{total} loss {loss:.4f}")


def _state(
    model: BulwarkBase, loss_fn: MultiTaskLoss, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    *, step: int, epoch: int, batches_done: int, total: int, best: float | None,
) -> dict[str, Any]:
    return {
        "model": model.trainable_state(),
        "loss": loss_fn.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "batches_done": batches_done,
        "total": total,
        "best": best,
        "rng": rng_state(),
    }
