"""Distil Bulwark-B1 into the Bulwark-E1 edge student.

Teacher outputs are precomputed once (`bulwark train cache` with `init` = the fine-tuned
B1 checkpoint, see configs/train/cache-b1.yaml), so the teacher never has to fit on the
GPU next to the student. Loss = the multi-task loss on the manifest's targets (ground
truth, and teacher labels at TEACHER_WEIGHT, as in the other stages) + distill_weight *
soft-target matching against the cached B1 logits.

Checkpointing, resume, validation and the non-finite loss guard work as in the LoRA
fine-tune: `last.pt` every `checkpoint_minutes` and at each epoch end, exact mid-epoch
resume, a clean stop before `time_limit_minutes` so a run can span several Kaggle
sessions, and `best.pt` from validation passes on ground-truth labels (without the val
rows of the pairs this run masks), followed at once by a last.pt when it is the first.

The student copies the teacher only where the teacher learned something: labels the
teacher records as supervised (its `supervised_labels`, in the teacher dir's meta.json),
minus the (label, media type) pairs this run masks (`bulwark.train.labels`).

The teacher's logits were cached on unaugmented clips, so the student gets photometric
augmentation only (DISTILL_AUGMENT_POLICY): a crop would show it a different view than
the one its soft targets describe (part of an eye crop, a shifted head pose).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader

from bulwark.data.manifest import ClipRecord
from bulwark.data.taxonomy import Taxonomy, default_taxonomy
from bulwark.model.bulwark import BulwarkEdge
from bulwark.model.losses import MultiTaskLoss, distill_loss
from bulwark.train.common import (
    Checkpointer,
    Deadline,
    TrainConfig,
    autocast_dtype,
    cosine_lr,
    data_parallel,
    find_shards,
    grad_scaler,
    load_records,
    load_teacher_logits,
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
from bulwark.train.labels import LabelPlan, label_plan, print_plan
from bulwark.train.provenance import (
    ProvenanceError,
    build_provenance,
    features_identity,
    read_features_meta,
    require_clean,
    require_releasable,
)
from bulwark.train.stream import PHOTOMETRIC_POLICY, shards_identity
from bulwark.train.validation import keep_best, predict, report, validate

console = Console()

MAX_NONFINITE = 10
DISTILL_AUGMENT_POLICY = PHOTOMETRIC_POLICY


def run(config: TrainConfig) -> Path:
    if not config.teacher:
        raise ValueError("Set `teacher` to a features dir produced by `bulwark train cache` with a fine-tuned init.")
    seed_everything(config.seed)
    device = pick_device(config.device)
    taxonomy = default_taxonomy()
    teacher = features_identity(Path(config.teacher))
    if config.commercial_only:
        require_clean(teacher, "teacher")
    if config.release_only:
        require_releasable(teacher, "teacher")
    if teacher["has_provenance"] and not teacher["with_logits"]:
        raise ProvenanceError(f"{config.teacher} holds frozen features only; re-run cache with init=<B1 checkpoint>.")
    clip_ids, teacher_logits = load_teacher_logits(Path(config.teacher))
    missing = sorted(set(taxonomy.labels) - set(teacher_logits))
    if missing:
        raise ValueError(f"{config.teacher} has no teacher logits for {missing}; re-run cache with init=<B1 checkpoint>.")
    if any(len(v) != len(clip_ids) for v in teacher_logits.values()):
        raise ValueError(f"{config.teacher}: teacher logits do not have one row per clip")
    row = {clip_id: i for i, clip_id in enumerate(clip_ids)}
    records = {k: v for k, v in load_records(config).items() if k in row}
    require_split(records, "train", config)
    shards = find_shards(config.shards)
    stream = train_stream(shards, records, config, policy=DISTILL_AUGMENT_POLICY)
    steps_per_epoch = stream.num_batches() // config.grad_accum
    if steps_per_epoch == 0:
        raise ValueError(
            f"{stream.num_clips} train clips give fewer than grad_accum={config.grad_accum} full batches of "
            f"{config.batch_size} per epoch; lower batch_size or grad_accum."
        )
    val = val_stream(shards, records, config)
    train_records = [records[c] for c in stream.clip_ids]
    plan = label_plan(train_records, taxonomy, config.mask_labels)
    print_plan(plan, taxonomy)
    distilled = _distilled_labels(config.teacher, plan, train_records, taxonomy)
    supervised = tuple(name for name in taxonomy.labels if name in plan.supervised or name in distilled)
    used = [*stream.clip_ids, *(val.clip_ids if val is not None else ())]  # val picks best.pt
    provenance = build_provenance(
        "distill", config, {c: records[c] for c in used},
        {"shards": shards_identity(shards), "teacher": teacher},
        extra={"augment": DISTILL_AUGMENT_POLICY if config.augment else None, "labels": plan.summary(), "distilled": distilled},
    )
    total = config.max_steps or config.epochs * steps_per_epoch
    out = Path(config.out)
    fields = {
        "supervised_labels": supervised, "masked_labels": plan.masked_media(),
        "backbone": config.edge_backbone, "img_size": config.edge_img_size,
    }
    checkpointer = Checkpointer(out, config.checkpoint_minutes, provenance, fields)
    state = checkpointer.resume(total, fresh=config.fresh, allow_mismatch=config.allow_mismatch)

    model = BulwarkEdge(taxonomy, backbone=config.edge_backbone, img_size=config.edge_img_size).to(device)
    loss_fn = MultiTaskLoss(taxonomy).to(device)
    optimizer = torch.optim.AdamW(
        [*param_groups(model.named_parameters(), config.weight_decay), *loss_param_group(loss_fn, config.loss_weight_lr)],
        lr=config.lr,
    )
    dtype = autocast_dtype(device, config.precision)
    scaler = grad_scaler(device, dtype)
    console.print(f"precision={dtype or torch.float32}, {steps_per_epoch} optimizer steps per epoch, {total} in total")

    step, epoch, batches_done = 0, 0, 0
    if state is not None:
        model.load_state_dict(state["model"])
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
        return {
            "model": model.state_dict(), "loss": loss_fn.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "batches_done": batches_done,
            "total": total, "best": checkpointer.best, "rng": rng_state(),
        }

    started = time.monotonic()  # set again when the training loop starts

    def evaluate() -> None:
        nonlocal started
        if val is None:
            return
        begun = time.monotonic()
        outputs, val_records = predict(net, model, val, device, dtype, config.num_workers)
        result = validate(outputs, val_records, taxonomy, supervised, plan)
        report(out, step, result, seconds=time.monotonic() - begun)
        keep_best(checkpointer, result, {"model": model.state_dict(), "loss": loss_fn.state_dict(), "step": step})
        started += time.monotonic() - begun  # s_per_step counts training time only

    net = data_parallel(model, device, config.multi_gpu, dtype)
    loader = DataLoader(stream, batch_size=None, num_workers=config.num_workers, generator=torch.Generator())
    model.train()
    deadline = Deadline(config.time_limit_minutes)
    nonfinite = skipped = 0
    started, first_step = time.monotonic(), step
    while step < total:
        stream.epoch, stream.skip = epoch, batches_done
        usable = (stream.num_batches(epoch) // config.grad_accum) * config.grad_accum
        seen, micro, has_grads = batches_done, 0, False
        optimizer.zero_grad(set_to_none=True)
        for frames, batch in loader:
            seen += 1
            index = np.array([row[r.clip_id] for r in batch])
            soft_targets = {k: torch.from_numpy(teacher_logits[k][index]).to(device) for k in distilled}
            soft_mask = {k: v.to(device) for k, v in plan.sample_mask(batch, distilled).items()}
            targets = plan.targets(batch, taxonomy).to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = net(frames.to(device, non_blocking=True))
            hard, parts = loss_fn(logits, targets)
            soft = distill_loss(logits, soft_targets, taxonomy, mask=soft_mask)
            loss = hard + config.distill_weight * soft
            if torch.isfinite(loss):
                nonfinite = 0
                scaler.scale(loss / config.grad_accum).backward()
                has_grads = True
            else:
                nonfinite += 1
                log_line(out, {"event": "nonfinite_loss", "step": step, "distill": float(soft.detach()), "parts": parts})
                if nonfinite >= MAX_NONFINITE:
                    raise FloatingPointError(
                        f"{nonfinite} non-finite losses in a row at step {step}; last.pt was not written since they began."
                    )
            micro += 1
            if micro < config.grad_accum:
                continue
            micro = 0
            set_lr(optimizer, cosine_lr(step, total, config.lr, config.warmup_steps))
            if not optimizer_step(optimizer, scaler, model.parameters(), has_grads=has_grads):
                skipped += 1
            has_grads = False
            step += 1
            batches_done = seen
            if step % config.log_every == 0:
                entry: dict[str, Any] = {
                    "step": step, "epoch": epoch, "loss": float(loss.detach()), "distill": float(soft.detach()),
                    "s_per_step": (time.monotonic() - started) / (step - first_step), "skipped_steps": skipped,
                }
                if scaler.is_enabled():
                    entry["loss_scale"] = scaler.get_scale()
                if device.type == "cuda":
                    entry["max_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                log_line(out, {**entry, **{f"loss/{k}": v for k, v in parts.items()}})
                console.print(f"step {step}/{total} loss {float(loss.detach()):.4f} distill {float(soft.detach()):.4f}")
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
    checkpointer.save("final.pt", {"model": model.state_dict(), "loss": loss_fn.state_dict(), "step": step})
    return out


def _distilled_labels(
    teacher: str, plan: LabelPlan, records: list[ClipRecord], taxonomy: Taxonomy
) -> list[str]:
    """Labels whose soft targets the student copies: supervised in the teacher, and not
    masked for every media type of the training clips."""
    recorded = (read_features_meta(Path(teacher)) or {}).get("supervised_labels")
    if recorded is None:
        console.print(
            f"[yellow]{teacher}/meta.json does not record the teacher's supervised_labels; distilling the labels "
            "this run's clips have targets for instead.[/yellow]"
        )
        recorded = plan.supervised
    media = {r.media_type for r in records}
    return [name for name in taxonomy.labels if name in recorded and any(not plan.is_masked(name, m) for m in media)]
