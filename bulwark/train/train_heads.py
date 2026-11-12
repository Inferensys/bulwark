"""Train the multi-task heads on cached summary tokens. Light enough for a local machine
(Apple silicon/MPS or CPU) or a T4.

`best.pt` is chosen on ground-truth labels only (see `bulwark.train.validation`), never on
teacher pseudo-labels, and every checkpoint carries the learned loss weights so the LoRA
stage starts from the same objective. Labels the training clips can teach only one class
of are masked per media type (see `bulwark.train.labels`), and their val rows are left out
of the score.

The val split is small (few subjects per dataset), so best.pt records how many clips and
subjects each label's score rests on (`select.support`), and the shipped heads configs
pick it by k-fold cross-validation instead (`select_folds: 5`; 0 selects on the val split):

- `select_folds: k` deals each dataset's train and val subjects into k folds
  (`bulwark.data.splits.subject_folds`; train-only datasets stay in training), and k
  additional models, which are not saved, train with one fold held out and are scored on
  it at every epoch end. best.pt is the main run's model after as many steps as the epoch
  with the best mean score over the folds took (`epoch` x steps per epoch, also across a
  resume). The main run still trains on the train split only. `select` then also records
  `folds`, the chosen `epoch`, `fold_scores` and `epoch_scores` (the mean over folds per
  epoch), and `support` sums, per label, the held-out clips and subjects of the folds that
  scored it.
- The LoRA stage can start from `final.pt` instead: the fixed epoch budget, with no
  selection.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader, WeightedRandomSampler

from bulwark.data.manifest import ClipRecord
from bulwark.data.splits import subject_folds
from bulwark.data.taxonomy import Taxonomy, default_taxonomy
from bulwark.model.bulwark import load_payload
from bulwark.model.heads import BulwarkHeads
from bulwark.model.losses import MultiTaskLoss
from bulwark.train.common import (
    Checkpointer,
    FeatureDataset,
    TrainConfig,
    cosine_lr,
    dataset_weights,
    load_feature_dir,
    load_records,
    log_line,
    loss_param_group,
    param_groups,
    pick_device,
    require_split,
    rng_state,
    seed_everything,
    set_lr,
    set_rng_state,
)
from bulwark.train.labels import LabelPlan, label_plan, print_plan
from bulwark.train.provenance import (
    build_provenance,
    checkpoint_identity,
    features_identity,
    jsonable,
    require_clean,
    require_releasable,
)
from bulwark.train.validation import SELECTION, Validation, keep_best, report, validate

console = Console()

FOLD_SPLITS = ("train", "val")  # the subjects k-fold selection deals into folds
FeatureLoader = DataLoader[tuple[torch.Tensor, ClipRecord]]


def _collate(batch: list[tuple[torch.Tensor, ClipRecord]]) -> tuple[torch.Tensor, list[ClipRecord]]:
    return torch.stack([t for t, _ in batch]), [r for _, r in batch]


def run(config: TrainConfig) -> Path:
    if config.select_folds < 0 or config.select_folds == 1:
        raise ValueError(f"select_folds must be 0 (select on the val split) or at least 2, got {config.select_folds}")
    seed_everything(config.seed)
    device = pick_device(config.device)
    taxonomy = default_taxonomy()
    records = load_records(config)
    require_split(records, "train", config)
    feature_source = features_identity(Path(config.features))
    init_payload = load_payload(Path(config.init)) if config.init else None
    init = checkpoint_identity(Path(config.init), init_payload) if init_payload is not None else None
    if config.commercial_only:
        require_clean(feature_source, "features")
        require_clean(init, "init")
    if config.release_only:
        require_releasable(feature_source, "features")
        require_releasable(init, "init")
    clip_ids, tokens, _ = load_feature_dir(Path(config.features))
    train = FeatureDataset(clip_ids, tokens, records, "train")
    val = FeatureDataset(clip_ids, tokens, records, "val")
    if len(train) == 0:
        raise ValueError("No training clips with cached features. Check the manifest splits and features dir.")
    without = {split: sum(r.split == split for r in records.values()) - len(d) for split, d in (("train", train), ("val", val))}
    console.print(
        f"train={len(train)} val={len(val)} clips with features ({without['train']} train and {without['val']} val "
        f"manifest clips have none), device={device}"
    )
    if not len(val) and not config.select_folds:
        console.print("[yellow]No val clips: best.pt is not written, only last.pt and final.pt.[/yellow]")
    train_loader, val_loader = _loaders(train, val, config)

    plan = label_plan(train.records, taxonomy, config.mask_labels)
    print_plan(plan, taxonomy)
    used = {r.clip_id: r for r in [*train.records, *val.records]}
    provenance = build_provenance(
        "heads", config, used, {"features": feature_source, "init": init}, extra={"labels": plan.summary()}
    )
    total = config.max_steps or config.epochs * len(train_loader)
    out = Path(config.out)
    fields = {"supervised_labels": plan.supervised, "masked_labels": plan.masked_media()}
    checkpointer = Checkpointer(out, config.checkpoint_minutes, provenance, fields)
    state = checkpointer.resume(total, fresh=config.fresh, allow_mismatch=config.allow_mismatch)

    kfold = state.get("kfold") if state is not None and config.select_folds else None  # made once per run
    if config.select_folds and kfold is None:
        kfold = cross_validate(config, clip_ids, tokens, records, plan, taxonomy, init_payload, device, out)
        seed_everything(config.seed)  # the main run starts as it would without cross-validation
    heads, loss_fn, optimizer = _model(config, tokens, train, init_payload, taxonomy, device)
    step = 0
    if state is not None:
        heads.load_state_dict(state["model"])
        loss_fn.load_state_dict(state["loss"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        total = state.get("total", total)
        set_rng_state(state.get("rng"))
        log_line(out, {"event": "resume", "step": step})
        console.print(f"Resumed at step {step}/{total}")
    # With k-fold selection, best.pt is the model after the chosen number of epochs' steps.
    best_step = min(total, kfold["epoch"] * len(train_loader)) if kfold and kfold.get("epoch") else None
    if kfold and best_step is None:
        console.print("[yellow]No fold could be scored: best.pt is not written, only last.pt and final.pt.[/yellow]")
    elif best_step is not None and checkpointer.best is None and step > best_step:
        console.print(
            f"[yellow]Resumed at step {step}, past the chosen step {best_step}, with no best.pt written: "
            "best.pt is not written, only last.pt and final.pt.[/yellow]"
        )

    def snapshot() -> dict[str, Any]:
        return {
            "model": heads.state_dict(), "loss": loss_fn.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "total": total, "best": checkpointer.best, "kfold": kfold, "rng": rng_state(),
        }

    def after_step(done: int, loss: torch.Tensor, parts: dict[str, float]) -> None:
        nonlocal step
        step = done
        if step % config.log_every == 0:
            log_line(out, {"step": step, "loss": float(loss.detach()), **{f"loss/{k}": v for k, v in parts.items()}})
        if kfold is not None and step == best_step and checkpointer.best is None:  # by step: passes end elsewhere after a resume
            checkpointer.best = kfold["score"]
            checkpointer.save("best.pt", {"model": heads.state_dict(), "loss": loss_fn.state_dict(), "step": step, "select": kfold})
        if checkpointer.due():
            checkpointer.save("last.pt", snapshot())

    while step < total:
        step = _train_pass(heads, loss_fn, optimizer, train_loader, plan, config, device, step, total, after_step)
        weights = {"model": heads.state_dict(), "loss": loss_fn.state_dict(), "step": step}
        if len(val):
            result = _validate(heads, val_loader, device, plan)
            report(out, step, result)
            if not config.select_folds:
                keep_best(checkpointer, result, weights)
        checkpointer.save("last.pt", snapshot())
    checkpointer.save("final.pt", {"model": heads.state_dict(), "loss": loss_fn.state_dict(), "step": step})
    return out


def cross_validate(
    config: TrainConfig, clip_ids: list[str], tokens: np.ndarray, records: dict[str, ClipRecord], plan: LabelPlan,
    taxonomy: Taxonomy, init_payload: dict[str, Any] | None, device: torch.device, out: Path,
) -> dict[str, Any]:
    """Train one model per fold (`config.select_folds`, not saved) on the train and val
    clips outside it, score it on the fold at every epoch end, and return the `select`
    record of the epoch with the best mean score (`epoch` None when no fold was scored).

    Every model starts like the main run (seed, init or input statistics, schedule) and
    scores the labels `plan` trains, on the media types it trains them on.
    """
    k = config.select_folds
    fold_of = subject_folds(records.values(), k)
    scores: list[list[Validation]] = []
    for fold in range(k):

        def held_out(record: ClipRecord, fold: int = fold) -> bool:
            return record.split in FOLD_SPLITS and fold_of.get(record.subject_id) == fold

        def kept(record: ClipRecord, fold: int = fold) -> bool:
            return record.split in FOLD_SPLITS and fold_of.get(record.subject_id) != fold

        train = FeatureDataset(clip_ids, tokens, records, kept)
        val = FeatureDataset(clip_ids, tokens, records, held_out)
        if not len(val):
            console.print(f"[yellow]Fold {fold + 1}/{k} has no clips with features; it is not scored.[/yellow]")
            scores.append([])
            continue
        seed_everything(config.seed)
        train_loader, val_loader = _loaders(train, val, config)
        heads, loss_fn, optimizer = _model(config, tokens, train, init_payload, taxonomy, device)
        total = config.max_steps or config.epochs * len(train_loader)
        step, results = 0, []
        while step < total:
            step = _train_pass(heads, loss_fn, optimizer, train_loader, plan, config, device, step, total)
            results.append(_validate(heads, val_loader, device, plan))
            log_line(out, {
                "event": "cv", "fold": fold, "epoch": len(results), "step": step, "val_select": results[-1].score,
                "val_select_labels": results[-1].per_label, "val_select_support": results[-1].support,
            })
        rounded = [None if r.score is None else round(r.score, 4) for r in results]
        console.print(f"fold {fold + 1}/{k}: {len(train)} train and {len(val)} held-out clips; {SELECTION} by epoch {rounded}")
        scores.append(results)
    select = fold_select(scores, k)
    log_line(out, {"event": "cv_select", "select": select})
    console.print(f"k-fold selection: epoch {select['epoch']}, mean {SELECTION} {select['score']} over {k} folds")
    return select


def fold_select(scores: list[list[Validation]], k: int) -> dict[str, Any]:
    """The `select` record of the epoch whose mean score over folds is highest (the earliest
    of equal ones). `scores` holds each fold's validations, one per epoch."""
    epochs = max((len(s) for s in scores), default=0)
    means: list[float | None] = []
    for epoch in range(epochs):
        values = [score for score in (s[epoch].score for s in scores if epoch < len(s)) if score is not None]
        means.append(float(np.mean(values)) if values else None)
    record: dict[str, Any] = {"criterion": SELECTION, "folds": k, "epoch": None, "score": None, "epoch_scores": means}
    scored = {epoch: mean for epoch, mean in enumerate(means) if mean is not None}
    if not scored:
        return jsonable(record)
    best = max(scored, key=scored.__getitem__)
    chosen = [s[best] for s in scores if best < len(s)]
    labels = list(dict.fromkeys(name for v in chosen for name in v.per_label))
    record.update(
        epoch=best + 1,
        score=scored[best],
        metrics={name: float(np.mean([v.per_label[name] for v in chosen if name in v.per_label])) for name in labels},
        support={
            name: {key: sum(v.support[name][key] for v in chosen if name in v.support) for key in ("clips", "subjects")}
            for name in labels
        },
        fold_scores=[s[best].score if best < len(s) else None for s in scores],
    )
    return jsonable(record)


def _loaders(train: FeatureDataset, val: FeatureDataset, config: TrainConfig) -> tuple[FeatureLoader, FeatureLoader]:
    sampler = WeightedRandomSampler(dataset_weights(train.records, config.dataset_power), num_samples=len(train), replacement=True)
    train_loader = DataLoader(train, batch_size=config.batch_size, sampler=sampler, collate_fn=_collate, drop_last=True)
    if len(train_loader) == 0:
        raise ValueError(f"{len(train)} training clips make no full batch of {config.batch_size}; lower batch_size.")
    return train_loader, DataLoader(val, batch_size=config.batch_size * 2, collate_fn=_collate)


def _model(
    config: TrainConfig, tokens: np.ndarray, train: FeatureDataset, init_payload: dict[str, Any] | None,
    taxonomy: Taxonomy, device: torch.device,
) -> tuple[BulwarkHeads, MultiTaskLoss, torch.optim.Optimizer]:
    """Fresh heads, loss weights and optimizer: from `init_payload`, or with input statistics
    fitted on a sample of the training tokens."""
    heads = BulwarkHeads(tokens.shape[-1], taxonomy, input_tokens=tokens.shape[1]).to(device)
    loss_fn = MultiTaskLoss(taxonomy).to(device)
    if init_payload is not None:
        heads.load_state_dict(init_payload["model"])
        if "loss" in init_payload:
            loss_fn.load_state_dict(init_payload["loss"])
    else:
        sample = torch.from_numpy(train.tokens[np.random.default_rng(config.seed).permutation(len(train))[:20000]].astype(np.float32))
        heads.fit_input_stats(sample)
    optimizer = torch.optim.AdamW(
        [*param_groups(heads.named_parameters(), config.weight_decay), *loss_param_group(loss_fn, config.loss_weight_lr)],
        lr=config.lr,
    )
    return heads, loss_fn, optimizer


def _train_pass(
    heads: BulwarkHeads, loss_fn: MultiTaskLoss, optimizer: torch.optim.Optimizer, loader: FeatureLoader, plan: LabelPlan,
    config: TrainConfig, device: torch.device, step: int, total: int,
    after_step: Callable[[int, torch.Tensor, dict[str, float]], None] | None = None,
) -> int:
    """One pass over `loader`, or until `total` steps; returns the step count."""
    heads.train()
    for features, batch in loader:
        set_lr(optimizer, cosine_lr(step, total, config.lr, config.warmup_steps))
        targets = plan.targets(batch, heads.taxonomy).to(device)
        loss, parts = loss_fn(heads(features.to(device)), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {parts}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        optimizer.step()
        step += 1
        if after_step is not None:
            after_step(step, loss, parts)
        if step >= total:
            break
    return step


@torch.no_grad()
def _validate(heads: BulwarkHeads, loader: FeatureLoader, device: torch.device, plan: LabelPlan) -> Validation:
    heads.eval()
    outputs: list[dict[str, torch.Tensor]] = []
    records: list[ClipRecord] = []
    for features, batch in loader:
        outputs.append({k: v.cpu() for k, v in heads(features.to(device)).items()})
        records.extend(batch)
    return validate(outputs, records, heads.taxonomy, plan.supervised, plan)
