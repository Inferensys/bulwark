"""k-fold best.pt selection for the heads stage (`select_folds`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_training_pipeline import data, run_cache, run_heads  # noqa: F401  (data: the fixture)

from bulwark.cli import _train_config
from bulwark.data.manifest import read_manifest
from bulwark.data.splits import subject_folds
from bulwark.model.bulwark import load_payload
from bulwark.train import train_heads
from bulwark.train.common import load_config
from bulwark.train.train_heads import fold_select
from bulwark.train.validation import Validation


def test_heads_pick_best_pt_by_k_fold_over_train_and_val_subjects(data: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    seen: list[tuple[set[str], set[str], set[str]]] = []  # per pass: subjects and splits trained on, held out
    train_pass, validate = train_heads._train_pass, train_heads._validate

    def record_pass(heads, loss_fn, optimizer, loader, *args, **kwargs):  # type: ignore[no-untyped-def]
        records = loader.dataset.records
        seen.append(({r.subject_id for r in records}, {r.split for r in records}, set()))
        return train_pass(heads, loss_fn, optimizer, loader, *args, **kwargs)

    def record_validate(heads, loader, *args):  # type: ignore[no-untyped-def]
        seen[-1][2].update(r.subject_id for r in loader.dataset.records)
        return validate(heads, loader, *args)

    monkeypatch.setattr(train_heads, "_train_pass", record_pass)
    monkeypatch.setattr(train_heads, "_validate", record_validate)
    run_cache(data)
    out = run_heads(data, select_folds=3)  # 3 epochs of 6 steps (52 train clips, batch 8)

    # Each fold model trains on train and val subjects outside its fold and is scored on
    # the fold; the folds together hold out every train and val subject once. The main
    # run (the last 3 passes) trains on the train split only.
    records = read_manifest(data / "manifest.parquet")
    folds = subject_folds(records, 3)
    assert set(folds) == {r.subject_id for r in records if r.split in ("train", "val")}
    held_out: list[str] = []
    for fold in range(3):
        passes = seen[fold * 3 : fold * 3 + 3]
        trained, splits, scored = passes[0]
        assert {s for s, f in folds.items() if f == fold} == scored and not trained & scored
        assert splits == {"train", "val"} and all(p[0] == trained for p in passes)
        held_out += sorted(scored)
    assert sorted(held_out) == sorted(folds)
    assert all(p[1] == {"train"} for p in seen[9:]) and len(seen) == 12

    best, last = load_payload(out / "best.pt"), load_payload(out / "last.pt")
    select = best["select"]
    assert select["criterion"] == "gt_mean_headline" and select["folds"] == 3 and len(select["fold_scores"]) == 3
    assert select["score"] == max(s for s in select["epoch_scores"] if s is not None)
    assert best["step"] == 6 * select["epoch"] and last["kfold"] == select and last["best"] == select["score"]
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    cv = [e for e in log if e.get("event") == "cv"]
    assert [(e["fold"], e["epoch"]) for e in cv] == [(f, e) for f in range(3) for e in (1, 2, 3)]
    chosen = [e for e in cv if e["epoch"] == select["epoch"]]
    for label, support in select["support"].items():
        assert support["clips"] == sum(e["val_select_support"].get(label, {}).get("clips", 0) for e in chosen)
    assert [e["step"] for e in log if e.get("event") == "val"] == [6, 12, 18]  # the val split is still reported
    assert load_payload(out / "final.pt")["step"] == 18


def test_k_fold_best_pt_is_the_chosen_step_after_a_mid_epoch_resume(data: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    calls: list[int] = []
    select = {"criterion": "gt_mean_headline", "folds": 3, "epoch": 1, "score": 0.5, "epoch_scores": [0.5, 0.4, 0.3]}

    def cross_validate(*args):  # type: ignore[no-untyped-def]
        calls.append(1)
        return select

    train_pass = train_heads._train_pass

    def evicted_at_step_four(*args):  # type: ignore[no-untyped-def]
        if len(args) == 10:  # the main run passes after_step
            after_step = args[9]

            def stop(done, loss, parts):  # type: ignore[no-untyped-def]
                after_step(done, loss, parts)
                if done == 4:
                    raise RuntimeError("session evicted")

            args = (*args[:9], stop)
        return train_pass(*args)

    monkeypatch.setattr(train_heads, "cross_validate", cross_validate)
    monkeypatch.setattr(train_heads, "_train_pass", evicted_at_step_four)
    run_cache(data)
    with pytest.raises(RuntimeError, match="evicted"):
        run_heads(data, select_folds=3, checkpoint_minutes=0)  # last.pt after every step
    assert load_payload(data / "heads" / "last.pt")["step"] == 4 and not (data / "heads" / "best.pt").exists()
    monkeypatch.setattr(train_heads, "_train_pass", train_pass)
    out = run_heads(data, select_folds=3, checkpoint_minutes=0)  # resumes at step 4; passes now end at 10, 16, 18

    best = load_payload(out / "best.pt")
    assert best["select"] == select and best["step"] == 6 * select["epoch"] and calls == [1]
    assert load_payload(out / "final.pt")["step"] == 18


def test_select_folds_must_be_zero_or_at_least_two(data: Path) -> None:  # noqa: F811
    with pytest.raises(ValueError, match="select_folds must be 0"):
        run_heads(data, select_folds=1)


@pytest.mark.parametrize("name", ["heads.yaml", "heads-clean.yaml"])
def test_shipped_heads_configs_select_best_pt_by_5_fold_cross_validation(name: str) -> None:
    # docs/training.md (section 2) describes 5-fold selection as the shipped default and
    # `--set select_folds=0` as the way back to the val split; changing either changes that text.
    path = Path(__file__).resolve().parents[1] / "configs" / "train" / name
    assert load_config(path).select_folds == 5
    assert _train_config(path, ["select_folds=0"], fresh=True).select_folds == 0  # the documented override
    training = " ".join((path.parents[2] / "docs" / "training.md").read_text(encoding="utf-8").split())
    assert f"`{name}`" in training and "(`select_folds: 5`)" in training and "`--set select_folds=0`" in training


def _validation(score: float | None, **per_label: float) -> Validation:
    return Validation(score, per_label, {name: {"clips": 10, "subjects": 2} for name in per_label}, {})


def test_fold_select_takes_the_earliest_best_mean_and_aggregates_the_folds() -> None:
    scores = [
        [_validation(0.6, a=0.6), _validation(0.8, a=0.7, b=0.9), _validation(0.8, a=0.8)],
        [_validation(None), _validation(0.6, a=0.6), _validation(0.6, a=0.6)],
        [],  # a fold without held-out clips
    ]
    select = fold_select(scores, 3)
    assert select["epoch_scores"] == pytest.approx([0.6, 0.7, 0.7]) and select["epoch"] == 2
    assert select["score"] == select["epoch_scores"][1]
    assert select["fold_scores"] == [0.8, 0.6, None]
    assert select["metrics"] == {"a": pytest.approx(0.65), "b": 0.9}
    assert select["support"] == {"a": {"clips": 20, "subjects": 4}, "b": {"clips": 10, "subjects": 2}}
    nothing = fold_select([[_validation(None)], []], 2)
    assert (nothing["epoch"], nothing["score"], nothing["epoch_scores"]) == (None, None, [None])
