"""Measure how an adapter's accuracy grows with the number of labelled clips per class.

Trains adapters on k clips per class (k = 1, 2, 4, ...) drawn from the training split of a
`bulwark train cache` features directory, scores each on a held-out split, and repeats
every k with different draws. The script refuses to run when a subject appears in both
splits, or when the evaluation split holds fewer than two classes (a constant prediction
would then score perfectly). Writes the curve as JSON and prints a table.

    PYTHONPATH=. python examples/fewshot_curve.py \\
        --features data/features/frozen --manifest data/manifest.parquet \\
        --label activity --dataset drive_and_act --out runs/fewshot/activity.json

When --train-split equals --eval-split, that split's subjects are divided instead:
--eval-subjects of them (default: half), drawn with --seed, are scored and the rest train.
This measures adapters on a dataset no Bulwark model was trained on, such as UTA-RLDD,
whose clips are all in the holdout split:

    PYTHONPATH=. python examples/fewshot_curve.py \\
        --features data/features/frozen --manifest data/manifest.parquet \\
        --label alertness --dataset uta_rldd --train-split holdout --eval-split holdout \\
        --out runs/fewshot/alertness-uta_rldd.json

Targets are ground-truth labels only. Clips are drawn independently, so draws can share a
subject or a video; read small-k results with the subject count printed next to them. The
spread over draws says nothing about the evaluation set's own sampling error; for that,
the adapter trained on all clips gets a 95% interval from resampling evaluation subjects
(with at least 10 of them; fewer make the interval meaningless).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bulwark.adapters import fit_adapter, load_cached_embeddings
from bulwark.eval.metrics import auroc, macro_f1

MIN_CI_SUBJECTS = 10  # fewer evaluation subjects make a subject bootstrap meaningless


def balanced_accuracy(pred: np.ndarray, truth: np.ndarray) -> float | None:
    """Mean recall over the classes present in `truth`; None with fewer than two."""
    present = np.unique(truth)
    if len(present) < 2:
        return None
    return float(np.mean([(pred[truth == c] == c).mean() for c in present]))


def score(probs: np.ndarray, targets: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    """Balanced accuracy, macro-F1 and, for two classes, AUROC; metrics undefined on `targets` are left out."""
    pred = probs.argmax(1)
    truth = np.array([classes.index(t) for t in targets])
    values = {
        "balanced_accuracy": balanced_accuracy(pred, truth),
        "macro_f1": macro_f1(pred, truth, len(classes)) if len(np.unique(truth)) > 1 else None,
        "auroc": auroc(probs[:, 1], (truth == 1).astype(int)) if len(classes) == 2 else None,
    }
    return {name: float(value) for name, value in values.items() if value is not None}


def summarise(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean, std, min and max of each metric over the runs that define it."""
    out = {}
    for metric in dict.fromkeys(name for run in runs for name in run):
        values = [run[metric] for run in runs if metric in run]
        out[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "runs": len(values),
        }
    return out


def subject_interval(
    probs: np.ndarray, targets: np.ndarray, classes: tuple[str, ...], subjects: Sequence[str], draws: int, seed: int
) -> dict[str, Any]:
    """95% percentile interval of each metric, resampling evaluation subjects with replacement."""
    names = np.array(subjects)
    groups = [np.flatnonzero(names == s) for s in np.unique(names)]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(draws):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for name, value in score(probs[rows], targets[rows], classes).items():
            samples.setdefault(name, []).append(value)
    # Resamples where a metric is undefined (one class only) are dropped; `draws` says how many remain.
    return {
        name: {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5)), "draws": len(values)}
        for name, values in samples.items()
    }


def split_subjects(subjects: Sequence[str], count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the subjects left for training, and rows of `count` subjects drawn with `seed` for evaluation."""
    names = sorted(set(subjects))
    held = set(np.random.default_rng(seed).choice(names, size=count, replace=False).tolist())
    scored = np.array([subject in held for subject in subjects])
    return np.flatnonzero(~scored), np.flatnonzero(scored)


def draw(labels: np.ndarray, classes: tuple[str, ...], shots: int, rng: np.random.Generator) -> np.ndarray:
    """Up to `shots` clips per class (all of a class's clips when it has fewer)."""
    picks = []
    for cls in classes:
        rows = np.flatnonzero(labels == cls)
        picks.append(rng.choice(rows, size=min(shots, len(rows)), replace=False))
    return np.concatenate(picks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True, help="features dir from `bulwark train cache`")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", required=True, help="a binary or categorical taxonomy label")
    parser.add_argument("--dataset", action="append", default=None, help="restrict to these datasets (repeatable)")
    parser.add_argument("--train-split", default="train", choices=["train", "val", "test", "holdout"])
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test", "holdout"])
    parser.add_argument(
        "--eval-subjects", type=int, default=None,
        help="with --train-split equal to --eval-split: subjects drawn for evaluation (default: half)",
    )
    parser.add_argument("--shots", default="1,2,4,8,16,32", help="clips per class, comma separated")
    parser.add_argument("--repeats", type=int, default=5, help="draws per shot count")
    parser.add_argument("--hidden", type=int, default=0, help="0 = linear adapter, >0 = MLP width")
    parser.add_argument("--pooling", default="mean", choices=["mean", "first", "flatten"])
    parser.add_argument("--bootstrap", type=int, default=1000, help="subject resamples for the all-clips interval (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write the curve as JSON here")
    args = parser.parse_args()

    partition = args.train_split == args.eval_split
    if args.eval_subjects is not None and not partition:
        parser.error("--eval-subjects divides one split; pass the same --train-split and --eval-split")
    x_train, y_train, train_records = load_cached_embeddings(
        args.features, args.manifest, args.label, split=args.train_split, datasets=args.dataset
    )
    if partition:
        everyone = sorted({r.subject_id for r in train_records})
        count = args.eval_subjects if args.eval_subjects is not None else (len(everyone) + 1) // 2
        if not 0 < count < len(everyone):
            raise SystemExit(
                f"{args.label}: the {args.train_split} split has {len(everyone)} subject(s); --eval-subjects must "
                f"leave at least one for training and one for evaluation, got {count}"
            )
        train_rows, eval_rows = split_subjects([r.subject_id for r in train_records], count, args.seed)
        x_eval, y_eval, eval_records = x_train[eval_rows], [y_train[i] for i in eval_rows], [train_records[i] for i in eval_rows]
        x_train, y_train, train_records = x_train[train_rows], [y_train[i] for i in train_rows], [train_records[i] for i in train_rows]
    else:
        x_eval, y_eval, eval_records = load_cached_embeddings(
            args.features, args.manifest, args.label, split=args.eval_split, datasets=args.dataset
        )
    shared = {r.subject_id for r in train_records} & {r.subject_id for r in eval_records}
    if shared:
        raise SystemExit(
            f"{len(shared)} subject(s) are in both {args.train_split} and {args.eval_split} (for example "
            f"{sorted(shared)[0]}); a curve on seen subjects overstates accuracy. Use subject-disjoint splits."
        )
    train_labels = np.array(y_train)
    classes = tuple(sorted(set(y_train)))
    if len(classes) < 2:
        raise SystemExit(f"{args.label}: the training clips have only the classes {classes}; an adapter needs two")
    targets = np.array(y_eval)
    unseen = sorted(set(y_eval) - set(classes))
    if unseen:
        keep = ~np.isin(targets, unseen)
        print(f"Dropping {int((~keep).sum())} {args.eval_split} clips of classes absent from training: {unseen}")
        x_eval, targets = x_eval[keep], targets[keep]
        eval_records = [r for r, k in zip(eval_records, keep, strict=True) if k]
    if len(set(targets.tolist())) < 2:
        raise SystemExit(
            f"{args.label}: the {args.eval_split} split holds {len(targets)} clips of {sorted(set(targets.tolist()))} only. "
            "With one class a constant prediction scores perfectly, so there is nothing to measure."
        )
    fit = {"hidden": args.hidden, "pooling": args.pooling}

    rows: list[dict[str, Any]] = []
    for shots in [int(s) for s in args.shots.split(",")]:
        runs, sizes = [], []
        for repeat in range(args.repeats):
            rng = np.random.default_rng([args.seed, shots, repeat])
            index = draw(train_labels, classes, shots, rng)
            adapter = fit_adapter(x_train[index], train_labels[index].tolist(), classes=classes, seed=repeat, **fit)
            runs.append(score(adapter.predict_proba(x_eval), targets, classes))
            sizes.append(len(index))
        rows.append({"shots": shots, "train_clips": int(np.mean(sizes)), "repeats": args.repeats, **summarise(runs)})
    adapter = fit_adapter(x_train, y_train, classes=classes, seed=args.seed, **fit)
    probs = adapter.predict_proba(x_eval)
    subjects = [r.subject_id for r in eval_records]
    rows.append({"shots": "all", "train_clips": len(y_train), "repeats": 1, **summarise([score(probs, targets, classes)])})
    if args.bootstrap and len(set(subjects)) >= MIN_CI_SUBJECTS:
        rows[-1]["ci95_subject_bootstrap"] = subject_interval(probs, targets, classes, subjects, args.bootstrap, args.seed)

    # A constant prediction scores 1 / classes balanced accuracy and 0.5 AUROC.
    majority = Counter(y_train).most_common(1)[0][0]
    constant = np.tile(np.eye(len(classes))[classes.index(majority)], (len(targets), 1))
    baseline = score(constant, targets, classes)
    if "auroc" in baseline:
        baseline["auroc"] = 0.5
    result = {
        "label": args.label,
        "datasets": args.dataset or "all",
        "features": str(args.features),
        "manifest": str(args.manifest),
        "classes": list(classes),
        "train_clips_per_class": dict(sorted(Counter(y_train).items())),
        "train_split": args.train_split,
        "train_subjects": len({r.subject_id for r in train_records}),
        "eval_split": args.eval_split,
        # Set when train and evaluation subjects were drawn from one split, with this seed.
        "subject_draw_seed": args.seed if partition else None,
        "eval_clips": len(targets),
        "eval_subjects": len(set(subjects)),
        "eval_clips_per_class": dict(sorted(Counter(targets.tolist()).items())),
        "adapter": fit,
        "majority_class_baseline": baseline,
        "curve": rows,
    }
    drawn = f"; the {args.train_split} subjects divided with seed {args.seed}" if partition else ""
    print(
        f"{args.label}: training ({args.train_split}): {len(y_train)} clips, {result['train_subjects']} subject(s); "
        f"evaluation ({args.eval_split}): {result['eval_clips']} clips, {result['eval_subjects']} subject(s){drawn}; "
        f"majority-class baseline {json.dumps(baseline)}"
    )
    metrics = list(dict.fromkeys(m for row in rows for m in row if isinstance(row[m], dict) and "mean" in row[m]))
    print("Each cell: mean ± std [min] over the repeats.")
    print(f"{'shots':>6} {'clips':>6}  " + "  ".join(f"{m:>22}" for m in metrics))
    for row in rows:
        cells = "  ".join(
            (f"{row[m]['mean']:.3f} ± {row[m]['std']:.3f} [{row[m]['min']:.2f}]" if m in row else "-").rjust(22) for m in metrics
        )
        print(f"{row['shots']!s:>6} {row['train_clips']:>6}  {cells}")
    interval = rows[-1].get("ci95_subject_bootstrap")
    if interval:
        print("All clips, 95% subject-bootstrap interval: " + ", ".join(f"{m} [{v['low']:.3f}, {v['high']:.3f}]" for m, v in interval.items()))
    elif args.bootstrap:
        print(f"No subject-bootstrap interval: {result['eval_subjects']} evaluation subject(s), fewer than {MIN_CI_SUBJECTS}.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
