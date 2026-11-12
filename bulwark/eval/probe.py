"""Linear probes on cached frozen-encoder features.

These answer different questions from the main report, which scores a released model on
Bulwark's own splits. A probe fits a linear classifier on frozen V-JEPA summary tokens:

    cv                       leave-subjects-out k-fold per dataset, for Bulwark labels whose
                             test split holds only a few people (mean and std over folds);
                             datasets the split policy excludes or keeps train-only are skipped;
    drive_and_act_midlevel   Drive&Act's official protocol: the 34 native mid-level classes,
                             official participant splits 0-2, balanced accuracy on test, per view;
    uta_rldd_folds           UTA-RLDD's published fold folders, leave one fold out (alertness).

Features must come from the frozen encoder (cached without `init`): a fine-tuned encoder
has seen Bulwark's training subjects, which may be a protocol's test subjects.

`LinearProbe` is separate code from `bulwark.adapters.fit_adapter`, and a probe score is
not the score an adapter with default settings would get. Both standardise inputs and fit
one linear layer with full-batch L-BFGS and an L2 penalty on the weights. The probe also
fits binary labels with one logit and regression labels with a squared error (adapters
are multinomial classifiers only), picks the L2 strength from `L2_GRID` on a validation
set (adapters take a fixed `l2`), and does not reweight classes (adapters balance them by
default), so it learns the protocols' own class frequencies.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bulwark.data.manifest import read_manifest
from bulwark.data.taxonomy import default_taxonomy
from bulwark.eval import metrics as M
from bulwark.eval.provenance import public_path, run_meta, sha256_file

PROTOCOLS = ("cv", "drive_and_act_midlevel", "uta_rldd_folds")
L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
TOKEN_POOLS = ("global", "all")


# ---- the probe ------------------------------------------------------------------------------


class LinearProbe:
    """Standardised inputs -> one linear layer, fitted full-batch with L-BFGS and an L2 penalty."""

    def __init__(self, kind: str, classes: int, l2: float) -> None:
        self.kind, self.classes, self.l2 = kind, classes, l2

    def fit(self, x: np.ndarray, y: np.ndarray, max_iter: int = 200) -> LinearProbe:
        torch.manual_seed(0)
        self.mean = x.mean(0, keepdims=True)
        self.std = x.std(0, keepdims=True).clip(1e-6)
        inputs = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))
        target = torch.from_numpy(y)
        self.layer = nn.Linear(inputs.shape[1], self.classes if self.kind == "categorical" else 1)
        nn.init.zeros_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)
        optimizer = torch.optim.LBFGS(self.layer.parameters(), max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            out = self.layer(inputs)
            if self.kind == "binary":
                loss = F.binary_cross_entropy_with_logits(out[:, 0], target.float())
            elif self.kind == "categorical":
                loss = F.cross_entropy(out, target.long())
            else:
                loss = F.mse_loss(out[:, 0], target.float())
            loss = loss + self.l2 * self.layer.weight.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        out = self.layer(torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))).numpy()
        return out if self.kind == "categorical" else out[:, 0]


def probe_scores(kind: str, classes: int, out: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if kind == "binary":
        return {"auroc": M.auroc(out, y)}
    if kind == "categorical":
        pred = out.argmax(-1)
        return {
            "balanced_accuracy": M.balanced_accuracy(pred, y),
            "accuracy": float((pred == y).mean()),
            "macro_f1": M.macro_f1(pred, y, classes),
        }
    return {"mae": float(np.abs(out - y).mean())}


SELECT = {"binary": "auroc", "categorical": "balanced_accuracy", "regression": "mae"}


def fit_selected(
    kind: str, classes: int, train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray], *, refit_on_val: bool
) -> tuple[LinearProbe, float]:
    """Pick the L2 strength on val, then fit on train (or train + val)."""
    metric = SELECT[kind]
    best, best_l2 = None, L2_GRID[len(L2_GRID) // 2]
    for l2 in L2_GRID:
        value = probe_scores(kind, classes, LinearProbe(kind, classes, l2).fit(*train).predict(val[0]), val[1])[metric]
        if value is None:
            continue
        value = -value if kind == "regression" else value
        if best is None or value > best:
            best, best_l2 = value, l2
    x, y = train
    if refit_on_val:
        x, y = np.concatenate([train[0], val[0]]), np.concatenate([train[1], val[1]])
    return LinearProbe(kind, classes, best_l2).fit(x, y), best_l2


# ---- features ---------------------------------------------------------------------------------


def load_frozen_features(features: Path, pool: str = "global") -> tuple[dict[str, int], np.ndarray]:
    """{clip_id: row} and probe inputs. `global` is the summary token that averages the whole
    token grid; `all` flattens all 25 summary tokens (more detail, 25x the memory)."""
    from bulwark.eval.predict import check_features_source
    from bulwark.train.common import load_feature_dir

    if pool not in TOKEN_POOLS:
        raise ValueError(f"pool must be one of {TOKEN_POOLS}")
    check_features_source("heads", None, features)  # same rule as frozen heads: no fine-tuned init
    clip_ids, tokens, logits = load_feature_dir(features)
    if logits:
        raise ValueError(f"{features} was cached with a fine-tuned encoder (it holds logits); probes need frozen-encoder features.")
    x = tokens[:, 0] if pool == "global" else tokens.reshape(len(tokens), -1)
    return {c: i for i, c in enumerate(clip_ids)}, x.astype(np.float32)


def subject_folds(subjects: Sequence[str], k: int, salt: str = "bulwark-cv") -> dict[str, int]:
    """Subjects dealt round-robin into k folds in a stable hashed order."""
    ordered = sorted(set(subjects), key=lambda s: hashlib.sha256(f"{salt}:{s}".encode()).hexdigest())
    return {s: i % k for i, s in enumerate(ordered)}


def _summarise(values: list[float | None]) -> dict[str, Any]:
    defined = [v for v in values if v is not None]
    return {
        "mean": float(np.mean(defined)) if defined else None,
        "std": float(np.std(defined)) if len(defined) > 1 else None,
        "folds": values,
        "defined_folds": len(defined),
    }


# ---- protocols ----------------------------------------------------------------------------------


def cross_validate(features: Path, manifest: Path, *, datasets: Sequence[str], folds: int = 5, pool: str = "global") -> dict[str, Any]:
    """Leave-subjects-out k-fold linear probes on every ground-truth Bulwark label of each dataset.

    Datasets the manifest's split policy excludes (zenodo_driver) or keeps train-only (no
    person ids, so subject folds would leak, or prompted labels) are skipped, not scored.
    """
    from bulwark.eval.report import is_metadata, manifest_policy

    taxonomy = default_taxonomy()
    policy = manifest_policy(manifest)
    row, x = load_frozen_features(features, pool)
    records = [r for r in read_manifest(manifest) if r.clip_id in row]
    results: dict[str, Any] = {}
    for dataset in datasets:
        if dataset in policy.excluded_datasets:
            results[dataset] = {"skipped": "excluded by the split policy: neither trained on nor scored"}
            continue
        if dataset in policy.train_only_datasets:
            results[dataset] = {"skipped": "train-only in the split policy (no person ids, or prompted labels), so it is not scored"}
            continue
        group = [r for r in records if r.dataset == dataset]
        fold_of = subject_folds([r.subject_id for r in group], folds)
        if len(fold_of) < folds:
            results[dataset] = {"skipped": f"{len(fold_of)} subjects with features, fewer than {folds} folds"}
            continue
        entries: dict[str, Any] = {}
        for name, spec in taxonomy.labels.items():
            data = M.label_rows(group, spec)
            if data is None or is_metadata(spec):
                continue
            classes = len(spec.classes) if spec.kind == "categorical" else 1
            if spec.kind != "regression" and len(np.unique(data.target)) < 2:
                entries[name] = {"skipped": "one class only in this dataset"}
                continue
            inputs = x[[row[group[i].clip_id] for i in data.rows]]
            fold = np.array([fold_of[s] for s in data.subjects])
            scores: dict[str, list[float | None]] = {}
            for f in range(folds):
                inner = (f + 1) % folds
                test, val, train = fold == f, fold == inner, (fold != f) & (fold != inner)
                if not test.any() or not val.any() or not train.any():
                    continue
                probe, _ = fit_selected(spec.kind, classes, (inputs[train], data.target[train]), (inputs[val], data.target[val]), refit_on_val=True)
                for metric, value in probe_scores(spec.kind, classes, probe.predict(inputs[test]), data.target[test]).items():
                    scores.setdefault(metric, []).append(value)
            entries[name] = {
                "n": int(len(data.rows)),
                "n_subjects": len(set(data.subjects.tolist())),
                **{metric: _summarise(values) for metric, values in scores.items()},
            }
        results[dataset] = {"subjects": len(fold_of), "folds": folds, "labels": entries}
    return results


def _official_splits(root: Path, split: int) -> tuple[dict[str, set[str]], dict[tuple[str, str], int]]:
    """Participants per part and official row counts per (part, view) for one Drive&Act split."""
    participants: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    rows: dict[tuple[str, str], int] = {}
    for part in participants:
        for path in sorted(root.glob(f"activities_3s/*/midlevel.chunks_90.split_{split}.{part}.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                lines = list(csv.DictReader(handle))
            participants[part].update(f"drive_and_act:vp{int(r['participant_id'])}" for r in lines)
            rows[(part, path.parent.name)] = len(lines)
    if not participants["test"]:
        raise FileNotFoundError(f"No midlevel.chunks_90.split_{split}.*.csv under {root}/activities_3s")
    return participants, rows


def drive_and_act_midlevel(features: Path, manifest: Path, root: Path, *, splits: Sequence[int] = (0, 1, 2), pool: str = "global") -> dict[str, Any]:
    """Drive&Act mid-level activity recognition with the dataset's own splits and classes.

    Differences from the original protocol are reported with the result: clips are 16 frames
    sampled over each 3 s chunk, and chunks the manifest drops (shorter than 16 frames, or
    with no video) are counted as excluded rather than scored.
    """
    row, x = load_frozen_features(features, pool)
    records = [r for r in read_manifest(manifest) if r.dataset == "drive_and_act" and r.clip_id in row and "native_activity" in r.meta]
    if not records:
        raise ValueError("No Drive&Act clips with native activities and cached features")
    classes = sorted({str(r.meta["native_activity"]) for r in records})
    index = {c: i for i, c in enumerate(classes)}
    views: dict[str, Any] = {}
    for view in sorted({r.view for r in records}):
        group = [r for r in records if r.view == view]
        inputs = x[[row[r.clip_id] for r in group]]
        target = np.array([index[str(r.meta["native_activity"])] for r in group])
        subjects = np.array([r.subject_id for r in group])
        per_split: list[dict[str, Any]] = []
        for split in splits:
            participants, official = _official_splits(root, split)
            masks = {part: np.isin(subjects, sorted(people)) for part, people in participants.items()}
            if not all(m.any() for m in masks.values()):
                per_split.append({"split": split, "skipped": "a part has no clips with features"})
                continue
            probe, l2 = fit_selected(
                "categorical", len(classes), (inputs[masks["train"]], target[masks["train"]]),
                (inputs[masks["val"]], target[masks["val"]]), refit_on_val=False,
            )
            scores = probe_scores("categorical", len(classes), probe.predict(inputs[masks["test"]]), target[masks["test"]])
            scored = int(masks["test"].sum())
            per_split.append({
                "split": split, **scores, "l2": l2, "test_clips": scored,
                "official_test_chunks": official.get(("test", view)),
                "excluded_test_chunks": (official[("test", view)] - scored) if ("test", view) in official else None,
            })
        defined = [s["balanced_accuracy"] for s in per_split if s.get("balanced_accuracy") is not None]
        views[view] = {
            "balanced_accuracy_mean": float(np.mean(defined)) if defined else None,
            "balanced_accuracy_std": float(np.std(defined)) if len(defined) > 1 else None,
            "splits": per_split,
        }
    view_means = [v["balanced_accuracy_mean"] for v in views.values() if v["balanced_accuracy_mean"] is not None]
    return {
        "classes": classes,
        "views": views,
        "mean_over_views": float(np.mean(view_means)) if view_means else None,
        "notes": [
            "Linear probe on frozen-encoder summary tokens; L2 strength chosen on each split's official val participants.",
            "Balanced accuracy = mean per-class recall over the classes present in each test split.",
            "Clips are 16 frames sampled across each 3 s chunk; chunks shorter than 16 frames are excluded and counted.",
            "Check the protocol against Martin et al., ICCV 2019 before comparing with published numbers.",
        ],
    }


def uta_rldd_folds(features: Path, manifest: Path, *, pool: str = "global") -> dict[str, Any]:
    """Leave-one-fold-out alertness probe over UTA-RLDD's published fold folders.

    Not the holdout result (which is zero-shot): here the probe trains on UTA-RLDD itself.
    """
    spec = default_taxonomy()["alertness"]
    row, x = load_frozen_features(features, pool)
    records = [r for r in read_manifest(manifest) if r.dataset == "uta_rldd" and r.clip_id in row and "fold" in r.meta]
    data = M.label_rows(records, spec)
    if data is None:
        raise ValueError("No UTA-RLDD clips with alertness labels, fold folders and cached features")
    fold_names = np.array([str(records[i].meta["fold"]).split("_part")[0] for i in data.rows])
    inputs = x[[row[records[i].clip_id] for i in data.rows]]
    names = sorted(set(fold_names.tolist()))
    classes = len(spec.classes)
    clip_scores: dict[str, list[float | None]] = {}
    video_scores: list[float | None] = []
    for position, held in enumerate(names):
        inner = names[(position + 1) % len(names)]
        test, val = fold_names == held, fold_names == inner
        train = ~test & ~val
        if not train.any() or not val.any():
            continue
        probe, _ = fit_selected("categorical", classes, (inputs[train], data.target[train]), (inputs[val], data.target[val]), refit_on_val=True)
        out = probe.predict(inputs[test])
        for metric, value in probe_scores("categorical", classes, out, data.target[test]).items():
            clip_scores.setdefault(metric, []).append(value)
        # Labels are per video: also score the mean clip probability per video.
        test_rows = np.flatnonzero(test)
        videos: dict[str, list[int]] = {}
        for j, i in enumerate(test_rows):
            videos.setdefault(records[data.rows[i]].media_path, []).append(j)
        mean_probs = np.stack([M.softmax(out[js]).mean(0) for js in videos.values()])
        video_target = np.array([data.target[test_rows[js[0]]] for js in videos.values()])
        video_scores.append(float((mean_probs.argmax(-1) == video_target).mean()))
    return {
        "folds": names,
        "clips": {metric: _summarise(values) for metric, values in clip_scores.items()},
        "video_accuracy": _summarise(video_scores),
        "notes": [
            "Linear probe on frozen-encoder summary tokens, trained on UTA-RLDD folds; not the zero-shot holdout result.",
            "Folds are the dataset's Fold<k> folders; check against Ghoddoosian et al., CVPRW 2019 before comparing.",
        ],
    }


def run_probe(protocol: str, features: Path, manifest: Path, out: Path, *, datasets: Sequence[str] = (), folds: int = 5, root: Path | None = None, pool: str = "global") -> Path:
    if protocol == "cv":
        results = cross_validate(features, manifest, datasets=datasets, folds=folds, pool=pool)
    elif protocol == "drive_and_act_midlevel":
        if root is None:
            raise ValueError("drive_and_act_midlevel needs the dataset root (official split CSVs)")
        results = drive_and_act_midlevel(features, manifest, root, pool=pool)
    elif protocol == "uta_rldd_folds":
        results = uta_rldd_folds(features, manifest, pool=pool)
    else:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    meta = {
        **run_meta(), "protocol": protocol, "features": public_path(features), "pool": pool,
        "manifest": {"path": public_path(manifest), "sha256": sha256_file(manifest)},
    }
    out.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "results": results}
    (out / f"probe-{protocol}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path = out / f"probe-{protocol}.html"
    path.write_text(_render(payload), encoding="utf-8")
    return path


def _mean_std(summary: dict[str, Any] | None) -> str:
    if not summary or summary.get("mean") is None:
        return "–"
    std = f" ± {summary['std']:.3f}" if summary.get("std") is not None else ""
    return f"{summary['mean']:.3f}{std} <small>({summary['defined_folds']} folds)</small>"


def _rows(header: Sequence[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _render(payload: dict[str, Any]) -> str:
    meta, results = payload["meta"], payload["results"]
    protocol = meta["protocol"]
    git = meta["git"]
    parts = [
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
        f"<title>Bulwark probe: {html.escape(protocol)}</title>",
        "<style>:root{--bg:#fff;--fg:#111;--line:#ddd;--muted:#666}"
        "@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--line:#333;--muted:#999}}"
        "body{font:14px/1.45 system-ui,sans-serif;margin:24px;background:var(--bg);color:var(--fg)}"
        "table{border-collapse:collapse;margin:8px 0 24px;display:block;overflow-x:auto}"
        "td,th{border:1px solid var(--line);padding:4px 8px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}small{color:var(--muted)}</style>",
        f"<h1>Linear probe: {html.escape(protocol)}</h1>",
        f"<p>Frozen-encoder features <code>{html.escape(meta['features'])}</code> ({html.escape(meta['pool'])} tokens); "
        f"manifest sha256 {meta['manifest']['sha256'][:12]}; code {git['commit']}{' (uncommitted changes)' if git['dirty'] else ''}; "
        f"{html.escape(meta['created_utc'])} UTC. A probe measures the frozen representation, not a released model.</p>",
    ]
    if protocol == "cv":
        for dataset, result in results.items():
            parts.append(f"<h2>{html.escape(dataset)}</h2>")
            if "skipped" in result:
                parts.append(f"<p>Skipped: {html.escape(result['skipped'])}</p>")
                continue
            rows = []
            for name, entry in result["labels"].items():
                if "skipped" in entry:
                    rows.append([name, f"<small>{html.escape(entry['skipped'])}</small>", "", ""])
                    continue
                metric = SELECT[default_taxonomy()[name].kind]
                rows.append([name, f"{metric}: {_mean_std(entry.get(metric))}", str(entry["n"]), str(entry["n_subjects"])])
            parts.append(f"<p>{result['subjects']} subjects in {result['folds']} folds; mean ± std over folds.</p>")
            parts.append(_rows(["label", "metric", "clips", "subjects"], rows))
    elif protocol == "drive_and_act_midlevel":
        rows = []
        for view, entry in results["views"].items():
            per_split = ", ".join(
                f"{s['split']}: {s['balanced_accuracy']:.3f}" if s.get("balanced_accuracy") is not None else f"{s['split']}: –"
                for s in entry["splits"]
            )
            scored = sum(s.get("test_clips", 0) for s in entry["splits"])
            excluded = sum(s.get("excluded_test_chunks") or 0 for s in entry["splits"])
            mean = entry["balanced_accuracy_mean"]
            std = entry["balanced_accuracy_std"]
            cell = "–" if mean is None else f"{mean:.3f}" + (f" ± {std:.3f}" if std is not None else "")
            rows.append([html.escape(view), cell, per_split, str(scored), str(excluded)])
        mean_over_views = "–" if results["mean_over_views"] is None else f"{results['mean_over_views']:.3f}"
        parts.append(f"<p>{len(results['classes'])} classes. Mean over views: {mean_over_views}.</p>")
        parts.append(_rows(["view", "balanced accuracy", "per split", "test clips scored", "test chunks excluded"], rows))
    else:
        rows = [[metric, _mean_std(summary)] for metric, summary in results["clips"].items()]
        rows.append(["video accuracy", _mean_std(results["video_accuracy"])])
        parts.append(f"<p>Folds: {', '.join(results['folds'])}.</p>")
        parts.append(_rows(["metric", "mean ± std over held-out folds"], rows))
    notes = results.get("notes", [])
    if notes:
        parts.append("<ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul>")
    return "\n".join(parts)
