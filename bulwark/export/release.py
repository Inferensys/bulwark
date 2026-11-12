"""Whether trained weights may be published, and under which licence.

A checkpoint records the datasets it was trained on and `commercial_only` in its
provenance (`bulwark.train.provenance`); an exported graph and a release directory carry
the same record. The weights also carry what they learned from upstream models: the
distillation teacher's logits, the checkpoint a run started from (`init`), and the model
behind cached features or logits. Their datasets, recorded under `provenance.inputs`,
count as training datasets here. `check_release` compares that record with the dataset
registry:

* every training dataset, own or upstream, must have `weights_release_ok=True` (cleared
  for publishing trained weights, on the grounds its `weights_release_basis` records: the
  terms, the owners' written permission or the maintainers' own decision), whatever the
  licence;
* no model up the chain may have unrecorded training data (an init, teacher or features
  dir whose provenance or datasets are missing, at any step up);
* the code must be a recorded commit without uncommitted changes (RELEASING.md);
* weights released under a licence that does not restrict use to non-commercial research
  (`research_only`) must also come from a `commercial_only` run whose datasets are all
  `commercial_ok`.

v0 weights are research-only: CC BY-NC 4.0 plus the terms of the training datasets.
The registry flags are our reading of each dataset's terms, or our own dated decision
(`weights_release_basis`), not legal advice and not a statement by any dataset's owners.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bulwark.data.registry import BY_NAME
from bulwark.train.provenance import unrecorded_inputs, unrecorded_reason

# The licence `bulwark release`, `bulwark export` and examples/make_release.py write by default.
RESEARCH_LICENCE = "CC-BY-NC-4.0 plus training-dataset terms; non-commercial research only (see MODEL_CARD.md)"
# A licence restricts use to non-commercial research when it says so ("non-commercial",
# CC BY-NC, "research use only") and neither mentions commercial use other than as
# non-commercial nor names a licence that allows it (MIT, Apache, BSD, CC BY without NC,
# ...). See `research_only` and `bulwark.inference.check_weights_licence`.
NON_COMMERCIAL_LICENCE = re.compile(
    r"non[- ]?commercial|\bNC\b|\bresearch(?:[- ](?:use|purposes))?[- ]only\b", re.IGNORECASE
)
COMMERCIAL_LICENCE = re.compile(
    r"(?<!non)(?<!non-)(?<!non )commercial"
    r"|\b(?:MIT|Apache|BSD|ISC|MPL|(?:A|L)?GPL|CC0|Unlicense|public[- ]domain)\b"
    r"|\bCC[- ]?BY\b(?![- ]NC\b)",
    re.IGNORECASE,
)
# Inputs of a training run that are models trained on data (bulwark.train.provenance).
UPSTREAM_INPUTS = ("teacher", "init", "features")
OWN = "training"  # `via` of a dataset in the run's own training clips
# The Hugging Face model repositories that host the released weights, one per model: B1 as
# a release directory (PyTorch safetensors), E1 as ONNX graphs. The README.md of each is
# that model's card (scripts/hf_release.py renders it from release/hf/). A release
# directory's config.json and an ONNX graph's metadata record their model's card,
# `model_card_url(kind)`; these two ids are the only place the repositories are named.
HUB_URL = "https://huggingface.co"
B1_REPO = "Inferensys/bulwark-b1-driver-monitoring"
E1_REPO = "Inferensys/bulwark-e1-driver-monitoring-onnx"
MODEL_REPOS = {"b1": B1_REPO, "e1": E1_REPO}
MODEL_RELEASE = "v0"  # also the Hub tag of each repository's release commit
# The sha256 of the B1 checkpoint that model release v0 publishes: the `best.pt` of the
# LoRA fine-tune with AUC Distracted Driver v2 (run directory checkpoints/b1-lora-auc;
# docs/training.md, "Measured for v0"). B1's release directory is built from it, and E1 v0
# must be distilled from its outputs, since E1's card names it as E1's teacher.
B1_CHECKPOINT_SHA256 = "de184992f7ed95b0c26fd1aee93c135a8fc9e03d7e3d8ac62abe9f53f96e191e"


@dataclass
class TrainingRecord:
    """What a model says about its training data and code."""

    source: str                                # where the record was read from
    commercial_only: bool | None               # None: not recorded
    datasets: dict[str, dict[str, Any]]        # the run's own clips: name -> recorded licence, commercial_ok, splits
    commit: str | None = None
    dirty: bool | None = None
    # Datasets of upstream models: name -> {"via": ["teacher", ...], "licence", "commercial_ok"}.
    upstream: dict[str, dict[str, Any]] = field(default_factory=dict)
    unrecorded: list[str] = field(default_factory=list)  # upstream inputs whose training data is not recorded


@dataclass
class ReleaseCheck:
    record: TrainingRecord
    commercial: bool                           # checked for a licence that is not research-only (`research_only`)
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> registry terms
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> dict[str, Any]:
        """JSON-ready, for ONNX metadata and release configs."""
        return {
            "checked_for": "commercial licence" if self.commercial else "research licence",
            "cleared": self.ok,
            "problems": list(self.problems),
            "datasets": {
                name: {k: entry[k] for k in ("licence", "commercial_ok", "weights_release_ok", "weights_release_basis", "in_registry", "via")}
                for name, entry in self.datasets.items()
            },
            "unrecorded_inputs": list(self.record.unrecorded),
        }


def released_model(kind: str) -> str:
    """The released model, "b1" or "e1", that a model kind belongs to (`TorchModel.kind`, a
    release's `kind`): B1 on the frozen encoder ("b1-frozen") is a B1."""
    if kind == "e1":
        return "e1"
    if kind.startswith("b1"):
        return "b1"
    raise ValueError(f"No released model has kind {kind!r}; B1 kinds start with 'b1', E1's is 'e1'")


def model_repo(kind: str) -> str:
    """The Hugging Face model repository of a model kind (`released_model`)."""
    return MODEL_REPOS[released_model(kind)]


def model_card_url(kind: str) -> str:
    """The model card of a model kind: its Hugging Face model repository's page."""
    return f"{HUB_URL}/{model_repo(kind)}"


def research_only(licence: str) -> bool:
    """Whether a weights licence restricts use to non-commercial research
    (NON_COMMERCIAL_LICENCE, and nothing COMMERCIAL_LICENCE matches). Weights under any
    other licence need `check_release(..., commercial=True)` to clear them."""
    return COMMERCIAL_LICENCE.search(licence) is None and NON_COMMERCIAL_LICENCE.search(licence) is not None


def prepare_release(
    checkpoint: Path, weights_licence: str = RESEARCH_LICENCE, *, kind: str
) -> tuple[ReleaseCheck, dict[str, Any]]:
    """The release check of a checkpoint, and the fields a release directory adds to
    config.json with it (`save_release(..., extra=...)`): `model_card` (the card of the
    model `kind`, as `bulwark.inference.load` reports it: `model_card_url`),
    `model_release` and `release_check`, the verdict of `bulwark release-check`.

    `bulwark release` and examples/make_release.py both use it. The check is the
    commercial one when `weights_licence` does not restrict use to non-commercial
    research (`research_only`); such a licence is refused (ValueError) unless the check
    clears the weights. A research licence is never refused here: the verdict, cleared or
    not, is recorded, and `bulwark release-check` says whether the weights may be
    published.
    """
    card = model_card_url(kind)
    commercial = not research_only(weights_licence)
    check = check_release(training_record(checkpoint), commercial=commercial)
    if commercial and not check.ok:
        raise ValueError(
            f"Licence {weights_licence!r} does not restrict use to non-commercial research, so it needs weights "
            f"cleared for commercial release: {' '.join(check.problems)}"
        )
    return check, {"model_card": card, "model_release": MODEL_RELEASE, "release_check": check.summary()}


def training_record(path: Path) -> TrainingRecord:
    """The training record of a checkpoint (.pt), an exported graph (.onnx) or a release directory."""
    path = Path(path)
    if path.suffix == ".onnx":
        from bulwark.export.onnx import read_contract

        checkpoint = read_contract(path).get("checkpoint") or {}
        return record_from_provenance(checkpoint.get("provenance") or {}, f"{path.name} (ONNX contract)")
    if path.is_dir():
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        source = config.get("source") or {}
        upstream, unrecorded = release_upstream(source, config.get("release_check") or {})
        return TrainingRecord(
            source=f"{path.name}/config.json",
            commercial_only=source.get("commercial_only"),
            datasets=_datasets(source.get("datasets")),
            commit=source.get("code_commit"),
            dirty=source.get("code_dirty"),
            upstream=upstream,
            unrecorded=unrecorded,
        )
    from bulwark.model.bulwark import load_payload

    return record_from_provenance(load_payload(path).get("provenance") or {}, path.name)


def release_upstream(source: dict[str, Any], stored: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Upstream datasets and unrecorded inputs of a release directory's config.json.

    `source` is its `source` block: `bulwark.inference.save_release` writes
    `upstream_datasets` ({name: {"via", "licence", "commercial_ok"}}) and
    `unrecorded_inputs` there (both None for a model without provenance). `stored` is the
    `release_check` that `bulwark release` adds; releases written before the source block
    carried them list the upstream datasets only there. Both are read, and a dataset
    either names counts.
    """
    upstream: dict[str, dict[str, Any]] = {}
    for entries in (stored.get("datasets"), source.get("upstream_datasets")):
        for name, entry in (entries if isinstance(entries, dict) else {}).items():
            entry = entry if isinstance(entry, dict) else {}
            via = [v for v in entry.get("via") or () if v != OWN]
            if not via:
                continue  # the run's own clips, recorded under `datasets`
            item = upstream.setdefault(name, {"via": [], "licence": entry.get("licence"), "commercial_ok": None})
            item["via"] += [v for v in via if v not in item["via"]]
            item["licence"] = item["licence"] or entry.get("licence")
            if entry.get("commercial_ok") is not None:  # False anywhere wins; None: not recorded
                item["commercial_ok"] = bool(entry["commercial_ok"]) and item["commercial_ok"] is not False
    unrecorded = list(dict.fromkeys([*(stored.get("unrecorded_inputs") or ()), *(source.get("unrecorded_inputs") or ())]))
    return dict(sorted(upstream.items())), unrecorded


def record_from_provenance(provenance: dict[str, Any], source: str) -> TrainingRecord:
    code = provenance.get("code") or {}
    upstream, unrecorded = upstream_datasets(provenance.get("inputs") or {})
    return TrainingRecord(
        source=source,
        commercial_only=provenance.get("commercial_only"),
        datasets=_datasets(provenance.get("datasets")),
        commit=code.get("commit"),
        dirty=code.get("dirty"),
        upstream=upstream,
        unrecorded=unrecorded,
    )


def upstream_datasets(inputs: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Datasets of the models a run learned from (`inputs.teacher`, `.init`, `.features`),
    and the inputs whose training data is not recorded.

    `init` names a checkpoint; `features` and the distillation `teacher` name a features
    dir, whose data is that of the checkpoint it was cached with. Tokens of the frozen
    V-JEPA encoder (cached without one) have no Bulwark training data behind them. Each
    input records its upstream model's own training datasets (`datasets`), those of the
    models that one was trained from in turn (`upstream_datasets`), and the inputs up that
    chain whose training data is not recorded (`unrecorded_inputs`), so both lists cover
    the whole chain (heads -> B1 -> teacher features -> E1). An unrecorded input is listed
    as "<key> (<reason>)", one further up as "<key> -> <item>"
    (`bulwark.train.provenance.unrecorded_inputs`). Inputs recorded before
    `upstream_datasets` and `unrecorded_inputs` existed (absent, or None for a features
    dir cached before them) contribute their own datasets and status only."""
    found: dict[str, dict[str, Any]] = {}
    for key in UPSTREAM_INPUTS:
        entry = inputs.get(key)
        if not isinstance(entry, dict) or unrecorded_reason(entry) is not None:
            continue
        for value in (entry.get("datasets"), entry.get("upstream_datasets")):
            for name, info in _datasets(value).items():
                item = found.setdefault(name, {"via": [], "licence": info.get("licence"), "commercial_ok": None})
                if key not in item["via"]:
                    item["via"].append(key)
                item["licence"] = item["licence"] or info.get("licence")
                if info.get("commercial_ok") is not None:  # False anywhere wins; None: not recorded
                    item["commercial_ok"] = bool(info["commercial_ok"]) and item["commercial_ok"] is not False
    return dict(sorted(found.items())), unrecorded_inputs(inputs, UPSTREAM_INPUTS)


def _datasets(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(k): dict(v) if isinstance(v, dict) else {} for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return {str(name): {} for name in sorted(value)}
    return {}


def check_release(record: TrainingRecord, *, commercial: bool = False) -> ReleaseCheck:
    """Problems that stop these weights from being published (none: cleared).

    Every dataset the record names, own or upstream and in the registry or not, is listed
    in `ReleaseCheck.datasets` with `via` (how it reached the weights), `in_registry` and
    `weights_release_basis` (on what the registry's clearance rests; None when it records
    none).
    """
    check = ReleaseCheck(record, commercial)
    if not record.datasets:
        check.problems.append(f"{record.source} records no training datasets, so their terms cannot be checked.")
    for item in record.unrecorded:
        check.problems.append(f"Upstream input {item}: the data behind it cannot be checked.")
    names = sorted(set(record.datasets) | set(record.upstream))
    unknown = [name for name in names if name not in BY_NAME]
    if unknown:
        check.problems.append(f"Datasets not in the registry: {unknown}.")
    for name in names:
        own, upstream = record.datasets.get(name), record.upstream.get(name)
        recorded = [entry for entry in (own, upstream) if entry is not None]
        via = ([OWN] if own is not None else []) + list((upstream or {}).get("via") or ())
        spec = BY_NAME.get(name)
        if spec is None:
            licence = next((e["licence"] for e in recorded if e.get("licence")), "not recorded")
            check.datasets[name] = {
                "licence": licence,
                "commercial_ok": all(e.get("commercial_ok") is True for e in recorded),
                "weights_release_ok": None,
                "weights_release_basis": None,
                "in_registry": False,
                "via": via,
                "terms": "not in the registry",
                "train_clips": ((own or {}).get("splits") or {}).get("train"),
            }
            continue
        check.datasets[name] = {
            "licence": spec.licence,
            "commercial_ok": spec.commercial_ok and all(e.get("commercial_ok", True) is not False for e in recorded),
            "weights_release_ok": spec.weights_release_ok,
            "weights_release_basis": spec.weights_release_basis or None,
            "in_registry": True,
            "via": via,
            "terms": spec.terms or "not yet reviewed",
            "train_clips": ((own or {}).get("splits") or {}).get("train"),
        }
    uncleared = {n: e["weights_release_ok"] for n, e in check.datasets.items() if e["in_registry"] and e["weights_release_ok"] is not True}
    if uncleared:
        states = ", ".join(f"{n} ({'not allowed' if v is False else 'not determined'})" for n, v in uncleared.items())
        check.problems.append(f"Publishing weights trained on these datasets is not cleared: {states}.")
    if not record.commit:
        check.problems.append("No code commit is recorded.")
    elif record.dirty is None:
        check.problems.append(f"Commit {record.commit[:12]} is recorded, but not whether the code had uncommitted changes.")
    elif record.dirty:
        check.problems.append(f"Trained from commit {record.commit[:12]} with uncommitted changes.")
    if commercial:
        if record.commercial_only is not True:
            check.problems.append(
                "A licence that does not restrict use to non-commercial research needs a commercial_only run; this one is not."
            )
        restricted = sorted(n for n, e in check.datasets.items() if not e["commercial_ok"])
        if restricted:
            check.problems.append(f"Datasets not licensed for commercial use: {restricted}.")
    return check
