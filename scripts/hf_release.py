#!/usr/bin/env python3
"""Stage each Bulwark model for its own Hugging Face model repository, upload it, open it.

Each released model has its own model repository (`bulwark.export.release.MODEL_REPOS`):
Bulwark-B1 as a release directory (PyTorch safetensors) in
Inferensys/bulwark-b1-driver-monitoring, Bulwark-E1 as two ONNX graphs, one for 16-frame
video clips and one for still images, in Inferensys/bulwark-e1-driver-monitoring-onnx.
Release directories and graphs link to their own repository (`model_card_url`). The
README.md of each repository is that model's card: Hub metadata (`front_matter`) over a
card rendered from the tracked templates in release/hf/ (`render_card`, whose placeholders
release/hf/README.md lists), with the tables of results, speed, parameters and outputs
filled from the staged files. Documentation links point at this repository on GitHub at
the code's release tag (CODE_TAG), since a gated repository shows only README.md and
LICENSE to visitors. A repository stays private until every artifact in it clears
`bulwark release-check`, and is then opened as a gated repository with automatic
approval. RELEASING.md gives the steps and the launch order.

Which models this launch releases is one list, `released` in release/hf/launch.yaml
(`release_set`): Bulwark-B1 and Bulwark-E1, released together. A model left out keeps its
repository private: `open`, `stage --public` and `upload --public` refuse it, an upload of
it stays an untagged private preview, no staged card, full card or organisation card may
name its repository (the templates' release blocks drop the lines that link it), and
`collection` leaves out its items. Being listed is not being public: a card that links
another model's repository goes public (`upload` to a public repository, `open`) only once
the Hub reports that repository public, or in the same `open` run as that repository, and
`collection --public` waits until every released model's repository is public.

stage       Builds one repository folder per model (--model b1 or e1; LAYOUT): config.json
            (B1: the release directory's; E1: the graphs' contracts), the weights
            (model.safetensors; bulwark-e1-video.onnx and bulwark-e1-image.onnx with their
            bench.json), LICENSE (the CC BY-NC 4.0 legal code), the upstream licence
            (LICENSE-vjepa2; LICENSE-timm, the Apache-2.0 text), NOTICE, MODEL_CARD.md (the
            full card, links pointed at GitHub), evaluation/<model>/ (report.html and
            metrics.json; E1 also its B1 teacher's under evaluation/b1/), provenance.json
            (local paths scrubbed), release-check.json (the verdict of `bulwark
            release-check` for each artifact), README.md and last SHA256SUMS. It refuses
            dataset material, pickles, local paths, secrets, dataset subject ids, links
            that reach neither a staged file nor this repository's files at CODE_TAG,
            unknown template placeholders, evaluations of a --limit run, of part of a split
            or of other weights, and an E1 evaluation that scored other clips than its
            teacher's; with --public, also artifacts that are not cleared, weights that do
            not come from the released B1 (B1_CHECKPOINT_SHA256), an incomplete release and
            *at release* items, and a model that is not in the release set. Nothing is
            written unless every check passes. --validate also sends the front matter to
            the Hub's validate-yaml endpoint, which only checks it.
upload      Re-checks the folder (it must be what `stage` builds from the staged inputs and
            this checkout), creates the model's repository private if it does not exist,
            and uploads the folder so that main mirrors it. It refuses a public repository
            unless --public is given, and then only a cleared stage of a released model
            whose cards link no other model's repository that is still private; it tags
            the commit (MODEL_RELEASE) only for a cleared stage.
open        The only step that changes a repository's visibility, run as a step of its own
            once an Inferensys owner has reviewed the private repository pages and the
            access-request form in the staged README.md. One run opens the repositories of
            every --model it is given, all or none (`open_repos`): it runs every check for
            every model first, and changes nothing if one fails. It refuses a model that is
            not in the release set, and cards that link another model's repository that is
            still private, unless the same run opens it (the B1 and E1 cards link each
            other, so they open together). It re-checks each folder, compares it with main,
            scans every revision for files other than the stage's, then turns on gating
            with automatic approval (GATED) and makes each repository public in turn. If a
            change does not take, that repository and those the run opened before it are
            made private again, and it says which repositories are public.
collection  Creates or updates the collection that groups the models (release/hf/
            collection.yaml): title, description, items and notes. Items of a model that
            is not released are deferred: left out, and removed from the collection if
            they are there. It stays private; --public, at launch, adds the items marked
            at_launch (the demo Space and the AUC paper) and makes it public, once the
            repository of every released model is public.
org-card    Writes the organisation card (the README.md of the Space Inferensys/README,
            from release/hf/org-card.md) into a local folder and prints the organisation
            settings (release/hf/org.yaml). Publishing them is manual, at launch.

    python scripts/hf_release.py stage --model b1 --release releases/bulwark-b1-v0 \\
        --eval runs/eval/b1-lora-auc --provenance checkpoints/b1-lora-auc/provenance.json [--public] [--validate]
    python scripts/hf_release.py stage --model e1 --onnx checkpoints/bulwark-e1-v0.onnx \\
        --onnx checkpoints/bulwark-e1-v0-image.onnx --eval runs/eval/e1 --teacher-eval runs/eval/b1-lora-auc \\
        --provenance checkpoints/e1-v0/provenance.json [--public] [--validate]
    python scripts/hf_release.py upload --model b1 [--stage runs/hf-release/b1] [--tag v0 | --no-tag]
        [--allow-weight-deletion] [--public]
    python scripts/hf_release.py open --model b1 --model e1 [--stage runs/hf-release/b1 --stage runs/hf-release/e1] [--yes]
    python scripts/hf_release.py collection [--public]
    python scripts/hf_release.py org-card [--out runs/hf-org-card] [--validate]

stage and org-card need no network (unless --validate) and stage needs the export extra
(onnx). upload, open and collection need huggingface_hub (`--extra ops`) and `hf auth
login` with a token that can write to the Inferensys organisation; they talk to
https://huggingface.co only, whatever HF_ENDPOINT says. The release check follows the
registry of this checkout, so `stage --public`, a cleared upload and `open` refuse
uncommitted changes. Exit codes: 0 done, 1 refused (every problem is printed), 2 usage
error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import yaml

from bulwark import __version__
from bulwark.data.registry import BY_NAME
from bulwark.data.taxonomy import LabelSpec, default_taxonomy
from bulwark.eval.provenance import git_state, scrub_paths, sha256_file
from bulwark.export.release import (
    B1_CHECKPOINT_SHA256,
    HUB_URL,
    MODEL_RELEASE,
    MODEL_REPOS,
    check_release,
    model_card_url,
    released_model,
    research_only,
    training_record,
)
from bulwark.inference import (
    E1_GRAPHS,
    ONNX_CONTRACT_KEY,
    ONNX_RELEASE_FORMAT,
    RELEASE_FORMAT,
    RELEASE_VERSION,
)

if TYPE_CHECKING:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import ModelInfo, RepoSibling

REPO = Path(__file__).resolve().parents[1]
MODELS = ("b1", "e1")
MODEL_NAMES = {"b1": "Bulwark-B1", "e1": "Bulwark-E1"}
DEFAULT_STAGE = Path("runs/hf-release")  # one folder per model: runs/hf-release/<model>
DEFAULT_ORG_CARD = Path("runs/hf-org-card")
# The host in the model cards' URLs. HfApi() alone follows HF_ENDPOINT, so a mirror setting
# would receive the maintainer's token and the weights.
ENDPOINT: Final = HUB_URL
VALIDATE_URL = f"{ENDPOINT}/api/validate-yaml"  # checks card metadata; creates nothing
# The code and its docs stay on GitHub; the cards link to them at the release tag of the code.
SOURCE_URL = "https://github.com/Inferensys/bulwark"
RAW_URL = "https://raw.githubusercontent.com/Inferensys/bulwark"  # files at a tag, for images in cards
CODE_TAG = f"v{__version__}"
# Templates and data of the cards, collection and organisation card (release/hf/README.md).
TEMPLATES = Path("release/hf")
SHARED = TEMPLATES / "shared"
# The models this launch releases (`release_set`): the one list the cards, the collection,
# `open` and a public stage or upload key off.
LAUNCH = TEMPLATES / "launch.yaml"
CONTACT_URL = "https://inferensys.com/contact"
CONTACT_EMAIL = "hello@inferensys.com"
PRIVACY_URL = "https://inferensys.com/privacy"
LICENCE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
LEGAL_CODE_URL = "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en"

# Repository files staged under fixed names: the weights licence that everyone can read on
# a gated repository (LICENSE), the upstream licences the weights carry (B1: Meta's notice,
# which `bulwark release` writes into the release directory; E1: Apache-2.0, which section
# 4(a) of that licence requires with the timm-derived weights), NOTICE and the full card.
FIXED_FILES = {"LICENSE": "LICENSES/CC-BY-NC-4.0.txt", "NOTICE": "NOTICE", "MODEL_CARD.md": "MODEL_CARD.md"}
UPSTREAM_LICENCE = {"b1": "LICENSE-vjepa2", "e1": "LICENSE-timm"}
TIMM_LICENCE_SOURCE = "LICENSE"  # the repository's Apache-2.0 text, the licence of timm's weights
RELEASE_FILES = {"model.safetensors", "config.json", "LICENSE-vjepa2"}
# E1's two graphs, by the number of frames they take (`bulwark.inference.load` of E1's
# repository id loads the 16-frame one).
GRAPH_NAMES = E1_GRAPHS
GRAPH_MEDIA = {16: "video", 1: "image"}
# Only these two files of an evaluation: predictions/ holds per-clip outputs keyed by clip
# id, and calibration.json is already in the release's config.json or the graphs' metadata.
EVAL_FILES = ("report.html", "metrics.json")
SUMS = "SHA256SUMS"
RECORDS = "release-check.json"
PROVENANCE = "provenance.json"
COMMON_FILES = {"README.md", "MODEL_CARD.md", "config.json", "LICENSE", "NOTICE", SUMS, RECORDS, PROVENANCE}
# Every path a stage of each model may hold.
LAYOUT = {
    "b1": COMMON_FILES | {"model.safetensors", "LICENSE-vjepa2"} | {f"evaluation/b1/{n}" for n in EVAL_FILES},
    "e1": COMMON_FILES | {"LICENSE-timm"} | set(GRAPH_NAMES.values()) | {f"{n}.bench.json" for n in GRAPH_NAMES.values()}
          | {f"evaluation/{m}/{n}" for m in MODELS for n in EVAL_FILES},  # evaluation/b1: E1's teacher
}  # fmt: skip
WEIGHT_SUFFIXES = (".safetensors", ".onnx", ".pt", ".pth", ".bin", ".ckpt", ".h5", ".gguf", ".tflite", ".msgpack")
# Frames, video, tables, shards and archives (dataset material), and pickles, which run code
# when loaded.
DATA_SUFFIXES = (
    ".parquet", ".tar", ".shard", ".npz", ".npy", ".jsonl", ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".webp", ".mp4", ".avi", ".mkv", ".mov", ".webm", ".zip", ".gz", ".tgz", ".pkl", ".pickle",
    ".pt", ".pth", ".ckpt", ".bin",
)  # fmt: skip
# Input folders that hold dataset material or private notes (as in .gitignore).
PRIVATE_DIRS = ("data", "ops", "docs/local")

# A local absolute path: home, notebook and cloud working directories, macOS volumes and
# temporary folders, Windows drives, also after file://. /tmp/ is not matched (RELEASING.md
# uses it in examples), and the lookbehind keeps relative paths such as
# infra/kaggle/push_data.sh from matching.
LOCAL_PATH = re.compile(
    r"(?<![\w.-])(?:/(?:Users|home|root|kaggle|content|mnt|workspace|private|Volumes|var/folders|opt/ml)/"
    r"|~/|[A-Za-z]:[\\/])"
)
# Access tokens, signed storage URLs, account keys, private keys and kaggle.json's key.
SECRET = re.compile(
    r"hf_[A-Za-z0-9]{30,}|[?&]sig=|AccountKey=|-----BEGIN [A-Z ]*PRIVATE KEY|\"key\"\s*:\s*\"[0-9a-f]{32}\""
)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")  # as tests/test_docs_project_files.py
# Inferensys's public contact address, which the cards give for commercially licensed
# models and removal requests; any other address is refused.
PUBLIC_EMAILS = frozenset({CONTACT_EMAIL})
# Credentials and the Kaggle username: their values must never be published.
SECRET_ENV = ("HF_TOKEN", "KAGGLE_USER", "KAGGLE_USERNAME", "KAGGLE_KEY")
# A dataset subject id (`<dataset>:<id>` in the manifest), which is part of the dataset.
SUBJECT_ID = re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, sorted(BY_NAME, key=len, reverse=True))) + r"):[A-Za-z0-9][\w.-]*")
# Subject ids that name a whole side of a dataset without per-person ids
# (bulwark.data.sources.auc_v2), which the data card names; any other id is refused.
GROUP_IDS = frozenset({"auc_v2:train_drivers", "auc_v2:test_drivers"})
# Items the cards and the data card complete before any weights are published; a card
# placeholder that has no value yet renders as one (`pending`).
AT_RELEASE = re.compile(r"\*at release|open item", re.IGNORECASE)
# Inline links [text](target), [text](<target>) and [text](target "title"), read as
# tests/test_docs_project_files.py reads them: group 1 or 2 is the target, 3 the title.
LINK = re.compile(r"""\]\(\s*(?:<([^<>\n]*)>|([^)\s<>]+))(\s+(?:"[^"\n]*"|'[^'\n]*'|\([^()\n]*\)))?\s*\)""")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
HTML_LINK = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"']+)""", re.IGNORECASE)
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
# A link into this repository on GitHub: (blob|tree, ref, path).
CODE_LINK = re.compile(re.escape(SOURCE_URL) + r"/(blob|tree)/([^/#?]+)/([^#?]*)")
RAW_LINK = re.compile(re.escape(RAW_URL) + r"/([^/#?]+)/([^#?]*)")
# Free-text fields of an ONNX model, its graph, nodes and functions (not tensor data).
ONNX_TEXT_FIELDS = {"doc_string", "producer_name", "producer_version", "domain", "key", "value", "s", "strings"}
# Template syntax (release/hf/README.md): {{> name }} includes release/hf/shared/<name>.md,
# {{ name }} is a value from `card_values`, and a release block keeps its lines only when a
# model is in the release set ({{#released e1}}) or only while it is not
# ({{#unreleased e1}}), each tag on a line of its own, closed by {{/released}} or
# {{/unreleased}} on a line of its own. Blocks do not nest.
INCLUDE = re.compile(r"\{\{>\s*([a-z][a-z0-9-]*)\s*\}\}")
PLACEHOLDER = re.compile(r"\{\{\s*([a-z][a-z0-9_]*)\s*\}\}")
BRACES = re.compile(r"\{\{|\}\}")
BLOCK = re.compile(r"^\{\{#(released|unreleased) ([a-z0-9]+)\}\}\n(.*?)^\{\{/\1\}\}\n", re.MULTILINE | re.DOTALL)
BLOCK_TAG = re.compile(r"\{\{\s*[#/]")  # an opening or closing block tag, anywhere

# Hub metadata of each model's card (https://huggingface.co/docs/hub/model-cards). Every tag
# names an output, domain or method the card shows; the Hub adds `pytorch`/`onnx` from
# library_name and `safetensors` from the files.
_TAGS = ["driver-monitoring", "drowsiness-detection", "fatigue-detection", "driver-distraction", "in-cabin", "perclos",
         "blink-detection", "yawn-detection", "head-pose-estimation", "phone-use", "action-recognition", "automotive"]  # fmt: skip
CARD_METADATA: dict[str, dict[str, Any]] = {
    "b1": {
        "license": "cc-by-nc-4.0",
        "library_name": "pytorch",
        "pipeline_tag": "video-classification",
        "tags": [*_TAGS, "v-jepa", "lora", "multi-task", "non-commercial"],
        "metrics": ["roc_auc", "f1", "mae"],
    },
    "e1": {
        "license": "cc-by-nc-4.0",
        "library_name": "onnx",
        # E1's initialisation; B1's V-JEPA 2.1 encoder has no official Hub repository, so
        # the B1 card cites it instead. A second base model would mean a merge.
        "base_model": "timm/efficientvit_b1.r224_in1k",
        "base_model_relation": "finetune",
        "pipeline_tag": "video-classification",
        "tags": [*_TAGS, "efficientvit", "knowledge-distillation", "onnxruntime", "cpu", "multi-task", "non-commercial"],
        "metrics": ["roc_auc", "f1", "mae"],
    },
}
# How `open` gates a repository: "auto" approves every access request as soon as it is
# sent (huggingface_hub `update_repo_settings(gated=...)`; "manual" would wait for a
# maintainer). Approved requests can still be rejected later, which removes the access.
GATED: Final = "auto"
# The access-request form of each gated repository. With automatic approval, filling it in
# and ticking the box is what grants access, so the heading and prompt above it state the
# terms; the full terms are in the card's licence section, which the prompt links. Keys
# may not hold "." or "$" (the Hub's validator). The heading never suggests that these
# weights can be licensed for commercial use: Inferensys's commercial offer is separate,
# commercially licensed models trained on data cleared for commercial use. The last
# question is an opt-in: only a Yes lets Inferensys contact the requester about those
# models (GATE_DESCRIPTION). Only `open` gates a repository, so the Hub does not show the
# form before then: it is reviewed in the front matter of the staged README.md
# (RELEASING.md).
GATE_HEADING = "Free for non-commercial research. These weights are not licensed for commercial use."
GATE_FIELDS: dict[str, Any] = {
    "Name": "text",
    "Company or affiliation": "text",
    "Intended use": "text",
    "I accept the CC BY-NC licence and the training datasets' terms, and will use these weights only for "
    "non-commercial research": "checkbox",
    "May Inferensys contact you about commercially licensed models?": {"type": "select", "options": ["Yes", "No"]},
}
GATE_DESCRIPTION = (
    "Inferensys receives your Hugging Face username, email address and answers. It uses them to manage access "
    "and, only if you answer Yes to being contacted, to contact you about commercially licensed models. "
    f"Privacy notice: {PRIVACY_URL}"
)
GATE_BUTTON = "Request access"
LOGIN_HINT = (
    "An Inferensys organisation admin gives the uploading maintainer a token that can write to the "
    "organisation, and the maintainer runs `hf auth login` with it (RELEASING.md)."
)


# ---- stage ---------------------------------------------------------------------------------


def stage(
    model: str,
    out: Path,
    *,
    release: Path | None = None,
    graphs: Iterable[Path] = (),
    evaluation: Path | None = None,
    teacher_evaluation: Path | None = None,
    provenance: Path | None = None,
    public: bool = False,
    validate: bool = False,
    root: Path = REPO,
) -> list[str]:
    """Stage one model's repository into `out`; the problems that refused it (nothing
    written), or []. A public stage of a model that is not in the release set
    (`release_set`) is refused before anything else is checked."""
    graphs = list(graphs)
    if public:
        released, found = release_set(root)
        if found or model not in released:
            return found or [not_released(model)]
    inputs = [p for p in (release, *graphs, evaluation, teacher_evaluation, provenance) if p is not None]
    problems = _input_problems(inputs, out, root)
    if model == "b1":
        problems += [] if release is not None else ["A B1 stage needs --release, a release directory of the B1 checkpoint."]
        problems += ["A B1 stage takes no --onnx and no --teacher-eval."] if graphs or teacher_evaluation else []
    else:
        problems += [] if graphs else ["An E1 stage needs --onnx, the graphs `bulwark export` wrote (16 frames and 1 frame)."]
        problems += ["An E1 stage takes no --release: its repository holds the ONNX graphs."] if release else []
    if problems:
        return problems
    if release is not None:
        problems += _release_problems(release, model)
    named: dict[str, Path] = {}
    for path in graphs:
        found = _graph_problems(path)
        problems += found
        if not found:
            frames = int(onnx_props(path).get("frames") or 0)
            if frames not in GRAPH_NAMES:
                problems.append(f"{path}: a graph for {frames} frames; the E1 repository holds a 16-frame and a 1-frame graph")
            elif GRAPH_NAMES[frames] in named:
                problems.append(f"{path}: a second {frames}-frame graph ({named[GRAPH_NAMES[frames]]})")
            else:
                named[GRAPH_NAMES[frames]] = path
    if problems:
        return problems
    problems += _same_checkpoint_problems(named)
    sources = artifact_sources(model, release, named)
    found = _eval_problems(evaluation, scored_weights(model, sources, named.values())) if evaluation is not None else []
    if teacher_evaluation is not None:
        found += _eval_problems(teacher_evaluation, _teachers(named.values()), what="the B1 teacher of these graphs")
        if evaluation is not None and not found:
            found += _comparison_problems(evaluation, teacher_evaluation)
    problems += found
    if provenance is not None:
        problems += _provenance_problems(provenance, sources)
    if problems:
        return problems

    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        if release is not None:
            for name in RELEASE_FILES:
                if (release / name).is_file():
                    shutil.copyfile(release / name, partial / name)
        for name, path in named.items():
            shutil.copyfile(path, partial / name)
            bench = path.with_name(path.name + ".bench.json")
            if bench.is_file():
                shutil.copyfile(bench, partial / f"{name}.bench.json")
        for folder, source in ((model, evaluation), ("b1", teacher_evaluation)):
            if source is not None:
                (partial / "evaluation" / folder).mkdir(parents=True)
                for name in EVAL_FILES:
                    shutil.copyfile(source / name, partial / "evaluation" / folder / name)
        if provenance is not None:
            data = scrub_paths(json.loads(provenance.read_text(encoding="utf-8")))
            (partial / PROVENANCE).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        records = {name: artifact_record(path, model, root) for name, path in staged_artifacts(partial, model).items()}
        (partial / RECORDS).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        derived, found = derived_files(partial, model, records, root)
        problems += found
        for rel, text in derived.items():
            (partial / rel).write_text(text, encoding="utf-8")
        _remove_ds_store(partial)
        problems += scan(partial, model)
        if public:
            problems += public_problems(partial, model, records, root) + checkout_problems(root)
        if validate and not problems:
            problems += validate_metadata((partial / "README.md").read_text(encoding="utf-8"), "model")
        if problems:
            shutil.rmtree(partial)
            return problems
        write_sums(partial)
        if out.exists():
            out.rmdir()  # empty, checked above
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    print(f"Staged {len(_files(out))} files in {out} for {model_card_url(model)}:")
    for rel in _files(out):
        print(f"  {rel}")
    for name, record in records.items():
        verdict = "cleared" if record["publishable"] else "NOT cleared: " + " ".join(record["problems"])
        print(f"{name} ({record['kind']}): {verdict}")
    if validate:
        print(f"{VALIDATE_URL}: no errors or warnings in the front matter.")
    pending = public_problems(out, model, records, root) if not public else []
    if pending:
        print("Only a private repository may hold this stage; `upload` keeps it private and `open` refuses it:")
        for problem in pending:
            print(f"  {problem}")
    return []


def _input_problems(paths: list[Path], out: Path, root: Path) -> list[str]:
    problems = []
    private = [(d, (root / d).resolve()) for d in PRIVATE_DIRS]
    target = out.resolve()
    for path in paths:
        if path.is_symlink():
            problems.append(f"{path}: a symbolic link; pass the file or directory itself")
        if not path.exists():
            problems.append(f"{path}: does not exist")
            continue
        resolved = path.resolve()
        for name, folder in private:
            if resolved.is_relative_to(folder):
                problems.append(f"{path}: under {name}/, which holds dataset material or private notes")
        if target.is_relative_to(resolved) or resolved.is_relative_to(target):
            problems.append(f"{path}: overlaps the output folder {out}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        problems.append(f"{out} exists and is not an empty folder; stage into a new one")
    return problems


def _release_problems(path: Path, model: str) -> list[str]:
    """A B1 release directory from `bulwark release`, as given or as staged (the stage root)."""
    if not path.is_dir():
        return [f"{path}: not a release directory (`bulwark release` writes one)"]
    staged = (path / RECORDS).is_file()
    names = {p.name for p in path.iterdir() if p.name != ".DS_Store"}
    problems = [f"{path}/{n}: a symbolic link" for n in sorted(names & RELEASE_FILES) if (path / n).is_symlink()]
    extra = sorted(names - RELEASE_FILES)
    if extra and not staged:
        problems.append(f"{path}: holds {', '.join(extra)}; a release directory holds only {', '.join(sorted(RELEASE_FILES))}")
    if not {"model.safetensors", "config.json"} <= names:
        return [*problems, f"{path}: needs model.safetensors and config.json (`bulwark release` writes them)"]
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [*problems, f"{path}/config.json: unreadable ({error})"]
    if config.get("format") != RELEASE_FORMAT or config.get("format_version") != RELEASE_VERSION:
        problems.append(f"{path}: not a {RELEASE_FORMAT} v{RELEASE_VERSION} directory")
    kind = str(config.get("kind", ""))
    if not kind.startswith("b1"):
        problems.append(f"{path}: a release of kind {kind!r}; the {model.upper()} repository holds a B1 release directory")
    elif "LICENSE-vjepa2" not in names:
        problems.append(f"{path}: a B1 release needs LICENSE-vjepa2, Meta's notice for the encoder weights")
    licence = str(config.get("weights_licence") or "")
    if not research_only(licence):
        problems.append(f"{path}: licence {licence!r} is not research-only, and this repository is licensed cc-by-nc-4.0")
    return problems


def _graph_problems(path: Path) -> list[str]:
    if path.suffix != ".onnx" or not path.is_file():
        return [f"{path}: not an ONNX file"]
    props = onnx_props(path)
    if ONNX_CONTRACT_KEY not in props:
        return [f"{path}: no {ONNX_CONTRACT_KEY} metadata, so not a graph from `bulwark export`"]
    problems = []
    licence = props.get("license", "")
    if not research_only(licence):
        problems.append(f"{path}: licence {licence!r} is not research-only, and this repository is licensed cc-by-nc-4.0")
    bench = path.with_name(path.name + ".bench.json")
    if bench.is_symlink():
        problems.append(f"{bench}: a symbolic link")
    elif bench.is_file():
        try:
            recorded = (json.loads(bench.read_text(encoding="utf-8")).get("onnx") or {}).get("sha256")
        except ValueError:
            recorded = None
        if recorded != sha256_file(path):
            problems.append(f"{bench}: measured on another graph (onnx.sha256 differs); measure again with `bulwark export`")
    return problems


def _contract(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(onnx_props(path).get(ONNX_CONTRACT_KEY) or "{}"))


def _same_checkpoint_problems(graphs: dict[str, Path]) -> list[str]:
    checkpoints = {name: (_contract(path).get("checkpoint") or {}).get("sha256") for name, path in graphs.items()}
    if len(set(checkpoints.values())) > 1:
        return [f"The graphs come from different checkpoints ({checkpoints}); export both from the released E1 checkpoint."]
    return []


def _eval_problems(path: Path, weights: set[str | None], *, what: str = "the staged artifacts") -> list[str]:
    if not path.is_dir():
        return [f"{path}: not an evaluation directory (`bulwark evaluate --out`)"]
    problems = [f"{path}: no {name}" for name in EVAL_FILES if not (path / name).is_file()]
    problems += [f"{path}/{name}: a symbolic link" for name in EVAL_FILES if (path / name).is_symlink()]
    if problems:
        return problems
    try:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    except ValueError as error:
        return [f"{path}/metrics.json: unreadable ({error})"]
    meta = metrics.get("meta") or {}
    if meta.get("limit_per_split") != 0:
        problems.append(f"{path}: a --limit run ({meta.get('limit_per_split')} clips per split); its numbers are pipeline checks, never results")
    checkpoint = (meta.get("checkpoint") or {}).get("sha256")
    if checkpoint is None or checkpoint not in weights:
        problems.append(f"{path}: an evaluation of other weights (sha256 {str(checkpoint)[:12]}) than {what}")
    return problems + _coverage_problems(path, metrics.get("coverage"))


def _coverage_problems(path: Path, coverage: Any) -> list[str]:
    """A result must score every test and holdout clip of its manifest: an evaluation that
    predicted part of a split (shards or cached features missing) is refused like a --limit
    run, and so is one that predicted clips its manifest puts in another split."""
    if not isinstance(coverage, dict) or not coverage.get("test"):
        return [f"{path}: metrics.json records no coverage of the test split, so whether it scored every clip is unknown"]
    short, elsewhere = [], []
    for split in ("test", "holdout"):
        for dataset, entry in sorted((coverage.get(split) or {}).items()):
            if int(entry.get("predicted") or 0) < int(entry.get("manifest") or 0):
                short.append(f"{split} {dataset}: {int(entry.get('predicted') or 0):,} of {int(entry['manifest']):,}")
            if entry.get("predicted_in_other_splits"):
                elsewhere.append(f"{split} {dataset}: {int(entry['predicted_in_other_splits']):,}")
    problems = []
    if short:
        problems.append(f"{path}: predicted only part of a split ({'; '.join(short)} clips); its numbers are not results")
    if elsewhere:
        problems.append(f"{path}: predicted clips its manifest puts in another split ({'; '.join(elsewhere)}); evaluate on the manifest of the predictions")
    return problems


def _scored_clips(metrics: dict[str, Any]) -> tuple[Any, str]:
    """What an evaluation scored: its manifest's sha256 and its split policy."""
    recorded = (metrics.get("meta") or {}).get("predicted_with") or {}
    return (recorded.get("manifest") or {}).get("sha256"), json.dumps(recorded.get("split_policy"), sort_keys=True)


def _comparison_problems(evaluation: Path, teacher: Path) -> list[str]:
    """E1's card sets its B1 teacher's test scores beside E1's, so both evaluations must have
    scored the same clips: the same manifest, split by the same policy."""
    ours, theirs = (_scored_clips(_json(path / "metrics.json") or {}) for path in (evaluation, teacher))
    if ours[0] is None or theirs[0] is None:
        return [f"{evaluation} and {teacher}: metrics.json must record the manifest it scored (meta.predicted_with.manifest)"]
    if ours != theirs:
        return [
            f"{evaluation} scored other clips than {teacher} (manifest {str(ours[0])[:12]} against {str(theirs[0])[:12]}, or "
            "another split policy), so the card cannot set their scores side by side; evaluate both on one manifest"
        ]
    return []


def artifact_sources(model: str, release: Path | None, graphs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Artifact name -> the checkpoint it was built from, as its own record has it: a
    release's config.json `source`, a graph's contract `checkpoint` (with its provenance)."""
    if model == "b1":
        if release is None:
            return {}
        return {"model.safetensors": json.loads((release / "config.json").read_text(encoding="utf-8")).get("source") or {}}
    return {name: _contract(path).get("checkpoint") or {} for name, path in graphs.items()}


def scored_weights(model: str, sources: dict[str, dict[str, Any]], graphs: Iterable[Path]) -> set[str | None]:
    """The weights an evaluation of the staged model may have scored: the checkpoint the
    artifacts were built from, or (E1) one of the graphs itself."""
    checkpoints = {((s.get("checkpoint") or {}) if model == "b1" else s).get("sha256") for s in sources.values()}
    return checkpoints | ({sha256_file(p) for p in graphs} if model == "e1" else set())


def _teachers(graphs: Iterable[Path]) -> set[str | None]:
    """The B1 checkpoints whose outputs the graphs' E1 checkpoint was distilled from."""
    return {((_contract(p).get("training") or {}).get("teacher_init_sha256")) for p in graphs} - {None}


def _provenance_key(data: dict[str, Any]) -> tuple[Any, Any, Any]:
    return data.get("job"), data.get("manifest_sha256"), (data.get("code") or {}).get("commit")


def _source_key(source: dict[str, Any]) -> tuple[Any, Any, Any]:
    """A release's `source`, or a graph's contract `checkpoint`, as `_provenance_key`."""
    if "provenance" in source:
        return _provenance_key(source.get("provenance") or {})
    return source.get("job"), source.get("manifest_sha256"), source.get("code_commit")


def _provenance_problems(path: Path, sources: dict[str, dict[str, Any]]) -> list[str]:
    """provenance.json must be that of the checkpoint the artifacts were built from."""
    try:
        key = _provenance_key(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        return [f"{path}: unreadable ({error})"]
    if None in key or not sources or any(_source_key(s) != key for s in sources.values()):
        return [f"{path}: not the provenance of the staged weights' checkpoint (job, manifest_sha256 and code commit differ)"]
    return []


def staged_artifacts(stage: Path, model: str) -> dict[str, Path]:
    """The release-checked artifacts of a stage: B1's release directory (the stage root,
    named by its weights file) or E1's graphs."""
    if model == "b1":
        found = (stage / "config.json").is_file() and (stage / "model.safetensors").is_file()
        return {"model.safetensors": stage} if found else {}
    return {name: stage / name for name in GRAPH_NAMES.values() if (stage / name).is_file()}


# ---- the files `stage` derives from the staged inputs and the checkout -----------------------


def derived_files(stage: Path, model: str, records: dict[str, dict[str, Any]], root: Path) -> tuple[dict[str, str], list[str]]:
    """The files of a stage that are not copied from an input: the fixed repository files,
    E1's config.json and README.md. `upload` and `open` compare a stage with them."""
    problems: list[str] = []
    tracked = tracked_files(root)
    files: dict[str, str] = {}
    sources = {**FIXED_FILES, **({"LICENSE-timm": TIMM_LICENCE_SOURCE} if model == "e1" else {})}
    for rel, source in sources.items():
        if source not in tracked:
            problems.append(f"{source}: not tracked by git, so not a published repository file")
            continue
        files[rel] = (root / source).read_text(encoding="utf-8")
    if "MODEL_CARD.md" in files:
        files["MODEL_CARD.md"], found = repository_doc_links("MODEL_CARD.md", files["MODEL_CARD.md"], root, tracked)
        problems += found
    if model == "e1":
        files["config.json"] = graphs_config_text(stage)
    card, found = render_card(model, stage, records, root)
    problems += found
    if card and "licence-and-dataset-terms" not in heading_anchors(card):
        problems.append("README.md: the access-request form links #licence-and-dataset-terms; the card needs that heading")
    readme = front_matter(model, _datasets(records), root) + card
    problems += card_link_problems(readme, stage, {*_files(stage), *files, "README.md", SUMS}, root, tracked, LAYOUT[model])
    files["README.md"] = readme
    return files, problems


def graphs_config_text(stage: Path) -> str:
    """E1's config.json as staged (`graphs_config`)."""
    return json.dumps(graphs_config(stage), indent=2) + "\n"


def graphs_config(stage: Path) -> dict[str, Any]:
    """E1's config.json: the contract of each staged graph (inputs, outputs and their
    decoding, trained and masked labels, temperatures, thresholds, licence, training
    record), by file name, so that the repository root has the configuration the Hub
    counts downloads by and a reader can decode the outputs without opening a graph."""
    graphs = {}
    for name, path in staged_artifacts(stage, "e1").items():
        props = onnx_props(path)
        graphs[name] = {"frames": int(props.get("frames") or 0), "sha256": sha256_file(path), ONNX_CONTRACT_KEY: _contract(path)}
    return {
        "format": ONNX_RELEASE_FORMAT, "format_version": 1, "model": "e1", "model_card": model_card_url("e1"),
        "model_release": MODEL_RELEASE, "graphs": graphs,
    }  # fmt: skip


def tracked_files(root: Path) -> set[str]:
    listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    return set(listing.decode("utf-8").split("\0")) - {""}


def _code_url(path: str, tracked: set[str], anchor: str = "") -> str | None:
    """This repository's file or folder `path` on GitHub at CODE_TAG; None when git does not
    track it (GitHub has only what was committed)."""
    if path in tracked:
        kind = "blob"
    elif any(t.startswith(f"{path}/") for t in tracked):
        kind = "tree"
    else:
        return None
    return f"{SOURCE_URL}/{kind}/{CODE_TAG}/{path}" + (f"#{anchor}" if anchor else "")


def _link_syntax_problems(doc: str, text: str) -> list[str]:
    """Links that `LINK` does not read: a relative reference-style or HTML link, or any
    other `](`. None is skipped silently."""
    problems = [
        f"{doc}: {kind} link {target}: only inline links [text](target) are resolved; write it as one"
        for pattern, kind in ((REFERENCE_LINK, "reference-style"), (HTML_LINK, "HTML"))
        for target in pattern.findall(text)
        if not SCHEME.match(target)
    ]
    read = {match.start() for match in LINK.finditer(text)}
    problems += [
        f"{doc}: link {text[start:start + 40]!r}: not readable as [text](target) or [text](target \"title\"); write it as one"
        for start in (m.start() for m in re.finditer(r"\]\(", text))
        if start not in read
    ]
    return problems


def repository_doc_links(doc: str, text: str, root: Path, tracked: set[str]) -> tuple[str, list[str]]:
    """A repository doc as staged: every relative link pointed at the file or folder on
    GitHub at CODE_TAG, and refused when git does not track it or its anchor has no heading.
    In-page anchors stay, and must reach a heading of the doc."""
    problems = _link_syntax_problems(doc, text)

    def resolve(match: re.Match[str]) -> str:
        target, title = match.group(1) or match.group(2) or "", match.group(3) or ""
        if SCHEME.match(target):
            problems.extend(_code_link_problems(doc, target, root, tracked))
            return match.group(0)
        file_part, _, anchor = target.partition("#")
        if not file_part:
            if anchor not in heading_anchors(text):
                problems.append(f"{doc}: no heading for #{anchor}")
            return match.group(0)
        path = posixpath.normpath(posixpath.join(posixpath.dirname(doc), file_part))
        url = None if path == ".." or path.startswith("../") else _code_url(path, tracked, anchor)
        if url is None:
            problems.append(f"{doc}: link {target}: {path} is not a file or folder git tracks in this repository")
            return match.group(0)
        if anchor and (not path.endswith(".md") or anchor not in heading_anchors((root / path).read_text(encoding="utf-8"))):
            problems.append(f"{doc}: no heading for #{anchor} in {path}")
        return f"]({url}{title})"

    return LINK.sub(resolve, text), problems


def _code_link_problems(doc: str, url: str, root: Path, tracked: set[str]) -> list[str]:
    """A link into this repository on GitHub must be at CODE_TAG, the code the release
    describes, and reach a tracked file (blob) or folder (tree) and an existing heading."""
    raw = RAW_LINK.match(url)
    if raw is not None:  # a file shown in the card, such as an image
        ref, path = raw.groups()
        if ref != CODE_TAG:
            return [f"{doc}: link {url}: links into {RAW_URL} go to the release tag {CODE_TAG}, not {ref}"]
        return [] if path in tracked else [f"{doc}: link {url}: {path} is not a file git tracks in this repository"]
    match = CODE_LINK.match(url)
    if match is None:
        return []
    kind, ref, path = match.groups()
    anchor = url.partition("#")[2]
    path = path.rstrip("/")
    if ref != CODE_TAG:
        return [f"{doc}: link {url}: links into {SOURCE_URL} go to the release tag {CODE_TAG}, not {ref}"]
    if _code_url(path, tracked) != f"{SOURCE_URL}/{kind}/{CODE_TAG}/{path}":
        return [f"{doc}: link {url}: {path} is not a {'file' if kind == 'blob' else 'folder'} git tracks in this repository"]
    if anchor and (not path.endswith(".md") or anchor not in heading_anchors((root / path).read_text(encoding="utf-8"))):
        return [f"{doc}: no heading for #{anchor} in {path}"]
    return []


def card_link_problems(
    readme: str, stage: Path, staged: set[str], root: Path, tracked: set[str], layout: Iterable[str] = ()
) -> list[str]:
    """The card's links: a relative link must reach a file of this stage (it opens in the
    model repository, to users with access unless it is LICENSE), or a file of the model's
    `layout` that is not staged yet, which `public_problems` reports instead; a link into
    this repository on GitHub, a tracked path at CODE_TAG; an anchor, a heading of the card."""
    layout = set(layout)
    doc = "README.md"
    problems = _link_syntax_problems(doc, readme)
    anchors = heading_anchors(readme)
    for match in LINK.finditer(readme):
        target = match.group(1) or match.group(2) or ""
        if SCHEME.match(target):
            problems += _code_link_problems(doc, target, root, tracked)
            continue
        file_part, _, anchor = target.partition("#")
        if not file_part:
            if anchor not in anchors:
                problems.append(f"{doc}: no heading for #{anchor}")
            continue
        path = posixpath.normpath(file_part)
        if path in layout and path not in staged:
            continue
        if path not in staged:
            problems.append(f"{doc}: link {target} reaches no file of this stage (link this repository's docs at {SOURCE_URL}/blob/{CODE_TAG}/...)")
        elif anchor and (not path.endswith(".md") or anchor not in heading_anchors((stage / path).read_text(encoding="utf-8"))):
            problems.append(f"{doc}: no heading for #{anchor} in {path}")
    return problems


def heading_anchors(text: str) -> set[str]:
    """GitHub-style heading anchors, as tests/test_docs_project_files.py computes them."""
    anchors = set()
    for line in _outside_fences(text):
        if match := re.match(r"#{1,6}\s+(.*)", line):
            anchors.add(re.sub(r"[^\w\- ]", "", match.group(1).strip().lower()).replace(" ", "-"))
    return anchors


def _outside_fences(text: str) -> list[str]:
    lines, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(line)
    return lines


class _CardDumper(yaml.SafeDumper):
    """Writes multi-line strings as YAML block scalars, so the gate prompt stays readable,
    and long one-line keys (the licence checkbox) as plain keys: PyYAML's emitter switches
    to the explicit `? key` form above 128 characters, YAML's limit is 1024."""

    def check_simple_key(self) -> bool:
        event = self.event
        if isinstance(event, yaml.ScalarEvent) and event.anchor is None:
            analysis = self.analyze_scalar(event.value)
            return len(analysis.scalar) < 1024 and not analysis.empty and not analysis.multiline
        return bool(super().check_simple_key())


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


_CardDumper.add_representer(str, _represent_str)


def gate_prompt(model: str, datasets: Iterable[str], root: Path = REPO) -> str:
    """The text above the access-request form: the licence, the training datasets, what
    Inferensys counts as commercial use and where to ask, and the card's licence section."""
    names = dataset_names(datasets, root)
    count = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(len(names), str(len(names)))
    terms = f"the terms of the {count} training datasets: {_join(names)}" if names else "the terms of every training dataset"
    return (
        f"These weights are for **non-commercial research only**, under [CC BY-NC 4.0]({LICENCE_URL}). "
        f"You must also follow {terms}.\n\n"
        "Inferensys treats products, demos, pilots, proofs of concept and customer evaluations as commercial use. "
        f"For those, ask about commercially licensed models: [{CONTACT_URL.removeprefix('https://')}]({CONTACT_URL}) or "
        f"{CONTACT_EMAIL}\n\n"
        f"Read the [licence and dataset terms]({model_card_url(model)}#licence-and-dataset-terms) before you request "
        "access. Access is granted as soon as you submit this form.\n"
    )


def front_matter(model: str, datasets: Iterable[str], root: Path = REPO) -> str:
    """The Hub metadata block of a model's README.md: CARD_METADATA and the access-request
    form (GATE_*), whose prompt names the training datasets of the staged weights."""
    metadata = {
        **CARD_METADATA[model],
        "extra_gated_heading": GATE_HEADING,
        "extra_gated_prompt": gate_prompt(model, datasets, root),
        "extra_gated_description": GATE_DESCRIPTION,
        "extra_gated_fields": GATE_FIELDS,
        "extra_gated_button_content": GATE_BUTTON,
    }
    return "---\n" + yaml.dump(metadata, Dumper=_CardDumper, sort_keys=False, allow_unicode=True, width=1000) + "---\n\n"


def validate_metadata(readme: str, repo_type: str) -> list[str]:
    """Problems the Hub's validate-yaml endpoint finds in a README's front matter (errors and
    warnings), or the reason it could not be asked. The endpoint only validates."""
    _, sep, rest = readme.partition("---\n")
    header = "---\n" + rest.partition("---\n")[0] + "---\n" if sep else readme
    body = json.dumps({"repoType": repo_type, "content": header}).encode("utf-8")
    request = urllib.request.Request(
        VALIDATE_URL, data=body, method="POST", headers={"Content-Type": "application/json", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - a fixed https URL
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:  # 400 with {"errors": [...]} for invalid metadata
        text = error.read().decode("utf-8", "replace")
        if error.code != 400:
            return [f"{VALIDATE_URL} answered {error.code}: {text[:500]}"]
    except (urllib.error.URLError, TimeoutError) as error:
        return [f"{VALIDATE_URL} could not check the front matter ({error})"]
    try:
        answer = json.loads(text or "{}")
    except ValueError:
        return [f"{VALIDATE_URL} answered something other than JSON: {text[:500]}"]

    def message(item: Any) -> str:
        return str(item.get("message", item)) if isinstance(item, dict) else str(item)

    found = [f"validate-yaml error: {message(item)}" for item in answer.get("errors") or []]
    return found + [f"validate-yaml warning: {message(item)}" for item in answer.get("warnings") or []]


# ---- the card: templates and the values that fill them --------------------------------------


def render_card(model: str, stage: Path, records: dict[str, dict[str, Any]], root: Path = REPO) -> tuple[str, list[str]]:
    """The card of a model, from release/hf/<model>.md: release blocks kept or dropped by
    the release set (`release_blocks`), includes ({{> name }}) expanded from
    release/hf/shared/, then placeholders ({{ name }}) filled from `card_values`.

    A shared section may use only the values that are the same for every model
    (`shared_values`), so the shared sections of the two cards are identical word for word.
    A value the stage cannot fill yet renders as an *at release* item (`pending`), which
    keeps the stage from a public repository. Unknown names, other `{{`/`}}`, a card that
    names the repository of another model that is not released and a card whose headings
    break the cards' structure rules (`structure_problems`) are problems."""
    path = root / TEMPLATES / f"{model}.md"
    if not path.is_file():
        return "", [f"{TEMPLATES}/{model}.md: missing; it is the template of the {MODEL_NAMES[model]} card"]
    released, problems = release_set(root)
    shared = shared_values(root)

    def include(match: re.Match[str]) -> str:
        name = match.group(1)
        source = root / SHARED / f"{name}.md"
        if not source.is_file():
            problems.append(f"{TEMPLATES}/{model}.md: includes {SHARED}/{name}.md, which does not exist")
            return ""
        text = release_blocks(source.read_text(encoding="utf-8"), released, f"{SHARED}/{name}.md", problems)
        if INCLUDE.search(text):
            problems.append(f"{SHARED}/{name}.md: a shared section cannot include another")
        return fill(text, shared, f"{SHARED}/{name}.md", problems, model_values=card_value_names())

    text = release_blocks(path.read_text(encoding="utf-8"), released, f"{TEMPLATES}/{model}.md", problems)
    text = INCLUDE.sub(include, text)
    text = fill(text, {**shared, **card_values(model, stage, records, root)}, f"{TEMPLATES}/{model}.md", problems)
    problems += unreleased_repo_problems(text, "README.md", released, model)
    return text, problems + structure_problems(text)


def release_blocks(text: str, released: Iterable[str], source: str, problems: list[str]) -> str:
    """`text` with each release block ({{#released m}} or {{#unreleased m}} ... {{/released}}
    or {{/unreleased}}, each tag on a line of its own) replaced by its lines when model m's
    place in the release set matches, or by nothing. A block that names no model is a
    problem, and so is one that holds another block tag (a nested block, or a closing tag
    of the other kind), whatever the release set: a block runs to the next closing tag of
    its own kind, so such a block would swallow the text up to it unseen whenever it is
    dropped. A tag left over outside any block (an unclosed block) is one for `fill`."""
    models = set(released)

    def block(match: re.Match[str]) -> str:
        kind, model, body = match.groups()
        if model not in MODELS:
            problems.append(f"{source}: {{{{#{kind} {model}}}}} names no model ({' or '.join(MODELS)})")
            return ""
        if BLOCK_TAG.search(body):
            problems.append(
                f"{source}: a {{{{#{kind} {model}}}}} block holds another block tag; blocks do not nest, "
                f"and each closes with {{{{/{kind}}}}} on a line of its own"
            )
            return ""
        return body if (model in models) == (kind == "released") else ""

    return BLOCK.sub(block, text)


def unreleased_repo_problems(text: str, doc: str, released: Iterable[str], own: str | None = None) -> list[str]:
    """`doc` names the repository of a model that is not released (other than `own`, the
    model whose card it is): that repository is private, so to visitors the link is dead."""
    models = set(released)
    return [
        f"{doc}: names {MODEL_REPOS[other]}, the repository of {MODEL_NAMES[other]}, which is not released ({LAUNCH}); "
        f"link it only once {other} is listed there"
        for other in MODELS if other != own and other not in models and MODEL_REPOS[other] in text
    ]  # fmt: skip


def fill(text: str, values: dict[str, str], source: str, problems: list[str], *, model_values: Iterable[str] = ()) -> str:
    """`text` with its placeholders replaced; unknown and model-specific names go to `problems`."""
    specific = set(model_values)

    def value(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in values:
            return values[name]
        why = "differs between the models, so a shared section cannot use it" if name in specific else "is not a placeholder"
        problems.append(f"{source}: {{{{ {name} }}}} {why} (release/hf/README.md lists them)")
        return match.group(0)

    filled = PLACEHOLDER.sub(value, text)
    if BRACES.search(PLACEHOLDER.sub("", INCLUDE.sub("", text))):
        problems.append(f"{source}: '{{{{' or '}}}}' outside a placeholder or include")
    return filled


def structure_problems(card: str) -> list[str]:
    """The cards' heading rules: one H1, no skipped level, short ASCII headings without
    '?' or parentheses (their anchors stay stable), and a language on every code fence."""
    problems = []
    headings = [(len(m.group(1)), m.group(2).strip()) for line in _outside_fences(card) if (m := re.match(r"(#{1,6})\s+(.*)", line))]
    if [level for level, _ in headings].count(1) != 1:
        problems.append("README.md: the card needs exactly one H1 heading")
    for (before, _), (level, title) in zip([(0, ""), *headings], headings, strict=False):
        if level > before + 1 and before:
            problems.append(f"README.md: heading {title!r} skips a level")
    for level, title in headings:
        if level > 1 and (not title.isascii() or re.search(r"[?()]", title)):
            problems.append(f"README.md: heading {title!r} is not a plain ASCII statement without '?' or parentheses")
    fences = [line.strip() for line in card.splitlines() if line.lstrip().startswith("```")]
    if any(fence == "```" for fence in fences[0::2]):
        problems.append("README.md: a code fence without a language (bash, python, bibtex, text)")
    return problems


def pending(reason: str) -> str:
    """The text of a placeholder the stage cannot fill yet: an *at release* item."""
    return f"*At release: {reason}.*"


def release_set(root: Path = REPO) -> tuple[frozenset[str], list[str]]:
    """The models this launch releases (release/hf/launch.yaml `released`) and what is wrong
    with that file. Adding a model there, and staging it, is how it joins a launch; a file
    that does not list them properly releases nothing."""
    listed = _yaml(root / LAUNCH).get("released")
    if not isinstance(listed, list) or not listed or len(set(map(str, listed))) != len(listed) or not set(map(str, listed)) <= set(MODELS):
        return frozenset(), [f"{LAUNCH}: `released` must list one or more of {', '.join(MODELS)}, each once"]
    return frozenset(map(str, listed)), []


def not_released(model: str) -> str:
    """Why a public step refuses a model that is not in the release set."""
    name = MODEL_NAMES[model]
    return (
        f"{name} is not released yet: {LAUNCH} does not list {model} under `released`, so its repository stays private. "
        f"Add {model} there once {name} is to be released, and stage it again (RELEASING.md, launch order)."
    )


def shared_values(root: Path = REPO) -> dict[str, str]:
    """Placeholder values that are the same for both models."""
    collection = _yaml(root / TEMPLATES / "collection.yaml")
    url, title = collection.get("url"), collection.get("title")
    return {
        "b1_repo": MODEL_REPOS["b1"], "e1_repo": MODEL_REPOS["e1"],
        "b1_url": model_card_url("b1"), "e1_url": model_card_url("e1"),
        "github": SOURCE_URL, "code_url": f"{SOURCE_URL}/blob/{CODE_TAG}", "code_tree_url": f"{SOURCE_URL}/tree/{CODE_TAG}",
        "code_raw_url": f"{RAW_URL}/{CODE_TAG}",
        "code_tag": CODE_TAG, "code_version": __version__, "model_release": MODEL_RELEASE,
        "contact_url": CONTACT_URL, "contact_email": CONTACT_EMAIL, "privacy_url": PRIVACY_URL,
        "licence_url": LICENCE_URL, "legal_code_url": LEGAL_CODE_URL,
        "collection_link": f"[{title}]({url})" if url and title else pending("the collection's address (release/hf/collection.yaml `url`)"),
    }  # fmt: skip


def card_value_names() -> list[str]:
    """The names `card_values` fills for each model (release/hf/README.md lists them)."""
    return [
        "model_name", "repo_id", "card_url", "weights_licence", "checkpoint_sha256", "parameters", "parameters_exact",
        "files_table", "labels_table", "untrained_outputs", "untrained_classes", "not_trained_on_images", "copied_on_images",
        "not_trained_on_video", "image_calibration", "results_table", "holdout_summary", "results_by_dataset",
        "holdout_table", "test_split_table", "test_people", "test_people_short", "outputs_as_b1", "evaluated",
        "datasets_table", "dataset_citations", "release_basis", "speed_table", "speed_scope", "speed_measured", "parity",
    ]  # fmt: skip


def card_values(model: str, stage: Path, records: dict[str, dict[str, Any]], root: Path = REPO) -> dict[str, str]:
    """Placeholder values of one model's card, from the staged files and release checks.

    The values are those of the staged weights and their evaluation only: results and
    labels from evaluation/<model>/metrics.json (E1: B1's column from its teacher's
    evaluation, evaluation/b1/), speed from the graphs' bench.json, parameter counts from
    the weights, trained and masked outputs from config.json or the graphs' metadata,
    datasets from the release check. A value without its source is `pending`."""
    names = _yaml(root / TEMPLATES / "names.yaml")
    taxonomy = default_taxonomy()
    metrics = _json(stage / "evaluation" / model / "metrics.json")
    teacher = _json(stage / "evaluation" / "b1" / "metrics.json") if model == "e1" else None
    trained, masked, status = _label_record(model, stage, metrics)
    artifacts = staged_artifacts(stage, model)
    weights = sorted(artifacts.values(), key=lambda path: path.name != GRAPH_NAMES[16])  # E1: the video graph first
    first = records[next(iter(records))] if records else {}
    values = {
        "model_name": MODEL_NAMES[model], "repo_id": MODEL_REPOS[model], "card_url": model_card_url(model),
        "weights_licence": str(first.get("licence") or pending("no artifact is staged")),
        "checkpoint_sha256": str(first.get("checkpoint_sha256") or "")[:12] or pending("the checkpoint is not recorded"),
    }  # fmt: skip
    count = _parameters(model, weights[0]) if weights else None
    values["parameters"] = f"{count / 1e6:.1f}M" if count is not None else pending("no weights are staged")
    values["parameters_exact"] = f"{count:,}" if count is not None else pending("no weights are staged")
    values["files_table"] = _files_table(model, stage) if weights else pending("no weights are staged")
    labels = _labels_table(model, taxonomy, names, trained, masked, status) if trained else None
    values["labels_table"] = labels or pending("the staged weights record no trained outputs")
    untrained = [n for n in taxonomy.labels if n not in trained]
    values["untrained_outputs"] = _join(_unique(_plain(names, n) for n in untrained)) if trained else pending("no weights are staged")
    values["not_trained_on_images"] = _masked_on(names, trained, masked, "image")
    values["not_trained_on_video"] = _masked_on(names, trained, masked, "video")
    values["copied_on_images"] = _copied_on_images(model, names, trained, masked)
    values["image_calibration"] = _image_calibration(model, stage, taxonomy, names) if weights else pending("no weights are staged")
    missing = pending(f"no evaluation of these weights is staged (evaluation/{model}/)")
    values["untrained_classes"] = _untrained_classes(names, taxonomy, trained, metrics) if metrics else missing
    if metrics and (model == "b1" or teacher):
        values["results_table"] = _results_table(model, taxonomy, names, metrics, teacher)
    else:
        values["results_table"] = missing if not metrics else pending("the evaluation of E1's B1 teacher is not staged (evaluation/b1/)")
    values["holdout_summary"] = _holdout_summary(taxonomy, names, metrics) if metrics else missing
    values["results_by_dataset"] = _by_dataset(model, taxonomy, names, metrics, "test") if metrics else missing
    values["holdout_table"] = _holdout(model, taxonomy, names, metrics) if metrics else missing
    values["test_split_table"] = _test_split(names, metrics) if metrics else missing
    values["test_people"], values["test_people_short"] = _test_people_sentences(names, metrics) if metrics else (missing, missing)
    values["outputs_as_b1"] = _outputs_as_b1(trained, teacher) if model == "e1" else "the same"
    created = str((((metrics or {}).get("meta") or {}).get("predicted_with") or {}).get("created_utc") or "")
    values["evaluated"] = created[:10] or missing
    datasets = _datasets(records)
    values["datasets_table"] = _datasets_table(names, datasets) if datasets else pending("no release check is staged")
    values["dataset_citations"] = _dataset_citations(names, datasets) if datasets else pending("no release check is staged")
    bases =list(dict.fromkeys(str(d.get("weights_release_basis")) for r in records.values() for d in r["release_check"]["datasets"].values() if d.get("weights_release_basis")))  # fmt: skip
    values["release_basis"] = " ".join(bases) or pending("no dataset records the basis of its clearance")
    found = {name: _json(stage / f"{name}.bench.json") for name in artifacts} if model == "e1" else {}
    benches = {name: bench for name, bench in found.items() if bench}
    no_speed = pending("no speed was measured for B1") if model == "b1" else pending("no graph's bench.json is staged")
    values["speed_table"] = _speed_table(benches) if benches else no_speed
    scopes = list(dict.fromkeys(str(r.get("scope")) for b in benches.values() for r in b.get("results") or [] if r.get("scope")))
    values["speed_scope"] = "; ".join(scopes) or no_speed
    measured = sorted({str(b.get("measured_utc") or "")[:10] for b in benches.values()} - {""})
    values["speed_measured"] = " and ".join(measured) or no_speed
    values["parity"] = _parity(artifacts) if model == "e1" else pending("B1 is not exported to ONNX")
    assert set(values) == set(card_value_names())
    return values


def _yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    return data if isinstance(data, dict) else {}


def _json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _datasets(records: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({d for record in records.values() for d in record["release_check"]["datasets"]})


def dataset_names(datasets: Iterable[str], root: Path = REPO) -> list[str]:
    """Short names of datasets (release/hf/names.yaml `datasets`; else the registry title),
    in the order of names.yaml, then by name."""
    names = _yaml(root / TEMPLATES / "names.yaml")
    return [_dataset_name(names, d) for d in _ranked(names, datasets)]


def _ranked(names: dict[str, Any], datasets: Iterable[str]) -> list[str]:
    """Datasets in the order of names.yaml, then by name: the order of every list and table."""
    order = list(names.get("datasets") or {})
    return sorted(datasets, key=lambda d: (order.index(d) if d in order else len(order), d))


def _dataset(names: dict[str, Any], dataset: str) -> dict[str, Any]:
    entry = (names.get("datasets") or {}).get(dataset)
    return entry if isinstance(entry, dict) else {}


def _dataset_name(names: dict[str, Any], dataset: str, *, first: bool = False) -> str:
    """A dataset's short name (names.yaml), or with `first` the name a card gives it at its
    first mention, which spells out an abbreviation; else the registry's title."""
    entry = _dataset(names, dataset)
    return str((first and entry.get("first")) or entry.get("name") or (BY_NAME[dataset].title if dataset in BY_NAME else dataset))


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else "none"


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _entry(names: dict[str, Any], label: str) -> dict[str, Any]:
    entry = (names.get("outputs") or {}).get(label)
    return entry if isinstance(entry, dict) else {"name": label.replace("_", " ").capitalize()}


def _name(names: dict[str, Any], label: str) -> str:
    return str(_entry(names, label)["name"])


def _plain(names: dict[str, Any], label: str) -> str:
    """The name of an output inside a sentence: `list` in names.yaml, or the name with a
    lower-case first letter unless it is an acronym."""
    entry = _entry(names, label)
    name = str(entry.get("list") or entry["name"])
    return name if entry.get("list") or name[1:2].isupper() else name[:1].lower() + name[1:]


def _label_record(model: str, stage: Path, metrics: dict[str, Any] | None) -> tuple[list[str], dict[str, set[str]], dict[str, str]]:
    """Trained outputs, the media types each was masked on, and whether each had ground
    truth ("gt", "teacher_only"), from the staged weights' own records and the evaluation."""
    trained: list[str] = []
    masked: dict[str, set[str]] = {}
    status: dict[str, str] = {}
    if model == "b1":
        config = _json(stage / "config.json") or {}
        trained = list(config.get("labels") or [])
        masked = {k: set(v) for k, v in (config.get("masked_labels") or {}).items()}
    else:
        for path in staged_artifacts(stage, "e1").values():
            contract = _contract(path)
            trained += [n for n in contract.get("supervised_labels") or [] if n not in trained]
            for name, media in (contract.get("masked_labels") or {}).items():
                masked.setdefault(name, set()).update(media)
            label_status = contract.get("label_status") or {}
            status.update(dict.fromkeys(label_status.get("teacher_only") or [], "teacher_only"))
            status.update(dict.fromkeys(label_status.get("ground_truth") or [], "gt"))
    for name, entry in ((metrics or {}).get("training_support") or {}).items():
        status.setdefault(name, str(entry.get("status")))
    return trained, masked, status


def _value_range(spec: LabelSpec) -> str:
    if spec.kind == "binary":
        return "Yes/no probability"
    if spec.kind == "categorical":
        return f"One of {len(spec.classes)} classes"
    low, high = spec.value_range
    if spec.unit and spec.unit_scale != 1:  # returned as the value in `unit` divided by unit_scale
        return f"{low:g} to {high:g} ({spec.unit} / {spec.unit_scale:g})"
    return f"{low:g} to {high:g}" + (f" {spec.unit}" if spec.unit else "")


def _checked(spec: LabelSpec, status: str | None, note: str = "") -> str:
    """Whether ground truth checked an output, then why after a colon; a `note` (names.yaml)
    replaces the reason."""
    from bulwark.eval.report import is_metadata

    if status == "gt":
        checked = "Sanity check only" if is_metadata(spec) else "Yes"
    elif status == "teacher_only":
        checked = "No: learned from another model's labels"
    else:
        checked = "Not recorded"
    return f"{checked.split(':')[0]}: {note}" if note else checked


def _labels_table(
    model: str, taxonomy: Any, names: dict[str, Any], trained: list[str], masked: dict[str, set[str]], status: dict[str, str]
) -> str:
    """The trained outputs, each row under its group's name in the first column (a screen
    reader reads the group with every row): B1 `Group | What it detects | Output key | Value
    | Checked against ground truth`; E1 the check, then the media types its graphs were
    trained for (`_trained_for`) in place of the value range. An output's `note` in
    names.yaml (one a reader must not use) replaces the reason in its check."""
    value, trained_for = (["Value"], []) if model == "b1" else ([], ["Trained for"])
    columns = ["Group", "What it detects", "Output key", *value, "Checked against ground truth", *trained_for]
    rows = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    groups: dict[str, list[str]] = {}
    for label in taxonomy.labels:
        if label in trained:
            groups.setdefault(str(_entry(names, label).get("group") or "Other outputs"), []).append(label)
    for group, labels in groups.items():
        for label in labels:
            spec = taxonomy.labels[label]
            cells = [group, _name(names, label), f"`{label}`"]
            checked = _checked(spec, status.get(label), str(_entry(names, label).get("note") or ""))
            if model == "b1":
                cells += [_value_range(spec), checked]
            else:
                cells += [checked, _trained_for(names, label, masked)]
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


MEDIA_NAMES = {"video": "video", "image": "still images"}


def _labelled(names: dict[str, Any], label: str) -> set[str] | None:
    """The media types whose training clips had a label for `label`, ground truth or the
    face model's (names.yaml `labelled`, from docs/data-card.md's coverage table), or None
    when names.yaml does not say."""
    found = _entry(names, label).get("labelled")
    return set(found) & set(MEDIA_NAMES) if isinstance(found, list) else None


def _trained_for(names: dict[str, Any], label: str, masked: dict[str, set[str]]) -> str:
    """E1's media types for an output: those it was not masked on, and on which of them it
    only copied B1 because no training clip of that media type had a label for it."""
    trained = set(MEDIA_NAMES) - masked.get(label, set())
    found = _labelled(names, label)
    if found is None:
        return pending(f"release/hf/names.yaml records no `labelled` media types for {label}")
    labelled, copied = found & trained, trained - found
    if not labelled:
        return "Copies B1 only" if trained else "Neither"
    main = "Video and still images" if len(labelled) == 2 else MEDIA_NAMES[next(iter(labelled))].capitalize()
    if copied:
        return f"{main} ({MEDIA_NAMES[next(iter(copied))]}: copies B1 only)"
    return main if len(labelled) == 2 else f"{main} only"


def _copied_on_images(model: str, names: dict[str, Any], trained: list[str], masked: dict[str, set[str]]) -> str:
    """E1: a sentence naming the outputs its image graph was trained only to copy from B1,
    because no training image had a label for them; "" when there are none, and for B1."""
    if model != "e1" or not trained:
        return ""
    found = _unique(
        _plain(names, n) for n in trained if "image" not in masked.get(n, set()) and "image" not in (_labelled(names, n) or {"image"})
    )  # fmt: skip
    if not found:
        return ""
    one = len(found) == 1
    return (
        f"On a still image, ignore {_join(found)}: no training image had a label for {'it' if one else 'them'}, so on still "
        f"images E1 only learned to copy Bulwark-B1's {'output' if one else 'outputs'}, which B1 learned from video."
    )


def _masked_on(names: dict[str, Any], trained: list[str], masked: dict[str, set[str]], media: str) -> str:
    """A sentence naming the trained outputs masked on `media`, or "" when there are none."""
    if not trained:
        return pending("no weights are staged")
    found = _unique(_plain(names, n) for n in trained if media in masked.get(n, set()))
    if not found:
        return ""
    one, where = len(found) == 1, "On still images" if media == "image" else "On video"
    return f"{where}, {'this output was' if one else 'these outputs were'} not trained and {'is' if one else 'are'} left out: {_join(found)}."


def _untrained_classes(names: dict[str, Any], taxonomy: Any, trained: list[str], metrics: dict[str, Any]) -> str:
    """A sentence naming the classes of trained categorical outputs with no ground-truth
    training clip, or "" when there are none."""
    found = []
    support = metrics.get("training_support") or {}
    for label in trained:
        spec = taxonomy.labels.get(label)
        counts = (support.get(label) or {}).get("classes_gt") or {}
        if spec is not None and spec.kind == "categorical":
            classes = _entry(names, label).get("classes") or {}
            found += [f"{classes.get(c) or c.replace('_', ' ')} ({_plain(names, label)})" for c in spec.classes if counts.get(c) == 0]
    return f"{'This class was' if len(found) == 1 else 'These classes were'} never trained: {_join(found)}." if found else ""


def _image_calibration(model: str, stage: Path, taxonomy: Any, names: dict[str, Any]) -> str:
    """Which outputs have a temperature or threshold fitted on still images: B1's
    config.json `media_calibration.image`, E1's image graph's contract."""
    if model == "b1":
        image = ((_json(stage / "config.json") or {}).get("media_calibration") or {}).get("image") or {}
    else:
        path = stage / GRAPH_NAMES[1]
        image = _contract(path) if path.is_file() else {}
    fitted = {*(image.get("temperature") or {}), *(image.get("thresholds") or {})}
    found = _unique(_plain(names, n) for n in taxonomy.labels if n in fitted)
    if not found:
        return "On still images, no temperature or threshold was fitted."
    return f"On still images, a temperature or threshold was fitted for {_join(found)} only."


METRIC_NAMES = {"auroc": "AUROC", "macro_f1": "Macro-F1", "mae": "MAE"}


def _metric(spec: LabelSpec) -> str:
    from bulwark.eval.metrics import HEADLINE

    return HEADLINE[spec.kind]


def _score(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) and math.isfinite(value) else "n/a"


def _scored_labels(taxonomy: Any, entries: dict[str, Any]) -> list[str]:
    """Labels with a score, in taxonomy order (by output group, never by score), without the
    metadata outputs, which evaluation uses as sanity checks only."""
    from bulwark.eval.report import is_metadata

    return [n for n, spec in taxonomy.labels.items() if n in entries and not is_metadata(spec)]


def _groups(metrics: dict[str, Any]) -> set[str]:
    """Datasets whose test side is one group of subjects (no per-person ids)."""
    policy = ((metrics.get("meta") or {}).get("predicted_with") or {}).get("split_policy") or (metrics.get("meta") or {}).get("split_policy") or {}
    return set(policy.get("subject_group_datasets") or [])


def _people(entry: dict[str, Any], groups: set[str]) -> str:
    """Test people of a score; a dataset that is one group of subjects counts as a group."""
    total = int(entry.get("n_subjects") or 0)
    grouped = len(set(entry.get("datasets") or []) & groups)
    if grouped:
        return f"{total - grouped} + {grouped} group" + ("s" if grouped > 1 else "")
    return str(total)


def _test_people(count: Any) -> str:
    return f"{int(count or 0)} test {'person' if int(count or 0) == 1 else 'people'}"


SPLIT_SPREAD = 0.25  # a summary row whose datasets' scores differ by more than this is given per dataset


def _results_table(model: str, taxonomy: Any, names: dict[str, Any], metrics: dict[str, Any], teacher: dict[str, Any] | None) -> str:
    """`What it detects | Metric | <model> (baseline) | [B1 |] Test people` from the test
    split's summary: each score to 2 decimals with the constant baseline in brackets. An
    output whose per-dataset scores differ by more than SPLIT_SPREAD gets one row per
    dataset, so that a mean never hides a score at chance. E1's B1 column takes the
    teacher's score only where it is the mean over the same datasets (`_comparison_problems`
    checks that both scored the same clips), and a sentence under E1's table explains it."""
    test = (metrics.get("splits") or {}).get("test") or {}
    summary, per_dataset, counts = test.get("summary") or {}, test.get("datasets") or {}, test.get("counts") or {}
    their_test = ((teacher or {}).get("splits") or {}).get("test") or {}
    theirs, their_datasets = their_test.get("summary") or {}, their_test.get("datasets") or {}
    head = f"| What it detects | Metric | {MODEL_NAMES[model].split('-')[1]} (baseline) |" + (" B1 |" if model == "e1" else "") + " Test people |"
    rows = [head, "|---|---|---|" + ("---|" if model == "e1" else "") + "---|"]
    groups, shown, split = _groups(metrics), set(), []
    for label in _scored_labels(taxonomy, summary):
        entry, metric = summary[label], str(summary[label].get("metric") or _metric(taxonomy.labels[label]))
        datasets = [d for d in _ranked(names, entry.get("datasets") or []) if isinstance(((per_dataset.get(d) or {}).get(label) or {}).get(metric), (int, float))]
        scores = [per_dataset[d][label][metric] for d in datasets]
        if len(datasets) == len(entry.get("datasets") or []) > 1 and max(scores) - min(scores) > SPLIT_SPREAD:
            split.append(label)
            for dataset in datasets:
                own = per_dataset[dataset][label]
                cells = [f"{_name(names, label)}: {_dataset_name(names, dataset)}", METRIC_NAMES.get(metric, metric),
                         f"{_score(own.get(metric))} ({_score((own.get('baseline') or {}).get(metric))})"]  # fmt: skip
                if model == "e1":
                    other = ((their_datasets.get(dataset) or {}).get(label) or {}).get(metric)
                    cells.append(_score(other) if isinstance(other, (int, float)) else "not scored")
                    shown.add(cells[-1])
                people = "1 group" if dataset in groups else str(int((counts.get(dataset) or {}).get("subjects") or 0))
                rows.append("| " + " | ".join([*cells, people]) + " |")
            continue
        cells = [_name(names, label), METRIC_NAMES.get(metric, metric), f"{_score(entry.get('value'))} ({_score((entry.get('baseline') or {}).get(metric))})"]
        if model == "e1":
            other = theirs.get(label) or {}
            same = sorted(other.get("datasets") or []) == sorted(entry.get("datasets") or []) and other.get("metric", metric) == metric
            cells.append(("not comparable" if not same else _score(other.get("value"))) if label in theirs else "not scored")
            shown.add(cells[-1])
        cells.append(_people(entry, groups))
        rows.append("| " + " | ".join(cells) + " |")
    notes = []
    if model == "e1":
        notes.append("The B1 column is Bulwark-B1's score on the same test clips, from its own evaluation.")
        if "not comparable" in shown:
            notes.append('It reads "not comparable" where the two models were scored on different datasets for that output.')
        if "not scored" in shown:
            notes.append('It reads "not scored" where B1\'s evaluation has no score for that output.')
    if split:
        each = _join([_plain(names, label) for label in split])
        notes.append(
            f"{each[0].upper() + each[1:]} {'is' if len(split) == 1 else 'are'} given per dataset, because {'its' if len(split) == 1 else 'their'} "
            f"scores differ by more than {SPLIT_SPREAD:g} between datasets and a mean would hide that."
        )
    return "\n".join(rows) + ("\n\n" + " ".join(notes) if notes else "")


def _dataset_table(model: str, taxonomy: Any, names: dict[str, Any], entries: dict[str, Any], unit: str = "clip") -> list[str]:
    """Scores on one dataset. `n` counts what the dataset was scored on: clips, or events
    when it records one moment from several cameras (metrics.json `units`)."""
    count = "Events" if unit == "event" else "Clips"
    rows = [f"| What it detects | Metric | {MODEL_NAMES[model].split('-')[1]} (baseline) | {count} |", "|---|---|---|---|"]
    for label in _scored_labels(taxonomy, entries):
        entry, metric = entries[label], _metric(taxonomy.labels[label])
        if entry.get(metric) is None:
            continue
        score = f"{_score(entry.get(metric))} ({_score((entry.get('baseline') or {}).get(metric))})"
        rows.append(f"| {_name(names, label)} | {METRIC_NAMES[metric]} | {score} | {int(entry.get('n') or 0):,} |")
    return rows


def _by_dataset(model: str, taxonomy: Any, names: dict[str, Any], metrics: dict[str, Any], split: str) -> str:
    """One table per test dataset, under a bold line with its test people and clips (and
    events, for a dataset scored per event)."""
    data = (metrics.get("splits") or {}).get(split) or {}
    groups, blocks, units = _groups(metrics), [], data.get("units") or {}
    for dataset, entries in ((d, data["datasets"][d]) for d in _ranked(names, data.get("datasets") or {})):
        counts = (data.get("counts") or {}).get(dataset) or {}
        people = "one group of drivers" if dataset in groups else _test_people(counts.get("subjects"))
        title, clips = _dataset_name(names, dataset), f"{int(counts.get('clips') or 0):,} clips"
        lines = [f"**{title}** ({people}, {clips})", ""]
        if units.get(dataset) == "event":
            lines[0] = f"**{title}** ({people}, {clips} of {int(counts.get('events') or 0):,} events)"
            lines.append(
                f"{title} films each moment from several cameras. Each moment, an event, counts once here, scored on the mean "
                "of its cameras' probabilities."
            )
            lines.append("")
        blocks.append("\n".join([*lines, *_dataset_table(model, taxonomy, names, entries, units.get(dataset, "clip"))]))
    return "\n\n".join(blocks) or pending(f"the evaluation has no {split} datasets")


def _holdout_groups(names: dict[str, Any], metrics: dict[str, Any]) -> list[tuple[str, str | None, dict[str, Any]]]:
    """The holdout groups that were scored, as (dataset, camera view or None, group): a
    camera view of a test dataset, filming its test people (split policy `holdout_views`;
    the view's plain name from names.yaml `views`), or a dataset never trained on. Groups
    without clips (a dataset not ingested) are left out."""
    policy = ((metrics.get("meta") or {}).get("predicted_with") or {}).get("split_policy") or {}
    views = {str(pair[0]): str(pair[1]) for pair in policy.get("holdout_views") or []}
    entries = ((metrics.get("splits") or {}).get("holdout") or {}).get("datasets") or {}
    found: list[tuple[str, str | None, dict[str, Any]]] = []
    for dataset, group in sorted((metrics.get("holdout_groups") or {}).items()):
        if group.get("clips") and dataset in entries:
            view = views.get(dataset)
            found.append((dataset, str((names.get("views") or {}).get(view) or view.replace("_", " ")) if view else None, group))
    return found


def _holdout(model: str, taxonomy: Any, names: dict[str, Any], metrics: dict[str, Any]) -> str:
    """The holdout groups (a new camera view, datasets never trained on), each on its own."""
    data = (metrics.get("splits") or {}).get("holdout") or {}
    blocks = []
    for dataset, view, group in _holdout_groups(names, metrics):
        name, clips, people = _dataset_name(names, dataset), f"{int(group['clips']):,} clips", int(group.get("subjects") or 0)
        if view:
            line = f"**{name}, {view}** (a camera that training never used, filming the {_test_people(people)}; {clips})"
        else:
            line = f"**{name}** (a dataset never used in training; {people} {'person' if people == 1 else 'people'}, {clips})"
        unit = (data.get("units") or {}).get(dataset, "clip")
        blocks.append("\n".join([line, "", *_dataset_table(model, taxonomy, names, data["datasets"][dataset], unit)]))
    return "\n\n".join(blocks) or pending("the evaluation scored no holdout group")


def _holdout_summary(taxonomy: Any, names: dict[str, Any], metrics: dict[str, Any]) -> str:
    """One visible sentence per held-out camera view: its scores against the same dataset's
    test cameras, for the outputs scored on both; "" when no view was held out. The two
    sides are scored in the same unit: a holdout scored per clip (one camera) is set beside
    the test cameras' per-clip scores (`per_clip`) where the test side counts events, whose
    score is the mean of several cameras' probabilities."""
    splits = metrics.get("splits") or {}
    test_split, holdout_split = splits.get("test") or {}, splits.get("holdout") or {}
    test, holdout = test_split.get("datasets") or {}, holdout_split.get("datasets") or {}
    sentences = []
    for dataset, view, group in _holdout_groups(names, metrics):
        units = ((test_split.get("units") or {}).get(dataset, "clip"), (holdout_split.get("units") or {}).get(dataset, "clip"))
        per_clip = units == ("event", "clip")
        side = ((test_split.get("per_clip") or {}).get(dataset) if per_clip else test.get(dataset)) or {}
        pairs = []
        for label in _scored_labels(taxonomy, holdout[dataset]) if view and (per_clip or units[0] == units[1]) else []:
            metric = _metric(taxonomy.labels[label])
            new, old = (holdout[dataset].get(label) or {}).get(metric), (side.get(label) or {}).get(metric)
            if isinstance(new, (int, float)) and isinstance(old, (int, float)):
                pairs.append((label, metric, new, old))
        if not pairs:
            continue
        worse = [(new > old) if metric == "mae" else (new < old) for _, metric, new, old in pairs]  # MAE: lower is better
        verdict = "every score is worse" if all(worse) else "no score is worse" if not any(worse) else "some scores are worse"
        name = _dataset_name(names, dataset)
        scores = [f"{_plain(names, label)} {METRIC_NAMES[metric].replace('Macro', 'macro')} {_score(new)} against {_score(old)}" for label, metric, new, old in pairs]
        cameras = f"the clips of {name}'s test cameras, each scored on its own as here" if per_clip else f"{name}'s test cameras"
        sentences.append(
            f"**A camera that training never used.** On {name}'s {view}, filming the {_test_people(group.get('subjects'))}, "
            f"{verdict} than on {cameras}: {_join(scores)}."
        )
    return "\n\n".join(sentences)


def _test_people_sentences(names: dict[str, Any], metrics: dict[str, Any]) -> tuple[str, str]:
    """Who the test split holds, for the Results caveat ("Drive&Act, DMD and MRL Eye each
    have 3 test people. ...") and the opening ("3 test people per dataset"); a dataset that
    is one group of subjects is named as such."""
    counts = ((metrics.get("splits") or {}).get("test") or {}).get("counts") or {}
    groups = _groups(metrics)
    ranked = _ranked(names, counts)
    people: dict[int, list[str]] = {}
    for dataset in ranked:
        if dataset not in groups:
            people.setdefault(int((counts[dataset] or {}).get("subjects") or 0), []).append(_dataset_name(names, dataset, first=True))
    sentences = [f"{_join(sets)} {'each have' if len(sets) > 1 else 'has'} {_test_people(n)}." for n, sets in sorted(people.items())]
    grouped = [_dataset_name(names, d, first=True) for d in ranked if d in groups]
    if grouped:
        many = len(grouped) > 1
        sentences.append(
            f"{_join(grouped)} {'do' if many else 'does'} not identify the person in each image, so {'their' if many else 'its'} "
            "test people count as one group."
        )
    if not people:
        return " ".join(sentences) or pending("the evaluation has no test datasets"), pending("the evaluation has no test people")
    low, high = min(people), max(people)
    short = f"{low} test {'person' if high == 1 else 'people'} per dataset" if low == high else f"{low} to {high} test people per dataset"
    return " ".join(sentences), short


def _outputs_as_b1(trained: list[str], teacher: dict[str, Any] | None) -> str:
    """How E1's trained outputs compare with its B1 teacher's (as the teacher's evaluation
    records them) in the card's second sentence: "the same", or "most of the same"."""
    theirs = (((teacher or {}).get("checkpoint_labels") or {}).get("supervised_labels"))
    if not theirs or not trained:
        return pending("the evaluation of E1's B1 teacher is not staged (evaluation/b1/)")
    return "the same" if set(trained) == set(theirs) else "most of the same"


def _test_split(names: dict[str, Any], metrics: dict[str, Any]) -> str:
    counts = ((metrics.get("splits") or {}).get("test") or {}).get("counts") or {}
    groups = _groups(metrics)
    rows = ["| Dataset | Test people | Test clips |", "|---|---|---|"]
    for dataset, entry in ((d, counts[d]) for d in _ranked(names, counts)):
        people = "one group, not identified per image" if dataset in groups else str(entry.get("subjects", 0))
        rows.append(f"| {_dataset_name(names, dataset)} | {people} | {int(entry.get('clips') or 0):,} |")
    return "\n".join(rows)


def _datasets_table(names: dict[str, Any], datasets: list[str]) -> str:
    """`Dataset | Terms | Official page` of the training datasets: the registry's licence
    summary, word for word, and its link, with names.yaml's `archive` copy of the page next
    to it where it has one; under it, names.yaml's `note` of each, which spells out what the
    summary abbreviates."""
    rows = ["| Dataset | Terms | Official page |", "|---|---|---|"]
    for dataset in _ranked(names, datasets):
        spec = BY_NAME.get(dataset)
        if spec is None:
            rows.append(f"| {dataset} | not in the registry | |")
            continue
        name, archive = _dataset_name(names, dataset), _dataset(names, dataset).get("archive")
        page = f"[{name} website]({spec.url})" + (f" ([archived copy]({archive}))" if archive else "")
        rows.append(f"| {name} | {spec.licence} | {page} |")
    notes = [str(_dataset(names, d)["note"]) for d in _ranked(names, datasets) if _dataset(names, d).get("note")]
    return "\n".join(rows) + "".join(f"\n\n{note}" for note in notes)


def _dataset_citations(names: dict[str, Any], datasets: list[str]) -> str:
    """A list item citing each training dataset's paper (names.yaml `cite`)."""
    items = []
    for dataset in _ranked(names, datasets):
        cite = _dataset(names, dataset).get("cite")
        items.append(f"- {cite}" if cite else f"- {pending(f'release/hf/names.yaml has no `cite` for {dataset}')}")
    return "\n".join(items)


def _speed_table(benches: dict[str, dict[str, Any]]) -> str:
    """`Device | Threads | Runtime and version | Input | Result` from each graph's bench.json."""
    rows = ["| Device | Threads | Runtime and version | Input | Result |", "|---|---|---|---|---|"]
    for _, bench in sorted(benches.items(), key=lambda item: item[0] != GRAPH_NAMES[16]):
        for result in bench.get("results") or []:
            timing = f"p50 {result.get('ms_p50', float('nan')):.0f} ms, p95 {result.get('ms_p95', float('nan')):.0f} ms per call"
            rows.append(f"| {result.get('cpu')} | {result.get('threads')} | onnxruntime {result.get('onnxruntime')} | {result.get('input')} | {timing} |")
    return "\n".join(rows)


def _parameters(model: str, path: Path) -> int:
    """Values stored in the weights: the safetensors header's shapes, or the graph's initializers."""
    if model == "b1":
        with (path / "model.safetensors").open("rb") as handle:
            header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
        return sum(math.prod(t["shape"]) for k, t in header.items() if k != "__metadata__")
    import onnx

    return sum(math.prod(t.dims) for t in onnx.load(str(path), load_external_data=False).graph.initializer)


def _files_table(model: str, stage: Path) -> str:
    rows = ["| File | What it is | Size |", "|---|---|---|"]
    if model == "b1":
        rows.append(f"| `model.safetensors` | all B1 weights, with the LoRA adapters merged into the encoder | {_size(stage / 'model.safetensors')} |")
        rows.append(f"| `config.json` | outputs, calibration, input and training record, read by `bulwark.inference.load` | {_size(stage / 'config.json')} |")
    else:
        for frames, name in GRAPH_NAMES.items():
            if (stage / name).is_file():
                shape = ", ".join(str(d) for d in (_contract(stage / name).get("input") or {}).get("shape") or [])
                what = f"{GRAPH_MEDIA[frames]} graph: input `frames` ({shape}) uint8 RGB"
                rows.append(f"| `{name}` | {what} | {_size(stage / name)} |")
        size = _bytes(len(graphs_config_text(stage).encode("utf-8")))  # derived with the card, so not on disk yet
        rows.append(f"| `config.json` | both graphs' contracts: inputs, outputs and their decoding, labels, calibration | {size} |")
    return "\n".join(rows)


def _parity(graphs: dict[str, Path]) -> str:
    """What `bulwark export` checked each graph against: its contract's `parity` record (the
    frames it used, the largest logit difference from PyTorch, the tolerance)."""
    records = {name: _contract(path).get("parity") or {} for name, path in graphs.items()}
    if not records or not all(r.get("batch") and r.get("max_abs_logit_diff") is not None for r in records.values()):
        return pending("a graph records no parity check (`bulwark export` writes one)")
    batches = _unique(str(r["batch"]) for r in records.values())
    on = batches[0] if len(batches) == 1 else _join([f"{r['batch']} ({GRAPH_MEDIA[int(onnx_props(graphs[n]).get('frames') or 0)]} graph)" for n, r in records.items()])
    worst = max(float(r["max_abs_logit_diff"]) for r in records.values())
    tolerances = _unique(f"{float(r['tolerance']):g}" for r in records.values() if r.get("tolerance") is not None)
    within = f", within the tolerance of {_join(tolerances)}" if tolerances else ""
    return f"`bulwark export` checked each graph against the PyTorch model on {on}: the logits differ by at most {worst:.1e}{within}."


def _size(path: Path) -> str:
    return _bytes(path.stat().st_size) if path.is_file() else ""


def _bytes(size: int) -> str:
    return f"{size / 1e6:.1f} MB" if size >= 100_000 else f"{size / 1e3:.1f} kB"


# ---- checks shared by stage, upload and open ------------------------------------------------


def onnx_props(path: Path) -> dict[str, str]:
    import onnx

    return {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}


def artifact_record(path: Path, model: str, root: Path = REPO) -> dict[str, Any]:
    """The release check of a staged B1 release directory or E1 graph, as stored in
    release-check.json.

    The verdict of `bulwark release-check` (`check_release(training_record(path))`, which
    follows the registry of `root`; with `commercial` for a licence that is not
    research-only, as `bulwark release` and `bulwark export` check it), plus what a public
    release also needs: the card of the model's own repository, the current model release,
    a research-only licence, and the artifact's own record (config.json's
    `release_check`, the graph's `release_cleared`) saying cleared. `checked_*` say when
    and with which code; every other field must be the same when upload and open check
    again."""
    if path.suffix == ".onnx":
        props = onnx_props(path)
        contract = json.loads(props.get(ONNX_CONTRACT_KEY) or "{}")
        frames = props.get("frames")
        identity = {
            "kind": "onnx graph", "model": "e1", "frames": int(frames) if frames else None,
            "sha256": sha256_file(path), "checkpoint_sha256": (contract.get("checkpoint") or {}).get("sha256"),
        }  # fmt: skip
        licence, card, release = props.get("license"), props.get("model_card"), props.get("model_release")
        stored = [json.loads(props.get("release_cleared") or "null"), (contract.get("release_check") or {}).get("cleared")]
        rebuild, kind = "export it again with `bulwark export`", "e1"
    else:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        identity = {
            "kind": "release directory", "model": config.get("kind"), "sha256": sha256_file(path / "model.safetensors"),
            "checkpoint_sha256": ((config.get("source") or {}).get("checkpoint") or {}).get("sha256"),
        }  # fmt: skip
        licence, card, release = config.get("weights_licence"), config.get("model_card"), config.get("model_release")
        stored = [(config.get("release_check") or {}).get("cleared")]
        rebuild, kind = "build it again with `bulwark release`", str(config.get("kind") or "")
    check = check_release(training_record(path), commercial=not research_only(licence or ""))
    problems = list(check.problems)
    try:
        own = released_model(kind)
    except ValueError:
        own = None
    if own != model:
        problems.append(f"It is a {kind or 'model of no recorded kind'}, not {MODEL_NAMES[model]}.")
    elif card != model_card_url(model):
        problems.append(f"It records model card {card}, not {model_card_url(model)}: {rebuild}.")
    if release != MODEL_RELEASE:
        problems.append(f"It records model release {release}, not {MODEL_RELEASE}: {rebuild}.")
    if not research_only(licence or ""):
        problems.append(f"Its licence {licence!r} is not research-only.")
    if not all(value is True for value in stored):
        problems.append(f"Its own record says it is not cleared for release: {rebuild} once `bulwark release-check` clears it.")
    state = git_state(root)
    record = {
        "artifact": path.name if path.suffix == ".onnx" else "model.safetensors", **identity, "licence": licence,
        "model_card": card, "model_release": release, "release_check": check.summary(), "problems": problems,
        "publishable": not problems,
        "checked_with": {"bulwark": __version__, "commit": state.get("commit"), "dirty": state.get("dirty")},
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }  # fmt: skip
    return cast("dict[str, Any]", json.loads(json.dumps(record)))


def recheck(stage: Path, model: str, root: Path = REPO) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Fresh release checks of the staged artifacts, and every way they differ from the
    records staged with them (a changed registry or code, or an edited record)."""
    records: dict[str, dict[str, Any]] = {}
    problems = []
    stored_all = _json(stage / RECORDS)
    if stored_all is None:
        return {}, [f"{RECORDS}: missing or unreadable; stage again"]
    for name, path in staged_artifacts(stage, model).items():
        records[name] = fresh = artifact_record(path, model, root)
        stored = stored_all.get(name)
        if not isinstance(stored, dict):
            problems.append(f"{name}: not in {RECORDS}; stage again")
            continue
        keys = sorted(k for k in {*fresh, *stored} if not k.startswith("checked_") and fresh.get(k) != stored.get(k))
        if keys:
            problems.append(
                f"{name}: its release check now differs from the one staged with it ({', '.join(keys)}): the "
                "registry or the code changed since staging; stage again"
            )
    problems += [f"{name}: in {RECORDS}, but not staged" for name in sorted(set(stored_all) - set(records))]
    return records, problems


def public_problems(stage: Path, model: str, records: dict[str, dict[str, Any]], root: Path = REPO) -> list[str]:
    """What keeps a stage from a public repository: an artifact that is not cleared, weights
    that do not come from the released B1 (`_released_b1_problems`), a part
    of the model's release missing (RELEASING.md, "Bundle": B1's release directory, or E1's
    16-frame and 1-frame graphs from one checkpoint; the evaluation of these weights, E1's
    also of its B1 teacher; the checkpoint's provenance), *at release* items in the
    card, the full card or the data card it links to, a model that is not in the release
    set (`release_set`), and a full card that names the repository of another model that
    is not released (the card itself is refused for that at any stage, `render_card`)."""
    released, problems = release_set(root)
    if released and model not in released:
        problems.append(not_released(model))
    problems += [f"{name}: {problem}" for name, record in records.items() for problem in record["problems"]]
    problems += _released_b1_problems(stage, model, records)
    if model == "b1" and not records:
        problems.append("The release needs B1's release directory (`bulwark release` of the B1 checkpoint); none is staged.")
    for frames, name in GRAPH_NAMES.items() if model == "e1" else ():
        if name not in records:
            problems.append(f"The release needs a {frames}-frame E1 graph (`bulwark export`{' --frames 1' if frames == 1 else ''}); none is staged.")
    if not (stage / "evaluation" / model / "metrics.json").is_file():
        problems.append(f"No evaluation of these weights is staged (--eval; evaluation/{model}/).")
    if model == "e1" and not (stage / "evaluation" / "b1" / "metrics.json").is_file():
        problems.append("No evaluation of E1's B1 teacher is staged (--teacher-eval; evaluation/b1/).")
    if not (stage / PROVENANCE).is_file():
        problems.append(f"No provenance of the checkpoint is staged (--provenance; {PROVENANCE}).")
    readme = (stage / "README.md").read_text(encoding="utf-8") if (stage / "README.md").is_file() else ""
    for match in LINK.finditer(readme):
        target = (match.group(1) or match.group(2) or "").partition("#")[0]
        if target and not SCHEME.match(target) and not (stage / posixpath.normpath(target)).is_file():
            problems.append(f"README.md: link {target} reaches no staged file.")
    texts = {"README.md": stage / "README.md", "MODEL_CARD.md": stage / "MODEL_CARD.md", "docs/data-card.md": root / "docs" / "data-card.md"}
    for doc, path in texts.items():
        count = len(AT_RELEASE.findall(path.read_text(encoding="utf-8"))) if path.is_file() else 0
        if count:
            problems.append(f"{doc}: {count} items marked *at release* or open items; complete them (RELEASING.md, Model card).")
    if (stage / "MODEL_CARD.md").is_file():
        problems += unreleased_repo_problems((stage / "MODEL_CARD.md").read_text(encoding="utf-8"), "MODEL_CARD.md", released, model)
    return problems


def _released_b1_problems(stage: Path, model: str, records: dict[str, dict[str, Any]]) -> list[str]:
    """Both cards name one B1 (B1_CHECKPOINT_SHA256): B1's weights must be built from it, and
    E1's graphs distilled from its outputs, since E1's card calls Bulwark-B1 its teacher and
    sets its scores beside E1's."""
    released = B1_CHECKPOINT_SHA256[:12]
    if model == "b1":
        return [
            f"{name}: built from checkpoint {str(record.get('checkpoint_sha256'))[:12]}, not the released Bulwark-B1 checkpoint "
            f"({released}, bulwark.export.release.B1_CHECKPOINT_SHA256); build the release directory from that checkpoint."
            for name, record in records.items() if record.get("checkpoint_sha256") != B1_CHECKPOINT_SHA256
        ]  # fmt: skip
    teachers = sorted(str(t) for t in _teachers(staged_artifacts(stage, "e1").values()))
    if not records or teachers == [B1_CHECKPOINT_SHA256]:
        return []
    found = ", ".join(t[:12] for t in teachers) or "none recorded"
    return [
        f"The graphs were distilled from B1 checkpoint {found}, not the released Bulwark-B1 ({released}, "
        "bulwark.export.release.B1_CHECKPOINT_SHA256), which E1's card names as its teacher; distil E1 from that B1's outputs."
    ]


def checkout_problems(root: Path) -> list[str]:
    """A release check that clears weights for the public must come from committed code."""
    state = git_state(root)
    if state.get("dirty") is False:
        return []
    changed = ", ".join(state.get("changed") or []) or "unknown"
    return [
        f"The checkout has uncommitted changes ({changed}) or is not a git checkout. Its registry decides the "
        "release check, so commit first."
    ]


def scan(stage: Path, model: str) -> list[str]:
    """Files a stage must never hold, and text that must never be published.

    Refused: symbolic links, dotfiles, files with a DATA_SUFFIXES suffix or "manifest" in
    the name, paths outside the model's LAYOUT, a graph's bench.json without the graph,
    and binary files other than weights. The text of every other file, an ONNX graph's
    free-text fields and a safetensors header must hold no local path, secret, email
    address other than PUBLIC_EMAILS, value of a SECRET_ENV variable or dataset subject id
    other than GROUP_IDS."""
    problems = []
    for dirpath, dirnames, filenames in os.walk(stage):
        for name in [*dirnames, *filenames]:
            if (Path(dirpath) / name).is_symlink():
                problems.append(f"{(Path(dirpath) / name).relative_to(stage).as_posix()}: a symbolic link")
    values = {k: os.environ[k] for k in SECRET_ENV if len(os.environ.get(k, "").strip()) >= 4}
    for rel in _files(stage):
        path, name = stage / rel, rel.rsplit("/", 1)[-1]
        if any(part.startswith(".") for part in rel.split("/")):
            problems.append(f"{rel}: a dotfile")
        if name.lower().endswith(DATA_SUFFIXES):
            problems.append(f"{rel}: dataset material or a pickle ({Path(name).suffix} files are never staged)")
        if "manifest" in name.lower():
            problems.append(f"{rel}: a manifest is part of the dataset")
        if rel not in LAYOUT[model]:
            problems.append(f"{rel}: not part of a {MODEL_NAMES[model]} stage")
        elif rel.endswith(".bench.json") and not (stage / rel.removesuffix(".bench.json")).is_file():
            problems.append(f"{rel}: belongs to no staged graph named {rel.removesuffix('.bench.json')}")
        try:
            texts = _texts(path)
        except (UnicodeDecodeError, ValueError, OSError) as error:
            problems.append(f"{rel}: unexpected binary or unreadable file ({type(error).__name__})")
            continue
        problems += [f"{rel}: {p}" for p in _text_problems(texts, values)]
    return problems


def _texts(path: Path) -> list[str]:
    if path.suffix == ".onnx":
        return onnx_text(path)
    if path.suffix == ".safetensors":
        with path.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            if size > 100 << 20:
                raise ValueError("safetensors header too large")
            return [handle.read(size).decode("utf-8")]
    return [path.read_bytes().decode("utf-8")]


def onnx_text(path: Path) -> list[str]:
    """Every free-text string of an ONNX model (ONNX_TEXT_FIELDS of the model, its graph,
    nodes, attributes and functions; not tensor data)."""
    import onnx

    found: list[str] = []

    def walk(message: Any) -> None:
        for field, value in message.ListFields():
            repeated = getattr(field, "is_repeated", None)
            items = value if (repeated if repeated is not None else field.label == field.LABEL_REPEATED) else [value]
            if field.type == field.TYPE_MESSAGE:
                for item in items:
                    walk(item)
            elif field.name in ONNX_TEXT_FIELDS:
                found.extend(i.decode("utf-8", "replace") if isinstance(i, bytes) else str(i) for i in items)

    walk(onnx.load(str(path), load_external_data=False))
    return found


def _text_problems(texts: list[str], values: dict[str, str]) -> list[str]:
    problems = []
    for text in texts:
        if match := LOCAL_PATH.search(text):
            problems.append(f"a local path ({text[max(0, match.start() - 20):match.end() + 30]!r})")
        if SECRET.search(text):
            problems.append("what looks like a token, key or signed URL")
        if {address.rstrip(".") for address in EMAIL.findall(text)} - PUBLIC_EMAILS:
            problems.append("an email address")
        problems += [f"the value of ${k}" for k, v in values.items() if v in text]
        if found := sorted({i.rstrip(".") for i in SUBJECT_ID.findall(text)} - GROUP_IDS):
            problems.append(
                f"dataset subject ids ({', '.join(found[:3])}); they are part of the dataset (an evaluation "
                "report: re-score it with `bulwark report`, which leaves them out)"
            )
    return list(dict.fromkeys(problems))


def _remove_ds_store(stage: Path) -> None:
    for path in stage.rglob(".DS_Store"):
        path.unlink()


def _files(stage: Path) -> list[str]:
    """Every regular file in a stage, relative, sorted (symbolic links are not followed)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(stage):
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
        found += [(Path(dirpath) / f).relative_to(stage).as_posix() for f in filenames if not (Path(dirpath) / f).is_symlink()]
    return sorted(found)


def write_sums(stage: Path) -> None:
    """SHA256SUMS of every other file, `shasum -a 256 -c` compatible; written last."""
    lines = [f"{sha256_file(stage / rel)}  {rel}\n" for rel in _files(stage) if rel != SUMS]
    (stage / SUMS).write_text("".join(lines), encoding="utf-8")


def verify_sums(stage: Path) -> list[str]:
    """The stage holds exactly the files SHA256SUMS lists, unchanged."""
    path = stage / SUMS
    if not path.is_file():
        return [f"{stage}: no {SUMS}; run `stage` first"]
    listed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, sep, rel = line.partition("  ")
        if not sep or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return [f"{SUMS}: unreadable line {line!r}; stage again"]
        listed[rel] = digest
    present = set(_files(stage)) - {SUMS}
    problems = [f"{rel}: not in {SUMS} (added after staging)" for rel in sorted(present - set(listed))]
    problems += [f"{rel}: in {SUMS} but missing" for rel in sorted(set(listed) - present)]
    problems += [f"{rel}: changed since staging" for rel in sorted(present & set(listed)) if sha256_file(stage / rel) != listed[rel]]
    return problems


def check_stage(stage: Path, model: str, root: Path = REPO) -> tuple[list[str], dict[str, dict[str, Any]], list[str]]:
    """What upload and open check before any call to the Hub: (problems that refuse any
    upload, the fresh release-check records, the problems that keep the stage private).
    The stage is cleared only when both lists are empty.

    Besides SHA256SUMS, the scan and the release checks, the input checks of `stage` run
    again on the staged copies, and every file `stage` derives (README.md, MODEL_CARD.md,
    the licences, NOTICE, E1's config.json) must be what it derives now, so that a stage
    edited after staging (with SHA256SUMS written again), or one staged from templates or
    docs that have changed since, is refused as `stage` would refuse it."""
    if not stage.is_dir():
        return [f"{stage}: no staged folder; run `stage --model {model}` first"], {}, []
    _remove_ds_store(stage)
    problems = verify_sums(stage) + scan(stage, model)
    if not problems:
        problems = staged_input_problems(stage, model)
    if problems:
        return problems, {}, []
    records, changed = recheck(stage, model, root)
    if not records:
        return [*changed, f"{stage}: no {MODEL_NAMES[model]} artifact is staged"], {}, []
    derived, found = derived_files(stage, model, records, root)
    changed += found
    for rel, text in derived.items():
        path = stage / rel
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            changed.append(f"{rel}: not what `stage` writes for these artifacts from this checkout's files; stage again")
    return changed, records, public_problems(stage, model, records, root)


def staged_input_problems(stage: Path, model: str) -> list[str]:
    """The input checks of `stage` on a staged folder: its release directory or graphs
    (with their bench.json), the evaluations (each of the staged weights, or of E1's
    teacher) and the provenance (of the staged weights' checkpoint)."""
    artifacts = staged_artifacts(stage, model)
    problems = [] if artifacts else [f"{stage}: no {MODEL_NAMES[model]} weights are staged"]
    for name, path in artifacts.items():
        found = _release_problems(path, model) if model == "b1" else _graph_problems(path)
        if model == "e1" and not found and int(onnx_props(path).get("frames") or 0) != {v: k for k, v in GRAPH_NAMES.items()}[name]:
            found.append(f"{name}: the graph's frames do not match its name")
        problems += found
    required = ["LICENSE", "NOTICE", "MODEL_CARD.md", "README.md", "config.json", UPSTREAM_LICENCE[model]]
    problems += [f"{rel}: missing; a {MODEL_NAMES[model]} stage holds it" for rel in required if not (stage / rel).is_file()]
    if problems:
        return problems
    graphs = {n: p for n, p in artifacts.items() if model == "e1"}
    problems += _same_checkpoint_problems(graphs)
    sources = artifact_sources(model, stage if model == "b1" else None, graphs)
    own, teacher = stage / "evaluation" / model, stage / "evaluation" / "b1"
    found = _eval_problems(own, scored_weights(model, sources, graphs.values())) if own.is_dir() else []
    if model == "e1" and teacher.is_dir():
        found += _eval_problems(teacher, _teachers(graphs.values()), what="the B1 teacher of these graphs")
        if own.is_dir() and not found:
            found += _comparison_problems(own, teacher)
    problems += found
    if (stage / PROVENANCE).is_file():
        problems += _provenance_problems(stage / PROVENANCE, sources)
    return problems


# ---- upload and open -------------------------------------------------------------------------


def upload(
    stage: Path, model: str, api: HfApi, *, tag: str | None = MODEL_RELEASE, allow_weight_deletion: bool = False,
    public: bool = False, root: Path = REPO,
) -> list[str]:  # fmt: skip
    """Upload a model's stage to its repository (MODEL_REPOS) so that main mirrors it; the
    problems that refused it, or [].

    The repository is created private when it does not exist, and nothing is written to an
    existing one before its visibility is known. A stage is cleared when `stage --public`
    would take it. A public repository is refused unless `public`, and then takes only a
    cleared stage, and only when it is gated as `open` gates it (GATED) and every other
    model's repository that the cards link is public (`linked_closed_problems`); removing a
    weight file from it needs `allow_weight_deletion`, and adding or changing one needs a new
    `tag`. A cleared stage is tagged `tag` (refused when the tag exists; the tag is never
    moved), an uncleared one is an untagged private preview. A model that is not in the
    release set is never cleared, and `public` refuses it at once. Visibility is never
    changed here."""
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    repo = MODEL_REPOS[model]
    if public:
        released, found = release_set(root)
        if found or model not in released:
            return found or [not_released(model)]
    problems, records, pending_public = check_stage(stage, model, root)
    if problems:
        return problems
    cleared = not pending_public
    problems = (checkout_problems(root) if cleared else []) + _endpoint_problems(api)
    if problems:
        return problems
    try:
        info = _info(api, repo, files_metadata=True)
    except GatedRepoError:
        raise
    except RepositoryNotFoundError:
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        info = _info(api, repo, files_metadata=True)
    private = info.private is True
    if not private:
        if not public:
            return [f"{repo} is public; pass --public to upload to a public repository (only a stage `stage --public` would take)."]
        if not cleared:
            return [*pending_public, f"{repo} is public, so only a stage that `stage --public` would take may go there."]
        if info.gated != GATED:
            return [f"{repo} is public but not gated with automatic approval (gated: {info.gated!r}); fix its settings on the Hub first."]
        linked = linked_closed_problems(api, stage, model)
        if linked:
            return linked
    remote ={s.rfilename: s for s in info.siblings or [] if s.rfilename != ".gitattributes"}
    removed = sorted(set(remote) - set(_files(stage)))
    if removed:
        print(f"Removes from main, which then mirrors the stage: {', '.join(removed)}")
    weights_removed = [r for r in removed if r.lower().endswith(WEIGHT_SUFFIXES)]
    if weights_removed and not private and not allow_weight_deletion:
        return [f"The upload would remove weights from the public repository: {', '.join(weights_removed)}. "
                "Pass --allow-weight-deletion if that is intended."]  # fmt: skip
    new_weights = [rel for rel in _files(stage) if rel.lower().endswith(WEIGHT_SUFFIXES) and not (rel in remote and _same_file(stage / rel, remote[rel]))]
    if new_weights and not private and not tag:
        return [f"The upload would add or change weights in the public repository without a tag: {', '.join(new_weights)}. "
                "Tag them with a new --tag; --no-tag is for a stage whose weights main already holds."]  # fmt: skip
    if cleared and tag:
        refs = api.list_repo_refs(repo, repo_type="model")
        if tag in {t.name for t in refs.tags}:
            return [f"Tag {tag} already exists in {repo}, and tags are never moved: upload with another --tag, "
                    "or with --no-tag if main already holds these weights."]  # fmt: skip
    if cleared and private and remote:
        history = history_problems(api, repo, model, cleared_identities(stage, records))
        if history:
            print("Warning: `open` will refuse this repository until it is deleted and uploaded again:", *history, sep="\n  ")
    lines = [f"{name}: {'cleared' if r['publishable'] else 'not cleared: ' + ' '.join(r['problems'])}" for name, r in records.items()]
    if not cleared:
        lines += ["", "Not ready for a public release:", *(f"- {p}" for p in pending_public)]
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=stage,
        commit_message=f"{MODEL_NAMES[model]} {MODEL_RELEASE}: {', '.join(records)}" + ("" if cleared else " (private preview)"),
        commit_description="\n".join(lines),
        delete_patterns="*",
    )
    if cleared and tag:
        api.create_tag(repo, repo_type="model", tag=tag, revision=commit.oid, tag_message=f"{MODEL_NAMES[model]} {MODEL_RELEASE}")
    print(f"Uploaded {stage} to {commit.commit_url}" + (f", tagged {tag}" if cleared and tag else ""))
    print(f"{repo} is {'private' if private else 'public, gated with automatic approval'}.")
    if not cleared:
        print("The repository stays private; `open` refuses it until the stage is ready for a public release:", *pending_public, sep="\n  ")
    return []


def open_repo(stage: Path, model: str, api: HfApi, *, yes: bool = False, root: Path = REPO) -> list[str]:
    """Make one model's repository public: `open_repos` of that model alone, so every other
    model's repository that its cards link must be public already."""
    return open_repos({model: stage}, api, yes=yes, root=root)


def open_repos(stages: dict[str, Path], api: HfApi, *, yes: bool = False, root: Path = REPO) -> list[str]:
    """Make the repositories of the models in `stages` (model -> its stage) public, gated
    with automatic approval (GATED), all or none; the problems that refused it, or [].

    Every check runs for every model before any setting changes, and one failure leaves
    every repository as it was. A model that is not in the release set (`release_set`) is
    refused before anything else. Each needs a cleared stage from committed code that main
    mirrors file for file, and no revision of its repository (every commit reachable from a
    branch, tag, PR or convert ref) holding weights other than the stage's, dataset
    material, a manifest, a dotfile other than .gitattributes or another path outside the
    model's LAYOUT: making it public publishes every revision. A card may link the
    repository of another model that this run opens (the B1 and E1 cards link each other,
    so they open together: RELEASING.md, launch order); every other model's repository that
    the cards link must be public already (`linked_closed_problems`). Asks for each
    repository id unless `yes`.

    The repositories are then opened in order, each settings change checked afterwards.
    When one did not take, or the change or its check failed (the Hub may have applied it
    before failing), that repository is made private again, and so is every repository this
    run opened before it, so that no card goes public while a repository it links is not;
    the problems say which repositories are public and how to recover (`_undo_open`). An
    interrupt or exit during a change (KeyboardInterrupt, SystemExit) undoes the run the
    same way, prints what it did, and is raised again."""
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    models = list(stages)

    def named(model: str, problems: list[str]) -> list[str]:  # which model a problem is about, when there are several
        return [f"{MODEL_NAMES[model]}: {p}" for p in problems] if len(models) > 1 else problems

    released, found = release_set(root)
    if found:
        return found
    problems = [problem for model in models if model not in released for problem in named(model, [not_released(model)])]
    if problems:
        return problems
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for model, stage in stages.items():
        found, records[model], pending_public = check_stage(stage, model, root)
        problems += named(model, found + pending_public)
    if not problems:
        problems += checkout_problems(root) + _endpoint_problems(api)
    if problems:
        return problems
    for model, stage in stages.items():
        repo = MODEL_REPOS[model]
        try:
            info = _info(api, repo, files_metadata=True)
        except GatedRepoError:
            raise
        except RepositoryNotFoundError:
            problems += named(model, [f"{repo} does not exist, or this token cannot see it: run `upload --model {model}` first."])
            continue
        if info.private is not True:
            problems += named(model, [f"{repo} is already public (gated: {info.gated!r}); open changes nothing."])
            continue
        differences = remote_differences(stage, info.siblings or [])
        if differences:
            problems += named(model, [*differences, f"Main does not mirror this stage: run `upload --model {model}`, then `open`."])
            continue
        history = history_problems(api, repo, model, cleared_identities(stage, records[model]))
        if history:
            problems += named(model, [
                *history,
                "Making the repository public publishes every revision, and these hold files that are not this "
                "release's. Delete the repository in its settings on the Hub, run `upload` again, then `open`.",
            ])  # fmt: skip
    if problems:
        return problems
    closed: dict[str, None] = {}
    for model, stage in stages.items():
        others = _linked_closed(api, stage, model, opening=models)
        problems += named(model, [_closed_problem(other) for other in others])
        closed.update(dict.fromkeys(others))
    if problems:
        together = " ".join(f"--model {m}" for m in MODELS if m in models or m in closed)
        return [*problems, f"Repositories whose cards link each other open together, in one run: `open {together}`."]
    urls = _join([model_card_url(m) for m in models])
    if not yes:
        each = "" if len(models) == 1 else " of each"
        print(
            f"This makes {urls} public, gated with automatic approval: the model card, LICENSE and the file list{each} "
            "become public, and the other files open to anyone who sends the access-request form, which the Hub "
            "approves at once; approved users can read every revision. Open them only once an Inferensys owner has "
            "reviewed the private repository pages and the access-request form in the staged README.md, which the "
            "Hub first shows once a repository is open."
        )
        for model in models:
            if input(f"Type {MODEL_REPOS[model]} to open it: ").strip() != MODEL_REPOS[model]:
                return ["Not confirmed; nothing changed."]
    opened: list[str] = []
    for model in models:
        repo, url = MODEL_REPOS[model], model_card_url(model)
        try:
            api.update_repo_settings(repo, repo_type="model", gated=GATED, private=False)
            info = _info(api, repo)
            if info.private is False and info.gated == GATED:
                opened.append(model)
                print(
                    f"{url} is public, gated with automatic approval: the Hub approves each access request as it comes. "
                    "Requests and their answers are listed in its settings on the Hub, where access can be revoked."
                )
                continue
            state = f"reported private={info.private}, gated={info.gated!r}"
        except Exception as error:  # the Hub may have applied the change before the error
            state = f"failed ({type(error).__name__}: {error})"
        except BaseException as error:  # an interrupt or exit, maybe after the change took: undo, say so, let it through
            undone = _undo_open(api, models, model, f"was interrupted ({type(error).__name__})", [m for m in opened if m != model])
            print("Interrupted:", *undone, sep="\n  ", file=sys.stderr)
            raise
        return _undo_open(api, models, model, state, opened)
    return []


def _undo_open(api: HfApi, models: list[str], failed: str, state: str, opened: list[str]) -> list[str]:
    """After the settings change of `failed`'s repository did not take (`state`), make it
    private again, then every repository the run opened before it (`opened`), last first,
    since the run opens all or none. Says what each repository of the run is now, and with
    several, which are public and how to recover."""
    repo = MODEL_REPOS[failed]
    try:
        api.update_repo_settings(repo, repo_type="model", private=True)
        problems = [f"The settings change {state}, so {repo} was made private again; check its settings on the Hub."]
        unknown = []
    except Exception as error:
        problems = [f"The settings change {state}, and making {repo} private again failed ({type(error).__name__}: "
                    f"{error}). It may be public without gating: check its settings on the Hub now."]  # fmt: skip
        unknown = [failed]
    if len(models) == 1:
        return problems
    public = []
    for model in reversed(opened):
        try:
            api.update_repo_settings(MODEL_REPOS[model], repo_type="model", private=True)
            problems.append(f"{MODEL_REPOS[model]} was opened in this run and was made private again: the run opens all of its repositories or none.")
        except Exception as error:
            public.append(model)
            problems.append(
                f"{MODEL_REPOS[model]} was opened in this run, and making it private again failed ({type(error).__name__}: "
                f"{error}): it is public, gated with automatic approval."
            )  # fmt: skip
    rest = [m for m in models if m != failed and m not in opened]
    problems += [f"{MODEL_REPOS[m]} was not changed and stays private." for m in rest]
    private = [m for m in models if m not in public and m not in unknown]
    summary = f"Public now: {', '.join(MODEL_REPOS[m] for m in public) or 'none'}. Private: {', '.join(MODEL_REPOS[m] for m in private) or 'none'}."
    if unknown:
        summary += f" Unknown, check on the Hub now: {MODEL_REPOS[failed]}."
    if public or unknown:
        waiting = " ".join(f"--model {m}" for m in models if m not in public)
        summary += (
            " A public card may now link a repository that is not public. Make each public repository above private in "
            f"its settings on the Hub, or, once the Hub takes the change, open the others: `open {waiting}`."
        )
    else:
        summary += f" Nothing is public. Fix the cause, then run `open {' '.join(f'--model {m}' for m in models)}` again."
    return [*problems, summary]


def _info(api: HfApi, repo: str, **kwargs: Any) -> ModelInfo:
    return cast("ModelInfo", api.repo_info(repo, repo_type="model", **kwargs))


def _space_is_public(api: HfApi, space: str) -> bool:
    """Whether the Space exists and is public, so `collection --public` can add it."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return api.repo_info(space, repo_type="space").private is False
    except RepositoryNotFoundError:
        return False


def linked_closed_problems(api: HfApi, stage: Path, model: str, *, opening: Iterable[str] = ()) -> list[str]:
    """The repositories of the other models that the stage's cards (README.md and
    MODEL_CARD.md) name but that are not public yet, asked of the Hub, except those of the
    models in `opening`, which the same `open` run makes public. The release set says which
    repositories a card may link; whether they are open yet is the Hub's to say. With both
    models in launch.yaml, each card links the other model's repository while it is still
    private (RELEASING.md, launch order), so a card that goes public, by an upload to a
    public repository or by `open`, waits until every repository it links is public, or
    opens in one `open` run with them."""
    return [_closed_problem(other) for other in _linked_closed(api, stage, model, opening=opening)]


def _closed_problem(model: str) -> str:
    return f"{MODEL_REPOS[model]} is not public yet; open it before publishing a card that links it (RELEASING.md, launch order)."


def _linked_closed(api: HfApi, stage: Path, model: str, *, opening: Iterable[str] = ()) -> list[str]:
    """The models of `linked_closed_problems`."""
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    text = "\n".join((stage / rel).read_text(encoding="utf-8") for rel in ("README.md", "MODEL_CARD.md") if (stage / rel).is_file())
    skip = {model, *opening}
    closed = []
    for other in MODELS:
        repo = MODEL_REPOS[other]
        if other in skip or repo not in text:
            continue
        try:
            public = _info(api, repo).private is False
        except GatedRepoError:
            raise
        except RepositoryNotFoundError:
            public = False
        if not public:
            closed.append(other)
    return closed


def _endpoint_problems(api: HfApi) -> list[str]:
    if getattr(api, "endpoint", None) != ENDPOINT:
        return [f"The Hub client talks to {getattr(api, 'endpoint', None)}, not {ENDPOINT}; nothing was sent."]
    return []


def git_blob_sha1(path: Path) -> str:
    """The git object id of a file, which the Hub reports as `blob_id` for files outside LFS."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(sibling: RepoSibling) -> tuple[str, str | None]:
    sha256 = getattr(sibling.lfs, "sha256", None)
    return ("sha256", sha256) if sha256 else ("blob", sibling.blob_id)


def cleared_identities(stage: Path, records: dict[str, dict[str, Any]]) -> set[tuple[str, str | None]]:
    """The weights of the release-checked artifacts (`records`, from `recheck`) as the Hub
    identifies them: LFS sha256 and git blob id."""
    found: set[tuple[str, str | None]] = set()
    for name, record in records.items():
        found |= {("sha256", record["sha256"]), ("blob", git_blob_sha1(stage / name))}
    return found


def _same_file(path: Path, sibling: RepoSibling) -> bool:
    kind, value = _identity(sibling)
    return value is not None and value == (sha256_file(path) if kind == "sha256" else git_blob_sha1(path))


def remote_differences(stage: Path, siblings: list[RepoSibling]) -> list[str]:
    """How main (`repo_info(files_metadata=True).siblings`) differs from the stage."""
    local = set(_files(stage))
    remote = {s.rfilename: s for s in siblings if s.rfilename != ".gitattributes"}
    problems = [f"{rel}: staged, not in the repository" for rel in sorted(local - set(remote))]
    problems += [f"{rel}: in the repository, not staged" for rel in sorted(set(remote) - local)]
    problems += [f"{rel}: differs from the staged file" for rel in sorted(local & set(remote)) if not _same_file(stage / rel, remote[rel])]
    return problems


def history_problems(api: HfApi, repo: str, model: str, allowed: set[tuple[str, str | None]]) -> list[str]:
    """Files in any revision of a model's repository that may not become public: weights
    other than `allowed`, dataset material, manifests, dotfiles other than .gitattributes,
    and any other path outside the model's LAYOUT, such as a table of labels."""
    refs = api.list_repo_refs(repo, repo_type="model", include_pull_requests=True)
    heads = [r.target_commit for r in (*refs.branches, *refs.tags, *refs.converts, *(refs.pull_requests or []))]
    commits = dict.fromkeys(c.commit_id for head in heads for c in api.list_repo_commits(repo, repo_type="model", revision=head))
    problems = []
    for commit in commits:
        for sibling in _info(api, repo, revision=commit, files_metadata=True).siblings or []:
            rel = sibling.rfilename
            name = rel.rsplit("/", 1)[-1].lower()
            if rel == ".gitattributes":
                continue
            if name.endswith(WEIGHT_SUFFIXES):
                reason = None if _identity(sibling) in allowed else "weights that are not this release's cleared weights"
            elif name.endswith(DATA_SUFFIXES):
                reason = "dataset material or a pickle"
            elif "manifest" in name:
                reason = "a manifest"
            elif any(part.startswith(".") for part in rel.split("/")):
                reason = "a dotfile"
            elif rel not in LAYOUT[model]:
                reason = f"not a file of a {MODEL_NAMES[model]} stage"
            else:
                reason = None
            if reason:
                problems.append(f"commit {commit[:12]}: {rel} ({reason})")
    return problems


# ---- the collection and the organisation card ------------------------------------------------

# Limits of the Hub (huggingface_hub `create_collection` and `add_collection_item`).
COLLECTION_LIMITS = {"title": 60, "description": 150, "note": 500}
# The item types collection.yaml may list, each the Hub's `item_type` (the demo Space, the
# models, the papers).
COLLECTION_TYPES = ("model", "space", "paper")


def collection_spec(root: Path = REPO) -> tuple[dict[str, Any], list[str]]:
    """release/hf/collection.yaml, with each item's `id` resolved (a model item names `b1`
    or `e1`; a space or paper item gives its Hub id) and the model it belongs to (`for`: a
    model item's own model, or the `for` of an item that only that model needs), and its
    problems: missing fields, texts over the Hub's limits, unknown types and models, and an
    item listed twice."""
    data = _yaml(root / TEMPLATES / "collection.yaml")
    problems = [f"collection.yaml: no {key}" for key in ("namespace", "title", "description") if not data.get(key)]
    for key in ("title", "description"):
        if len(str(data.get(key) or "")) > COLLECTION_LIMITS[key]:
            problems.append(f"collection.yaml: the {key} is longer than {COLLECTION_LIMITS[key]} characters")
    items = []
    seen: dict[tuple[Any, str], int] = {}
    for number, item in enumerate(data.get("items") or [], 1):
        kind, note = item.get("type"), str(item.get("note") or "").strip()
        item_id = MODEL_REPOS.get(str(item.get("model"))) if kind == "model" else str(item.get("id") or "")
        owner = str(item.get("model")) if kind == "model" else item.get("for")
        if kind not in COLLECTION_TYPES or not item_id:
            problems.append(f"collection.yaml: item {number} needs type model (with model b1 or e1), space or paper (with id)")
        elif (kind, item_id) in seen:
            # One copy deferred and one wanted would delete the item the run then expects to find.
            problems.append(f"collection.yaml: item {number} repeats item {seen[kind, item_id]} ({kind} {item_id})")
        else:
            seen[kind, item_id] = number
        if owner is not None and owner not in MODELS:
            problems.append(f"collection.yaml: item {number} is `for` {owner}, which is not a model ({' or '.join(MODELS)})")
        if len(note) > COLLECTION_LIMITS["note"]:
            problems.append(f"collection.yaml: the note of item {number} is longer than {COLLECTION_LIMITS['note']} characters")
        items.append({"type": kind, "id": item_id, "note": note, "at_launch": bool(item.get("at_launch")), "for": owner})
    return {**data, "items": items}, problems


def collection(api: HfApi, *, public: bool = False, root: Path = REPO) -> list[str]:
    """Create or update the collection of release/hf/collection.yaml: its title,
    description, items in order and their notes. It is created private and stays private;
    with `public` (at launch) it gets the items marked `at_launch` (the demo Space among
    them) and is made public, once the repository of every model in the release set
    (`release_set`) is public. A public collection is refused without `public`.

    The items of a model that is not in the release set (its model item, and papers `for`
    it) are deferred: left out, and removed from the collection when they are there, each
    removal printed. That is the only deletion: items not in the file are left in place and
    listed, and come back only once the model is released."""
    spec, problems = collection_spec(root)
    released, found = release_set(root)
    problems += found + _endpoint_problems(api)
    if problems:
        return problems
    if public:
        closed = [MODEL_REPOS[m] for m in MODELS if m in released and _info(api, MODEL_REPOS[m]).private is not False]
        if closed:
            return [f"{repo} is not public yet; open it before the collection (RELEASING.md, launch order)." for repo in closed]
        spaces = [item["id"] for item in spec["items"] if item["type"] == "space" and (item["for"] is None or item["for"] in released)]
        missing = [space for space in spaces if not _space_is_public(api, space)]
        if missing:
            return [
                f"The Space {space} does not exist or is not public; create it (RELEASING.md, launch order), or change "
                f"or remove its item in {TEMPLATES}/collection.yaml, before collection --public."
                for space in missing
            ]
    found_collection = api.create_collection(
        spec["title"], namespace=spec["namespace"], description=spec["description"], private=True, exists_ok=True
    )
    slug = found_collection.slug
    if found_collection.private is not True and not public:
        return [f"The collection {slug} is public; pass --public to update a public collection."]
    api.update_collection_metadata(slug, title=spec["title"], description=spec["description"])
    deferred = [item for item in spec["items"] if item["for"] is not None and item["for"] not in released]
    wanted = [item for item in spec["items"] if item not in deferred and (public or not item["at_launch"])]
    current = {(i.item_type, i.item_id): i for i in api.get_collection(slug).items}
    for item in deferred:
        existing = current.get((item["type"], item["id"]))
        if existing is not None:
            api.delete_collection_item(slug, existing.item_object_id)
            print(f"Removed {item['type']} {item['id']}: it waits for {MODEL_NAMES[item['for']]}, which is not released ({LAUNCH}).")
    for item in wanted:
        existing = current.get((item["type"], item["id"]))
        if existing is None:
            api.add_collection_item(slug, item["id"], item["type"], note=item["note"] or None, exists_ok=True)
        elif (existing.note or "") != item["note"]:
            api.update_collection_item(slug, existing.item_object_id, note=item["note"])
    items = api.get_collection(slug).items
    by_key = {(i.item_type, i.item_id): i for i in items}
    for position, item in enumerate(wanted):
        existing = by_key.get((item["type"], item["id"]))
        if existing is not None and existing.position != position:
            api.update_collection_item(slug, existing.item_object_id, position=position)
    listed = {(item["type"], item["id"]) for item in spec["items"]}
    others = [f"{i.item_type} {i.item_id}" for i in items if (i.item_type, i.item_id) not in listed]
    if others:
        print(f"Left in place (not in collection.yaml): {', '.join(others)}")
    if public:
        api.update_collection_metadata(slug, private=False)
    url = f"{ENDPOINT}/collections/{slug}"
    print(f"{url} holds {len(wanted)} items and is {'public' if public else 'private'}.")
    if deferred:
        waiting = ", ".join(f"{item['type']} {item['id']}" for item in deferred)
        print(f"Deferred until their model is released ({LAUNCH}): {waiting}")
    if spec.get("url") != url:
        print(f"Record its address in {TEMPLATES}/collection.yaml (url: {url}), so the cards link to it.")
    return []


def org_card(out: Path, *, validate: bool = False, root: Path = REPO) -> list[str]:
    """Write the organisation card, the README.md of the Space Inferensys/README (a public
    Space of that name is what the Hub shows on the organisation page), into `out`, and print
    the organisation's settings (release/hf/org.yaml). Its release blocks follow the release
    set, and it may name no repository of a model that is not released. Nothing is sent to
    the Hub except, with `validate`, the card's metadata to the validate-yaml endpoint, so
    it cannot check that the repositories it links are public yet: it lists them, for the
    admin who publishes the Space."""
    problems = _input_problems([], out, root)
    template = root / TEMPLATES / "org-card.md"
    settings = _yaml(root / TEMPLATES / "org.yaml")
    released, found = release_set(root)
    problems += found
    if not template.is_file():
        problems.append(f"{TEMPLATES}/org-card.md: missing")
    problems += [f"{TEMPLATES}/org.yaml: no {key}" for key in ("details", "website") if not settings.get(key)]
    if problems:
        return problems
    source = f"{TEMPLATES}/org-card.md"
    body = release_blocks(template.read_text(encoding="utf-8"), released, source, problems)
    body = fill(body, shared_values(root), source, problems)
    problems += unreleased_repo_problems(body, "README.md", released)
    metadata = {"title": "README", "sdk": "static", "pinned": False}
    card = "---\n" + yaml.dump(metadata, Dumper=_CardDumper, sort_keys=False) + "---\n\n" + body
    problems += card_link_problems(card, out, set(), root, tracked_files(root))  # absolute links only: a Space has no stage
    problems += [f"README.md: {p}" for p in _text_problems([card, *map(str, settings.values())], {})]
    problems += structure_problems(body)
    if validate and not problems:
        problems += validate_metadata(card, "space")
    if problems:
        return problems
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(card, encoding="utf-8")
    print(f"Wrote {out / 'README.md'}, the card of the Space {MODEL_REPOS['b1'].split('/')[0]}/README.")
    if count := len(AT_RELEASE.findall(body)):
        print(f"Not ready to publish: {count} *at release* items (the collection link needs collection.yaml `url`).")
    if linked := [MODEL_REPOS[m] for m in MODELS if MODEL_REPOS[m] in body]:
        # Offline, so it cannot ask the Hub as `upload` and `open` do (`linked_closed_problems`).
        print(f"It links {', '.join(linked)}: make the Space public, or update it, only once each is public (RELEASING.md, launch order).")
    print("Organisation settings, entered by hand at launch (RELEASING.md, launch order):")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    return []


# ---- command line --------------------------------------------------------------------------


def main(argv: list[str] | None = None, api: HfApi | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage", help="stage one model's repository folder")
    staging.add_argument("--model", choices=MODELS, required=True)
    staging.add_argument("--release", type=Path, help="B1: the release directory from `bulwark release`")
    staging.add_argument("--onnx", type=Path, action="append", default=[], help="E1: a graph from `bulwark export` (16 frames, and 1 frame); its <graph>.bench.json comes along")
    staging.add_argument("--eval", type=Path, help="`bulwark evaluate` output of these weights; only report.html and metrics.json are staged")
    staging.add_argument("--teacher-eval", type=Path, help="E1: `bulwark evaluate` output of the B1 checkpoint E1 was distilled from")
    staging.add_argument("--provenance", type=Path, help="provenance.json of the released checkpoint")
    staging.add_argument("--out", type=Path, help=f"folder to write (default {DEFAULT_STAGE}/<model>); must be new or empty")
    staging.add_argument("--public", action="store_true", help="refuse anything a public repository may not take")
    staging.add_argument("--validate", action="store_true", help=f"also check the front matter with {VALIDATE_URL} (sends it; creates nothing)")
    uploading = commands.add_parser("upload", help="upload a stage to its model's repository (created private if missing)")
    uploading.add_argument("--model", choices=MODELS, required=True)
    uploading.add_argument("--stage", type=Path)
    tagging = uploading.add_mutually_exclusive_group()
    tagging.add_argument("--tag", default=MODEL_RELEASE, help=f"tag of a cleared upload (default {MODEL_RELEASE})")
    tagging.add_argument("--no-tag", dest="tag", action="store_const", const=None, help="do not tag (a later card fix; a public repository refuses new weights without a tag)")
    uploading.add_argument("--allow-weight-deletion", action="store_true", help="let an upload to the public repository remove weight files")
    uploading.add_argument("--public", action="store_true", help="allow an upload to a repository that is already public (a cleared stage only)")
    opening = commands.add_parser("open", help="make the models' repositories public, gated with automatic approval, all or none")
    opening.add_argument("--model", choices=MODELS, action="append", required=True, help="a model to open; repeat it to open several in one run")
    opening.add_argument("--stage", type=Path, action="append", help=f"the stage of each --model, in the same order (default {DEFAULT_STAGE}/<model>)")
    opening.add_argument("--yes", action="store_true", help="do not ask for the repository ids")
    grouping = commands.add_parser("collection", help="create or update the collection of the models (private)")
    grouping.add_argument("--public", action="store_true", help="also add the at-launch items and make it public (launch)")
    carding = commands.add_parser("org-card", help="write the organisation card to a local folder")
    carding.add_argument("--out", type=Path, default=DEFAULT_ORG_CARD, help=f"folder to write (default {DEFAULT_ORG_CARD}); must be new or empty")
    carding.add_argument("--validate", action="store_true", help=f"also check its metadata with {VALIDATE_URL}")
    args = parser.parse_args(argv)
    if args.command == "open":
        if len(set(args.model)) != len(args.model):
            parser.error("open: name each --model once")
        if args.stage is not None and len(args.stage) != len(args.model):
            parser.error("open: give one --stage per --model, in the same order, or none")

    if args.command == "stage":
        out = args.out or DEFAULT_STAGE / args.model
        problems = stage(
            args.model, out, release=args.release, graphs=args.onnx, evaluation=args.eval,
            teacher_evaluation=args.teacher_eval, provenance=args.provenance, public=args.public, validate=args.validate,
        )  # fmt: skip
    elif args.command == "org-card":
        problems = org_card(args.out, validate=args.validate)
    else:
        problems = _hub_command(args, api)
    if problems:
        print("Refused:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    return 0


def _hub_command(args: argparse.Namespace, api: HfApi | None) -> list[str]:
    from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError, OfflineModeIsEnabled

    try:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(endpoint=ENDPOINT)
        commands: dict[str, Callable[[HfApi], list[str]]] = {
            "upload": lambda hub: upload(
                args.stage or DEFAULT_STAGE / args.model, args.model, hub, tag=args.tag,
                allow_weight_deletion=args.allow_weight_deletion, public=args.public,
            ),  # fmt: skip
            "open": lambda hub: open_repos(
                dict(zip(args.model, args.stage or [DEFAULT_STAGE / m for m in args.model], strict=True)), hub, yes=args.yes
            ),
            "collection": lambda hub: collection(hub, public=args.public),
        }
        return commands[args.command](api)
    except OfflineModeIsEnabled as error:
        return [f"{error}", "HF_HUB_OFFLINE is set; unset it to reach the Hub."]
    except (HfHubHTTPError, LocalTokenNotFoundError) as error:
        return [f"{type(error).__name__}: {error}", LOGIN_HINT]


if __name__ == "__main__":
    sys.exit(main())
