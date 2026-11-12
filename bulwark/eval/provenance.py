"""What a result was computed from: code version, file hashes, data and training scope.

Paths are recorded relative to the repository root, or as a bare file name outside it, so
published results (metrics.json, report.html, ONNX metadata) never carry local usernames
or directory layouts; the sha256 next to each path identifies the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from bulwark.data.manifest import read_manifest_metadata
from bulwark.data.splits import SPLIT_POLICY_KEY, SplitPolicy, policy_record
from bulwark.data.taxonomy import Taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]
# Written by `bulwark train cache` next to cached features (see bulwark/train/cache_features.py).
FEATURES_META = "meta.json"
# Checkpoint entries that are weights or optimiser state, not provenance.
_STATE_KEYS = {"model", "optimizer", "loss", "scaler"}


def git_state(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit, dirty flag and changed files (at most 20) of the code that produced a result.

    Remote jobs run from an archive without .git; `scripts/pack_code.sh` records the commit
    in PROVENANCE.json at the archive root, which wins when present. The fields are those
    the training jobs record (`bulwark.train.provenance.code_version`), except that `dirty`
    here covers every tracked file but ignores untracked ones, while `code_version` covers
    tracked and untracked files under bulwark/, configs/, scripts/ and pyproject.toml.
    """
    packed = root / "PROVENANCE.json"
    if packed.exists():
        try:
            data = json.loads(packed.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        commit = data.get("commit", data.get("git_commit"))
        dirty = data.get("dirty", data.get("git_dirty"))
        changed = list(data.get("dirty_files", data.get("changed")) or [])[:20]
        if commit:
            return {"commit": commit, "dirty": dirty, "changed": changed, "source": "PROVENANCE.json"}
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "source": "unknown"}
    changed = [line[3:] for line in status.splitlines() if line.strip()][:20]
    return {"commit": commit, "dirty": bool(changed), "changed": changed, "source": "git"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def public_path(path: str | Path | None) -> str | None:
    """A path safe to publish: relative to the repository root, else only its last component."""
    if path is None or str(path) == "":
        return None
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    for candidate in (absolute, absolute.resolve()):  # the checkout itself may sit behind a symlink
        if candidate.is_relative_to(REPO_ROOT):
            return candidate.relative_to(REPO_ROOT).as_posix()
    return absolute.name


def scrub_paths(value: Any) -> Any:
    """`public_path` applied to every absolute path string in a JSON-like value."""
    if isinstance(value, dict):
        return {k: scrub_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_paths(v) for v in value]
    if isinstance(value, str) and (value.startswith(("/", "~/")) or (len(value) > 2 and value[1:3] == ":\\")):
        return public_path(value)
    return value


def taxonomy_digest(taxonomy: Taxonomy) -> str:
    spec = [(s.name, s.head, s.kind, list(s.classes), list(s.value_range)) for s in taxonomy.labels.values()]
    return hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:16]


def jsonable(value: Any) -> Any:
    """Plain JSON types only; tensors and other objects become short descriptions."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else f"tensor{tuple(value.shape)}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def checkpoint_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Everything a checkpoint records besides its weights (step, config, provenance...)."""
    return scrub_paths({k: jsonable(v) for k, v in payload.items() if k not in _STATE_KEYS})


def training_scope(meta: dict[str, Any]) -> dict[str, Any]:
    """`commercial_only` and `datasets` the model was trained with, when the checkpoint says.

    Looks at the top level, then under `provenance` and `config` (and `provenance.config`).
    `datasets` may be a list of names or a per-dataset summary; it is returned as names.
    """
    sources = [meta, meta.get("provenance") or {}, meta.get("config") or {}]
    sources.append((meta.get("provenance") or {}).get("config") or {})
    scope: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("commercial_only", "datasets"):
            if source.get(key) not in (None, [], {}) and key not in scope:
                scope[key] = source[key]
    if isinstance(scope.get("datasets"), dict):
        scope["datasets"] = sorted(scope["datasets"])
    return scope


def features_meta(features: Path) -> dict[str, Any] | None:
    """The provenance `bulwark train cache` writes next to cached features, if any."""
    path = Path(features) / FEATURES_META
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("run"), dict) else None


def features_init(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """The checkpoint a features dir was cached with (`init`), or None for a frozen encoder."""
    if not meta:
        return None
    return (meta.get("run") or {}).get("init") or None


def features_summary(features: Path) -> dict[str, Any]:
    """What a features dir was cached from, without local paths."""
    meta = features_meta(features)
    out: dict[str, Any] = {"path": public_path(features), "has_provenance": meta is not None}
    if meta is None:
        return out
    run = meta.get("run") or {}
    init = features_init(meta)
    out.update(
        commercial_only=meta.get("commercial_only"),
        with_logits=run.get("with_logits"),
        precision=run.get("precision"),
        encoder=scrub_paths(run.get("encoder")),
        init=None if init is None else {"path": public_path(meta.get("init_path")), **scrub_paths(init)},
        code=(meta.get("code") or {}).get("commit"),
        shards=len(meta.get("shards") or {}),
    )
    return out


def split_policy(manifest: Path) -> dict[str, Any]:
    """The split policy the manifest records (`bulwark split` writes it), else the code
    default, marked as such. Removed subject ids are never published, only their count."""
    try:
        recorded = read_manifest_metadata(manifest).get(SPLIT_POLICY_KEY)
    except (OSError, ValueError):
        recorded = None
    if recorded:
        return {"source": "manifest", **json.loads(recorded)}
    default = jsonable(policy_record(SplitPolicy()))
    return {"source": "code default (the manifest does not record its split policy)", **default}


def run_meta() -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
