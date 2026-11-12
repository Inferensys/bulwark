"""Where a training artifact came from, and the checks that keep runs from mixing.

Every checkpoint (`last.pt`, `best.pt`, `final.pt`) records its provenance: the resolved
config, the data selection (commercial_only, datasets and their licences, manifest hash,
split salt), its inputs (shards, features, init checkpoint, teacher), the labels it could
learn, the taxonomy and package versions, the code commit and library versions. Each
model input (init, features, teacher) names the datasets of the model behind it
(`datasets`) and of every model that one was trained from in turn (`upstream_datasets`),
and the inputs up that chain whose training data is not recorded (`unrecorded_inputs`).
Every features directory records the encoder, init and precision behind it in its
meta.json (see `bulwark.train.cache_features`). Uses:

- resume: `last.pt` also stores a run fingerprint, the part of the provenance that must
  match for training to continue (see `fingerprint`);
- the clean chain: with `commercial_only`, every upstream artifact must itself record
  `commercial_only: true` (see `require_clean`). This check has no override;
- releasable weights: with `release_only`, every upstream artifact must itself record
  `release_only: true` and have been trained only on datasets cleared for publishing
  trained weights (registry `weights_release_ok`; see `require_releasable`). This check
  has no override either.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bulwark.data.manifest import ClipRecord
    from bulwark.train.common import TrainConfig

SCHEMA = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_META = "meta.json"

# Config keys that may change between sessions of one run: where it runs and how often it
# saves, not what it trains. Input paths are exempt too: `fingerprint` compares inputs by
# content hash, not by path.
RESUME_EXEMPT = frozenset({
    "out", "device", "num_workers", "multi_gpu", "time_limit_minutes", "checkpoint_minutes",
    "log_every", "eval_every", "fresh", "allow_mismatch", "overwrite",
    "manifest", "shards", "features", "init", "teacher",
})


# Inputs that are models trained on Bulwark data (a checkpoint, or a features dir cached with
# one), whose datasets a run inherits.
UPSTREAM_INPUTS = ("init", "features", "teacher")
# What an input's identity records about the models behind it. It is read from the input
# file, whose hash is compared already, and older runs lack it, so comparisons of two
# identities (resume, a features dir's run) leave it out.
CHAIN_KEYS = ("upstream_datasets", "unrecorded_inputs")


class ProvenanceError(RuntimeError):
    """An artifact cannot be used here: wrong run, missing provenance, or not clean."""


# ---- hashing ----------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """sha256 of a file, cached per (path, size, mtime) within the process."""
    stat = Path(path).stat()
    return _sha256_cached(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def jsonable(value: Any) -> Any:
    """Plain JSON types only, so payloads load with `torch.load(weights_only=True)`."""
    return json.loads(json.dumps(value, default=_plain))


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):  # numpy and torch scalars
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return sorted(value) if isinstance(value, (set, frozenset)) else list(value)
    return str(value)


# ---- code and environment ---------------------------------------------------------------


@lru_cache(maxsize=1)
def code_version() -> dict[str, Any]:
    """The commit that produced an artifact and whether code paths had local changes.

    Remote jobs run from a packed archive without .git; `scripts/pack_code.sh` writes
    PROVENANCE.json at the archive root (`commit`, `dirty`, `changed`, `packed_at`), which
    wins when present. Archives packed before that use `git_commit`, `git_dirty` and
    `dirty_files`, which are still read.
    """
    packed = REPO_ROOT / "PROVENANCE.json"
    if packed.exists():
        data = json.loads(packed.read_text(encoding="utf-8"))
        return {
            "commit": data.get("commit", data.get("git_commit")),
            "dirty": data.get("dirty", data.get("git_dirty")),
            "changed": list(data.get("changed", data.get("dirty_files")) or [])[:20],
            "packed_at": data.get("packed_at"),
            "source": "PROVENANCE.json",
        }
    try:
        commit = _git("rev-parse", "HEAD").strip()
        changes = _git("status", "--porcelain", "--", "bulwark", "configs", "scripts", "pyproject.toml").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "source": "unknown"}
    return {"commit": commit, "dirty": bool(changes), "changed": [line[3:] for line in changes[:20]], "source": "git"}


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30)
    return result.stdout  # unstripped: porcelain lines start with a status column that may be a space


def bulwark_version() -> str | None:
    """The installed distribution's version (None when the code runs without being installed)."""
    try:
        return importlib.metadata.version("inferensys-bulwark")
    except importlib.metadata.PackageNotFoundError:
        return None


def library_versions() -> dict[str, str | None]:
    import numpy
    import torch

    versions: dict[str, str | None] = {
        "bulwark": bulwark_version(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": numpy.__version__,
    }
    if torch.cuda.is_available():
        versions["gpu"] = torch.cuda.get_device_name(0)
    for name in ("timm", "onnx"):
        with contextlib.suppress(ImportError):
            versions[name] = importlib.import_module(name).__version__
    return versions


# ---- identities of inputs ---------------------------------------------------------------


def encoder_identity(encoder: Any) -> dict[str, Any]:
    """The V-JEPA code commit and weights a B1 encoder (`VJepaEncoder`) was built from.

    `weights` is None for a randomly initialised encoder. The weights' sha256 is the one
    `load_pretrained` checked, so the 1.7 GB file is not hashed twice.
    """
    from bulwark.model.backbone import VJEPA_COMMIT

    weights = getattr(encoder, "weights_path", None)
    if weights is None:
        return {"vjepa_commit": VJEPA_COMMIT, "weights": None, "weights_sha256": None}
    sha256 = getattr(encoder, "weights_sha256", None) or file_sha256(weights)
    return {"vjepa_commit": VJEPA_COMMIT, "weights": Path(weights).name, "weights_sha256": sha256}


def checkpoint_identity(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """How a downstream run refers to a checkpoint it starts from.

    `datasets` are the checkpoint's own training datasets; `upstream_datasets` those of
    the models it was trained from (its init, features or teacher), all the way up the
    chain (see `upstream_datasets`); `unrecorded_inputs` the inputs up that chain whose
    training data is not recorded (see `unrecorded_inputs`). Both are None when the
    checkpoint records no provenance.
    """
    recorded = payload.get("provenance") or {}
    inputs = recorded.get("inputs") or {}
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "has_provenance": bool(recorded),
        "commercial_only": recorded.get("commercial_only"),
        "release_only": recorded.get("release_only"),
        "datasets": recorded.get("datasets"),
        "upstream_datasets": upstream_datasets(inputs) if recorded else None,
        "unrecorded_inputs": unrecorded_inputs(inputs) if recorded else None,
        "manifest_sha256": recorded.get("manifest_sha256"),
        "commit": (recorded.get("code") or {}).get("commit"),
    }


def upstream_datasets(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The datasets behind a run's model inputs (UPSTREAM_INPUTS): each input's own
    `datasets` and its `upstream_datasets`, so the list covers the whole chain (an E1's
    teacher -> that B1 -> the heads it started from).

    {dataset: {"licence", "commercial_ok"}}; commercial_ok is False when any input records
    False. Inputs written before `upstream_datasets` was recorded contribute their own
    datasets only.
    """
    found: dict[str, dict[str, Any]] = {}
    for key in UPSTREAM_INPUTS:
        entry = inputs.get(key)
        if not isinstance(entry, dict):
            continue
        for value in (entry.get("datasets"), entry.get("upstream_datasets")):
            items = value.items() if isinstance(value, dict) else ((name, {}) for name in value or ())
            for name, info in items:
                info = info if isinstance(info, dict) else {}
                item = found.setdefault(name, {"licence": info.get("licence"), "commercial_ok": info.get("commercial_ok")})
                if info.get("commercial_ok") is False:
                    item["commercial_ok"] = False
    return dict(sorted(found.items()))


def unrecorded_reason(entry: dict[str, Any]) -> str | None:
    """Why the training data behind one model input (a `checkpoint_identity` or
    `features_identity`) is not recorded, or None when it is recorded or no Bulwark
    training is behind it (tokens of the frozen V-JEPA encoder, cached without an init).

    A features dir stands for the checkpoint it was cached with: its `datasets` are that
    checkpoint's.
    """
    if entry.get("has_provenance") is False:
        return "records no provenance"
    if "with_logits" in entry:  # a features dir (features_identity)
        init = entry.get("init")
        if not init and not entry.get("init_sha256") and not entry.get("with_logits"):
            return None  # frozen-encoder tokens
        if isinstance(init, dict) and init.get("has_provenance") is False:
            return "cached with a checkpoint that records no provenance"
    if not entry.get("datasets"):
        return "records no training datasets"
    return None


def unrecorded_inputs(inputs: dict[str, Any], keys: tuple[str, ...] = UPSTREAM_INPUTS) -> list[str]:
    """The model inputs of a run whose training data is not recorded, all the way up the chain.

    A direct input is listed as "<key> (<reason>)" (see `unrecorded_reason`). The items an
    input itself records under `unrecorded_inputs` follow as "<key> -> <item>", so an E1
    distilled from a B1 whose init records no provenance lists
    "teacher -> init (records no provenance)". Inputs recorded before `unrecorded_inputs`
    existed contribute their own status only.
    """
    found: list[str] = []
    for key in keys:
        entry = inputs.get(key)
        if not isinstance(entry, dict):
            continue
        reason = unrecorded_reason(entry)
        if reason is not None:
            found.append(f"{key} ({reason})")
        found += [f"{key} -> {item}" for item in entry.get("unrecorded_inputs") or ()]
    return found


def read_features_meta(path: Path) -> dict[str, Any] | None:
    meta = Path(path) / FEATURES_META
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None


def features_identity(path: Path) -> dict[str, Any]:
    """How a run refers to a features dir (cached tokens or teacher logits) it reads.

    It carries the encoder and init identities from meta.json, so a checkpoint trained on
    the dir still names the V-JEPA weights and the B1 behind it after the dir itself is
    gone, with that B1's datasets (`datasets`), those of the models it was trained from
    (`upstream_datasets`) and the inputs up that chain whose training data is not recorded
    (`unrecorded_inputs`). The last two are None for a dir cached before they were
    recorded.
    """
    meta = read_features_meta(path)
    if meta is None:  # written before features provenance existed
        files = sorted((f.name, f.stat().st_size) for f in Path(path).glob("features-*.npz"))
        return {"path": str(path), "has_provenance": False, "commercial_only": None, "files_sha256": json_sha256(files)}
    run = meta.get("run") or {}
    init = run.get("init") or {}
    return {
        "path": str(path),
        "has_provenance": True,
        "commercial_only": meta.get("commercial_only"),
        "release_only": meta.get("release_only"),
        "datasets": init.get("datasets"),
        "upstream_datasets": init.get("upstream_datasets"),
        "unrecorded_inputs": init.get("unrecorded_inputs"),
        "run_sha256": json_sha256(run),
        "shards_sha256": json_sha256(meta.get("shards")),
        "with_logits": run.get("with_logits"),
        "init_sha256": init.get("sha256"),
        "encoder": run.get("encoder"),
        "init": run.get("init"),
    }


# ---- provenance and fingerprints --------------------------------------------------------


def weights_release_ok(dataset: str) -> bool | None:
    """The registry's `weights_release_ok` for a dataset; None when undetermined or unknown."""
    from bulwark.data.registry import BY_NAME

    spec = BY_NAME.get(dataset)
    return spec.weights_release_ok if spec is not None else None


def dataset_summary(records: dict[str, ClipRecord]) -> dict[str, dict[str, Any]]:
    """Per dataset: clips per split, licence, commercial_ok and weights_release_ok, for the
    records a run used."""
    summary: dict[str, dict[str, Any]] = {}
    for record in records.values():
        entry = summary.setdefault(record.dataset, {
            "licence": record.licence, "commercial_ok": record.commercial_ok,
            "weights_release_ok": weights_release_ok(record.dataset), "splits": {},
        })
        entry["commercial_ok"] = entry["commercial_ok"] and record.commercial_ok
        entry["splits"][record.split] = entry["splits"].get(record.split, 0) + 1
    return dict(sorted(summary.items()))


def build_provenance(
    job: str,
    config: TrainConfig,
    records: dict[str, ClipRecord] | None,
    inputs: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to tell where an artifact came from. JSON-safe."""
    from bulwark.data.splits import SplitPolicy
    from bulwark.data.taxonomy import default_taxonomy

    manifest = Path(config.manifest)
    datasets = dataset_summary(records) if records is not None else None
    return jsonable({
        "schema": SCHEMA,
        "job": job,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "commercial_only": bool(config.commercial_only),
        "release_only": bool(config.release_only),
        "datasets": datasets,
        "manifest_sha256": file_sha256(manifest) if records is not None and manifest.exists() else None,
        "split_salt": SplitPolicy().salt,
        "taxonomy_version": default_taxonomy().version,
        "bulwark_version": bulwark_version(),
        "inputs": inputs,
        "code": code_version(),
        "versions": library_versions(),
        **(extra or {}),
    })


def fingerprint(provenance: dict[str, Any]) -> dict[str, Any]:
    """The part of a run's provenance that must match for `last.pt` to be resumed.

    Inputs are compared by content (sha256 and run hashes), not by path, so the same init
    or teacher under another mount (/kaggle/input vs checkpoints/) still resumes.
    `upstream_datasets` and `unrecorded_inputs` are read from the input itself, whose hash
    is compared already, so they are left out: a last.pt written before they were
    recorded still resumes.
    """
    config = {k: v for k, v in provenance["config"].items() if k not in RESUME_EXEMPT}
    inputs = {
        name: {k: v for k, v in value.items() if k != "path" and k not in CHAIN_KEYS} if isinstance(value, dict) else value
        for name, value in (provenance.get("inputs") or {}).items()
    }
    return jsonable({
        "job": provenance["job"],
        "config": config,
        "manifest_sha256": provenance.get("manifest_sha256"),
        "inputs": inputs,
    })


def differences(expected: Any, found: Any, prefix: str = "") -> list[str]:
    """Dotted paths where two fingerprints differ."""
    if isinstance(expected, dict) and isinstance(found, dict):
        keys = sorted(set(expected) | set(found))
        return [d for key in keys for d in differences(expected.get(key), found.get(key), f"{prefix}{key}.")]
    return [] if expected == found else [prefix.rstrip(".") or "<root>"]


def require_clean(source: dict[str, Any] | None, what: str) -> None:
    """With commercial_only, refuse an upstream artifact that does not record clean data only.

    `source` is a `checkpoint_identity` or `features_identity`; None means no upstream.
    `what` is "init", "teacher", "features" or "last.pt", and picks the hint.
    """
    if source is None:
        return
    path = source.get("path")
    if not source.get("has_provenance"):
        raise ProvenanceError(
            f"commercial_only=true, but {what} {path} records no provenance, so it cannot be shown to "
            f"come from commercially usable data. {_hint(what, missing=True)}"
        )
    if source.get("commercial_only") is not True:
        raise ProvenanceError(
            f"commercial_only=true, but {what} {path} was produced with commercial_only=false "
            f"(it saw research-only data). {_hint(what, missing=False)}"
        )
    unclean = sorted(name for name, info in _chain_datasets(source).items() if not info.get("commercial_ok"))
    if unclean:
        raise ProvenanceError(f"{what} {path} was trained on datasets that are not commercially usable: {unclean}")


def _chain_datasets(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """An upstream artifact's own datasets and those of the models it was trained from."""
    found: dict[str, dict[str, Any]] = {}
    for value in (source.get("upstream_datasets"), source.get("datasets")):
        if isinstance(value, dict):
            found.update({name: info if isinstance(info, dict) else {} for name, info in value.items()})
    return found


def require_releasable(source: dict[str, Any] | None, what: str) -> None:
    """With release_only, refuse an upstream artifact that does not record `release_only:
    true`, or that was trained on a dataset whose terms do not (yet) allow publishing
    trained weights.

    `source` and `what` are as in `require_clean`. Both the artifact's own `datasets` and
    its `upstream_datasets` are checked. The recorded flag carries the check through the
    whole chain (heads -> B1 -> teacher features -> E1) for artifacts written before
    `upstream_datasets` was recorded. Frozen features (cached without an init, so no
    Bulwark training behind them) pass, including dirs cached before release_only was
    recorded.
    """
    if source is None:
        return
    path = source.get("path")
    if not source.get("has_provenance"):
        raise ProvenanceError(
            f"release_only=true, but {what} {path} records no provenance, so the datasets behind it are unknown. "
            f"{_release_hint(what)}"
        )
    frozen = source.get("with_logits") is False and not source.get("init")
    if source.get("release_only") is not True and not frozen:
        raise ProvenanceError(
            f"release_only=true, but {what} {path} was not produced with release_only=true, so the datasets "
            f"behind its own init or teacher are not known to allow published weights. {_release_hint(what)}"
        )
    blocked = sorted(name for name in _chain_datasets(source) if weights_release_ok(name) is not True)
    if blocked:
        raise ProvenanceError(
            f"release_only=true, but {what} {path} was trained on datasets not cleared for published weights "
            f"(registry weights_release_ok): {blocked}"
        )


def _release_hint(what: str) -> str:
    """What to do about an upstream artifact `require_releasable` refused."""
    if what == "last.pt":
        return "Use a new `out`, or --set fresh=true to move the old run aside."
    if what == "features":
        return "Use frozen features (configs/train/cache.yaml) or a dir cached from a release_only init."
    return "Rebuild it, and the artifacts it was trained from, with release_only=true."


def _hint(what: str, *, missing: bool) -> str:
    """What to do about an upstream artifact `require_clean` refused."""
    if what == "features":
        # Frozen features are clean by construction (no Bulwark training behind the
        # encoder); a dir without meta.json predates features provenance and only needs
        # re-caching.
        if missing:
            return "Re-cache it with configs/train/cache.yaml and --set overwrite=true."
        return "Use frozen features (configs/train/cache.yaml) or a dir cached from a clean init."
    if what == "teacher":
        return "Use a dir written by configs/train/cache-b1-clean.yaml."
    if what == "last.pt":
        return "Use a new `out`, or --set fresh=true to move the old run aside."
    return "Rebuild it with the *-clean configs." if missing else "Point it at an output of the *-clean configs."


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
