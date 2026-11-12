"""scripts/purge_cloud_copies.sh against stub `az`, `azcopy` and `kaggle` commands."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash scripts")

# Blob storage as a JSON list of blob names in $AZ_STATE/blobs.json.
AZ_STUB = r'''#!/usr/bin/env python3
import fnmatch, json, os, sys
from pathlib import Path

state = Path(os.environ["AZ_STATE"])
args = sys.argv[1:]
with open(state / "calls.jsonl", "a") as log:
    log.write(json.dumps(args) + "\n")
blobs = json.loads((state / "blobs.json").read_text())


def opt(name):
    return args[args.index(name) + 1]


if args[:4] == ["storage", "account", "blob-service-properties", "show"]:
    print(os.environ.get("RETENTION", "False") + "\nFalse")
elif args[:3] == ["storage", "blob", "list"]:
    print("\n".join(name for name in blobs if name.startswith(opt("--prefix"))))
elif args[:3] == ["storage", "blob", "delete-batch"]:
    kept = [name for name in blobs if not fnmatch.fnmatch(name, opt("--pattern"))]
    (state / "blobs.json").write_text(json.dumps(kept))
elif args[:3] == ["storage", "blob", "delete"]:
    (state / "blobs.json").write_text(json.dumps([name for name in blobs if name != opt("-n")]))
elif args[:3] == ["storage", "container", "generate-sas"]:
    print("se=2026-10-30&sp=racwl&sig=SECRETSIG")
else:
    sys.exit(f"az stub: unhandled {args}")
'''
LOG_STUB = """#!/usr/bin/env bash
echo "$(basename "$0") $*" >> "$AZ_STATE/tools.log"
"""
# Logs each call and keeps a copy of every uploaded file under $AZ_STATE/blob/<blob path>.
# AZCOPY_FAIL makes every upload fail, as a run interrupted mid-upload would.
AZCOPY_STUB = r"""#!/usr/bin/env bash
echo "azcopy $*" >> "$AZ_STATE/tools.log"
if [ -n "${AZCOPY_FAIL:-}" ]; then exit 1; fi
if [ "$1" = copy ]; then
  dest=${3%%\?*}; dest=${dest#*.blob.core.windows.net/*/}
  mkdir -p "$AZ_STATE/blob/$(dirname "$dest")" && cp "$2" "$AZ_STATE/blob/$dest"
fi
"""
BLOBS = [
    "data/shards/dmd-00000.tar",
    "data/shards/dmd-r001-00000.tar",
    "data/shards/dmd-r002-00000.tar",  # no longer local: the purge left it empty, or a reshard repacked it
    "data/shards/mrl_eye-00000.tar",
    "data/features/frozen/features-dmd-00000.npz",
    "data/features/frozen/features-mrl_eye-00000.npz",
    "data/features/frozen/meta.json",
    "data/manifest.parquet",
    "data/labels/face.jsonl",
    "checkpoints/heads-v0/best.pt",
]


# No dataset is cleared for Azure Blob in the registry yet (DMD and MRL Eye are cleared for
# Kaggle only), so the fixture names both in ALLOW_UNCLEARED, which the script treats exactly
# as a registry clearance; a test that sets ALLOW_UNCLEARED itself replaces this list.
BLOB_CLEARED = "dmd mrl_eye"


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A repo root with a purged dmd, a removal list naming one dmd subject, and stubs."""
    root, state, stubs = tmp_path / "repo", tmp_path / "state", tmp_path / "bin"
    for path in (root / "scripts", state, stubs):
        path.mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "purge_cloud_copies.sh", root / "scripts")
    (root / "infra").symlink_to(REPO / "infra")
    shutil.copytree(REPO / "bulwark", root / "bulwark", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "bulwark" / "data" / "removed_subjects.yaml").write_text(
        'removed:\n  "dmd:s1": {date: 2026-10-01, source: "subject request"}\n'
    )
    data = root / "data"
    for name in ("dmd-00000.tar", "dmd-r001-00000.tar", "mrl_eye-00000.tar"):
        (data / "shards").mkdir(parents=True, exist_ok=True)
        (data / "shards" / name).write_bytes(name.encode())
    (data / "labels").mkdir()
    (data / "labels" / "face.jsonl").write_text("{}\n")
    table = pa.table({"clip_id": ["d/2", "m/1"], "subject_id": ["dmd:s2", "mrl_eye:s1"], "dataset": ["dmd", "mrl_eye"]})
    pq.write_table(table, data / "manifest.parquet")
    (state / "blobs.json").write_text(json.dumps(BLOBS))
    (stubs / "az").write_text(AZ_STUB)
    (stubs / "azcopy").write_text(AZCOPY_STUB)
    (stubs / "kaggle").write_text(LOG_STUB)
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "AZ_STATE": str(state),
        "PYTHON": sys.executable,
        "KAGGLE": str(stubs / "kaggle"),
        "BULWARK_STORAGE": "teststore",
        "BULWARK_RG": "rg",
        "BULWARK_ENV_LOCAL": str(tmp_path / "no-env.local.sh"),  # a developer's own settings stay out
        "ALLOW_UNCLEARED": BLOB_CLEARED,
    }
    return root, env


def purge(
    root: Path, env: dict[str, str], *flags: str, dataset: str = "dmd", **extra: str
) -> subprocess.CompletedProcess[str]:
    command = ["bash", str(root / "scripts" / "purge_cloud_copies.sh"), *flags, dataset]
    return subprocess.run(command, env={**env, **extra}, capture_output=True, text=True, timeout=120)


def blobs(env: dict[str, str]) -> list[str]:
    return json.loads((Path(env["AZ_STATE"]) / "blobs.json").read_text())


def tools(env: dict[str, str]) -> list[str]:
    path = Path(env["AZ_STATE"]) / "tools.log"
    return path.read_text().splitlines() if path.exists() else []


def uploaded(env: dict[str, str]) -> list[str]:
    """Blob paths that `azcopy copy` wrote to."""
    base = "https://teststore.blob.core.windows.net/bulwark/"
    return sorted(line.split()[3].removeprefix(base).split("?")[0] for line in tools(env) if line.startswith("azcopy copy"))


def uploaded_datasets(env: dict[str, str]) -> list[str]:
    """The dataset column of the manifest that went up to Blob."""
    return pq.read_table(Path(env["AZ_STATE"]) / "blob" / "data" / "manifest.parquet").column("dataset").to_pylist()


def test_purge_replaces_only_the_dataset_in_blob_and_logs_it(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    result = purge(root, env, KAGGLE_SLUGS="bulwark-dmd bulwark-meta", KAGGLE_USER="someone")
    assert result.returncode == 0, result.stderr
    # The local shards went up over their blobs; only the blob no local shard replaces is gone,
    # with the dataset's features and the label blobs, which no job reads.
    assert sorted(blobs(env)) == sorted(set(BLOBS) - {
        "data/shards/dmd-r002-00000.tar", "data/features/frozen/features-dmd-00000.npz", "data/labels/face.jsonl"
    })  # fmt: skip
    assert uploaded(env) == ["data/manifest.parquet", "data/shards/dmd-00000.tar", "data/shards/dmd-r001-00000.tar"]
    assert "kaggle datasets delete -y someone/bulwark-dmd" in tools(env)
    assert "kaggle datasets delete -y someone/bulwark-meta" in tools(env)

    log = (root / "docs" / "purge-log.md").read_text()
    row = log.splitlines()[-1]
    assert row.startswith("| ") and "| dmd | 1 |" in row
    assert ("shard blobs: 2 replaced by local shards, 1 without a local shard deleted; feature blobs: 1 deleted; "
            "manifest: replaced; label blobs: 1 deleted") in row  # fmt: skip
    assert "SHARD_GLOB must list every source the" in result.stdout
    assert "s1" not in log  # no subject ids
    # The log lives in docs/, whose files must hold no amounts, addresses or ticket numbers.
    assert not re.search(r"\$\s?\d|[\w.+-]+@[\w-]+\.[\w.-]+|(?<![\w.])\d{12,}(?![\w.])", log)

    purge(root, env, AZURE="0")
    assert len((root / "docs" / "purge-log.md").read_text().splitlines()) == len(log.splitlines()) + 1


def test_purge_refuses_until_the_local_purge_is_done(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    table = pa.table({"clip_id": ["d/1"], "subject_id": ["dmd:s1"]})
    pq.write_table(table, root / "data" / "manifest.parquet")
    result = purge(root, env)
    assert result.returncode != 0 and "purge locally first" in result.stderr
    assert blobs(env) == BLOBS and not tools(env)
    assert not (root / "docs" / "purge-log.md").exists()


def test_dry_run_changes_nothing(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    result = purge(root, env, DRY_RUN="1", RETENTION="True", KAGGLE_SLUGS="bulwark-dmd", KAGGLE_USER="someone")
    assert result.returncode == 0, result.stderr
    assert "soft delete or versioning" in result.stderr and "would run: az storage blob delete-batch" in result.stderr
    assert blobs(env) == BLOBS and not tools(env)
    assert "SECRETSIG" not in result.stderr + result.stdout
    assert not (root / "docs" / "purge-log.md").exists()


def test_an_interrupted_upload_deletes_nothing(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    result = purge(root, env, AZCOPY_FAIL="1")
    assert result.returncode != 0
    assert blobs(env) == BLOBS  # uploads come first, so Blob still holds the whole dataset
    assert not (root / "docs" / "purge-log.md").exists()


@pytest.mark.parametrize(
    ("missing", "message"),
    [("data/manifest.parquet", "data/manifest.parquet is missing"), ("data/shards", "none to replace them")],
)
def test_purge_refuses_without_the_local_data_to_replace_blob(checkout, missing: str, message: str) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    path = root / missing
    shutil.rmtree(path) if path.is_dir() else path.unlink()
    result = purge(root, env)
    assert result.returncode != 0 and message in result.stderr
    assert blobs(env) == BLOBS and not tools(env)
    assert not (root / "docs" / "purge-log.md").exists()


def test_purge_needs_a_removed_subject_unless_resharding(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    (root / "bulwark" / "data" / "removed_subjects.yaml").write_text("removed: {}\n")
    result = purge(root, env)
    assert result.returncode != 0 and "lists no subject of dmd" in result.stderr
    assert blobs(env) == BLOBS and not tools(env)

    assert purge(root, env, "--reshard", KAGGLE_SLUGS="bulwark-dmd", KAGGLE_USER="someone").returncode == 2
    assert blobs(env) == BLOBS and not tools(env)

    result = purge(root, env, "--reshard")
    assert result.returncode == 0, result.stderr
    assert sorted(blobs(env)) == sorted(set(BLOBS) - {
        "data/shards/dmd-r002-00000.tar", "data/features/frozen/features-dmd-00000.npz", "data/labels/face.jsonl"
    })  # fmt: skip
    assert "new version" in result.stdout and not any("kaggle" in line for line in tools(env))
    assert not (root / "docs" / "purge-log.md").exists()  # a reshard is not a purge


def test_purge_checks_the_manifest_as_upload_does(checkout) -> None:  # type: ignore[no-untyped-def]
    # zenodo_driver is cleared for neither Kaggle nor Azure Blob in the registry.
    root, env = checkout
    table = pa.table({"clip_id": ["d/2", "z/1"], "subject_id": ["dmd:s2", "zenodo_driver:z1"],
                      "dataset": ["dmd", "zenodo_driver"]})  # fmt: skip
    pq.write_table(table, root / "data" / "manifest.parquet")
    result = purge(root, env)
    assert result.returncode != 0 and "zenodo_driver (1 rows)" in result.stderr
    assert blobs(env) == BLOBS and not tools(env)  # refused before anything changed
    assert not (root / "docs" / "purge-log.md").exists()

    result = purge(root, env, DROP_UNCLEARED="1")
    assert result.returncode == 0, result.stderr
    assert "data/manifest.parquet" in uploaded(env)
    assert uploaded_datasets(env) == ["dmd"]  # the staged manifest, not the local one


ZENODO = ["data/shards/zenodo_driver-00000.tar", "data/features/frozen/features-zenodo_driver-00000.npz"]


def with_zenodo(root: Path, env: dict[str, str]) -> None:
    """Blob holding shards and features of zenodo_driver (not cleared for any third-party
    storage), uploaded before the clearance check, and a manifest with its rows."""
    (Path(env["AZ_STATE"]) / "blobs.json").write_text(json.dumps(BLOBS + ZENODO))
    table = pa.table({"clip_id": ["d/2", "z/1"], "subject_id": ["dmd:s2", "zenodo_driver:z1"],
                      "dataset": ["dmd", "zenodo_driver"]})  # fmt: skip
    pq.write_table(table, root / "data" / "manifest.parquet")


def test_shards_of_an_uncleared_dataset_are_deleted_not_uploaded(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    with_zenodo(root, env)
    (root / "data" / "shards" / "zenodo_driver-00000.tar").write_bytes(b"zenodo")
    result = purge(root, env, "--reshard", dataset="zenodo_driver", DROP_UNCLEARED="1")
    assert result.returncode == 0, result.stderr
    assert uploaded(env) == ["data/manifest.parquet"] and uploaded_datasets(env) == ["dmd"]
    assert sorted(blobs(env)) == sorted(set(BLOBS) - {"data/labels/face.jsonl"})
    assert "shard blobs: 1 deleted, none uploaded (not cleared for storage on Azure Blob)" in result.stdout
    assert "stays off Kaggle" in result.stdout and "except zenodo_driver-*.tar" in result.stdout

    # Named in ALLOW_UNCLEARED, it is handled as a cleared dataset.
    with_zenodo(root, env)
    (Path(env["AZ_STATE"]) / "tools.log").unlink()
    result = purge(root, env, "--reshard", dataset="zenodo_driver", ALLOW_UNCLEARED=f"{BLOB_CLEARED} zenodo_driver")
    assert result.returncode == 0, result.stderr
    assert uploaded(env) == ["data/manifest.parquet", "data/shards/zenodo_driver-00000.tar"]
    assert uploaded_datasets(env) == ["dmd", "zenodo_driver"]
    assert "stays off Kaggle" not in result.stdout


def test_uncleared_shard_blobs_go_without_local_shards(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    with_zenodo(root, env)
    (root / "bulwark" / "data" / "removed_subjects.yaml").write_text(
        'removed:\n  "zenodo_driver:z9": {date: 2026-10-01, source: "subject request"}\n'
    )
    result = purge(root, env, dataset="zenodo_driver", DROP_UNCLEARED="1")
    assert result.returncode == 0, result.stderr  # no local shard needed to replace them
    assert not set(ZENODO) & set(blobs(env))
    assert uploaded(env) == ["data/manifest.parquet"]
    row = (root / "docs" / "purge-log.md").read_text().splitlines()[-1]
    assert "| zenodo_driver | 1 |" in row and "shard blobs: 1 deleted, none uploaded" in row

    result = purge(root, env, "--reshard", dataset="zenodo")
    assert result.returncode != 0 and "not a dataset in bulwark/data/registry.py" in result.stderr


def test_a_manifest_blob_is_deleted_when_no_dataset_is_cleared_for_blob(checkout) -> None:  # type: ignore[no-untyped-def]
    # With the registry as it is, no dataset is cleared for Azure Blob, so a manifest that
    # Blob still holds cannot be replaced by a staged copy (it would have no rows).
    root, env = checkout
    result = purge(root, env, ALLOW_UNCLEARED="")
    assert result.returncode == 1 and "Set DROP_UNCLEARED=1 to delete the manifest blob" in result.stderr
    assert blobs(env) == BLOBS and not tools(env)  # refused before anything changed
    assert not (root / "docs" / "purge-log.md").exists()

    result = purge(root, env, ALLOW_UNCLEARED="", DROP_UNCLEARED="1")
    assert result.returncode == 0, result.stderr
    assert uploaded(env) == []
    gone = {name for name in BLOBS if name.startswith("data/shards/dmd-")} | {
        "data/manifest.parquet", "data/features/frozen/features-dmd-00000.npz", "data/labels/face.jsonl"
    }  # fmt: skip
    assert sorted(blobs(env)) == sorted(set(BLOBS) - gone)
    row = (root / "docs" / "purge-log.md").read_text().splitlines()[-1]
    assert "| dmd | 1 |" in row and "shard blobs: 3 deleted, none uploaded" in row
    assert "manifest: deleted (no dataset in it is cleared for storage on Azure Blob)" in row


def test_blob_and_kaggle_are_cleared_separately(checkout) -> None:  # type: ignore[no-untyped-def]
    # With the registry as it is, DMD may be kept on Kaggle but not yet on Azure Blob: its
    # shard blobs are deleted rather than replaced, while the Kaggle steps keep its shards.
    root, env = checkout
    result = purge(root, env, "--reshard", ALLOW_UNCLEARED="mrl_eye", DROP_UNCLEARED="1")
    assert result.returncode == 0, result.stderr
    assert uploaded(env) == ["data/manifest.parquet"] and uploaded_datasets(env) == ["mrl_eye"]
    assert not [name for name in blobs(env) if name.startswith("data/shards/dmd-")]
    assert "shard blobs: 3 deleted, none uploaded (not cleared for storage on Azure Blob)" in result.stdout
    assert "stays off Kaggle" not in result.stdout and "including dmd-*.tar" in result.stdout
