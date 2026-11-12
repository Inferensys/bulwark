"""infra/azure/upload.sh against stub `az` and `azcopy`: what goes up to Blob, and what stays local."""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_infra_azure import AZURE, azure  # noqa: F401  (the fixture: stub az and azcopy)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash scripts")

# Logs each call and keeps a copy of every file `azcopy copy` uploads, under its Blob path.
AZCOPY_STUB = r"""#!/usr/bin/env bash
printf '%s\n' "$@" -- >> "$AZ_STATE/azcopy.log"
if [ "$1" = copy ]; then
  dest=${3%%\?*}; dest=${dest#*.blob.core.windows.net/*/}
  mkdir -p "$AZ_STATE/blob/$(dirname "$dest")" && cp "$2" "$AZ_STATE/blob/$dest"
fi
"""


# No dataset is cleared for Azure Blob in the registry yet (DMD is cleared for Kaggle only),
# so tests that need one name DMD in ALLOW_UNCLEARED: upload.sh treats a dataset named
# there exactly as one the registry clears.
BLOB_CLEARED = {"ALLOW_UNCLEARED": "dmd"}


@pytest.fixture
def checkout(tmp_path: Path, azure: dict[str, str]) -> tuple[Path, dict[str, str]]:  # noqa: F811
    """A repo root holding shards and a manifest of DMD (cleared for Kaggle, not for Blob)
    and the Zenodo driver set (cleared for neither), plus files that must never go up."""
    (tmp_path / "bin" / "azcopy").write_text(AZCOPY_STUB)
    root = tmp_path / "repo"
    shards = root / "data" / "shards"
    shards.mkdir(parents=True)
    for name in ("dmd-00000.tar", "zenodo_driver-00000.tar", "dmd-00001.tar.partial", "dmd.shards.json"):
        (shards / name).write_bytes(name.encode())
    table = pa.table({"clip_id": ["dmd/1", "zenodo_driver/1"], "dataset": ["dmd", "zenodo_driver"]})
    pq.write_table(table, root / "data" / "manifest.parquet")
    (root / "checkpoints" / "heads-v0").mkdir(parents=True)
    (root / "checkpoints" / "heads-v0" / "best.pt").write_bytes(b"heads")
    return root, azure


def upload(root: Path, env: dict[str, str], *paths: str, **extra: str) -> subprocess.CompletedProcess[str]:
    command = ["bash", str(AZURE / "upload.sh"), *paths]
    return subprocess.run(command, cwd=root, env={**env, **extra}, capture_output=True, text=True, timeout=120)


def azcopy_calls(env: dict[str, str]) -> list[list[str]]:
    log = Path(env["AZ_STATE"]) / "azcopy.log"
    if not log.exists():
        return []
    made: list[list[str]] = [[]]
    for line in log.read_text().splitlines():
        made[-1].append(line) if line != "--" else made.append([])
    return made[:-1]


def synced(root: Path, env: dict[str, str]) -> set[str]:
    """Shard files the `azcopy sync` of data/shards would upload, applying its --exclude-pattern
    as azcopy does (`;`-separated wildcards matched against file names)."""
    (sync,) = [args for args in azcopy_calls(env) if args[:2] == ["sync", "data/shards"]]
    patterns = sync[sync.index("--exclude-pattern") + 1].split(";")
    names = [path.name for path in (root / "data" / "shards").iterdir()]
    return {name for name in names if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns)}


def test_uncleared_datasets_and_unfinished_files_stay_local(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    result = upload(root, env, "data/shards", "data/manifest.parquet", "checkpoints/heads-v0", **BLOB_CLEARED)
    # The manifest holds rows of a dataset that is not cleared: nothing goes up, no SAS is made.
    assert result.returncode == 1 and "zenodo_driver (1 rows)" in result.stderr
    assert not azcopy_calls(env)
    assert not (Path(env["AZ_STATE"]) / "calls.jsonl").exists()  # no az call, so no SAS was made

    result = upload(
        root, env, "data/shards", "data/manifest.parquet", "checkpoints/heads-v0", DROP_UNCLEARED="1", **BLOB_CLEARED
    )
    assert result.returncode == 0, result.stderr
    assert synced(root, env) == {"dmd-00000.tar"}
    assert "leaving out the files of zenodo_driver" in result.stderr
    manifest = pq.read_table(Path(env["AZ_STATE"]) / "blob" / "data" / "manifest.parquet")
    assert manifest.column("dataset").to_pylist() == ["dmd"]
    assert ["sync", "checkpoints/heads-v0"] in [args[:2] for args in azcopy_calls(env)]


def test_allow_uncleared_uploads_a_named_dataset(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    result = upload(root, env, "data/shards", "data/manifest.parquet", ALLOW_UNCLEARED="dmd zenodo_driver")
    assert result.returncode == 0, result.stderr
    assert synced(root, env) == {"dmd-00000.tar", "zenodo_driver-00000.tar"}
    # Nothing left out, so the manifest goes up byte for byte (runs record its sha256).
    uploaded = Path(env["AZ_STATE"]) / "blob" / "data" / "manifest.parquet"
    assert uploaded.read_bytes() == (root / "data" / "manifest.parquet").read_bytes()


@pytest.mark.parametrize(
    ("path", "code", "message"),
    [
        ("data/raw", 2, "Azure jobs read only"),
        ("data/labels", 2, "Azure jobs read only"),
        ("checkpoints/../data/raw", 2, "Azure jobs read only"),
        ("checkpoints/missing", 2, "not found"),
        ("data/shards/zenodo_driver-00000.tar", 1, "not cleared"),
        ("data/shards/dmd-00001.tar.partial", 1, "not uploaded"),
        ("checkpoints/heads-v0/test.tmp.npz", 1, "not uploaded"),
        ("checkpoints/", 2, "data/features and checkpoints/<dir>"),
    ],
)
def test_upload_refuses_before_anything_goes_up(checkout, path: str, code: int, message: str) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    (root / "data" / "raw").mkdir()
    (root / "data" / "labels").mkdir()
    (root / "checkpoints" / "heads-v0" / "test.tmp.npz").write_bytes(b"partial")
    result = upload(root, env, "data/shards/dmd-00000.tar", path, **BLOB_CLEARED)
    assert result.returncode == code and message in result.stderr
    assert not azcopy_calls(env)


def test_a_single_cleared_shard_goes_up(checkout) -> None:  # type: ignore[no-untyped-def]
    root, env = checkout
    result = upload(root, env, "data/shards/dmd-00000.tar", **BLOB_CLEARED)
    assert result.returncode == 0, result.stderr
    assert (Path(env["AZ_STATE"]) / "blob" / "data" / "shards" / "dmd-00000.tar").read_bytes() == b"dmd-00000.tar"


def test_kaggle_clearance_does_not_cover_blob(checkout) -> None:  # type: ignore[no-untyped-def]
    # With the registry as it is, DMD may be kept on Kaggle but not yet on Azure Blob.
    root, env = checkout
    result = upload(root, env, "data/shards/dmd-00000.tar")
    assert result.returncode == 1
    assert "dmd, which is not cleared for storage on Azure Blob (registry azure_storage_ok)" in result.stderr
    result = upload(root, env, "data/shards", "data/manifest.parquet", DROP_UNCLEARED="1")
    assert result.returncode == 1 and "no dataset in the manifest is cleared for storage on Azure Blob" in result.stderr
    assert not azcopy_calls(env)
