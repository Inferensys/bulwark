"""scripts/pack_code.sh: what goes into the job code archive, and its provenance record."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash scripts")


def git(repo: Path, *args: str) -> str:
    command = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args]
    return subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal committed checkout with pack_code.sh and a few ignored files."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "pack_code.sh", root / "scripts" / "pack_code.sh")
    for path in (".gitignore", "pyproject.toml", "uv.lock"):  # the lock is checked against pyproject
        shutil.copy(REPO / path, root / path)
    for path in ("README.md", "bulwark/data/manifest.py", "configs/a.yaml", "docs/x.md",
                 "infra/azure/run_job.sh", "infra/kaggle/covers/x.png", "tests/test_x.py"):  # fmt: skip
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(f"# {path}\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    # Ignored files that must never be packed.
    (root / "infra/azure/ledger.jsonl").write_text("{}\n")
    (root / "infra/azure/env.local.sh").write_text("export BULWARK_ALERT_EMAIL=me@example.com\n")
    (root / "bulwark/data/__pycache__").mkdir()
    (root / "bulwark/data/__pycache__/manifest.cpython-311.pyc").write_bytes(b"\0")
    return root


def pack(repo: Path, out: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(repo / "scripts" / "pack_code.sh"), str(out)],
        capture_output=True, text=True, env={**os.environ, **env},
    )  # fmt: skip


def test_pack_code_records_the_commit_and_skips_ignored_files(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "code.tgz"
    result = pack(repo, out)
    assert result.returncode == 0, result.stderr
    with tarfile.open(out) as archive:
        names = set(archive.getnames())
        provenance = json.loads(archive.extractfile("PROVENANCE.json").read())  # type: ignore[union-attr]
    assert {"bulwark/data/manifest.py", "uv.lock", "infra/azure/run_job.sh", "PROVENANCE.json"} <= names
    assert not any("ledger" in n or "env.local" in n or "__pycache__" in n for n in names)
    # Jobs never run tests or read docs or dataset covers; they would only make the archive
    # (inside a Kaggle job's script) larger.
    assert not {"docs/x.md", "tests/test_x.py", "infra/kaggle/covers/x.png"} & names
    # The keys bulwark/train/provenance.py reads.
    assert provenance["commit"] == git(repo, "rev-parse", "HEAD").strip()
    assert provenance["dirty"] is False and provenance["changed"] == []


def test_pack_code_refuses_uncommitted_changes_unless_allowed(repo: Path, tmp_path: Path) -> None:
    (repo / "bulwark/data/manifest.py").write_text("# edited\n")
    (repo / "bulwark/data/new.py").write_text("# untracked\n")
    refused = pack(repo, tmp_path / "a.tgz")
    assert refused.returncode == 1 and "bulwark/data/manifest.py" in refused.stderr
    assert not (tmp_path / "a.tgz").exists()

    allowed = pack(repo, tmp_path / "b.tgz", ALLOW_DIRTY="1")
    assert allowed.returncode == 0, allowed.stderr
    with tarfile.open(tmp_path / "b.tgz") as archive:
        provenance = json.loads(archive.extractfile("PROVENANCE.json").read())  # type: ignore[union-attr]
        assert archive.extractfile("bulwark/data/manifest.py").read() == b"# edited\n"  # type: ignore[union-attr]
    assert provenance["dirty"] is True
    assert set(provenance["changed"]) == {"bulwark/data/manifest.py", "bulwark/data/new.py"}


def test_pack_code_refuses_a_stale_lock(repo: Path, tmp_path: Path) -> None:
    # A renamed project (or any dependency change) without `uv lock` would fail `uv sync
    # --frozen` on the job machine.
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(re.sub(r'(?m)^name = "[^"]+"', 'name = "renamed-bulwark"', pyproject.read_text(), count=1))
    git(repo, "commit", "-q", "-am", "rename")
    result = pack(repo, tmp_path / "c.tgz")
    assert result.returncode == 1 and "uv lock" in result.stderr
    assert not (tmp_path / "c.tgz").exists()


def test_shell_scripts_parse() -> None:
    scripts = sorted([*REPO.glob("scripts/*.sh"), *REPO.glob("infra/**/*.sh")])
    assert scripts
    for script in scripts:
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, f"{script}: {result.stderr}"

