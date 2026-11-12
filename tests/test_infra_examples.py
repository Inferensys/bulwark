"""Usage examples in the infra and scripts headers name only the v0 Kaggle slugs and datasets
that may leave the machine, so copying one never uploads, or reshards, an excluded set."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bulwark.data.registry import DATASETS
from bulwark.data.splits import SplitPolicy

REPO = Path(__file__).parents[1]
# The private Kaggle datasets docs/training.md uploads for v0, AUC Distracted Driver v2's
# shards (bulwark-auc) among them.
V0_SLUGS = {"bulwark-drive-and-act", "bulwark-dmd", "bulwark-mrl", "bulwark-auc", "bulwark-meta"}
EXAMPLE_FILES = sorted(
    [
        *REPO.glob("infra/**/*.sh"),
        *REPO.glob("infra/**/*.py"),
        *REPO.glob("scripts/*.sh"),
        REPO / "docs" / "training.md",
    ]
)
# Variables whose values are Kaggle dataset slugs (NAME and PREVIOUS_KERNELS name kernels).
SLUG_VARIABLES = r"(?:SLUG|DATASETS|KAGGLE_SLUGS|META_SLUG|MANIFEST_DATASET|D)"
ASSIGNMENT = re.compile(rf"\b{SLUG_VARIABLES}=(\"[^\"]*\"|'[^']*'|\S+)")
SLUG = re.compile(r"\bbulwark-[a-z0-9-]*[a-z0-9]")
# A quoted list of shard globs, as SHARD_GLOB takes it: 'drive_and_act-*.tar dmd-*.tar'.
SHARD_GLOBS = re.compile(r"'((?:[a-z0-9_]+-\*\.tar ?)+)'")
GLOB_DATASET = re.compile(r"\b([a-z0-9_]+)-\*\.tar\b")
RESHARD = re.compile(r"\bRESHARD=(\"[^\"]*\"|'[^']*'|\S+)")


def cleared() -> set[str]:
    excluded = set(SplitPolicy().excluded_datasets)
    return {
        spec.name
        for spec in DATASETS
        if spec.kaggle_storage_ok is True and spec.name not in excluded
    }


def test_the_example_scan_reads_the_files_it_checks() -> None:
    names = {path.relative_to(REPO).as_posix() for path in EXAMPLE_FILES}
    assert {
        "infra/kaggle/push_data.sh",
        "infra/kaggle/push_job.sh",
        "scripts/prepare_local.sh",
    } <= names
    assert {"docs/training.md", "infra/azure/upload.sh", "scripts/purge_cloud_copies.sh"} <= names
    push_data = (REPO / "infra" / "kaggle" / "push_data.sh").read_text()
    globs = {
        name for value in SHARD_GLOBS.findall(push_data) for name in GLOB_DATASET.findall(value)
    }
    assert {"dmd", "drive_and_act", "mrl_eye"} <= globs


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.relative_to(REPO).as_posix())
def test_examples_name_only_the_v0_kaggle_slugs(path: Path) -> None:
    text = path.read_text()
    slugs = {slug for value in ASSIGNMENT.findall(text) for slug in SLUG.findall(value)}
    slugs |= set(re.findall(r"e\.g\. (bulwark-[a-z0-9-]*[a-z0-9])", text))
    assert slugs <= V0_SLUGS, f"{path.name}: {sorted(slugs - V0_SLUGS)}"


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.relative_to(REPO).as_posix())
def test_shard_and_reshard_examples_name_cleared_datasets(path: Path) -> None:
    text = path.read_text()
    named = {name for value in SHARD_GLOBS.findall(text) for name in GLOB_DATASET.findall(value)}
    # $RESHARD and <dataset> are the variable and a placeholder, not examples.
    named |= {
        name
        for value in RESHARD.findall(text)
        for name in value.strip("'\"").split()
        if re.fullmatch(r"[a-z0-9_]+", name)
    }
    assert named <= cleared(), f"{path.name}: {sorted(named - cleared())}"


def test_the_excluded_zenodo_set_is_not_cleared() -> None:
    # The two checks above rely on this: an example naming the Zenodo set fails them.
    assert "zenodo_driver" in SplitPolicy().excluded_datasets
    assert "zenodo_driver" not in cleared()


def test_push_job_attaches_the_documented_v0_slugs_by_default() -> None:
    """push_job.sh attaches the five v0 slugs by default, AUC Distracted Driver v2's
    (bulwark-auc) among them: the set docs/training.md passes as DATASETS."""
    script = (REPO / "infra" / "kaggle" / "push_job.sh").read_text()
    training = (REPO / "docs" / "training.md").read_text()
    default = re.search(r'^DATASETS="\$\{DATASETS:-([^}]*)\}"$', script, re.MULTILINE)
    documented = re.search(r'^D="([^"]*)"$', training, re.MULTILINE)
    assert default and documented
    assert default[1].split() == documented[1].split()
    assert set(default[1].split()) == V0_SLUGS
