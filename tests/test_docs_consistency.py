"""The docs state the versions, configs and model sizes that the code actually has."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import re
from pathlib import Path

import pytest

import bulwark
from bulwark.data.splits import SplitPolicy
from bulwark.data.taxonomy import default_taxonomy
from bulwark.train.common import TrainConfig
from bulwark.train.provenance import RESUME_EXEMPT

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _prose(name: str) -> str:
    """The file with line breaks folded, so a phrase may wrap."""
    return " ".join(_read(name).split())


def _package_version() -> str:
    return tomllib.loads(_read("pyproject.toml"))["project"]["version"]


def test_version_comes_from_the_installed_distribution() -> None:
    try:
        installed = metadata.version("inferensys-bulwark")
    except metadata.PackageNotFoundError:
        pytest.skip("inferensys-bulwark is not installed")
    assert bulwark.__version__ == installed


def test_version_falls_back_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    try:
        assert importlib.reload(bulwark).__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(bulwark)


def test_docs_name_the_current_versions_and_split_salt() -> None:
    version = _package_version()
    taxonomy = default_taxonomy().version
    salt = SplitPolicy().salt
    readme, card = _prose("README.md"), _prose("MODEL_CARD.md")
    assert f"version {version}" in readme
    assert f"`inferensys-bulwark` {version}" in card
    assert f"package version ({version})" in card
    assert f"taxonomy has its own version ({taxonomy})" in readme
    assert f"taxonomy version ({taxonomy})" in card
    for name in ("README.md", "MODEL_CARD.md", "docs/evaluation.md"):
        assert f"`{salt}`" in _prose(name), name


def test_changelog_has_an_entry_for_the_package_version() -> None:
    # RELEASING.md: the release replaces "Unreleased" in this heading with the release date.
    assert re.search(rf"^## \[{re.escape(_package_version())}\] - ", _read("CHANGELOG.md"), re.MULTILINE)


def test_readme_downloads_mrl_eye_from_the_registry_url() -> None:
    from bulwark.data.registry import get_spec

    url = get_spec("mrl_eye").download_url
    readme = _read("README.md")
    assert f"MRL={url}" in readme
    assert re.search(r"https://web\.archive\.org/web/\d{14}id_/\$MRL", readme), "Internet Archive fallback"


def test_training_doc_covers_every_training_config() -> None:
    training = _read("docs/training.md")
    configs = sorted(p.name for p in (ROOT / "configs" / "train").glob("*.yaml"))
    assert configs
    missing = [c for c in configs if not re.search(rf"(?<![\w-]){re.escape(c)}", training)]
    assert not missing, f"docs/training.md does not mention {missing}"


def test_training_doc_names_the_resume_exempt_keys() -> None:
    training = _read("docs/training.md")
    fields = set(TrainConfig.__dataclass_fields__)
    assert not RESUME_EXEMPT - fields, RESUME_EXEMPT - fields
    missing = sorted(key for key in RESUME_EXEMPT if f"`{key}`" not in training and f"{key}=" not in training)
    assert not missing, f"docs/training.md does not name the resume-exempt keys {missing}"


def test_documented_e1_size_matches_the_model() -> None:
    from bulwark.model.bulwark import BulwarkEdge

    model = BulwarkEdge(pretrained=False)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    heads = sum(p.numel() for p in model.heads.parameters()) / 1e6
    assert f"About {total:.1f}M parameters in total" in _prose("docs/architecture.md")
    assert f"{heads:.1f}M heads" in _prose("docs/architecture.md")
    assert f"about {total:.1f}M parameters" in _prose("MODEL_CARD.md")


@pytest.mark.vjepa
def test_documented_b1_trainable_parameters_match_the_model() -> None:
    from bulwark.model.bulwark import BulwarkBase

    model = BulwarkBase(default_taxonomy(), lora_blocks=4, lora_rank=16)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    encoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith("heads.")) / 1e6
    printed = f"LoRA layers: {len(model.lora_layers)}, trainable params: {trainable:.2f}M"
    for name in ("docs/training.md", "docs/architecture.md"):
        assert printed in _prose(name), name
    assert f"{trainable:.2f}M parameters in all" in _prose("MODEL_CARD.md")
    for name in ("docs/architecture.md", "MODEL_CARD.md"):
        assert f"about {encoder:.2f}M trainable" in _prose(name), name
