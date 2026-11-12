"""The public project files agree with each other and with the code they describe."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

from bulwark.export.release import MODEL_REPOS, model_card_url
from bulwark.model.backbone import VJEPA_COMMIT

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Inferensys/bulwark"

# Markdown files a reader of the public repository sees. docs/datasets.md is generated
# from the registry and checked against it separately; its text is checked with the other
# public files, since the weights release stages it (the model card links to it).
PUBLIC_MARKDOWN = [
    "README.md",
    "MODEL_CARD.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "RELEASING.md",
    "CODE_OF_CONDUCT.md",
    *(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "docs").glob("*.md")) if p.name != "datasets.md"),
    *(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / ".github").rglob("*.md"))),
    "release/hf/README.md",
]
# The Hugging Face card templates and data (release/hf/): published once rendered. Their
# links are template placeholders, which scripts/hf_release.py resolves and checks.
CARD_TEMPLATES = sorted(
    p.relative_to(ROOT).as_posix() for p in (ROOT / "release" / "hf").rglob("*") if p.is_file() and p.name != "README.md"
)
PUBLIC_FILES = [*PUBLIC_MARKDOWN, "docs/datasets.md", "NOTICE", "CITATION.cff", ".github/ISSUE_TEMPLATE/config.yml", *CARD_TEMPLATES]
# Inferensys's public contact address, which the cards give for commercially licensed
# models and removal requests; any other address is an internal detail.
PUBLIC_EMAILS = {"hello@inferensys.com"}
# sha256 of https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt, downloaded 2026-10-30.
CC_BY_NC_SHA256 = "41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_code_is_apache_2_everywhere() -> None:
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert licence.lstrip().startswith("Apache License\n                           Version 2.0, January 2004")
    assert "END OF TERMS AND CONDITIONS" in licence
    project = _pyproject()["project"]
    # The wheel also ships Meta's MIT-licensed encoder code (bulwark/model/vjepa21) with its
    # licence text; SPDX expressions replace the licence classifiers (PEP 639).
    assert project["license"] == "Apache-2.0 AND MIT"
    assert project["license-files"] == ["LICENSE", "NOTICE", "bulwark/model/vjepa21/LICENSE"]
    assert not any(c.startswith("License ::") for c in project["classifiers"])
    assert "einops" not in " ".join(project["dependencies"])  # only Meta's left-out AudioPatchEmbed used it
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["license"] == "Apache-2.0"
    assert "Inferensys" in (ROOT / "NOTICE").read_text(encoding="utf-8").splitlines()[1]


def test_distribution_name_differs_but_import_and_cli_stay_bulwark() -> None:
    project = _pyproject()["project"]
    assert project["name"] == "inferensys-bulwark"
    assert project["scripts"] == {"bulwark": "bulwark.cli:app"}
    assert _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["bulwark"]


def test_one_tagline_and_one_repository_url() -> None:
    project = _pyproject()["project"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_paragraph = " ".join(readme.split("\n\n")[1].split())
    assert first_paragraph == project["description"]
    assert project["urls"]["Repository"] == REPO_URL
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["repository-code"] == REPO_URL
    for field in ("cff-version", "message", "title", "authors"):
        assert citation[field], field


def _released() -> set[str]:
    """The models the launch releases: release/hf/launch.yaml, which scripts/hf_release.py
    follows (release/hf/README.md, "Release set")."""
    return set(yaml.safe_load((ROOT / "release" / "hf" / "launch.yaml").read_text(encoding="utf-8"))["released"])


# Public files that may name the repository of a model that is not released: the release
# steps, which describe that future repository for maintainers.
MAINTAINER_DOCS = {"RELEASING.md"}


def test_weights_links_are_the_hugging_face_repository_of_each_released_model() -> None:
    """The code stays on GitHub (REPO_URL); the card that travels with each model's weights
    is the README of that model's Hugging Face model repository. The public docs link the
    repository of every released model, and never that of a model that is not released,
    which stays private, so its link would be dead to every visitor."""
    assert MODEL_REPOS == {"b1": "Inferensys/bulwark-b1-driver-monitoring", "e1": "Inferensys/bulwark-e1-driver-monitoring-onnx"}
    urls = {"b1": "https://huggingface.co/Inferensys/bulwark-b1-driver-monitoring", "e1": "https://huggingface.co/Inferensys/bulwark-e1-driver-monitoring-onnx"}
    assert model_card_url("b1") == model_card_url("b1-frozen") == urls["b1"] and model_card_url("e1") == urls["e1"]
    with pytest.raises(ValueError, match="No released model"):
        model_card_url("onnx")
    released = _released()
    assert "b1" in released and released <= set(MODEL_REPOS)
    for name in ("README.md", "MODEL_CARD.md", "RELEASING.md", "SECURITY.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for model in sorted(released):
            assert urls[model] in text, (name, model)
    for name in PUBLIC_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "blob/main/MODEL_CARD.md" not in text, name
        assert not re.search(r"huggingface\.co/Inferensys/bulwark(?![\w-])", text), f"{name}: the old single repository"
        for model in sorted(set(MODEL_REPOS) - released):
            assert name in MAINTAINER_DOCS or MODEL_REPOS[model] not in text, f"{name}: names {MODEL_REPOS[model]}, which is not released"


def test_where_a_reader_chooses_a_model_e1_v0_is_said_to_score_well_below_b1() -> None:
    """E1 v0 is released as a research preview that scores well below B1 (RELEASING.md,
    release set). Wherever a reader chooses between the models, the text says so and
    points to B1 for research: README.md, MODEL_CARD.md, the collection's note on E1, and
    every line of B1's card and the organisation card that links E1. E1's own card says so
    first (tests/test_infra_hf_release.py, on the staged card)."""
    if "e1" not in _released():
        pytest.skip("E1 is not released")
    for name in ("README.md", "MODEL_CARD.md"):
        text = " ".join((ROOT / name).read_text(encoding="utf-8").split())
        assert "research preview that scores well below B1" in text and "use b1 for research" in text.lower(), name
    collection = yaml.safe_load((ROOT / "release" / "hf" / "collection.yaml").read_text(encoding="utf-8"))
    (note,) = [item["note"] for item in collection["items"] if item.get("model") == "e1"]
    assert "E1 v0 is a research preview that scores well below B1" in note and "B1 for research" in note
    for name in ("b1.md", "org-card.md"):
        lines = [line for line in (ROOT / "release" / "hf" / name).read_text(encoding="utf-8").splitlines() if "{{ e1_url }}" in line]
        assert lines and all("scores well below" in line for line in lines), name


def test_cc_by_nc_legal_code_is_the_official_text() -> None:
    path = ROOT / "LICENSES" / "CC-BY-NC-4.0.txt"
    assert path.read_text(encoding="utf-8").splitlines()[0] == "Attribution-NonCommercial 4.0 International"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CC_BY_NC_SHA256


def test_notice_pins_the_same_vjepa_commit_as_the_code() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    fetch = (ROOT / "scripts" / "fetch_pretrained.sh").read_text(encoding="utf-8")
    assert VJEPA_COMMIT in notice
    assert f"VJEPA_COMMIT={VJEPA_COMMIT}" in fetch


@pytest.mark.parametrize("name", PUBLIC_FILES)
def test_public_files_hold_no_internal_details(name: str) -> None:
    """Personal addresses, ticket numbers, budgets and local paths belong in private notes."""
    text = (ROOT / name).read_text(encoding="utf-8")
    addresses = {address.rstrip(".") for address in re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)}
    assert not addresses - PUBLIC_EMAILS, "email address"
    assert not re.search(r"(?<![\w.])\d{12,}(?![\w.])", text), "long number such as a support ticket"
    assert not re.search(r"\$\s?\d", text), "money amount"
    assert "/Users/" not in text and "~/" not in text, "machine-specific path"
    assert not re.search(r"drafted but|not yet sent|unsent", text, re.IGNORECASE), "correspondence with dataset owners"


# A claim that a dataset's owners consented to the weights release, or were asked about it.
OWNER_CONSENT = re.compile(
    r"\b(?:owners?|authors?|Vicomtech|Fraunhofer IOSB|MI-AUC group)\b[^.]{0,60}?\b(?:agreed|agree to|consented|"
    r"approved|gave permission|granted permission|were asked|was asked|have been asked|answered)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("name", PUBLIC_FILES)
def test_public_files_never_claim_the_dataset_owners_consent(name: str) -> None:
    """Releasing the v0 weights is the maintainers' decision (the registry's
    MAINTAINERS_RELEASE_DECISION), not the datasets' owners'."""
    match = OWNER_CONSENT.search((ROOT / name).read_text(encoding="utf-8"))
    assert match is None, match and match.group(0)


# "The owner" seeing or reviewing something, which next to the datasets' owners reads as a
# dataset owner having been shown the release. Inferensys reviews the private page.
BARE_OWNER_REVIEW = re.compile(r"\bthe owner\b[^.]{0,40}?\b(?:seen|sees?|saw|reviews?|reviewed|checks?|checked)\b", re.IGNORECASE)


# Wording that offers the research weights for a commercial licence, or for a reader's own
# evaluation (Inferensys counts customer evaluations as commercial use). The commercial offer
# is separate: commercially licensed models, trained on data cleared for commercial use
# (release/hf/shared/product.md).
WEIGHTS_OFFERED = re.compile(
    r"(?<!non-)commercial licen(?:sing|ces?)\b|licence from Inferensys|Using Bulwark in a product"
    r"|\b(?:use|using)\b[^.]{0,60}\bfor evaluation\b|\bfor evaluation,? use\b|\bnot for evaluation\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("name", [*PUBLIC_FILES, "scripts/hf_release.py"])
def test_public_files_never_offer_the_research_weights_for_a_licence_or_an_evaluation(name: str) -> None:
    match = WEIGHTS_OFFERED.search(" ".join((ROOT / name).read_text(encoding="utf-8").split()))
    assert match is None, match and match.group(0)


def test_weights_offered_matches_the_old_wording_only() -> None:
    for old in ("May Inferensys contact you about commercial licensing?", "Commercial licences: inferensys.com",
                "Commercial use needs a licence from Inferensys.", "## Using Bulwark in a product", "Use B1 for evaluation.",
                "use it to try CPU deployment, and Bulwark-B1 for evaluation", "For evaluation, use Bulwark-B1",
                "it is for trying CPU deployment, not for evaluation"):  # fmt: skip
        assert WEIGHTS_OFFERED.search(old), old
    for new in ("Commercially licensed models: contact Inferensys", "a non-commercial licence agreement",
                "the commercial-licence track", "are never used for evaluation", "Use B1 for research.",
                "These weights are not licensed for commercial use."):  # fmt: skip
        assert not WEIGHTS_OFFERED.search(new), new


@pytest.mark.parametrize("name", [*PUBLIC_FILES, "scripts/hf_release.py"])
def test_public_files_say_who_reviews_the_release_before_it_opens(name: str) -> None:
    match = BARE_OWNER_REVIEW.search(" ".join((ROOT / name).read_text(encoding="utf-8").split()))
    assert match is None, match and match.group(0)


def test_bare_owner_review_matches_the_ambiguous_wording_only() -> None:
    assert BARE_OWNER_REVIEW.search("run by a maintainer once the owner has seen the private page")
    assert BARE_OWNER_REVIEW.search("the owner reviews the private repository page")
    assert not BARE_OWNER_REVIEW.search("once Inferensys has reviewed the private page")
    assert not BARE_OWNER_REVIEW.search("an Inferensys owner reviews the private repository page")
    assert not BARE_OWNER_REVIEW.search("if the owners of a training dataset object")


@pytest.mark.parametrize("name", ["README.md", "MODEL_CARD.md", "RELEASING.md", "docs/data-card.md"])
def test_the_release_decision_is_stated_as_the_maintainers_own(name: str) -> None:
    text = " ".join((ROOT / name).read_text(encoding="utf-8").split())
    assert "Inferensys's decision, not a statement by the datasets' owners" in text
    assert "on each dataset's research terms" in text and "must also follow each dataset's terms" in text
    assert "withdrawn and the models retrained without it" in text


def _link_targets(text: str) -> list[str]:
    """Targets of inline links: [text](target), [text](<target>) and [text](target "title")."""
    pattern = r"""\]\(\s*(?:<([^<>\n]*)>|([^)\s<>]+))(?:\s+(?:"[^"\n]*"|'[^'\n]*'|\([^()\n]*\)))?\s*\)"""
    return [angled or bare for angled, bare in re.findall(pattern, text)]


def test_link_targets_include_titled_and_angle_bracket_links() -> None:
    text = '[a](docs/a.md) [b](docs/b.md "B") [c](<docs/c.md>) [d](https://x.org \'D\')'
    assert _link_targets(text) == ["docs/a.md", "docs/b.md", "docs/c.md", "https://x.org"]


@pytest.mark.parametrize("name", PUBLIC_MARKDOWN)
def test_relative_links_resolve(name: str) -> None:
    path = ROOT / name
    text = path.read_text(encoding="utf-8")
    assert len(_link_targets(text)) == text.count("]("), f"{name}: a link that is not [text](target) or [text](target \"title\")"
    for target in _link_targets(text):
        if re.match(r"[a-z]+:", target):
            continue
        file_part, _, anchor = target.partition("#")
        linked = (path.parent / file_part).resolve() if file_part else path
        assert linked.exists(), f"{name}: broken link {target}"
        if anchor and linked.suffix == ".md":
            assert anchor in _anchors(linked), f"{name}: no heading for #{anchor} in {linked.name}"


def _anchors(path: Path) -> set[str]:
    """GitHub-style heading anchors."""
    anchors = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := re.match(r"#{1,6}\s+(.*)", line):
            slug = re.sub(r"[^\w\- ]", "", match.group(1).strip().lower()).replace(" ", "-")
            anchors.add(slug)
    return anchors
