"""Registry terms, label lists and the generated dataset doc."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from bulwark.data.registry import (
    DATASETS,
    MAINTAINERS_RELEASE_DECISION,
    STORAGE_FIELDS,
    DatasetSpec,
    datasets_markdown,
    get_spec,
    uncleared_for_release,
)
from bulwark.data.sources import get_adapter
from bulwark.data.sources.auc_v2 import CLASS_LABELS as AUC_CLASSES
from bulwark.data.sources.auc_v2 import COMMON as AUC_COMMON
from bulwark.data.sources.dmd import DRIVER_ACTIONS
from bulwark.data.sources.drive_and_act import ACTIVITY_LABELS
from bulwark.data.sources.images import ZENODO_CLASSES
from bulwark.data.sources.state_farm import CLASS_LABELS, COMMON
from bulwark.data.taxonomy import default_taxonomy


@pytest.mark.filterwarnings("ignore:MRL Eye")  # the fixture is not the full release
def test_provides_lists_every_ground_truth_label_an_adapter_writes(
    tmp_path: Path, make_image, make_video
) -> None:  # type: ignore[no-untyped-def]
    emitted: dict[str, set[str]] = {
        "auc_v2": {k for labels in AUC_CLASSES.values() for k in labels} | set(AUC_COMMON),
        "drive_and_act": {k for labels in ACTIVITY_LABELS.values() for k in labels}
        | {"modality", "context"},
        "state_farm": {k for labels in CLASS_LABELS.values() for k in labels} | set(COMMON),
        "zenodo_driver": {k for labels in ZENODO_CLASSES.values() for k in labels}
        | {"modality", "context"},
    }
    make_image(tmp_path / "mrl" / "s0001" / "s0001_00001_0_1_0_0_0_01.png")
    make_image(tmp_path / "cew" / "ClosedFace" / "a.jpg")
    nthu = tmp_path / "nthu" / "001" / "night_glasses"
    make_video(nthu / "yawning.avi", frames=60)
    for kind, values in (("drowsiness", "1"), ("eye", "1"), ("head", "2"), ("mouth", "2")):
        (nthu / f"001_night_glasses_yawning_{kind}.txt").write_text(values * 60)
    make_video(tmp_path / "uta" / "Fold1_part1" / "01" / "5.mp4", frames=90)
    make_video(tmp_path / "yawdd" / "Mirror" / "1-FemaleGlasses-Normal.avi", frames=90)
    make_image(tmp_path / "sh17" / "images" / "p.jpg", size=(400, 400))
    (tmp_path / "sh17" / "labels").mkdir(parents=True)
    (tmp_path / "sh17" / "labels" / "p.txt").write_text("0 0.3 0.5 0.4 0.8\n")
    for name, root in (
        ("mrl_eye", "mrl"),
        ("cew", "cew"),
        ("nthu_ddd", "nthu"),
        ("uta_rldd", "uta"),
        ("yawdd", "yawdd"),
        ("sh17", "sh17"),
    ):
        records = list(get_adapter(name, tmp_path / root).records())
        assert records, name
        emitted[name] = {k for r in records for k in r.labels_gt}
    for name, labels in emitted.items():
        assert labels <= set(get_spec(name).provides), (name, labels - set(get_spec(name).provides))
    taxonomy = default_taxonomy()
    for spec in DATASETS:
        assert set(spec.provides) <= set(taxonomy.labels), spec.name


def test_adapters_can_override_terms_per_record(tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    adapter = get_adapter("cew", tmp_path)
    path = make_image(tmp_path / "ClosedFace" / "a.jpg")
    default = adapter.image_record(path, "a", {})
    assert (default.licence, default.commercial_ok) == (get_spec("cew").licence, False)
    custom = adapter.image_record(
        path, "a", {}, licence="CC BY 4.0 (project x)", commercial_ok=True
    )
    assert (custom.licence, custom.commercial_ok) == ("CC BY 4.0 (project x)", True)


def test_unverified_terms_are_not_marked_commercial_or_releasable() -> None:
    assert not get_spec("roboflow_forklift_ppe").commercial_ok
    assert not get_spec("stock_video").commercial_ok
    # Only the maintainers' release decision of 2026-10-30 clears datasets; terms nobody has
    # decided on, and the excluded Zenodo set, are not cleared.
    assert {s.name for s in DATASETS if s.weights_release_ok} == {"dmd", "drive_and_act", "mrl_eye", "auc_v2"}
    assert uncleared_for_release(["drive_and_act", "zenodo_driver", "mrl_eye", "state_farm"]) == [
        "state_farm",
        "zenodo_driver",
    ]
    mrl = get_spec("mrl_eye")
    assert "Kaggle mirror" in mrl.source_copy
    # The structured fingerprint the adapter compares is the one the description quotes.
    assert re.fullmatch(r"[0-9a-f]{64}", mrl.source_copy_fingerprint)
    assert f"sha256:{mrl.source_copy_fingerprint}" in mrl.source_copy
    assert all(spec.source_copy for spec in DATASETS if spec.source_copy_fingerprint)


def test_markdown_shows_release_terms_and_citations() -> None:
    text = datasets_markdown()
    assert "| Release |" in text and "## Terms relied on" in text and "## Citations" in text
    assert "doi          = {10.5281/zenodo.14908802}" in text
    assert "can be used in a model we sell" not in text
    committed = Path(__file__).resolve().parents[1] / "docs" / "datasets.md"
    assert committed.read_text(encoding="utf-8") == text, (
        "regenerate: bulwark datasets --markdown docs/datasets.md"
    )


def test_dmd_provides_exactly_what_its_face_clips_carry(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    def session(name: str, kind: str, tracks: dict[str, str]) -> None:
        folder = tmp_path / "gA" / "1" / name
        stem = f"gA_1_{name}_2019-03-08T09;31;15+01;00"
        make_video(folder / f"{stem}_rgb_face.mp4", frames=90)
        actions = {
            str(i): {
                "type": f"{track}/{value}",
                "frame_intervals": [{"frame_start": 0, "frame_end": 89}],
            }
            for i, (track, value) in enumerate(tracks.items())
        }
        (folder / f"{stem}_rgb_ann_{kind}.json").write_text(
            json.dumps({"openlabel": {"actions": actions}})
        )

    session("s1", "distraction", {"driver_actions": "safe_drive", "hands_using_wheel": "both"})
    session("s5", "drowsiness", {"eyes_state": "close", "blinks": "blinking", "yawning": "yawning"})
    session("s6", "gaze", {"gaze_zone": "front"})
    records = list(get_adapter("dmd", tmp_path).records())
    emitted = {k for r in records for k in r.labels_gt}
    emitted |= {k for labels in DRIVER_ACTIONS.values() for k in labels}
    assert emitted == set(get_spec("dmd").provides)
    assert {k for r in records for k in r.labels_teacher} == {
        "alertness"
    }  # derived, not ground truth


def test_storage_is_cleared_per_provider_and_shown_in_the_doc() -> None:
    fields = {field.name for field in dataclasses.fields(DatasetSpec)}
    assert set(STORAGE_FIELDS) == {"kaggle", "azure"} and set(STORAGE_FIELDS.values()) <= fields
    doc = datasets_markdown()
    assert "| Kaggle | Azure |" in doc
    flag = {None: "unknown", True: "yes", False: "no"}
    for spec in DATASETS:
        kaggle, azure = spec.kaggle_storage_ok, spec.azure_storage_ok
        assert kaggle in flag and azure in flag, spec.name
        row = next(line for line in doc.splitlines() if line.startswith(f"| [{spec.title}]("))
        assert row.split(" | ")[4:6] == [flag[kaggle], flag[azure]], spec.name


def test_cleared_storage_is_backed_by_a_dated_decision_in_the_terms() -> None:
    """A True storage flag names its provider in `terms` with the date it was approved, and
    approval for one provider does not stand in for the other."""
    names = {"kaggle": "Kaggle", "azure": "Azure Blob"}
    for spec in DATASETS:
        for storage, field in STORAGE_FIELDS.items():
            if getattr(spec, field) is True:
                assert re.search(rf"{names[storage]}[^.]*approved[^.]*\d{{4}}-\d{{2}}-\d{{2}}", spec.terms), (
                    spec.name, storage
                )
    for name in ("drive_and_act", "dmd", "mrl_eye"):
        spec = get_spec(name)
        assert spec.kaggle_storage_ok is True and spec.azure_storage_ok is None, name
        assert "Azure Blob not yet approved" in spec.terms, name
    zenodo = get_spec("zenodo_driver")
    assert zenodo.kaggle_storage_ok is False and zenodo.azure_storage_ok is False


def test_auc_v2_entry_matches_its_adapter_and_its_storage_decision() -> None:
    spec = get_spec("auc_v2")
    emitted = {k for labels in AUC_CLASSES.values() for k in labels} | set(AUC_COMMON)
    assert spec.adapter and set(spec.provides) == emitted and "food" not in emitted
    assert not spec.commercial_ok and not spec.download_url
    # A private Kaggle copy kept although the MI-AUC group has not authorized it; the terms
    # say so.
    assert spec.kaggle_storage_ok is True and spec.azure_storage_ok is False
    assert "the group has not authorized the copy" in spec.terms
    assert "prior authorization of the MI-AUC group" in spec.terms
    # The reference the V2 licence agreement asks users to cite.
    assert spec.citation.startswith('H. Eraqi, Y. Abouelnaga, M. Saad, M. Moustafa, "Driver ')
    assert "10,555 train / 1,123 test" in spec.notes and "2,000 train / 800 test" in spec.notes
    # The one driver on both sides of the authors' split is moved to train (the adapter).
    assert "moves those 83 to train (camera 1 here: 10,638 train / 1,040 test)" in spec.notes


V0_SOURCES = {
    "dmd": "Vicomtech",
    "drive_and_act": "Fraunhofer IOSB",
    "mrl_eye": "the dataset's authors",
    "auc_v2": "the MI-AUC group",
}


def test_v0_sources_are_cleared_by_the_maintainers_decision_not_by_their_owners() -> None:
    """The maintainers decided on 2026-10-30 to release the v0 weights for non-commercial
    research. The registry records that as their own decision: never as the owners'
    consent, and with no claim that the owners were asked."""
    for name, owner in V0_SOURCES.items():
        spec = get_spec(name)
        assert spec.weights_release_ok is True and spec.weights_release_basis == MAINTAINERS_RELEASE_DECISION, name
        assert not spec.commercial_ok and spec.kaggle_storage_ok is True, name  # unchanged by the decision
        assert f"is Inferensys's decision of 2026-10-30, not a statement by {owner}." in spec.terms, name
        for stale in ("permission is to be requested", "answers", "ask the authors"):
            assert stale not in spec.terms, (name, stale)
    assert uncleared_for_release(V0_SOURCES) == []
    assert uncleared_for_release(["dmd", "yawdd", "uta_rldd"]) == ["uta_rldd", "yawdd"]
    basis = MAINTAINERS_RELEASE_DECISION
    assert basis.startswith("Inferensys's decision of 2026-10-30") and "not a statement by the dataset's owners" in basis
    assert "non-commercial research only, under CC BY-NC 4.0" in basis
    assert "must also follow the dataset's terms" in basis and "withdrawn and the models retrained" in basis
    claims = r"permission|consent|agree|approv|asked|request|grant|authori[sz]"
    assert not re.search(claims, basis, re.IGNORECASE)


def test_the_release_decision_does_not_say_what_v0_is_trained_on() -> None:
    """The decision clears weights trained on each dataset; it does not say v0 was trained
    on it, so the same basis can clear a dataset whatever a release is trained on. What v0
    is trained on (the four sources, AUC Distracted Driver v2 among them) is stated in each
    dataset's own terms, as for DMD."""
    assert "that v0 weights trained on the dataset may be published" in MAINTAINERS_RELEASE_DECISION
    assert "the v0 weights trained on the dataset" not in MAINTAINERS_RELEASE_DECISION
    auc = get_spec("auc_v2")
    assert "Releasing the v0 weights trained on AUC Distracted Driver v2 for non-commercial research" in auc.terms
    assert "If the v0 release includes" not in auc.terms
    doc = datasets_markdown()
    assert "that v0 weights trained on it may be published for non-commercial research" in doc
    assert "decision to publish the v0 weights" not in doc


def test_every_summary_of_the_flags_allows_for_the_maintainers_decision() -> None:
    """Not every flag is a reading of the dataset's terms: for the v0 sources,
    `weights_release_ok` is the maintainers' own decision (`weights_release_basis`)."""
    from bulwark import cli
    from bulwark.data import registry
    from bulwark.export import release

    for doc in (registry.__doc__, release.__doc__, cli.release_check.__doc__):
        text = " ".join((doc or "").split())
        assert "reading of each dataset's terms" in text
        assert re.search(r"reading of each dataset's terms,? or[^.]*\bdecision\b", text), text[:120]


def test_a_set_release_flag_says_what_it_rests_on() -> None:
    """`weights_release_basis` is filled exactly when `weights_release_ok` is set, and is
    dated; the doc and its Release column show it."""
    doc = datasets_markdown()
    for spec in DATASETS:
        assert bool(spec.weights_release_basis) == (spec.weights_release_ok is not None), spec.name
        row = next(line for line in doc.splitlines() if line.startswith(f"| [{spec.title}]("))
        release = row.split(" | ")[3]
        if spec.weights_release_basis:
            assert re.search(r"\d{4}-\d{2}-\d{2}", spec.weights_release_basis), spec.name
            flag = "yes" if spec.weights_release_ok else "no"
            assert f"  Weights release ({flag}): {spec.weights_release_basis}" in doc.splitlines(), spec.name
        decided = spec.weights_release_basis == MAINTAINERS_RELEASE_DECISION
        assert release == ("yes (maintainers' decision)" if decided else {None: "unknown", True: "yes", False: "no"}[spec.weights_release_ok])
    assert "it is not a statement by the dataset's owners" in doc


def test_registry_text_reads_as_public_documentation() -> None:
    for spec in DATASETS:
        own = " ".join((spec.licence, spec.notes, spec.source_copy))  # `terms` also quotes others
        assert not re.search(r"\b(?:we|our)\b", own, re.IGNORECASE), spec.name
        text = f"{own} {spec.terms} {spec.weights_release_basis}"
        assert not re.search(r"\bowner(?:'s)? decision|\bthe owner'?s?\b", text, re.IGNORECASE), spec.name
        # Correspondence with dataset owners (what was drafted or sent) stays in private notes.
        assert not re.search(r"drafted|not yet sent|unsent|e-?mail(?:ed)? to", text, re.IGNORECASE), spec.name
    # A licence summary does not claim restrictions the terms say were not reviewed.
    state_farm = get_spec("state_farm")
    assert "not yet reviewed" in state_farm.terms and "not yet reviewed" in state_farm.licence
    assert "non-commercial" not in state_farm.licence


def test_adapter_datasets_have_a_citation_and_terms() -> None:
    # State Farm's citation is behind a Kaggle sign-in; it must be filled before ingest.
    # synthetic_wan is generated by this project.
    missing = {s.name for s in DATASETS if s.adapter and not (s.citation and s.terms)}
    assert missing <= {"state_farm", "synthetic_wan"}
