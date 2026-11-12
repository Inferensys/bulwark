"""Provenance records and the checks built on them: code version, resume fingerprints,
features identities and the clean chain."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from bulwark.data.taxonomy import default_taxonomy
from bulwark.model.bulwark import load_payload
from bulwark.train import provenance as prov
from bulwark.train.cache_features import logit_layout_sha256
from bulwark.train.common import Checkpointer, ResumeError
from bulwark.train.provenance import (
    ProvenanceError,
    code_version,
    features_identity,
    fingerprint,
    require_clean,
)


def provenance(*, commercial_only: bool = False, init_path: str = "checkpoints/heads/best.pt", init_sha: str = "s1",
               code: dict | None = None, created: str = "2026-01-01T00:00:00+00:00") -> dict:
    return {
        "job": "lora",
        "created": created,
        "config": {"lr": 1e-3, "commercial_only": commercial_only, "init": init_path},
        "commercial_only": commercial_only,
        "manifest_sha256": "m",
        "inputs": {"shards": {"a.tar": "1"}, "init": {"path": init_path, "sha256": init_sha}},
        "code": code or {"commit": "c1", "dirty": False},
    }


@pytest.fixture
def packed_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(prov, "REPO_ROOT", tmp_path)
    code_version.cache_clear()
    yield tmp_path
    code_version.cache_clear()


def test_code_version_reads_pack_code_provenance(packed_root: Path) -> None:
    # The keys scripts/pack_code.sh writes into the archive that Kaggle and Azure jobs run from.
    (packed_root / "PROVENANCE.json").write_text(json.dumps({
        "commit": "0123abc", "dirty": True, "changed": ["bulwark/train/common.py"],
        "packed_at": "2026-10-01T12:00:00Z",
    }))
    assert code_version() == {
        "commit": "0123abc", "dirty": True, "changed": ["bulwark/train/common.py"],
        "packed_at": "2026-10-01T12:00:00Z", "source": "PROVENANCE.json",
    }


def test_code_version_reads_the_older_git_keys(packed_root: Path) -> None:
    # Archives packed before pack_code.sh switched to the short keys.
    (packed_root / "PROVENANCE.json").write_text(json.dumps({"git_commit": "0123abc", "git_dirty": False}))
    version = code_version()
    assert (version["commit"], version["dirty"], version["changed"]) == ("0123abc", False, [])


def test_resume_compares_inputs_by_content_not_path(tmp_path: Path) -> None:
    Checkpointer(tmp_path, 15, provenance(init_path="/kaggle/input/heads/best.pt")).save("last.pt", {"model": {}, "step": 1})
    assert "path" not in fingerprint(provenance())["inputs"]["init"]
    # The same init under another mount resumes; a different init does not.
    assert Checkpointer(tmp_path, 15, provenance(init_path="checkpoints/heads/best.pt")).resume()["step"] == 1
    with pytest.raises(ResumeError, match=r"inputs\.init\.sha256"):
        Checkpointer(tmp_path, 15, provenance(init_sha="s2")).resume()


def test_commercial_only_never_resumes_a_last_pt_without_clean_provenance(tmp_path: Path) -> None:
    torch.save({"model": {}, "step": 3}, tmp_path / "last.pt")  # written before provenance existed
    with pytest.raises(ProvenanceError, match=r"last\.pt .*records no provenance.*fresh=true"):
        Checkpointer(tmp_path, 15, provenance(commercial_only=True)).resume(allow_mismatch=True)
    # Without commercial_only, allow_mismatch may still continue it.
    assert Checkpointer(tmp_path, 15, provenance()).resume(allow_mismatch=True)["step"] == 3


def test_sessions_record_each_code_version_and_a_change_is_logged(tmp_path: Path) -> None:
    first = Checkpointer(tmp_path, 15, provenance(code={"commit": "c1", "dirty": False}))
    assert first.resume() is None
    first.save("last.pt", {"model": {}, "step": 1})
    second = Checkpointer(tmp_path, 15, provenance(code={"commit": "c2", "dirty": False}, created="2026-01-02T00:00:00+00:00"))
    assert second.resume()["step"] == 1
    second.save("final.pt", {"model": {}, "step": 2})
    sessions = load_payload(tmp_path / "final.pt")["provenance"]["sessions"]
    assert [s["code"]["commit"] for s in sessions] == ["c1", "c2"]
    events = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    changed = [e for e in events if e.get("event") == "code_changed"]
    assert len(changed) == 1 and changed[0]["before"]["commit"] == "c1" and changed[0]["now"]["commit"] == "c2"


def test_features_identity_names_the_encoder_init_and_datasets_behind_a_dir(tmp_path: Path) -> None:
    init = {"sha256": "b1", "has_provenance": True, "commercial_only": True,
            "datasets": {"img": {"commercial_ok": True}, "vid": {"commercial_ok": False}}}
    encoder = {"vjepa_commit": "v", "weights": "w.pt", "weights_sha256": "ws"}
    meta = {"commercial_only": True, "run": {"encoder": encoder, "init": init, "with_logits": True}, "shards": {}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    identity = features_identity(tmp_path)
    assert identity["datasets"] == init["datasets"] and identity["encoder"] == encoder and identity["init"] == init
    with pytest.raises(ProvenanceError, match=r"not commercially usable: \['vid'\]"):
        require_clean(identity, "teacher")


def test_require_clean_hints_at_the_right_fix() -> None:
    missing = {"path": "x", "has_provenance": False}
    research = {"path": "x", "has_provenance": True, "commercial_only": False}
    with pytest.raises(ProvenanceError, match=r"cache\.yaml and --set overwrite=true"):
        require_clean(missing, "features")
    with pytest.raises(ProvenanceError, match=r"Use frozen features"):
        require_clean(research, "features")
    with pytest.raises(ProvenanceError, match=r"cache-b1-clean\.yaml"):
        require_clean(missing, "teacher")
    with pytest.raises(ProvenanceError, match=r"\*-clean configs"):
        require_clean(research, "init")


def test_logit_layout_ignores_label_docs_but_not_classes() -> None:
    taxonomy = default_taxonomy()
    name, spec = next((n, s) for n, s in taxonomy.labels.items() if s.kind == "categorical")
    redocumented = dataclasses.replace(taxonomy, labels={**taxonomy.labels, name: dataclasses.replace(spec, doc="reworded")})
    reclassed = dataclasses.replace(taxonomy, labels={**taxonomy.labels, name: dataclasses.replace(spec, classes=(*spec.classes, "new"))})
    assert logit_layout_sha256(redocumented) == logit_layout_sha256(taxonomy)
    assert logit_layout_sha256(reclassed) != logit_layout_sha256(taxonomy)
