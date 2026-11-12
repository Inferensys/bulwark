"""The datasets a chain of models inherits (`upstream_datasets`), and what a features dir's
meta.json records about the code that wrote it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_training_pipeline import cfg, data, run_cache, run_heads, run_lora  # noqa: F401  (data: the fixture)

from bulwark.data.splits import SplitPolicy
from bulwark.data.taxonomy import default_taxonomy
from bulwark.model.bulwark import load_payload
from bulwark.train.provenance import (
    ProvenanceError,
    checkpoint_identity,
    fingerprint,
    require_clean,
    require_releasable,
)

IMG = {"licence": "test", "commercial_ok": True}
VID = {"licence": "test", "commercial_ok": False}


def test_upstream_datasets_follow_the_whole_chain(data: Path) -> None:  # noqa: F811
    from bulwark.train.cache_features import run as cache
    from bulwark.train.distill import run as distill

    frozen = run_cache(data)
    meta = json.loads((frozen / "meta.json").read_text())
    assert meta["taxonomy_version"] == default_taxonomy().version and meta["split_salt"] == SplitPolicy().salt
    assert "bulwark_version" in meta

    heads = run_heads(data, datasets=["img"])  # the heads see img only; B1 and E1 see both
    lora = run_lora(data, init=str(heads / "final.pt"))
    init = load_payload(lora / "final.pt")["provenance"]["inputs"]["init"]
    assert set(init["datasets"]) == {"img"} and init["upstream_datasets"] == {}  # frozen features: nothing upstream
    b1 = checkpoint_identity(lora / "final.pt", load_payload(lora / "final.pt"))
    assert set(b1["datasets"]) == {"img", "vid"} and b1["upstream_datasets"] == {"img": IMG}

    teacher = cache(cfg(data, init=str(lora / "final.pt"), features=str(data / "features" / "b1"), batch_size=8))
    e1 = distill(cfg(data, teacher=str(teacher), out=str(data / "e1"), max_steps=2))
    payload = load_payload(e1 / "final.pt")
    recorded = payload["provenance"]["inputs"]["teacher"]
    assert set(recorded["datasets"]) == {"img", "vid"} and recorded["upstream_datasets"] == {"img": IMG}
    assert checkpoint_identity(e1 / "final.pt", payload)["upstream_datasets"] == {"img": IMG, "vid": VID}

    # A teacher dir cached before upstream_datasets was recorded still resumes, untouched.
    stamp = (teacher / "features-img-00000.npz").stat().st_mtime_ns
    older = json.loads((teacher / "meta.json").read_text())
    del older["run"]["init"]["upstream_datasets"]
    (teacher / "meta.json").write_text(json.dumps(older))
    cache(cfg(data, init=str(lora / "final.pt"), features=str(teacher), batch_size=8))
    assert (teacher / "features-img-00000.npz").stat().st_mtime_ns == stamp


def test_a_last_pt_from_before_upstream_datasets_still_resumes() -> None:
    def provenance(init: dict) -> dict:
        return {"job": "lora", "config": {"lr": 1e-3}, "inputs": {"init": {"path": "a/best.pt", **init}}}

    older = provenance({"sha256": "s1"})
    newer = provenance({"sha256": "s1", "upstream_datasets": {"dmd": {"commercial_ok": False}}})
    assert fingerprint(older) == fingerprint(newer)


def test_clean_and_release_checks_cover_the_upstream_datasets() -> None:
    source = {
        "path": "x", "has_provenance": True, "commercial_only": True, "release_only": True,
        "datasets": {"open_images": {"commercial_ok": True}}, "upstream_datasets": {"dmd": {"commercial_ok": False}},
    }
    with pytest.raises(ProvenanceError, match=r"not commercially usable: \['dmd'\]"):
        require_clean(source, "init")
    # DMD is cleared for a weights release (the maintainers' decision); YawDD, upstream only, is not.
    releasable = {**source, "datasets": {"dmd": {"commercial_ok": False}}, "upstream_datasets": {"yawdd": {"commercial_ok": False}}}
    with pytest.raises(ProvenanceError, match=r"published weights.*\['yawdd'\]"):
        require_releasable(releasable, "init")
