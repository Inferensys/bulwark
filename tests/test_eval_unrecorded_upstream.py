"""An upstream model whose training data is not recorded blocks a release from any step
down the chain, not only when it is a direct input.

A B1 fine-tuned from a heads checkpoint that records no provenance names that init under
`unrecorded_inputs`; a features dir cached with the B1 and an E1 distilled from it carry
it on, so the E1's release check still reports the unknown data two steps up.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from test_training_pipeline import cfg, data, run_cache, run_heads, run_lora  # noqa: F401  (data: the fixture)

from bulwark.export.release import (
    check_release,
    record_from_provenance,
    training_record,
    upstream_datasets,
)
from bulwark.inference import load, release_source
from bulwark.model.bulwark import load_payload
from bulwark.train.provenance import (
    FEATURES_META,
    checkpoint_identity,
    features_identity,
    fingerprint,
    unrecorded_inputs,
)

GRANDPARENT = "teacher -> init (records no provenance)"
DMD = {"dmd": {"licence": "CC BY-NC 4.0", "commercial_ok": False, "splits": {"train": 8}}}


def test_an_unrecorded_init_is_carried_through_a_features_dir_to_the_student(tmp_path: Path) -> None:
    # A B1 whose init records no provenance (a checkpoint from before provenance existed).
    b1_provenance = {
        "commercial_only": False, "datasets": DMD,
        "inputs": {"init": {"has_provenance": False, "sha256": "ab" * 32}},
    }
    torch.save({"provenance": b1_provenance}, tmp_path / "b1.pt")
    b1 = checkpoint_identity(tmp_path / "b1.pt", {"provenance": b1_provenance})
    assert b1["has_provenance"] is True and b1["upstream_datasets"] == {}
    assert b1["unrecorded_inputs"] == ["init (records no provenance)"]

    # The teacher dir cached with that B1 (meta.json as bulwark.train.cache_features writes it).
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    run = {"init": {k: v for k, v in b1.items() if k != "path"}, "with_logits": True}
    (teacher / FEATURES_META).write_text(json.dumps({"run": run, "commercial_only": False, "shards": {}}))
    identity = features_identity(teacher)
    assert identity["unrecorded_inputs"] == ["init (records no provenance)"]

    # The E1 distilled from it: its own datasets are recorded, the grandparent's are not.
    found, unrecorded = upstream_datasets({"teacher": identity})
    assert set(found) == {"dmd"} and unrecorded == [GRANDPARENT]
    e1_provenance = {
        "commercial_only": False, "datasets": {"mrl_eye": {"commercial_ok": True}},
        "inputs": {"teacher": identity}, "code": {"commit": "0123abc", "dirty": False},
    }
    check = check_release(record_from_provenance(e1_provenance, "e1.pt"))
    assert check.summary()["unrecorded_inputs"] == [GRANDPARENT]
    assert any(GRANDPARENT in problem for problem in check.problems)

    # One more step down (a model started from that E1), and the chain still names it.
    torch.save({"provenance": e1_provenance}, tmp_path / "e1.pt")
    e1 = checkpoint_identity(tmp_path / "e1.pt", {"provenance": e1_provenance})
    assert e1["unrecorded_inputs"] == [GRANDPARENT]
    assert unrecorded_inputs({"init": e1}) == [f"init -> {GRANDPARENT}"]


def test_the_chain_record_does_not_change_resume_fingerprints() -> None:
    def provenance(init: dict) -> dict:
        return {"job": "distill", "config": {"lr": 1e-3}, "inputs": {"teacher": {"path": "a", "run_sha256": "r1", **init}}}

    newer = provenance({"upstream_datasets": {}, "unrecorded_inputs": ["init (records no provenance)"]})
    assert fingerprint(provenance({})) == fingerprint(newer)


def test_a_student_two_steps_below_an_unrecorded_init_is_not_cleared(data: Path) -> None:  # noqa: F811
    from bulwark.train.cache_features import run as cache
    from bulwark.train.distill import run as distill

    run_cache(data)
    heads = run_heads(data)
    payload = load_payload(heads / "final.pt")
    del payload["provenance"]  # as a heads checkpoint written before provenance existed
    torch.save(payload, data / "heads-legacy.pt")

    lora = run_lora(data, init=str(data / "heads-legacy.pt"))
    b1 = load_payload(lora / "final.pt")
    assert check_release(training_record(lora / "final.pt")).record.unrecorded == ["init (records no provenance)"]
    assert checkpoint_identity(lora / "final.pt", b1)["unrecorded_inputs"] == ["init (records no provenance)"]

    teacher = cache(cfg(data, init=str(lora / "final.pt"), features=str(data / "features" / "b1"), batch_size=8))
    meta = json.loads((teacher / FEATURES_META).read_text())
    assert meta["run"]["init"]["unrecorded_inputs"] == ["init (records no provenance)"]
    e1 = distill(cfg(data, teacher=str(teacher), out=str(data / "e1"), max_steps=2))

    check = check_release(training_record(e1 / "final.pt"))
    assert check.record.unrecorded == [GRANDPARENT] and not check.ok
    assert any(GRANDPARENT in problem for problem in check.problems)
    # A release directory publishes it, so `bulwark release-check` on the directory sees it too.
    assert release_source(load(e1 / "final.pt", device="cpu"))["unrecorded_inputs"] == [GRANDPARENT]

    # A teacher dir cached before unrecorded_inputs was recorded still resumes, untouched.
    stamp = (teacher / "features-img-00000.npz").stat().st_mtime_ns
    del meta["run"]["init"]["unrecorded_inputs"]
    (teacher / FEATURES_META).write_text(json.dumps(meta))
    cache(cfg(data, init=str(lora / "final.pt"), features=str(teacher), batch_size=8))
    assert (teacher / "features-img-00000.npz").stat().st_mtime_ns == stamp
