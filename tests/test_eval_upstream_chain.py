"""The release check covers the whole chain of upstream models (heads -> B1 -> teacher
features -> E1), from checkpoints and from release directories written by `save_release`."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from bulwark.export.release import check_release, training_record, upstream_datasets
from bulwark.inference import load, save_release
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint
from bulwark.train.provenance import upstream_datasets as chain_datasets

EDGE = {"backbone": "test_efficientnet", "img_size": 64}
RESEARCH = "CC BY-NC 4.0 (research use only)"


def teacher_features(*, upstream: dict | None) -> dict:
    """`inputs.teacher` as `features_identity` records it: a B1 trained on DMD whose own init
    (a heads checkpoint) was trained on MRL Eye. `upstream` None: cached before
    `upstream_datasets` was recorded."""
    return {
        "has_provenance": True, "with_logits": True, "init_sha256": "ef" * 32, "path": "/data/features/b1",
        "datasets": {"dmd": {"licence": "CC BY-NC 4.0", "commercial_ok": False, "splits": {"train": 8}}},
        "upstream_datasets": upstream,
    }


def test_upstream_datasets_include_each_inputs_own_upstream_chain() -> None:
    inputs = {
        "teacher": teacher_features(upstream={
            "mrl_eye": {"licence": "CC BY 4.0", "commercial_ok": True},
            "dmd": {"licence": "CC BY-NC 4.0", "commercial_ok": False},
        }),
        # A checkpoint recorded before `upstream_datasets` existed: its own datasets only.
        "init": {"has_provenance": True, "sha256": "ab" * 32, "datasets": {"drive_and_act": {"commercial_ok": False}}},
    }
    found, unrecorded = upstream_datasets(inputs)
    assert unrecorded == []
    assert found == {
        "dmd": {"via": ["teacher"], "licence": "CC BY-NC 4.0", "commercial_ok": False},
        "drive_and_act": {"via": ["init"], "licence": None, "commercial_ok": False},
        "mrl_eye": {"via": ["teacher"], "licence": "CC BY 4.0", "commercial_ok": True},
    }
    # The same datasets as the chain training records (bulwark.train.provenance).
    assert set(found) == set(chain_datasets(inputs))

    # A dataset the upstream chain records as not commercially usable stays so, whichever entry names it.
    inputs = {"teacher": teacher_features(upstream={"dmd": {"commercial_ok": True}, "mrl_eye": {"commercial_ok": False}})}
    found, _ = upstream_datasets(inputs)
    assert found["dmd"]["commercial_ok"] is False and found["mrl_eye"]["commercial_ok"] is False

    # A features dir cached before upstream_datasets was recorded (None) gives its own datasets.
    found, _ = upstream_datasets({"teacher": teacher_features(upstream=None)})
    assert list(found) == ["dmd"]


def distilled(path: Path) -> Path:
    """An E1 whose distillation teacher records its own and its upstream datasets."""
    torch.manual_seed(0)
    provenance = {
        "job": "distill",
        "commercial_only": False,
        "datasets": {"drive_and_act": {"licence": "test", "commercial_ok": False, "splits": {"train": 4}}},
        "inputs": {"teacher": teacher_features(upstream={"mrl_eye": {"licence": "CC BY 4.0", "commercial_ok": True}})},
        "code": {"commit": "0123abc", "dirty": False},
    }
    extra = {"step": 1, **EDGE, "supervised_labels": ["yawn"], "provenance": provenance}
    return save_checkpoint(path, BulwarkEdge(pretrained=False, **EDGE), extra)


def test_release_directories_from_save_release_record_the_teachers_datasets(tmp_path: Path) -> None:
    checkpoint = distilled(tmp_path / "e1.pt")
    release = save_release(load(checkpoint, device="cpu"), tmp_path / "release", weights_licence=RESEARCH)
    config = json.loads((release / "config.json").read_text())
    assert "release_check" not in config  # written by save_release, not `bulwark release`

    record = training_record(release)
    assert record.upstream == {
        "dmd": {"via": ["teacher"], "licence": "CC BY-NC 4.0", "commercial_ok": False},
        "mrl_eye": {"via": ["teacher"], "licence": "CC BY 4.0", "commercial_ok": True},
    }
    assert record.unrecorded == [] and record.upstream == training_record(checkpoint).upstream
    check = check_release(record)
    assert check.datasets["dmd"]["via"] == ["teacher"] and check.datasets["mrl_eye"]["via"] == ["teacher"]
    assert check.datasets["drive_and_act"]["via"] == ["training"]

    # A release whose source block predates upstream_datasets still reads them from its release check.
    stored = check_release(training_record(checkpoint)).summary()
    del config["source"]["upstream_datasets"], config["source"]["unrecorded_inputs"]
    (release / "config.json").write_text(json.dumps({**config, "release_check": stored}))
    assert set(training_record(release).upstream) == {"dmd", "mrl_eye"}
    assert training_record(release).upstream["dmd"]["via"] == ["teacher"]

    # Upstream inputs whose training data is not recorded travel through the source block too.
    config["source"].update(upstream_datasets={}, unrecorded_inputs=["teacher (records no training datasets)"])
    (release / "config.json").write_text(json.dumps(config))
    check = check_release(training_record(release))
    assert any("teacher (records no training datasets)" in problem for problem in check.problems)
