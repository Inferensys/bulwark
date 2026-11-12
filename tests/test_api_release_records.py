"""Records that travel with a model: calibration.json from `bulwark evaluate`, and upstream datasets in releases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bulwark.data.manifest import ClipRecord
from bulwark.data.taxonomy import default_taxonomy
from bulwark.eval.predict import Predictions
from bulwark.eval.report import calibrate, calibration_file
from bulwark.export.release import training_record
from bulwark.inference import load, read_temperatures, release_source, save_release
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint

EDGE = {"backbone": "test_efficientnet", "img_size": 64}


def edge_checkpoint(path: Path, **extra: object) -> Path:
    torch.manual_seed(0)
    return save_checkpoint(path, BulwarkEdge(pretrained=False, **EDGE), {"step": 1, **EDGE, **extra})


def evaluate_calibration(path: Path, checkpoint_sha256: str) -> dict:
    """calibration.json exactly as `bulwark evaluate` writes it, from val logits of yawn."""
    rng = np.random.default_rng(0)
    targets = np.arange(60) % 2
    records = {
        f"dmd/{i}": ClipRecord(
            clip_id=f"dmd/{i}", dataset="dmd", subject_id=f"dmd:{i % 6}", media_path="x.mp4", media_type="video",
            licence="test", commercial_ok=False, split="val", labels_gt={"yawn": float(targets[i])},
        )
        for i in range(60)
    }
    logits = {"yawn": (4.0 * targets - 2.0 + rng.normal(size=60)).astype(np.float32), "phone": np.zeros(60, np.float32)}
    entries = calibrate(Predictions(list(records), logits), records, default_taxonomy(), ["yawn", "phone"])
    meta = {
        "model": "e1",
        "checkpoint": {"path": "e1.pt", "sha256": checkpoint_sha256},
        "manifest": {"sha256": "ab" * 32},
        "scored_with": {"git": {"commit": "0123abc"}},
    }
    data = calibration_file(meta, entries)
    path.write_text(json.dumps(data))
    return data


def test_load_applies_the_calibration_file_bulwark_evaluate_writes(tmp_path: Path) -> None:
    checkpoint = edge_checkpoint(tmp_path / "e1.pt")
    sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    data = evaluate_calibration(tmp_path / "calibration.json", sha256)
    yawn = data["labels"]["yawn"]
    assert yawn["threshold_from"] == "val" and yawn["temperature"] is not None and data["labels"]["phone"]["temperature"] is None

    model = load(checkpoint, device="cpu", temperature=tmp_path / "calibration.json")
    assert model.temperature == {"yawn": yawn["temperature"]} and model.raw_thresholds == {"yawn": yawn["threshold"]}
    assert read_temperatures(tmp_path / "calibration.json") == {"yawn": yawn["temperature"]}

    evaluate_calibration(tmp_path / "other.json", "cd" * 32)
    with pytest.raises(ValueError, match="fitted on weights with sha256 cdcdcdcdcdcd"):
        load(checkpoint, device="cpu", temperature=tmp_path / "other.json")


def test_calibration_per_media_type_round_trips_from_calibrate_to_predict(tmp_path: Path) -> None:
    """calibration.json from `calibrate` and `calibration_file`, read back by `load`: each
    media type gets its own fit, and a label with val clips of one media type only gets
    nothing on the other."""
    rng = np.random.default_rng(1)
    records, yawn, phone = {}, [], []
    for i in range(80):
        media = "image" if i < 40 else "video"
        y = float(i % 2)
        records[f"c/{i}"] = ClipRecord(
            clip_id=f"c/{i}", dataset="auc_v2" if media == "image" else "dmd", subject_id=f"s:{i % 8}",
            media_path=f"{i}.{'jpg' if media == 'image' else 'mp4'}", media_type=media, licence="test", commercial_ok=False,
            split="val", labels_gt={"yawn": y, **({"phone": y} if media == "video" else {})},
        )
        centre = 2.0 if media == "image" else -2.0  # confidence differs between the media types
        yawn.append(centre + 2.0 * y - 1.0 + rng.normal(scale=0.7))
        phone.append(3.0 * y - 1.5 + rng.normal())
    logits = {"yawn": np.array(yawn, np.float32), "phone": np.array(phone, np.float32)}
    entries = calibrate(Predictions(list(records), logits), records, default_taxonomy(), ["yawn", "phone"])
    checkpoint = edge_checkpoint(tmp_path / "e1.pt")
    meta = {
        "model": "e1",
        "checkpoint": {"path": "e1.pt", "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "manifest": {"sha256": "ab" * 32},
        "scored_with": {"git": {"commit": "0123abc"}},
    }
    (tmp_path / "calibration.json").write_text(json.dumps(calibration_file(meta, entries)))
    model = load(checkpoint, device="cpu", temperature=tmp_path / "calibration.json")
    for media in ("image", "video"):
        fitted = entries["yawn"]["media"][media]
        assert model.temperature_for(media)["yawn"] == fitted["temperature"]
        assert model.calibration.for_media(media).thresholds["yawn"] == fitted["threshold"]
    assert entries["yawn"]["media"]["video"]["threshold"] < entries["yawn"]["media"]["image"]["threshold"]
    assert "phone" in model.temperature_for("video") and "phone" not in model.temperature_for("image")
    assert model.calibration.for_media("image").no_val_clips == {"phone"}
    assert model.temperature == {name: entries[name]["temperature"] for name in ("yawn", "phone")}


def distilled(path: Path, *, teacher_ok: bool | None, commercial_only: bool = True) -> Path:
    """An E1 from a commercial_only run whose distillation teacher (a B1) was trained on `teacher_set`.

    Dataset names and their commercial_ok values are test fixtures, not registry entries.
    """
    teacher = {"has_provenance": True, "with_logits": True, "init_sha256": "ef" * 32, "path": "/data/features/b1"}
    if teacher_ok is not None:
        teacher["datasets"] = {"teacher_set": {"licence": "research only", "commercial_ok": teacher_ok, "splits": {"train": 8}}}
    provenance = {
        "job": "distill",
        "commercial_only": commercial_only,
        "datasets": {"own_set": {"licence": "test", "commercial_ok": True, "splits": {"train": 4}}},
        "inputs": {"teacher": teacher},
        "code": {"commit": "0123abc", "dirty": False},
    }
    return edge_checkpoint(path, supervised_labels=["yawn"], provenance=provenance)


def test_releases_publish_upstream_datasets_and_count_them_for_the_licence(tmp_path: Path) -> None:
    model = load(distilled(tmp_path / "e1.pt", teacher_ok=False), device="cpu")
    source = release_source(model)
    assert source["upstream_datasets"] == {"teacher_set": {"via": ["teacher"], "licence": "research only", "commercial_ok": False}}
    assert source["unrecorded_inputs"] == []
    with pytest.raises(ValueError, match="not commercially usable: teacher_set"):
        save_release(model, tmp_path / "commercial", weights_licence="Apache-2.0")
    release = save_release(model, tmp_path / "release", weights_licence="CC BY-NC 4.0 (research use only)")
    config = json.loads((release / "config.json").read_text())
    assert config["source"]["upstream_datasets"] == source["upstream_datasets"] and "/data/" not in json.dumps(config)
    # A release of a release keeps them, and refuses a commercial licence the same way.
    again = load(release, device="cpu")
    assert release_source(again)["upstream_datasets"] == source["upstream_datasets"]
    with pytest.raises(ValueError, match="teacher_set"):
        save_release(again, tmp_path / "again", weights_licence="Apache-2.0")

    unrecorded = load(distilled(tmp_path / "unrecorded.pt", teacher_ok=None), device="cpu")
    assert release_source(unrecorded)["unrecorded_inputs"] == ["teacher (records no training datasets)"]
    with pytest.raises(ValueError, match="upstream inputs whose training data is not recorded"):
        save_release(unrecorded, tmp_path / "unrecorded", weights_licence="Apache-2.0")

    clean = load(distilled(tmp_path / "clean.pt", teacher_ok=True), device="cpu")
    save_release(clean, tmp_path / "clean", weights_licence="Apache-2.0")
    assert training_record(tmp_path / "clean.pt").upstream["teacher_set"]["commercial_ok"] is True
