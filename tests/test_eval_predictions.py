from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.data.shards import ShardWriter
from bulwark.data.taxonomy import default_taxonomy
from bulwark.eval.predict import (
    Predictions,
    load_predictions,
    predict_features,
    shard_batches,
    write_predictions,
)
from bulwark.eval.provenance import sha256_file
from bulwark.eval.report import evaluate
from bulwark.model.heads import BulwarkHeads
from bulwark.model.pooling import NUM_SUMMARY_TOKENS


def preds(ids: list[str], value: float) -> Predictions:
    return Predictions(ids, {"yawn": np.full(len(ids), value, dtype=np.float32)})


def test_rerun_with_fewer_splits_drops_the_earlier_files(tmp_path: Path) -> None:
    directory = tmp_path / "predictions"
    write_predictions({"val": preds(["v"], 1.0), "test": preds(["t"], 1.0), "holdout": preds(["h"], 1.0)}, directory, {"model": "b1"})
    write_predictions({"test": preds(["t"], 2.0)}, directory, {"model": "e1"})
    loaded = load_predictions(directory)
    assert set(loaded) == {"test"} and loaded["test"].logits["yawn"][0] == 2.0
    assert sorted(p.name for p in directory.glob("*.npz")) == ["test.npz"]


def test_unlisted_or_changed_prediction_files_are_refused(tmp_path: Path) -> None:
    directory = tmp_path / "predictions"
    write_predictions({"test": preds(["t"], 1.0)}, directory, {"model": "e1"})
    preds(["h"], 0.0).save(directory / "holdout.npz")  # copied in from another run
    with pytest.raises(ValueError, match="does not list"):
        load_predictions(directory)
    (directory / "holdout.npz").unlink()
    preds(["t"], 5.0).save(directory / "test.npz")  # overwritten after meta.json was written
    with pytest.raises(ValueError, match="differs"):
        load_predictions(directory)


def test_shard_batches_read_wanted_clips_with_frames_in_numeric_order(tmp_path: Path) -> None:
    frames = np.stack([np.full((8, 8, 3), 2 * i, dtype=np.uint8) for i in range(110)])  # names 00 .. 109
    with ShardWriter(tmp_path, prefix="t") as writer:
        writer.add("long", frames)
        writer.add("other", frames[:2])
    [(ids, batch)] = list(shard_batches([tmp_path / "t-00000.tar"], {"long"}, batch_size=4))
    assert ids == ["long"] and np.abs(batch[0, :, 0, 0, 0].astype(int) - 2 * np.arange(110)).max() <= 2


# ---- cached features and the checkpoint behind them -------------------------------------------


def record(clip_id: str, split: str, commercial: bool, labels: dict) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset="mrl_eye" if commercial else "drive_and_act", subject_id=f"s:{clip_id}",
        media_path=f"{clip_id}.jpg", media_type="image", licence="t", commercial_ok=commercial, split=split, labels_gt=labels,
    )


def features_dir(path: Path, ids: list[str], init: dict | None) -> Path:
    path.mkdir()
    tokens = np.zeros((len(ids), NUM_SUMMARY_TOKENS, 8), dtype=np.float16)
    arrays = {"logit__eyes_closed": np.linspace(-2, 2, len(ids)).astype(np.float32)} if init else {}
    # Laid out as `bulwark train cache` writes it: one file per shard listed in meta.json.
    init_sha = np.array((init or {}).get("sha256") or "")
    np.savez(path / "features-t.npz", clip_ids=np.array(ids), tokens=tokens, init_sha256=init_sha, **arrays)
    meta = {"run": {"init": init, "with_logits": bool(init), "precision": "float16"}, "init_path": "/home/someone/b1/final.pt" if init else None,
            "commercial_only": bool(init and init.get("commercial_only")), "code": {"commit": "abc"}, "shards": {"t.tar": "x"}}
    (path / "meta.json").write_text(json.dumps(meta))
    return path


def test_logits_must_come_from_the_named_checkpoint(tmp_path: Path) -> None:
    torch.save({"model": {"heads.x": torch.zeros(1)}}, tmp_path / "b1.pt")
    torch.save({"model": {"heads.x": torch.ones(1)}}, tmp_path / "other.pt")
    init = {"sha256": sha256_file(tmp_path / "b1.pt"), "has_provenance": True, "commercial_only": True, "datasets": {"mrl_eye": {}}}
    cached = features_dir(tmp_path / "b1-features", ["a", "b"], init)
    with pytest.raises(ValueError, match="cached from a checkpoint"):
        predict_features("logits", tmp_path / "other.pt", cached, {"a"})
    assert predict_features("logits", tmp_path / "b1.pt", cached, {"a"}).clip_ids == ["a"]

    # frozen heads: tokens from a dir whose meta.json records a fine-tuned init are refused
    torch.save({"model": BulwarkHeads(8, default_taxonomy(), input_tokens=NUM_SUMMARY_TOKENS).state_dict()}, tmp_path / "heads.pt")
    frozen = features_dir(tmp_path / "frozen", ["a"], None)
    assert predict_features("heads", tmp_path / "heads.pt", frozen, {"a"}).clip_ids == ["a"]
    tuned = tmp_path / "tuned"
    tuned.mkdir()
    np.savez(tuned / "features-0.npz", clip_ids=np.array(["a"]), tokens=np.zeros((1, NUM_SUMMARY_TOKENS, 8), dtype=np.float16))
    (tuned / "meta.json").write_text(json.dumps({"run": {"init": init, "with_logits": False}}))
    with pytest.raises(ValueError, match="fine-tuned init"):
        predict_features("heads", tmp_path / "heads.pt", tuned, {"a"})


def test_logits_without_checkpoint_take_scope_and_provenance_from_the_features(tmp_path: Path) -> None:
    records = [
        record("tr-clean-0", "train", True, {"eyes_closed": 0.0}),
        record("tr-clean-1", "train", True, {"eyes_closed": 1.0}),
        record("tr-research", "train", False, {"eyes_closed": 1.0}),
        *(record(f"te-{i}", "test", True, {"eyes_closed": float(i % 2)}) for i in range(6)),
    ]
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    torch.save({"model": {"heads.x": torch.zeros(1)}}, tmp_path / "b1-clean.pt")
    init = {"sha256": sha256_file(tmp_path / "b1-clean.pt"), "has_provenance": True, "commercial_only": True,
            "datasets": {"mrl_eye": {"licence": "t", "commercial_ok": True}}}
    cached = features_dir(tmp_path / "features", [f"te-{i}" for i in range(6)], init)

    path = evaluate("logits", None, manifest=manifest, out=tmp_path / "eval", features=cached, splits=["test"], bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    meta = metrics["meta"]
    assert meta["checkpoint"]["sha256"] == init["sha256"] and meta["checkpoint"]["path"] == "final.pt"
    assert meta["training_scope"] == {"commercial_only": True, "datasets": ["mrl_eye"]}
    assert metrics["training_support"]["eyes_closed"]["gt"] == 2  # the research-only clip is outside the clean scope
    assert meta["features"]["init"]["path"] == "final.pt" and "/home/" not in json.dumps(metrics)

    saved = json.loads((tmp_path / "eval" / "predictions" / "meta.json").read_text())
    made = saved["predicted_with"]
    assert made["manifest"]["sha256"] == sha256_file(manifest) and made["manifest"]["path"] == "manifest.parquet"
    assert "commit" in made["git"] and made["taxonomy"]["digest"] and made["split_policy"]["source"].startswith("code default")
    assert meta["predicted_with"] == made and meta["scored_with"]["created_utc"]
    assert not [w for w in meta["warnings"] if "manifest" in w]
