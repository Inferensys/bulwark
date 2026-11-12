from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bulwark.data.manifest import ClipRecord, read_manifest, write_manifest
from bulwark.data.taxonomy import default_taxonomy
from bulwark.eval import metrics as M
from bulwark.eval.predict import (
    Predictions,
    detect_kind,
    load_predictions,
    predict_features,
    write_predictions,
)
from bulwark.eval.report import VIDEO_LEVEL, _aggregate, build_report, score_split
from bulwark.model.heads import BulwarkHeads
from bulwark.model.pooling import NUM_SUMMARY_TOKENS


def rec(clip_id: str, dataset: str, subject: str, split: str, labels: dict, **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset=dataset, subject_id=f"{dataset}:{subject}", media_path=kwargs.pop("media_path", f"{clip_id}.mp4"),  # type: ignore[arg-type]
        media_type=kwargs.pop("media_type", "video"), licence="t", commercial_ok=False, split=split, labels_gt=labels, **kwargs,  # type: ignore[arg-type]
    )


def synthetic(tmp_path: Path) -> tuple[Path, dict[str, Predictions]]:
    """Two evaluable datasets (one multi-view), a video-labelled holdout, and train/val records."""
    rng = np.random.default_rng(0)
    records: list[ClipRecord] = []
    signal: dict[str, dict[str, float]] = {}
    for split, subjects in (("train", 6), ("val", 4), ("test", 12)):
        for s in range(subjects):
            for e in range(4):
                positive = (s + e) % 2
                for view in ("kinect_color", "ceiling"):  # one event, two cameras
                    cid = f"da/{split}/{s}/{e}/{view}"
                    records.append(rec(
                        cid, "drive_and_act", f"vp{split}{s}", split,
                        {"phone_at_ear": float(positive), "phone": 1.0, "activity": "phone_call" if positive else "attentive",
                         "context": "vehicle_cabin", "eyeglasses": float(s % 2)},
                        view=view, meta={"event_id": f"{split}/{s}/{e}"},
                    ))
                    signal[cid] = {"phone_at_ear": 4.0 * positive - 2 + rng.normal(), "eyeglasses": 3.0 * (s % 2) - 1.5}
                cid = f"mrl/{split}/{s}/{e}"
                records.append(rec(
                    cid, "mrl_eye", f"m{split}{s}", split,
                    {"eyes_closed": float(positive), "phone": float(e == 0), "low_light": float(e % 2), "eyeglasses": float(e < 2)},
                    labels_teacher={"yawn": 1.0},
                ))
                signal[cid] = {"eyes_closed": 3.0 * positive - 1.5 + rng.normal(), "eyeglasses": 3.0 * (e < 2) - 1.5}
    for s in range(3):  # teacher-only alertness supervision would not count; give two classes as GT
        records.append(rec(f"nthu/{s}", "nthu_ddd", f"n{s}", "train", {"alertness": "alert" if s else "drowsy"}))
    for s in range(10):
        for state, alert in (("0", "alert"), ("5", "low_vigilance")):
            for w in range(3):
                cid = f"uta/{s}/{state}/{w}"
                records.append(rec(
                    cid, "uta_rldd", f"u{s}", "holdout", {"alertness": alert, "kss": 0.5},
                    media_path=f"uta/{s}/{state}.mp4", meta={"label_quality": "video_level_self_report"},
                ))
                signal[cid] = {}
    manifest = write_manifest(records, tmp_path / "manifest.parquet")

    taxonomy = default_taxonomy()
    predictions = {}
    for split in ("val", "test", "holdout"):
        ids = [r.clip_id for r in records if r.split == split]
        logits = {}
        for name, spec in taxonomy.labels.items():
            shape = (len(ids), spec.width) if spec.kind == "categorical" else (len(ids),)
            logits[name] = np.zeros(shape, dtype=np.float32)
        for row, cid in enumerate(ids):
            for name, value in signal[cid].items():
                logits[name][row] = value
        predictions[split] = Predictions(ids, logits)
    return manifest, predictions


def test_report_scores_only_supervised_ground_truth_with_context(tmp_path: Path) -> None:
    manifest, predictions = synthetic(tmp_path)
    reference = {k: Predictions(v.clip_ids, {n: a + 0.1 for n, a in v.logits.items()}) for k, v in predictions.items()}
    path = build_report(predictions, manifest, tmp_path / "eval", reference=reference, bootstrap=100, meta={"model": "test"})
    metrics = json.loads((path.parent / "metrics.json").read_text())
    test = metrics["splits"]["test"]
    da, mrl = test["datasets"]["drive_and_act"], test["datasets"]["mrl_eye"]

    # no GT training labels: kss and yawn (teacher only) are listed, never scored
    assert "kss" in metrics["not_scored"] and "kss" not in test["summary"]
    assert "yawn" not in mrl and "yawn" not in metrics["not_scored"]  # teacher labels are not GT anywhere
    # metadata labels go to the sanity table only
    assert "context" not in da and "context" in test["sanity"]["drive_and_act"]
    # counts: two cameras per event, 12 subjects; the headline fuses the views of each event
    assert test["counts"]["drive_and_act"] == {"clips": 96, "subjects": 12, "events": 48, "event_ids_recorded": True}
    assert test["units"] == {"drive_and_act": "event", "mrl_eye": "clip"}
    entry = da["phone_at_ear"]
    assert entry["n"] == 48 and entry["n_subjects"] == 12 and entry["n_events"] == 48
    assert entry["auroc"] > 0.8 and entry["baseline"]["auroc"] == 0.5 and entry["ci"] is not None
    # Tuned on val clips of the clips' media type (all video here).
    fitted = metrics["calibration"]["phone_at_ear"]
    assert entry["threshold"] == {"video": fitted["media"]["video"]["threshold"]} and entry["threshold"]["video"] != 0.5
    assert entry["temperature"] == {"video": fitted["media"]["video"]["temperature"]}
    assert fitted["source"] == "val" and list(fitted["media"]) == ["video"]
    per_clip = test["per_clip"]["drive_and_act"]["phone_at_ear"]
    assert per_clip["n"] == 96 and per_clip["n_events"] == 48 and "mrl_eye" not in test["per_clip"]
    assert per_clip["threshold"] == entry["threshold"] and per_clip["temperature"] == entry["temperature"] and "ece_calibrated" in per_clip
    reliability = entry["reliability"]
    assert sum(reliability["count"]) == entry["n"] and len(reliability["edges"]) == len(reliability["count"]) + 1
    # one-sided in Drive&Act: AUROC undefined there, so the mean and the pooled column skip it
    assert da["phone"]["auroc"] is None and test["summary"]["phone"]["datasets"] == ["mrl_eye"]
    assert "phone" not in test["pooled"] and "eyeglasses" in test["pooled"]
    eyeglasses = test["summary"]["eyeglasses"]
    assert eyeglasses["datasets"] == ["drive_and_act", "mrl_eye"]
    assert abs(eyeglasses["value"] - np.mean([da["eyeglasses"]["auroc"], mrl["eyeglasses"]["auroc"]])) < 1e-9
    assert eyeglasses["ci"] is not None
    # slices by view and by condition (per clip)
    assert set(test["slices"]["drive_and_act"]) >= {"view=ceiling", "view=kinect_color"}
    assert "low_light=0.0" in test["slices"]["mrl_eye"]
    # holdout labelled per video: clip and video level; a class without training labels is flagged
    holdout = metrics["splits"]["holdout"]
    assert holdout["videos"]["uta_rldd"]["alertness"]["n"] == 20
    uta = holdout["datasets"]["uta_rldd"]["alertness"]
    assert uta["untrained_classes"] == ["low_vigilance"] and "macro_f1_trained_classes" in uta
    # holdout groups are never averaged; a mean over datasets carries the flag too
    assert holdout["summary"] == {} and "pooled" not in holdout
    records = {r.clip_id: r for r in read_manifest(manifest)}
    combined = score_split(
        predictions["holdout"], records, default_taxonomy(), split="holdout", scored=["alertness"], sanity=[], calibration={},
        priors={"*": {}}, trained={"alertness": [0, 2]}, bootstrap=0,
    )
    assert combined["summary"]["alertness"]["untrained_classes"] == ["low_vigilance"]
    assert metrics["training_support"]["alertness"]["classes_gt"]["low_vigilance"] == 0
    # reference model agreement and provenance
    assert test["reference"]["agreement"]["phone_at_ear"]["decision_agreement"] > 0.9
    assert len(metrics["meta"]["manifest"]["sha256"]) == 64 and "commit" in metrics["meta"]["scored_with"]["git"]
    assert any("record no code version" in w for w in metrics["meta"]["warnings"])  # no predicted_with here
    calibration = json.loads((path.parent / "calibration.json").read_text())
    assert calibration["meta"]["model"] == "test" and calibration["meta"]["manifest_sha256"] == metrics["meta"]["manifest"]["sha256"]
    assert calibration["labels"]["phone_at_ear"]["source"] == "val" and calibration["meta"]["val_clips"]["phone_at_ear"] > 0
    html = path.read_text()
    assert "Not scored" in html and "mean over datasets" in html and "Sanity checks" in html
    assert "Training support" in html and "Per clip" in html and "trained classes only" in html


def test_predictions_roundtrip_and_report_from_saved_files(tmp_path: Path) -> None:
    manifest, predictions = synthetic(tmp_path)
    write_predictions(predictions, tmp_path / "preds", {"model": "test"})
    loaded = load_predictions(tmp_path / "preds")
    assert set(loaded) == {"val", "test", "holdout"}
    assert np.array_equal(loaded["test"].logits["eyes_closed"], predictions["test"].logits["eyes_closed"])
    meta = json.loads((tmp_path / "preds" / "meta.json").read_text())
    assert meta["files"]["test"]["clips"] == len(predictions["test"].clip_ids) and len(meta["files"]["test"]["sha256"]) == 64
    without_val = {k: v for k, v in loaded.items() if k != "val"}
    path = build_report(without_val, manifest, tmp_path / "eval", bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    assert metrics["calibration"]["phone_at_ear"]["source"] == "default"
    assert "ci" not in metrics["splits"]["test"]["datasets"]["mrl_eye"]["eyes_closed"]


def test_checkpoint_kinds_and_mismatch_refusals(tmp_path: Path) -> None:
    taxonomy = default_taxonomy()
    heads = BulwarkHeads(16, taxonomy, input_tokens=NUM_SUMMARY_TOKENS)
    torch.save({"model": heads.state_dict()}, tmp_path / "heads.pt")
    torch.save({"model": {f"heads.{k}": v for k, v in heads.state_dict().items()}}, tmp_path / "b1.pt")
    torch.save({"model": {"frame_encoder.stem.weight": torch.zeros(1)}}, tmp_path / "e1.pt")
    assert detect_kind(tmp_path / "heads.pt") == "heads"
    assert detect_kind(tmp_path / "b1.pt") == "b1"
    assert detect_kind(tmp_path / "e1.pt") == "e1"
    assert detect_kind(tmp_path / "x.onnx") == "onnx" and detect_kind(None) == "logits"

    frozen, tuned = tmp_path / "frozen", tmp_path / "tuned"
    frozen.mkdir()
    tuned.mkdir()
    tokens = np.zeros((3, NUM_SUMMARY_TOKENS, 16), dtype=np.float16)
    ids = np.array(["a", "b", "c"])
    np.savez(frozen / "features-0.npz", clip_ids=ids, tokens=tokens)
    np.savez(tuned / "features-0.npz", clip_ids=ids, tokens=tokens, logit__yawn=np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="B1 checkpoint"):
        predict_features("heads", tmp_path / "b1.pt", frozen, {"a"})
    with pytest.raises(ValueError, match="fine-tuned encoder"):
        predict_features("heads", tmp_path / "heads.pt", tuned, {"a"})
    with pytest.raises(ValueError, match="no logit"):
        predict_features("logits", None, frozen, {"a"})
    assert predict_features("heads", tmp_path / "heads.pt", frozen, {"a", "c"}).clip_ids == ["a", "c"]
    cached = predict_features("logits", None, tuned, {"b"})
    assert cached.clip_ids == ["b"] and set(cached.logits) == {"yawn"}


def test_report_header_says_how_best_pt_was_chosen(tmp_path: Path) -> None:
    records = [rec(f"mrl/{split}/{s}", "mrl_eye", f"m{split}{s}", split, {"eyes_closed": float(s % 2)})
               for split in ("train", "test") for s in range(4)]
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    ids = [r.clip_id for r in records if r.split == "test"]
    predictions = {"test": Predictions(ids, {"eyes_closed": np.linspace(-1, 1, len(ids)).astype(np.float32)})}

    def header(select: dict) -> str:
        meta = {"model": "heads", "checkpoint": {"path": "best.pt", "sha256": "ab" * 32, "meta": {"step": 120, "select": select}}}
        page = build_report(predictions, manifest, tmp_path / str(len(select)), meta=meta, bootstrap=0).read_text()
        return page.split("<td>checkpoint chosen by</td><td>")[1].split("</td>")[0]

    assert header({"criterion": "val_select", "score": 0.81234}) == "val_select = 0.8123 on val, step 120"
    # select_folds: the score is a mean over held-out folds of train and val subjects, not a val score.
    assert header({"criterion": "val_select", "score": 0.7, "folds": 5, "epoch": 3}) == (
        "val_select = 0.7000, mean over 5 held-out folds of train and val subjects (epoch 3), step 120"
    )


def media_synthetic(tmp_path: Path, *, auc_test: bool = False) -> tuple[Path, dict[str, Predictions]]:
    """eyes_closed on MRL Eye images and DMD video, whose logits run low, as B1's do; with
    `auc_test`, phone_at_ear test images (AUC v2-like) while val has phone_at_ear on video only."""
    rng = np.random.default_rng(0)
    records: list[ClipRecord] = []
    signal: dict[str, dict[str, float]] = {}
    for split, subjects in (("train", 6), ("val", 6), ("test", 6)):
        for s in range(subjects):
            for e in range(6):
                closed = float(e % 3 == 0)
                cid = f"mrl/{split}/{s}/{e}"
                records.append(rec(cid, "mrl_eye", f"m{split}{s}", split, {"eyes_closed": closed},
                                   media_type="image", media_path=f"{cid}.png"))
                signal[cid] = {"eyes_closed": 3.0 * closed + 0.5 + rng.normal(scale=0.5)}
                cid = f"dmd/{split}/{s}/{e}"
                labels = {"eyes_closed": closed, "phone_at_ear": float(e % 2)}
                records.append(rec(cid, "dmd", f"d{split}{s}", split, labels, view="face" if e % 2 else "body"))
                signal[cid] = {"eyes_closed": 2.0 * closed - 4.0 + rng.normal(scale=0.5), "phone_at_ear": 3.0 * (e % 2) - 1.5}
    if auc_test:
        for s in range(6):
            for e in range(2):
                cid = f"auc/{s}/{e}"
                records.append(rec(cid, "auc_v2", f"a{s}", "test", {"phone_at_ear": float(e)}, media_type="image",
                                   media_path=f"{cid}.jpg"))
                signal[cid] = {"phone_at_ear": 3.0 * e - 1.5}
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    taxonomy = default_taxonomy()
    predictions = {}
    for split in ("val", "test"):
        ids = [r.clip_id for r in records if r.split == split]
        logits = {
            name: np.zeros((len(ids), spec.width) if spec.kind == "categorical" else len(ids), dtype=np.float32)
            for name, spec in taxonomy.labels.items()
        }
        for row, cid in enumerate(ids):
            for name, value in signal[cid].items():
                logits[name][row] = value
        predictions[split] = Predictions(ids, logits)
    return manifest, predictions


def test_each_clip_is_scored_with_the_threshold_and_temperature_of_its_media_type(tmp_path: Path) -> None:
    manifest, predictions = media_synthetic(tmp_path)
    path = build_report(predictions, manifest, tmp_path / "eval", bootstrap=0, meta={"model": "test"})
    metrics = json.loads((path.parent / "metrics.json").read_text())
    fitted = metrics["calibration"]["eyes_closed"]
    image, video = fitted["media"]["image"], fitted["media"]["video"]
    assert video["threshold"] < 0.2 < fitted["threshold"]  # video logits run low; the fit on all val clips suits images
    test = metrics["splits"]["test"]
    dmd, mrl = test["datasets"]["dmd"]["eyes_closed"], test["datasets"]["mrl_eye"]["eyes_closed"]
    assert dmd["threshold"] == {"video": video["threshold"]} and dmd["temperature"] == {"video": video["temperature"]}
    assert mrl["threshold"] == {"image": image["threshold"]}

    records = {r.clip_id: r for r in read_manifest(manifest)}
    dmd_ids = [c for c in predictions["test"].clip_ids if c.startswith("dmd/")]
    rows = [predictions["test"].clip_ids.index(c) for c in dmd_ids]
    p = M.sigmoid(predictions["test"].logits["eyes_closed"][rows].astype(np.float64))
    y = np.array([int(records[c].labels_gt["eyes_closed"]) for c in dmd_ids])
    assert dmd["f1"] == M.f1((p >= video["threshold"]).astype(int), y) and dmd["f1"] > 0.5
    assert M.f1((p >= fitted["threshold"]).astype(int), y) == 0.0  # what one threshold for all clips gave
    assert dmd["ece_calibrated"] == M.ece_binary(M.sigmoid(np.log(p / (1 - p)) / video["temperature"]), y)

    # The pooled column scores each clip with its own media type's threshold, in one F1.
    assert test["pooled"]["eyes_closed"]["threshold"] == {"image": image["threshold"], "video": video["threshold"]}
    assert set(test["pooled"]["eyes_closed"]["temperature"]) == {"image", "video"}
    # Condition slices use the media type's values too.
    sliced = test["slices"]["dmd"]["view=face"]["eyes_closed"]
    assert sliced["threshold"] == {"video": video["threshold"]}

    calibration = json.loads((path.parent / "calibration.json").read_text())
    assert calibration["labels"]["eyes_closed"]["media"] == fitted["media"]
    assert calibration["meta"]["val_clips"]["eyes_closed"] == fitted["n_val"] == image["n_val"] + video["n_val"]
    page = path.read_text()
    assert "Thresholds and temperatures (fitted on val)" in page and "<th>all val clips</th><th>image</th><th>video</th>" in page
    assert f"{video['threshold']:.3f} <small>(val)</small>" in page
    # The calibration table names the temperature each dataset's clips were scored with.
    assert f"Brier {dmd['brier']:.3f} · T {video['temperature']:.2f}</small>" in page
    assert f"Brier {mrl['brier']:.3f} · T {image['temperature']:.2f}</small>" in page
    assert not any("Val has" in w for w in metrics["meta"]["warnings"])


def test_a_calibration_without_media_entries_applies_to_every_media_type(tmp_path: Path) -> None:
    manifest, predictions = media_synthetic(tmp_path)
    records = {r.clip_id: r for r in read_manifest(manifest)}
    # A calibration.json written before media entries: one fit for all clips.
    old = {"eyes_closed": {"threshold": 0.8, "threshold_from": "val", "temperature": 1.5, "source": "val", "n_val": 72}}
    result = score_split(
        predictions["test"], records, default_taxonomy(), split="test", scored=["eyes_closed"], sanity=[],
        calibration=old, priors={"*": {}}, bootstrap=0,
    )
    assert result["datasets"]["dmd"]["eyes_closed"]["threshold"] == {"video": 0.8}
    assert result["datasets"]["mrl_eye"]["eyes_closed"]["temperature"] == {"image": 1.5}
    assert result["pooled"]["eyes_closed"]["threshold"] == {"image": 0.8, "video": 0.8}
    assert result["datasets"]["dmd"]["eyes_closed"]["f1"] == 0.0


def test_events_clips_videos_and_slices_are_scored_with_the_values_of_their_media_type() -> None:
    # Two cameras per event and labels per video: every calibrated table takes the video
    # entry, not the fit on all val clips (which, as for B1's eyes_closed, flags no video clip).
    rng = np.random.default_rng(0)
    records, raw = [], []
    for s in range(6):
        for e in range(4):
            closed = float(e % 2)
            value = 2.0 * closed - 4.0 + rng.normal(scale=0.5)
            for view in ("face", "body"):
                records.append(rec(f"dmd/{s}/{e}/{view}", "dmd", f"d{s}", "test", {"eyes_closed": closed}, view=view,
                                   media_path=f"dmd/{s}/{e}.mp4", meta={"event_id": f"{s}/{e}", "label_quality": VIDEO_LEVEL}))
                raw.append(value + rng.normal(scale=0.1))
    fit = {"threshold_from": "val", "source": "val", "n_val": 40}
    calibration = {"eyes_closed": {**fit, "threshold": 0.8, "temperature": 1.5, "media": {
        "video": {**fit, "threshold": 0.1, "temperature": 3.0}, "image": {**fit, "threshold": 0.8, "temperature": 1.5},
    }}}
    preds = Predictions([r.clip_id for r in records], {"eyes_closed": np.array(raw, dtype=np.float32)})
    result = score_split(
        preds, {r.clip_id: r for r in records}, default_taxonomy(), split="test", scored=["eyes_closed"], sanity=[],
        calibration=calibration, priors={"*": {}}, bootstrap=0,
    )
    assert result["units"]["dmd"] == "event"
    tables = {
        "event": result["datasets"]["dmd"], "clip": result["per_clip"]["dmd"], "video": result["videos"]["dmd"],
        "slice": result["slices"]["dmd"]["view=face"],
    }
    for table, entries in tables.items():
        entry = entries["eyes_closed"]
        assert entry["threshold"] == {"video": 0.1} and entry["temperature"] == {"video": 3.0}, table
        assert "ece_calibrated" in entry, table
    z = preds.logits["eyes_closed"].astype(np.float64)
    y = np.array([int(r.labels_gt["eyes_closed"]) for r in records])
    clip = tables["clip"]["eyes_closed"]
    assert clip["n"] == 48 and clip["f1"] == M.f1((M.sigmoid(z) >= 0.1).astype(int), y) > 0.5
    assert M.f1((M.sigmoid(z) >= 0.8).astype(int), y) == 0.0
    assert clip["ece_calibrated"] == M.ece_binary(M.sigmoid(z / 3.0), y)


def test_test_clips_of_a_media_type_without_val_clips_get_no_fit_and_a_warning(tmp_path: Path) -> None:
    manifest, predictions = media_synthetic(tmp_path, auc_test=True)
    path = build_report(predictions, manifest, tmp_path / "eval", bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    assert list(metrics["calibration"]["phone_at_ear"]["media"]) == ["video"]
    auc = metrics["splits"]["test"]["datasets"]["auc_v2"]["phone_at_ear"]
    # Never the video fit: images are scored at 0.5, without a temperature.
    assert auc["threshold"] == {"image": 0.5} and auc["temperature"] == {"image": None} and "ece_calibrated" not in auc
    warnings = [w for w in metrics["meta"]["warnings"] if w.startswith("Val has no")]
    assert warnings == [
        "Val has no image clips with ground truth for phone_at_ear, so no threshold or temperature was fitted for its "
        "image clips: they are scored at the 0.5 threshold and without a temperature."
    ]
    page = path.read_text()
    row = page.split("<tr><td>phone_at_ear</td>")[1].split("</tr>")[0]
    assert "<td>no val clips of this media type: 0.5, no temperature</td>" in row
    # The checkpoint masked phone_at_ear on images: the masked warning covers it instead.
    meta = {"checkpoint": {"path": "b1.pt", "sha256": "ab" * 32, "meta": {"masked_labels": {"phone_at_ear": ["image"]}}}}
    path = build_report(predictions, manifest, tmp_path / "masked", bootstrap=0, meta=meta)
    warnings = json.loads((path.parent / "metrics.json").read_text())["meta"]["warnings"]
    assert not any(w.startswith("Val has no") for w in warnings) and any("phone_at_ear was masked on image" in w for w in warnings)


def test_media_types_with_too_few_or_no_val_clips_get_no_fit_and_a_warning(tmp_path: Path) -> None:
    manifest, predictions = media_synthetic(tmp_path, auc_test=True)
    records = read_manifest(manifest)
    for r in records:
        if r.split == "val":
            # phone_at_ear ground truth from AUC v2 only, which has no val clips; eyes_closed
            # on six DMD val clips of one subject (2 closed).
            r.labels_gt.pop("phone_at_ear", None)
            if r.dataset == "dmd" and r.subject_id != "dmd:dval0":
                r.labels_gt.pop("eyes_closed")
    write_manifest(records, manifest)
    path = build_report(predictions, manifest, tmp_path / "eval", bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    eyes = metrics["calibration"]["eyes_closed"]
    video = eyes["media"]["video"]
    assert (video["threshold"], video["threshold_from"], video["temperature"]) == (0.5, "default", None)
    assert (video["n_val"], video["n_val_positive"], video["not_fitted"]) == (6, 2, "fewer than 10 clips of each of two classes")
    assert eyes["threshold_from"] == "val" and "not_fitted" not in eyes["media"]["image"]
    assert metrics["calibration"]["phone_at_ear"]["media"] == {}
    test = metrics["splits"]["test"]["datasets"]
    # Never the fit on all val clips: video clips without a video fit are scored at 0.5, without a temperature.
    dmd = test["dmd"]["eyes_closed"]
    assert dmd["threshold"] == {"video": 0.5} and dmd["temperature"] == {"video": None} and "ece_calibrated" not in dmd
    assert test["auc_v2"]["phone_at_ear"]["threshold"] == {"image": 0.5} and test["dmd"]["phone_at_ear"]["temperature"] == {"video": None}
    scored_without = "so no threshold or temperature was fitted for its {} clips: they are scored at the 0.5 threshold and without a temperature."
    assert sorted(w for w in metrics["meta"]["warnings"] if w.startswith("Val has")) == [
        "Val has fewer than 10 clips of each of two classes among its video clips with ground truth for eyes_closed, "
        + scored_without.format("video"),
        f"Val has no image clips with ground truth for phone_at_ear, {scored_without.format('image')}",
        f"Val has no video clips with ground truth for phone_at_ear, {scored_without.format('video')}",
    ]
    page = path.read_text()
    row = page.split("<tr><td>eyes_closed</td>")[1].split("</tr>")[0]
    assert "<td>not fitted (fewer than 10 clips of each of two classes): 0.5, no temperature<br><small>6 clips · 2 positive · 1 subjects</small></td>" in row
    # Without val predictions the header says every threshold is the default; no warning per label.
    path = build_report({"test": predictions["test"]}, manifest, tmp_path / "noval", bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    assert not any(w.startswith("Val has") for w in metrics["meta"]["warnings"])
    assert metrics["meta"]["threshold_source"] == "default 0.5 (no val predictions)"


def test_events_and_videos_never_mix_media_types() -> None:
    taxonomy = default_taxonomy()
    records = [
        rec("a/0", "drive_and_act", "s", "test", {}, meta={"event_id": "e"}),
        rec("a/1", "drive_and_act", "s", "test", {}, meta={"event_id": "e"}, media_type="image"),
    ]
    logits = {"eyes_closed": np.zeros(2, dtype=np.float32)}
    with pytest.raises(ValueError, match="e groups clips of several media types"):
        _aggregate(logits, records, taxonomy, M.event_id)
    fused, kept = _aggregate(logits, records[:1], taxonomy, M.event_id)
    assert kept[0].media_type == "video" and fused["eyes_closed"].shape == (1,)
