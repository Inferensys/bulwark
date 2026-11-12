"""The split policy record across `split` and `label merge`, the labels a checkpoint records
as trained or masked (report, ONNX metadata), val thresholds in exported graphs, rolling
PERCLOS in `bulwark predict`, empty holdout groups, and datasets the probe leaves out."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import plain
from typer.testing import CliRunner

from bulwark.cli import app
from bulwark.data.manifest import ClipRecord, read_manifest, write_manifest
from bulwark.data.taxonomy import default_taxonomy
from bulwark.eval.predict import Predictions
from bulwark.eval.probe import cross_validate
from bulwark.eval.provenance import sha256_file, split_policy
from bulwark.eval.report import build_report, evaluate
from bulwark.export.onnx import read_contract
from bulwark.model.pooling import NUM_SUMMARY_TOKENS

runner = CliRunner()
REPO = Path(__file__).resolve().parents[1]


def clip(clip_id: str, dataset: str, subject: str, split: str, labels: dict | None = None, **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset=dataset, subject_id=f"{dataset}:{subject}", media_path=kwargs.pop("media_path", f"{clip_id}.mp4"),  # type: ignore[arg-type]
        media_type=kwargs.pop("media_type", "video"), licence="t", commercial_ok=False, split=split,  # type: ignore[arg-type]
        labels_gt=labels or {}, **kwargs,  # type: ignore[arg-type]
    )


def test_split_records_its_policy_and_label_merge_keeps_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "manifests").mkdir()
    dmd = [clip(f"dmd/{s}", "dmd", f"g{s}", "unassigned", {"eyes_closed": float(s % 2)}) for s in range(12)]
    zenodo = [clip("zenodo_driver/a", "zenodo_driver", "a", "unassigned", {"phone": 1.0}, media_type="image", media_path="a.jpg")]
    write_manifest(dmd, tmp_path / "manifests" / "dmd.parquet")
    write_manifest(zenodo, tmp_path / "manifests" / "zenodo_driver.parquet")
    result = runner.invoke(app, ["split", "--manifests", "manifests", "--out", "m.parquet", "--no-dedup"])
    assert result.exit_code == 0, result.output
    manifest = tmp_path / "m.parquet"
    assert {r.clip_id: r.split for r in read_manifest(manifest)}["zenodo_driver/a"] == "excluded"
    recorded = split_policy(manifest)
    assert recorded["source"] == "manifest" and recorded["excluded_datasets"] == ["zenodo_driver"]

    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "objects.jsonl").write_text(json.dumps({"clip_id": "dmd/0", "labels": {"phone": 0.0}}) + "\n")
    result = runner.invoke(app, ["label", "merge", "--manifest", "m.parquet", "--labels", "labels"])
    assert result.exit_code == 0, result.output
    assert split_policy(manifest) == recorded  # the policy survives the rewrite
    assert {r.clip_id: r for r in read_manifest(manifest)}["dmd/0"].labels_teacher == {"phone": 0.0}


def report_manifest(tmp_path: Path) -> tuple[Path, dict[str, Predictions]]:
    """Drive&Act-like clips: phone and eyes_closed ground truth in train and test, yawn in
    train only, and the held-out view's clips of one test subject (no UTA-RLDD)."""
    records = []
    for split, subjects in (("train", 4), ("test", 6)):
        for s in range(subjects):
            for e in range(2):
                labels = {"eyes_closed": float(e), "phone": float((s + e) % 2), **({"yawn": float(e)} if split == "train" else {})}
                records.append(clip(f"da/{split}/{s}/{e}", "drive_and_act", f"vp{split}{s}", split, labels, view="kinect_color"))
    records.append(clip("da/holdout/0", "drive_and_act", "vptest0", "holdout", {"eyes_closed": 1.0},
                        view="inner_mirror", meta={"subject_split": "test"}))
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    predictions = {}
    for split in ("test", "holdout"):
        ids = [r.clip_id for r in records if r.split == split]
        logits = {}
        for name, spec in default_taxonomy().labels.items():
            logits[name] = np.zeros((len(ids), spec.width) if spec.kind == "categorical" else len(ids), dtype=np.float32)
        logits["eyes_closed"] = np.array([2.0 if r.labels_gt["eyes_closed"] else -2.0 for r in records if r.split == split], dtype=np.float32)
        predictions[split] = Predictions(ids, logits)
    return manifest, predictions


def test_report_scores_only_what_the_checkpoint_trained_and_lists_empty_holdout_groups(tmp_path: Path) -> None:
    manifest, predictions = report_manifest(tmp_path)
    checkpoint = {"path": "b1.pt", "sha256": "ab" * 32,
                  "meta": {"supervised_labels": ["eyes_closed"], "masked_labels": {"eyes_closed": ["image"]}}}
    path = build_report(predictions, manifest, tmp_path / "eval", meta={"model": "b1", "checkpoint": checkpoint}, bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    test = metrics["splits"]["test"]["datasets"]["drive_and_act"]
    assert test["eyes_closed"]["auroc"] == 1.0 and "phone" not in test
    assert metrics["not_scored"]["phone"] == "not trained by this checkpoint (not in its supervised_labels)"
    # yawn has training ground truth but no test clips: still not scored, and calibration.json
    # holds no threshold for it that `bulwark export --calibration` could carry into a graph.
    assert list(json.loads((path.parent / "calibration.json").read_text())["labels"]) == ["eyes_closed"]
    assert "yawn" not in metrics["not_scored"]
    assert metrics["checkpoint_labels"] == {"supervised_labels": ["eyes_closed"], "masked_labels": {"eyes_closed": ["image"]}}
    assert any("eyes_closed was masked on image clips" in w for w in metrics["meta"]["warnings"])
    page = path.read_text()
    assert "<td>yes; masked on image</td>" in page and "<th>trained</th>" in page
    # UTA-RLDD is not ingested: the holdout split says so instead of leaving it out.
    assert "<td>uta_rldd</td><td>0</td><td colspan=3>not available (not ingested)</td>" in page

    # Without a record of trained labels, every label with ground truth is scored, as before.
    path = build_report(predictions, manifest, tmp_path / "plain", bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    assert "phone" in metrics["splits"]["test"]["datasets"]["drive_and_act"] and metrics["checkpoint_labels"] is None
    assert "<td>not recorded</td>" in path.read_text()


def features(path: Path, ids: list[str], *, init: dict | None = None, meta: dict | None = None) -> Path:
    """A features dir as `bulwark train cache` writes it (one shard), with logits when `init` is set."""
    path.mkdir()
    arrays = {"logit__eyes_closed": np.linspace(-2, 2, len(ids)).astype(np.float32)} if init else {}
    np.savez(path / "features-t.npz", clip_ids=np.array(ids), tokens=np.zeros((len(ids), NUM_SUMMARY_TOKENS, 8), dtype=np.float16),
             init_sha256=np.array((init or {}).get("sha256") or ""), **arrays)
    record = {"run": {"init": init, "with_logits": bool(init)}, "shards": {"t.tar": "x"}, **(meta or {})}
    (path / "meta.json").write_text(json.dumps(record))
    return path


def test_cached_logits_carry_the_init_labels_and_nonfinite_clips_are_reported(tmp_path: Path) -> None:
    records = [clip(f"te-{i}", "mrl_eye", f"s{i}", "test", {"eyes_closed": float(i % 2)}) for i in range(7)]
    records.append(clip("tr-0", "mrl_eye", "t", "train", {"eyes_closed": 1.0}))
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    init = {"sha256": "cd" * 32, "has_provenance": True, "datasets": {"mrl_eye": {}}}
    cached = features(tmp_path / "features", [f"te-{i}" for i in range(6)], init=init,
                      meta={"supervised_labels": ["eyes_closed"], "masked_labels": {}, "nonfinite": {"t.tar": ["te-6"]}})
    path = evaluate("logits", None, manifest=manifest, out=tmp_path / "eval", features=cached, splits=["test"], bootstrap=0)
    metrics = json.loads((path.parent / "metrics.json").read_text())
    assert metrics["meta"]["nonfinite_clips"] == {"test": 1} and metrics["meta"]["unpredicted_clips"] == {"test": 1}
    assert metrics["checkpoint_labels"]["supervised_labels"] == ["eyes_closed"]
    assert "non-finite features" in path.read_text()


def test_probe_cv_leaves_out_excluded_and_train_only_datasets(tmp_path: Path) -> None:
    records = [clip("zenodo_driver/a", "zenodo_driver", "a", "excluded", {"phone": 1.0}),
               clip("cew/a", "cew", "a", "train", {"eyes_closed": 1.0})]
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    frozen = features(tmp_path / "frozen", ["zenodo_driver/a", "cew/a"])
    results = cross_validate(frozen, manifest, datasets=["zenodo_driver", "cew"], folds=2)
    assert results["zenodo_driver"]["skipped"].startswith("excluded by the split policy")
    assert results["cew"]["skipped"].startswith("train-only")


# ---- an E1 checkpoint that records trained and masked labels ---------------------------------


@pytest.fixture(scope="module")
def e1(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from bulwark.model.bulwark import BulwarkEdge, save_checkpoint

    root = tmp_path_factory.mktemp("e1")
    torch.manual_seed(0)
    model = BulwarkEdge(default_taxonomy(), pretrained=False, img_size=64)
    extra = {"img_size": 64, "supervised_labels": ["eyes_closed", "perclos"], "masked_labels": {"perclos": ["image"]}}
    checkpoint = save_checkpoint(root / "e1.pt", model, extra, full=True)
    meta = {"checkpoint": {"path": "e1.pt", "sha256": sha256_file(checkpoint)}}
    calibration = root / "calibration.json"  # written before calibration.json had media entries
    calibration.write_text(json.dumps({
        "meta": meta,
        "labels": {"eyes_closed": {"temperature": 2.0, "threshold": 0.3, "threshold_from": "val"},
                   "perclos": {"temperature": None, "threshold": 0.5, "threshold_from": "default"}},
    }))
    per_media = root / "per-media.json"  # as `bulwark evaluate` writes it now; yawn had image val clips only
    per_media.write_text(json.dumps({
        "meta": meta,
        "labels": {
            "eyes_closed": {"temperature": 2.0, "threshold": 0.3, "threshold_from": "val", "media": {
                "image": {"temperature": 1.25, "threshold": 0.6, "threshold_from": "val"},
                "video": {"temperature": 3.0, "threshold": 0.1, "threshold_from": "val"},
            }},
            "yawn": {"temperature": 1.5, "threshold": 0.2, "threshold_from": "val", "media": {
                "image": {"temperature": 1.5, "threshold": 0.2, "threshold_from": "val"},
            }},
            "perclos": {"temperature": None, "threshold": 0.5, "threshold_from": "default"},
        },
    }))
    return {"root": root, "checkpoint": checkpoint, "calibration": calibration, "per_media": per_media}


def test_export_stores_masked_labels_and_val_thresholds(e1: dict[str, Path], tmp_path: Path) -> None:
    import onnx

    from bulwark.inference import load

    graph = tmp_path / "e1.onnx"
    command = ["export", "--checkpoint", str(e1["checkpoint"]), "--out", str(graph), "--frames", "1", "--input-size", "96",
               "--no-bench", "--calibration", str(e1["calibration"])]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    props = {p.key: p.value for p in onnx.load(str(graph), load_external_data=False).metadata_props}
    assert json.loads(props["masked_labels"]) == {"perclos": ["image"]}
    assert json.loads(props["thresholds"]) == {"eyes_closed": 0.3} and json.loads(props["temperature"]) == {"eyes_closed": 2.0}
    # A file without media entries: its fit on all val clips, and the graph says so.
    assert read_contract(graph)["calibration"] == {"media": "image", "pooled": ["eyes_closed"], "no_val_clips": []}
    assert "has no image entries for eyes_closed" in " ".join(result.output.split())
    model = load(graph)
    assert model.raw_thresholds == {"eyes_closed": 0.3} and model.labels == ("eyes_closed", "perclos")
    # The embeddings adapters read are exported by default (--no-embeddings leaves them out).
    assert [o.name for o in onnx.load(str(graph)).graph.output][-2:] == ["tokens", "state"]
    assert model.embed(np.zeros((1, 1, 96, 96, 3), dtype=np.uint8), "state").shape[0] == 1

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"meta": {"checkpoint": {"sha256": "ef" * 32}}, "labels": {"eyes_closed": {"temperature": 1.0}}}))
    result = runner.invoke(app, [*command[:-1], str(other)])
    assert result.exit_code == 1 and "fitted on weights" in plain(result.output)


@pytest.mark.parametrize(("frames", "media"), [(1, "image"), (4, "video")])
def test_export_stores_the_calibration_of_the_graphs_media_type(e1: dict[str, Path], tmp_path: Path, frames: int, media: str) -> None:
    import onnx

    from bulwark.inference import load

    graph = tmp_path / f"e1-{frames}.onnx"
    result = runner.invoke(app, ["export", "--checkpoint", str(e1["checkpoint"]), "--out", str(graph), "--frames", str(frames),
                                 "--input-size", "96", "--no-bench", "--no-embeddings", "--calibration", str(e1["per_media"])])
    assert result.exit_code == 0, result.output
    props = {p.key: p.value for p in onnx.load(str(graph), load_external_data=False).metadata_props}
    terms = read_contract(graph)
    # One threshold and one temperature per label, as before: those of the graph's media type.
    expected = {
        "image": ({"eyes_closed": 1.25, "yawn": 1.5}, {"eyes_closed": 0.6, "yawn": 0.2}, []),
        "video": ({"eyes_closed": 3.0}, {"eyes_closed": 0.1}, ["yawn"]),
    }[media]
    assert json.loads(props["temperature"]) == terms["temperature"] == expected[0]
    assert json.loads(props["thresholds"]) == terms["thresholds"] == expected[1]
    assert terms["calibration"] == {"media": media, "pooled": [], "no_val_clips": expected[2]}
    assert "calibration" not in props  # in the contract only
    output = " ".join(result.output.split())
    assert f"Calibration for {media} clips" in output and ("Val had no video clips for yawn" in output) == (media == "video")
    model = load(graph)
    # The graph applies its values to its own media type; yawn is not a supervised label, so it is not returned.
    assert model.temperature == expected[0] and model.temperature_for(media) == {"eyes_closed": expected[0]["eyes_closed"]}
    assert model.metadata["contract"]["calibration"] == terms["calibration"]


def test_export_refuses_the_calibration_of_another_media_type(e1: dict[str, Path], tmp_path: Path) -> None:
    from bulwark.export.onnx import export_edge

    with pytest.raises(ValueError, match="for 'video' clips, but a 1-frame graph takes image clips"):
        export_edge(e1["checkpoint"], tmp_path / "x.onnx", frames=1, input_size=96, temperature={"eyes_closed": 3.0},
                    calibration={"media": "video", "pooled": []})
    assert not (tmp_path / "x.onnx").exists()


def test_export_with_a_manifest_keeps_the_supervised_labels_the_checkpoint_records(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnx

    from bulwark.export.onnx import export_edge, read_contract
    from bulwark.inference import load
    from bulwark.model.bulwark import BulwarkEdge, save_checkpoint

    torch.manual_seed(0)
    model = BulwarkEdge(default_taxonomy(), pretrained=False, img_size=64)
    masked = {"attention_direction": ["image", "video"]}  # a v0 config's mask_labels
    extra = {"img_size": 64, "supervised_labels": ["eyes_closed"], "masked_labels": masked}
    checkpoint = save_checkpoint(tmp_path / "e1.pt", model, extra, full=True)
    records = [clip(f"da/{i}", "drive_and_act", f"vp{i}", "train", {"eyes_closed": float(i % 2), "attention_direction": "left"})
               for i in range(4)]
    manifest = write_manifest(records, tmp_path / "manifest.parquet")
    graph = export_edge(checkpoint, tmp_path / "e1.onnx", frames=1, input_size=96, manifest=manifest)

    props = {p.key: p.value for p in onnx.load(str(graph), load_external_data=False).metadata_props}
    terms = read_contract(graph)
    assert json.loads(props["supervised_labels"]) == terms["supervised_labels"] == ["eyes_closed"]
    assert terms["supervised_labels_from"] == "checkpoint"
    assert json.loads(props["masked_labels"]) == masked
    assert json.loads(props["ground_truth"]) == ["eyes_closed"]
    assert json.loads(props["not_trained_by_checkpoint"]) == ["attention_direction"]
    assert terms["label_status"]["not_trained_by_checkpoint"] == ["attention_direction"]
    assert load(graph).labels == ("eyes_closed",)


def test_predict_adds_rolling_perclos_and_thresholds(e1: dict[str, Path], tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    from bulwark.cli import predict
    from bulwark.inference import PERCLOS_SECONDS

    assert inspect.signature(predict).parameters["perclos_seconds"].default == PERCLOS_SECONDS
    video = make_video(tmp_path / "cabin.mp4", frames=100)  # 3.3 s: windows at 0-2 s and 1-3 s
    out = tmp_path / "timeline.json"
    result = runner.invoke(app, ["predict", str(e1["checkpoint"]), str(video), "--out", str(out), "--device", "cpu",
                                 "--perclos-seconds", "3", "--temperature", str(e1["calibration"])])
    assert result.exit_code == 0, result.output
    timeline = json.loads(out.read_text())
    assert timeline["perclos_seconds"] == 3 and timeline["temperature"] == {"eyes_closed": 2.0}
    assert set(timeline["thresholds"]) == {"eyes_closed"} and timeline["media"] == "video"
    first, second = timeline["windows"]
    assert first["rolling"] == {"perclos_3s": None}
    assert second["rolling"]["perclos_3s"] == pytest.approx((first["outputs"]["perclos"] + second["outputs"]["perclos"]) / 2)


def test_predict_runs_a_one_frame_graph_over_a_video_with_the_image_calibration(e1: dict[str, Path], tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    from bulwark.export.onnx import export_edge

    def moved(p: float, t: float) -> float:
        return float(1 / (1 + np.exp(-np.log(p / (1 - p)) / t)))

    # A --frames 1 graph cuts 1-frame windows from a video, which count as images, so the
    # timeline takes the image entries: `bulwark predict` and the example alike.
    graph = export_edge(e1["checkpoint"], tmp_path / "e1-image.onnx", frames=1, input_size=96, embeddings=False)
    video = make_video(tmp_path / "cabin.mp4", frames=40)
    cli_out, example_out = tmp_path / "cli.json", tmp_path / "example.json"
    result = runner.invoke(app, ["predict", str(graph), str(video), "--out", str(cli_out), "--calibration", str(e1["per_media"])])
    assert result.exit_code == 0, result.output
    done = subprocess.run(
        [sys.executable, "examples/predict_video.py", str(graph), str(video), "--temperature", str(e1["per_media"]), "--out", str(example_out)],
        cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr
    for out in (cli_out, example_out):
        timeline = json.loads(out.read_text())
        # perclos is masked on images, so only eyes_closed is returned, at the image entry's values.
        assert timeline["media"] == "image" and timeline["labels"] == ["eyes_closed"], out.name
        assert timeline["temperature"] == {"eyes_closed": 1.25}, out.name
        assert timeline["thresholds"] == pytest.approx({"eyes_closed": moved(0.6, 1.25)}), out.name
        assert all(set(w["outputs"]) == {"eyes_closed"} for w in timeline["windows"]), out.name


def test_shard_skips_partial_files_of_an_interrupted_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_image) -> None:  # type: ignore[no-untyped-def]
    from bulwark.data.shards import shard_index

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data.yaml").write_text("roots:\n  state_farm: raw\n")
    monkeypatch.setattr("bulwark.cli.DATA_CONFIG", tmp_path / "configs" / "data.yaml")
    make_image(tmp_path / "raw" / "a.jpg")
    write_manifest([clip("state_farm/a", "state_farm", "a", "train", media_type="image", media_path="a.jpg")], tmp_path / "m.parquet")
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "state_farm-00000.tar.partial").write_bytes(b"cut off")  # killed mid-write
    result = runner.invoke(app, ["shard", "--manifest", "m.parquet", "--out", "shards"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "shards").iterdir()) == ["state_farm-00000.tar", "state_farm.shards.json"]
    assert set(shard_index(tmp_path / "shards" / "state_farm-00000.tar").values()) == {"state_farm/a"}


def test_predict_uses_the_calibration_of_the_inputs_media_type(e1: dict[str, Path], tmp_path: Path, make_video, make_image) -> None:  # type: ignore[no-untyped-def]
    def moved(p: float, t: float) -> float:
        return float(1 / (1 + np.exp(-np.log(p / (1 - p)) / t)))

    video = make_video(tmp_path / "cabin.mp4", frames=70)
    image = make_image(tmp_path / "face.jpg")
    for source, media, temperature, threshold in ((video, "video", 3.0, 0.1), (image, "image", 1.25, 0.6)):
        out = tmp_path / f"{media}.json"
        result = runner.invoke(app, ["predict", str(e1["checkpoint"]), str(source), "--out", str(out), "--device", "cpu",
                                     "--calibration", str(e1["per_media"])])
        assert result.exit_code == 0, result.output
        timeline = json.loads(out.read_text())
        assert timeline["media"] == media
        # Only the returned labels: yawn is not a supervised label, perclos is masked on images.
        assert timeline["labels"] == (["eyes_closed", "perclos"] if media == "video" else ["eyes_closed"])
        assert timeline["temperature"] == {"eyes_closed": temperature}
        assert timeline["thresholds"] == pytest.approx({"eyes_closed": moved(threshold, temperature)})
