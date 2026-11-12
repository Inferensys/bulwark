"""The example scripts' guards: single-class evaluation sets, shared subjects, mixed E1 inputs, releases."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from bulwark.adapters import Adapter
from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.inference import load
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
EDGE = {"backbone": "test_efficientnet", "img_size": 64}


def run(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)


def example(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "examples" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache(tmp_path: Path, rows: list[tuple[str, str, str, str]]) -> tuple[Path, Path]:
    """A features dir and manifest from (clip_id, split, subject, eyeglasses label) rows."""
    rng = np.random.default_rng(0)
    records = [
        ClipRecord(
            clip_id=clip_id, dataset="mrl_eye", subject_id=f"mrl_eye:{subject}", media_path="x.jpg", media_type="image",
            licence="test", commercial_ok=False, split=split, labels_gt={"eyeglasses": float(label)},
        )
        for clip_id, split, subject, label in rows
    ]
    write_manifest(records, tmp_path / "manifest.parquet")
    tokens = rng.normal(size=(len(rows), 25, 8)).astype(np.float32)
    for row, (*_, label) in enumerate(rows):
        tokens[row, :, 0] += 3.0 * float(label)
    features = tmp_path / "features"
    features.mkdir()
    np.savez(features / "features-a.npz", clip_ids=np.array([r[0] for r in rows]), tokens=tokens.astype(np.float16))
    return features, tmp_path / "manifest.parquet"


def fewshot(features: Path, manifest: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return run(
        "examples/fewshot_curve.py", "--features", str(features), "--manifest", str(manifest),
        "--label", "eyeglasses", "--shots", "1,2", "--repeats", "2", *extra,
    )


def test_fewshot_curve_refuses_a_single_class_evaluation_split(tmp_path: Path) -> None:
    rows = [(f"tr{i}", "train", f"a{i}", str(i % 2)) for i in range(12)]
    rows += [(f"te{i}", "test", f"b{i}", "1") for i in range(6)]
    done = fewshot(*cache(tmp_path, rows))
    assert done.returncode != 0 and "constant prediction scores perfectly" in done.stderr


def test_fewshot_curve_refuses_subjects_shared_with_training(tmp_path: Path) -> None:
    rows = [(f"tr{i}", "train", f"a{i}", str(i % 2)) for i in range(12)]
    rows += [(f"te{i}", "test", f"b{i}", str(i % 2)) for i in range(5)] + [("te9", "test", "a3", "1")]
    done = fewshot(*cache(tmp_path, rows))
    assert done.returncode != 0 and "1 subject(s) are in both train and test" in done.stderr


def test_fewshot_curve_bootstraps_subjects_and_tolerates_undefined_metrics(tmp_path: Path) -> None:
    rows = [(f"tr{i}", "train", f"a{i}", str(i % 2)) for i in range(20)]
    rows += [(f"te{i}", "test", f"b{i // 2}", str(i % 2)) for i in range(24)]  # 12 subjects, 2 clips each
    out = tmp_path / "curve.json"
    done = fewshot(*cache(tmp_path, rows), "--bootstrap", "200", "--out", str(out))
    assert done.returncode == 0, done.stderr
    curve = json.loads(out.read_text())
    assert curve["eval_subjects"] == 12 and curve["train_subjects"] == 20
    interval = curve["curve"][-1]["ci95_subject_bootstrap"]
    assert interval["balanced_accuracy"]["low"] <= curve["curve"][-1]["balanced_accuracy"]["mean"] <= interval["balanced_accuracy"]["high"]

    module = example("fewshot_curve")
    classes = ("0", "1")
    single = module.score(np.array([[0.2, 0.8]] * 3), np.array(["1"] * 3), classes)
    assert single == {}  # every metric is undefined on one class
    summary = module.summarise([{"balanced_accuracy": 0.5, "auroc": 0.6}, {"balanced_accuracy": 0.7}])
    assert summary["balanced_accuracy"]["runs"] == 2 and summary["auroc"] == {"mean": 0.6, "std": 0.0, "min": 0.6, "max": 0.6, "runs": 1}


def test_fewshot_curve_divides_the_subjects_of_one_split(tmp_path: Path) -> None:
    # A holdout-only dataset (as UTA-RLDD): 24 subjects, 2 clips each, both classes per subject.
    rows = [(f"h{i}", "holdout", f"s{i // 2}", str(i % 2)) for i in range(48)]
    features, manifest = cache(tmp_path, rows)
    out = tmp_path / "curve.json"
    same = ("--train-split", "holdout", "--eval-split", "holdout")
    done = fewshot(features, manifest, *same, "--eval-subjects", "12", "--bootstrap", "100", "--out", str(out))
    assert done.returncode == 0, done.stderr
    curve = json.loads(out.read_text())
    assert curve["train_split"] == curve["eval_split"] == "holdout" and curve["subject_draw_seed"] == 0
    assert curve["train_subjects"] == 12 and curve["eval_subjects"] == 12 and curve["eval_clips"] == 24
    assert "ci95_subject_bootstrap" in curve["curve"][-1]
    assert fewshot(features, manifest, *same).returncode == 0  # default: half the subjects

    split_subjects = example("fewshot_curve").split_subjects
    subjects = [f"s{i // 2}" for i in range(48)]
    train, scored = split_subjects(subjects, 5, seed=3)
    assert len({subjects[i] for i in scored}) == 5 and not {subjects[i] for i in train} & {subjects[i] for i in scored}
    assert sorted([*train, *scored]) == list(range(48))
    assert np.array_equal(split_subjects(subjects, 5, seed=3)[1], scored)  # the draw is seeded

    whole = fewshot(features, manifest, *same, "--eval-subjects", "24")
    assert whole.returncode != 0 and "leave at least one for training" in whole.stderr
    mixed = fewshot(features, manifest, "--eval-subjects", "4")
    assert mixed.returncode != 0 and "pass the same --train-split and --eval-split" in mixed.stderr


def edge_checkpoint(path: Path, **extra: object) -> Path:
    torch.manual_seed(0)
    return save_checkpoint(path, BulwarkEdge(pretrained=False, **EDGE), {"step": 1, **EDGE, **extra})


def test_train_adapter_mixes_images_and_videos_on_e1(tmp_path: Path, make_image, make_video) -> None:  # type: ignore[no-untyped-def]
    held_out = example("train_adapter").held_out
    subjects = [f"s{i}" for i in range(40)]
    test = [s for s in subjects if held_out(s, 0.2)]
    train = [s for s in subjects if not held_out(s, 0.2)]
    assert test and len(train) >= 2
    rows = [{"path": f"img{i}.png", "label": "a" if i % 2 else "b", "subject": s} for i, s in enumerate(train[:4])]
    rows += [{"path": "clip.avi", "label": "a", "subject": test[0]}]  # held-out subjects cover one class only
    for row in rows:
        if row["path"].endswith(".png"):
            make_image(tmp_path / row["path"], value=60 if row["label"] == "a" else 180)
    make_video(tmp_path / "clip.avi", frames=60, fps=30.0)
    with (tmp_path / "clips.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "subject"])
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = edge_checkpoint(tmp_path / "e1.pt")
    done = run("examples/train_adapter.py", str(checkpoint), str(tmp_path / "clips.csv"), "--out", str(tmp_path / "a.pt"), "--device", "cpu")
    assert done.returncode == 0, done.stderr
    assert "only have class 'a'; skipping the held-out score" in done.stdout
    adapter = Adapter.load(tmp_path / "a.pt")
    assert adapter.config.source == "e1:e1.pt" and adapter.config.in_dim == load(checkpoint, device="cpu").embed(np.zeros((1, 32, 32, 3), np.uint8)).shape[-1]
    assert dict(adapter.config.model) == {"family": "e1", "weights": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def test_make_release_requires_labels_and_writes_a_loadable_release(tmp_path: Path) -> None:
    checkpoint = edge_checkpoint(tmp_path / "e1.pt")
    out = tmp_path / "release"
    refused = run("examples/make_release.py", str(checkpoint), str(out), "--weights-licence", "research use only")
    assert refused.returncode != 0 and "--labels" in refused.stderr and not out.exists()

    permissive = run("examples/make_release.py", str(checkpoint), str(out), "--weights-licence", "MIT", "--labels", "yawn")
    assert permissive.returncode != 0 and "does not restrict use to non-commercial research" in permissive.stderr
    assert "Traceback" not in permissive.stderr and not out.exists()

    labels = {"yawn": {"temperature": 1.5, "threshold": 0.3, "threshold_from": "val"}, "phone": {"temperature": None}}
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calibration = {"meta": {"model": "e1", "checkpoint": {"path": "e1.pt", "sha256": sha}}, "labels": labels}
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    done = run(
        "examples/make_release.py", str(checkpoint), str(out), "--weights-licence", "research use only",
        "--labels", "yawn,phone", "--temperature", str(tmp_path / "calibration.json"),
    )
    assert done.returncode == 0, done.stderr
    config = json.loads((out / "config.json").read_text())
    assert config["labels"] == ["yawn", "phone"] and config["temperature"] == {"yawn": 1.5} and config["thresholds"] == {"yawn": 0.3}
    assert str(tmp_path) not in (out / "config.json").read_text()
    model = load(out, device="cpu")
    assert model.labels == ("yawn", "phone") and model.temperature == {"yawn": 1.5}


def test_predict_video_writes_a_timeline_with_rolling_perclos(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    checkpoint = edge_checkpoint(tmp_path / "e1.pt")
    video = make_video(tmp_path / "v.avi", frames=150, fps=30.0)  # 5 s: windows start at 0, 1, 2, 3 s
    out = tmp_path / "timeline.json"
    done = run(
        "examples/predict_video.py", str(checkpoint), str(video), "--labels", "perclos,yawn",
        "--perclos-seconds", "3", "--device", "cpu", "--out", str(out),
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(out.read_text())
    windows = result["windows"]
    assert [w["start"] for w in windows] == [0.0, 1.0, 2.0, 3.0] and result["perclos_seconds"] == 3.0
    rolled = [w["rolling"]["perclos_3s"] for w in windows]
    assert rolled[0] is None and rolled[1] == pytest.approx(np.mean([w["outputs"]["perclos"] for w in windows[:2]]))
    assert rolled[3] == pytest.approx(np.mean([w["outputs"]["perclos"] for w in windows[2:]]))  # starts 2 and 3 s

    without = run("examples/predict_video.py", str(checkpoint), str(video), "--labels", "yawn", "--device", "cpu", "--out", str(out))
    assert without.returncode == 0, without.stderr
    assert "no rolling PERCLOS" in without.stdout and "rolling" not in json.loads(out.read_text())["windows"][0]

    short = run(
        "examples/predict_video.py", str(checkpoint), str(video), "--labels", "perclos",
        "--perclos-seconds", "1", "--device", "cpu", "--out", str(out),
    )
    assert short.returncode == 0, short.stderr
    assert "shorter than one 2 s window, so no rolling PERCLOS" in short.stdout
    assert "rolling" not in json.loads(out.read_text())["windows"][0]


def test_predict_video_refuses_a_calibration_file_of_other_weights(tmp_path: Path, make_video) -> None:  # type: ignore[no-untyped-def]
    checkpoint = edge_checkpoint(tmp_path / "e1.pt")
    video = make_video(tmp_path / "v.avi", frames=60, fps=30.0)
    calibration = {"meta": {"model": "e1", "checkpoint": {"path": "other.pt", "sha256": "ab" * 32}}, "labels": {"yawn": {"temperature": 1.5}}}
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    done = run("examples/predict_video.py", str(checkpoint), str(video), "--temperature", str(tmp_path / "calibration.json"), "--device", "cpu")
    assert done.returncode != 0 and "was fitted on weights with sha256 abababababab" in done.stderr
    assert "Traceback" not in done.stderr


def test_predict_video_runs_images_as_one_clip_and_leaves_out_masked_labels(tmp_path: Path, make_video, make_image) -> None:  # type: ignore[no-untyped-def]
    # phone was trained on images and masked on video, as the training jobs record it.
    checkpoint = edge_checkpoint(tmp_path / "e1.pt", supervised_labels=["phone", "yawn"], masked_labels={"phone": ["video"]})
    out = tmp_path / "timeline.json"
    video = make_video(tmp_path / "v.avi", frames=60, fps=30.0)
    done = run("examples/predict_video.py", str(checkpoint), str(video), "--device", "cpu", "--out", str(out))
    assert done.returncode == 0, done.stderr
    result = json.loads(out.read_text())
    assert result["labels"] == ["yawn"] and result["masked_labels"] == {"phone": ["video"]}
    assert [set(w["outputs"]) for w in result["windows"]] == [{"yawn"}]

    image = make_image(tmp_path / "face.jpg")
    done = run("examples/predict_video.py", str(checkpoint), str(image), "--device", "cpu", "--out", str(out))
    assert done.returncode == 0, done.stderr
    result = json.loads(out.read_text())
    assert result["labels"] == ["phone", "yawn"] and len(result["windows"]) == 1
    assert result["windows"][0]["start"] == result["windows"][0]["end"] == 0.0
    assert set(result["windows"][0]["outputs"]) == {"phone", "yawn"}


def test_examples_use_the_calibration_of_each_media_type(tmp_path: Path, make_video, make_image) -> None:  # type: ignore[no-untyped-def]
    checkpoint = edge_checkpoint(tmp_path / "e1.pt")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    yawn = {"temperature": 1.5, "threshold": 0.3, "threshold_from": "val", "media": {
        "video": {"temperature": 2.0, "threshold": 0.25, "threshold_from": "val"},
        "image": {"temperature": 1.2, "threshold": 0.4, "threshold_from": "val"},
    }}
    calibration = {"meta": {"model": "e1", "checkpoint": {"path": "e1.pt", "sha256": sha}}, "labels": {"yawn": yawn}}
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))

    def moved(p: float, t: float) -> float:
        return float(1 / (1 + np.exp(-np.log(p / (1 - p)) / t)))

    out = tmp_path / "timeline.json"
    for source, media, t, p in ((make_video(tmp_path / "v.avi", frames=60, fps=30.0), "video", 2.0, 0.25),
                                (make_image(tmp_path / "face.jpg"), "image", 1.2, 0.4)):
        done = run("examples/predict_video.py", str(checkpoint), str(source), "--labels", "yawn,phone", "--device", "cpu",
                   "--temperature", str(tmp_path / "calibration.json"), "--out", str(out))
        assert done.returncode == 0, done.stderr
        result = json.loads(out.read_text())
        assert result["media"] == media and result["temperature"] == {"yawn": t}
        assert result["thresholds"] == pytest.approx({"yawn": moved(p, t)})

    release = tmp_path / "release"
    done = run("examples/make_release.py", str(checkpoint), str(release), "--weights-licence", "research use only",
               "--labels", "yawn,phone", "--temperature", str(tmp_path / "calibration.json"))
    assert done.returncode == 0, done.stderr
    assert "Calibration for image clips: 1 temperatures, 1 thresholds" in done.stdout
    config = json.loads((release / "config.json").read_text())
    assert config["temperature"] == {"yawn": 1.5} and config["thresholds"] == {"yawn": 0.3}
    assert config["media_calibration"]["video"] == {"temperature": {"yawn": 2.0}, "thresholds": {"yawn": 0.25}}
    model = load(release, device="cpu")
    assert model.temperature_for("image") == {"yawn": 1.2} and model.temperature_for("video") == {"yawn": 2.0}
