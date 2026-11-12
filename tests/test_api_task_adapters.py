"""Task adapters (`bulwark.adapters`) and the adapter examples, on synthetic embeddings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bulwark.adapters import Adapter, features_source, fit_adapter, load_cached_embeddings, pool
from bulwark.data.manifest import ClipRecord, write_manifest

REPO = Path(__file__).resolve().parents[1]


def separable(n: int, classes: tuple[str, ...] = ("a", "b", "c"), dim: int = 16, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """(n, 25, dim) tokens whose mean carries the class."""
    rng = np.random.default_rng(seed)
    labels = [classes[i % len(classes)] for i in range(n)]
    tokens = rng.normal(size=(n, 25, dim)).astype(np.float32)
    for row, label in enumerate(labels):
        tokens[row, :, classes.index(label)] += 2.0
    return tokens, labels


def test_linear_adapter_learns_and_is_deterministic() -> None:
    train, labels = separable(30)
    test, truth = separable(90, seed=1)
    adapter = fit_adapter(train, labels)
    assert adapter.config.classes == ("a", "b", "c") and adapter.config.in_dim == 16 and adapter.config.hidden == 0
    assert np.mean(np.array(adapter.predict(test)) == np.array(truth)) > 0.9
    probs = adapter.predict_proba(test)
    assert probs.shape == (90, 3) and np.allclose(probs.sum(1), 1, atol=1e-5)
    np.testing.assert_array_equal(fit_adapter(train, labels).predict_proba(test), probs)


def test_pooling_options() -> None:
    tokens = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(pool(tokens, "mean"), tokens.mean(1))
    assert torch.equal(pool(tokens, "first"), tokens[:, 0])
    assert pool(tokens.numpy(), "flatten").shape == (2, 12)
    assert pool(np.ones((5, 7), np.float16)).shape == (5, 7)  # vectors pass through
    with pytest.raises(ValueError):
        pool(tokens, "max")
    train, labels = separable(12)
    assert fit_adapter(train, labels, pooling="flatten", steps=5).config.in_dim == 25 * 16
    assert fit_adapter(train.mean(1), labels, steps=5).predict_proba(train).shape == (12, 3)  # pools tokens at predict time


def test_mlp_adapter_saves_and_loads_with_its_config(tmp_path: Path) -> None:
    train, labels = separable(30)
    adapter = fit_adapter(train, labels, hidden=8, classes=("c", "b", "a"), embedding="state", source="unit test")
    path = adapter.save(tmp_path / "adapters" / "x.pt")
    reloaded = Adapter.load(path)
    assert reloaded.config == adapter.config and reloaded.config.classes == ("c", "b", "a")
    np.testing.assert_allclose(reloaded.predict_proba(train), adapter.predict_proba(train), atol=1e-6)
    payload = torch.load(path, weights_only=True)  # plain tensors and types only
    assert payload["format"] == "bulwark-adapter" and payload["config"]["source"] == "unit test"
    torch.save({"format": "something-else"}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a bulwark-adapter"):
        Adapter.load(tmp_path / "other.pt")


def test_adapter_records_and_checks_the_model_behind_its_embeddings(tmp_path: Path) -> None:
    train, labels = separable(12)
    source = {"family": "b1", "weights": "ab" * 32}
    adapter = fit_adapter(train, labels, steps=5, model=source)
    reloaded = Adapter.load(adapter.save(tmp_path / "a.pt"))
    assert dict(reloaded.config.model) == source
    reloaded.check(source)
    reloaded.check({"family": "b1", "weights": None})  # unknown on the model side: not compared
    with pytest.raises(ValueError, match="weights="):
        reloaded.check({"family": "b1", "weights": "frozen"})
    with pytest.raises(ValueError, match="family="):
        reloaded.check({"family": "e1", "weights": "ab" * 32})
    fit_adapter(train, labels, steps=5).check({"family": "e1", "weights": "frozen"})  # fitted without model=

    # Adapter files written before the `model` field existed still load.
    payload = torch.load(tmp_path / "a.pt", weights_only=True)
    del payload["config"]["model"]
    torch.save(payload, tmp_path / "old.pt")
    assert dict(Adapter.load(tmp_path / "old.pt").config.model) == {}


def test_features_source_reads_the_cache_meta(tmp_path: Path) -> None:
    assert features_source(tmp_path) == {}
    (tmp_path / "meta.json").write_text(json.dumps({"run": {"init": None, "with_logits": False}}))
    assert features_source(tmp_path) == {"family": "b1", "weights": "frozen"}
    (tmp_path / "meta.json").write_text(json.dumps({"run": {"init": {"sha256": "cd" * 32}}}))
    assert features_source(tmp_path) == {"family": "b1", "weights": "cd" * 32}


def test_fit_adapter_rejects_bad_inputs() -> None:
    train, labels = separable(6)
    with pytest.raises(ValueError, match="not in classes"):
        fit_adapter(train, labels, classes=("a", "b"))
    with pytest.raises(ValueError, match="two classes"):
        fit_adapter(train, ["a"] * 6)
    with pytest.raises(ValueError, match="labels"):
        fit_adapter(train, labels[:5])
    adapter = fit_adapter(train, labels, steps=1)
    with pytest.raises(ValueError, match="16 features"):
        adapter.predict(np.zeros((2, 25, 8), np.float32))


def clip(clip_id: str, split: str, **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id, dataset="mrl_eye" if clip_id.startswith("m") else "drive_and_act",
        subject_id=f"{'mrl_eye' if clip_id.startswith('m') else 'drive_and_act'}:{clip_id}", media_path="x.jpg",
        media_type="image", licence="test", commercial_ok=False, split=split, **kwargs,  # type: ignore[arg-type]
    )


def fake_cache(tmp_path: Path, n: int = 40) -> tuple[Path, Path]:
    """Features dir + manifest: eyeglasses in ground truth on m* clips, teacher-only on t* clips."""
    tokens, labels = separable(n, classes=("0", "1"))
    records = [
        clip(f"m{i}", "train" if i < n * 3 // 4 else "test", labels_gt={"eyeglasses": float(label), "kss": 0.5})
        for i, label in enumerate(labels)
    ]
    records += [clip("t0", "train", labels_teacher={"eyeglasses": 1.0}), clip("d0", "train", labels_gt={"activity": "eating"})]
    write_manifest(records, tmp_path / "manifest.parquet")
    features = tmp_path / "features"
    features.mkdir()
    extra = np.zeros((2, 25, 16), np.float32)
    np.savez(
        features / "features-a.npz",
        clip_ids=np.array([r.clip_id for r in records]),
        tokens=np.concatenate([tokens, extra]).astype(np.float16),
    )
    return features, tmp_path / "manifest.parquet"


def test_load_cached_embeddings_uses_ground_truth_only(tmp_path: Path) -> None:
    features, manifest = fake_cache(tmp_path)
    x, labels, records = load_cached_embeddings(features, manifest, "eyeglasses", split="train")
    assert x.shape == (30, 25, 16) and x.dtype == np.float32
    assert set(labels) == {"0", "1"} and all(r.clip_id.startswith("m") for r in records)  # teacher-only t0 excluded
    _, test_labels, _ = load_cached_embeddings(features, manifest, "eyeglasses", split="test")
    assert len(test_labels) == 10
    _, mapped, _ = load_cached_embeddings(features, manifest, {"d0": "custom", "m0": "other"}, split=None)
    assert sorted(mapped) == ["custom", "other"]
    with pytest.raises(ValueError, match="regression"):
        load_cached_embeddings(features, manifest, "kss")
    with pytest.raises(ValueError, match="No cached clips"):
        load_cached_embeddings(features, manifest, "eyeglasses", datasets=["drive_and_act"])


def run_example(*args: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    done = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_adapter_quickstart_example_runs() -> None:
    assert "same probabilities" in run_example("examples/adapter_quickstart.py")


def test_fewshot_curve_example_writes_a_curve(tmp_path: Path) -> None:
    features, manifest = fake_cache(tmp_path)
    out = tmp_path / "curve.json"
    run_example(
        "examples/fewshot_curve.py", "--features", str(features), "--manifest", str(manifest),
        "--label", "eyeglasses", "--shots", "1,4", "--repeats", "2", "--out", str(out),
    )
    curve = json.loads(out.read_text())
    assert [row["shots"] for row in curve["curve"]] == [1, 4, "all"]
    assert curve["eval_clips"] == 10 and curve["eval_subjects"] == 10
    assert curve["majority_class_baseline"]["balanced_accuracy"] == 0.5
    assert curve["curve"][-1]["auroc"]["mean"] > 0.9
