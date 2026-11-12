"""Fit, save and reload a task adapter on synthetic embeddings. Needs no weights or data.

    PYTHONPATH=. python examples/adapter_quickstart.py

The arrays stand in for Bulwark-B1 summary tokens (N, 25, 768). With a model, replace
them with `load(checkpoint).embed(clips)`, or with `load_cached_embeddings` on a features
directory from `bulwark train cache`; examples/train_adapter.py and
examples/fewshot_curve.py do exactly that.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from bulwark.adapters import Adapter, fit_adapter

CLASSES = ("safe", "reaching_into_machine", "leaning_out")


def synthetic_tokens(rng: np.random.Generator, labels: list[str], directions: np.ndarray) -> np.ndarray:
    """Noise plus a class-specific shift: a stand-in for real embeddings."""
    tokens = rng.normal(size=(len(labels), 25, 768)).astype(np.float32)
    for row, label in enumerate(labels):
        tokens[row] += 0.03 * directions[CLASSES.index(label)]
    return tokens


def main() -> None:
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(len(CLASSES), 768)).astype(np.float32)
    train_labels = [cls for cls in CLASSES for _ in range(8)]       # 8 labelled clips per class
    test_labels = [cls for cls in CLASSES for _ in range(200)]
    train, test = synthetic_tokens(rng, train_labels, directions), synthetic_tokens(rng, test_labels, directions)

    adapter = fit_adapter(train, train_labels, source="synthetic example")
    accuracy = np.mean(np.array(adapter.predict(test)) == np.array(test_labels))
    print(f"Linear adapter on {len(train_labels)} synthetic clips: held-out accuracy {accuracy:.3f} (chance {1 / len(CLASSES):.3f})")

    with tempfile.TemporaryDirectory() as tmp:
        path = adapter.save(Path(tmp) / "adapter.pt")
        reloaded = Adapter.load(path)
        assert np.allclose(reloaded.predict_proba(test), adapter.predict_proba(test))
        print(f"Saved and reloaded {path.name}: classes {reloaded.config.classes}, same probabilities")


if __name__ == "__main__":
    main()
