"""The ONNX example in docs/architecture.md decodes an exported E1 exactly as PyTorch does."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bulwark.export.onnx import export_edge  # noqa: E402
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint  # noqa: E402
from bulwark.model.heads import activate  # noqa: E402

DOC = Path(__file__).resolve().parents[1] / "docs" / "architecture.md"


def test_documented_onnx_example_matches_pytorch(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = BulwarkEdge(pretrained=False).eval()
    checkpoint = save_checkpoint(tmp_path / "e1.pt", model, {"backbone": "efficientvit_b1.r224_in1k"}, full=True)
    onnx_path = export_edge(checkpoint, tmp_path / "bulwark-e1.onnx")

    example = re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)[-1]
    assert '"bulwark-e1.onnx"' in example
    namespace: dict = {}
    exec(example.replace('"bulwark-e1.onnx"', repr(str(onnx_path))), namespace)

    # Odd-sized BGR frames, as OpenCV would deliver them, exercise the letterbox step.
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (120, 160, 3), dtype=np.uint8) for _ in range(16)]
    predicted = namespace["predict"](frames)

    batch = namespace["prepare"](frames)
    size = namespace["size"]
    assert batch.shape == (1, 16, size, size, 3) and batch.dtype == np.uint8
    with torch.no_grad():
        reference = activate(model(torch.from_numpy(batch)), model.taxonomy)
    assert list(predicted) == model.heads.label_names
    for name, value in reference.items():
        np.testing.assert_allclose(predicted[name], value.numpy(), atol=1e-4, err_msg=name)
