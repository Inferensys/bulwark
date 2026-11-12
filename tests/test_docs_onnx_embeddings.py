"""The ONNX example in docs/architecture.md also decodes a graph exported with embeddings."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bulwark.export.onnx import EMBEDDING_OUTPUTS, export_edge  # noqa: E402
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint  # noqa: E402
from bulwark.model.heads import activate  # noqa: E402

DOC = Path(__file__).resolve().parents[1] / "docs" / "architecture.md"


def test_documented_onnx_example_reads_a_graph_with_embeddings(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = BulwarkEdge(pretrained=False).eval()
    checkpoint = save_checkpoint(tmp_path / "e1.pt", model, {"backbone": "efficientvit_b1.r224_in1k"}, full=True)
    onnx_path = export_edge(checkpoint, tmp_path / "bulwark-e1.onnx", embeddings=True)

    example = re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)[-1]
    namespace: dict = {}
    exec(example.replace('"bulwark-e1.onnx"', repr(str(onnx_path))), namespace)
    outputs = [o.name for o in namespace["session"].get_outputs()]
    assert outputs[-len(EMBEDDING_OUTPUTS):] == list(EMBEDDING_OUTPUTS)

    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, (90, 160, 3), dtype=np.uint8) for _ in range(16)]
    predicted = namespace["predict"](frames)

    assert list(predicted) == model.heads.label_names  # the embeddings are not decoded as labels
    with torch.no_grad():
        reference = activate(model(torch.from_numpy(namespace["prepare"](frames))), model.taxonomy)
    for name, value in reference.items():
        np.testing.assert_allclose(predicted[name], value.numpy(), atol=1e-4, err_msg=name)
