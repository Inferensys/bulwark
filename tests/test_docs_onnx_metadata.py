"""docs/architecture.md names every metadata entry an exported E1 graph carries.

The "E1 ONNX contract" section lists the entries of `bulwark_contract` in a table and the
single-key metadata_props after it. A graph exported with everything `bulwark export` can
add (a training manifest, recorded supervised and masked labels, temperatures and
thresholds with the record of the media type they were fitted for) must not carry an
entry that section leaves out.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx  # noqa: E402

from bulwark.data.manifest import ClipRecord, write_manifest  # noqa: E402
from bulwark.export.onnx import CONTRACT_KEY, export_edge, read_contract  # noqa: E402
from bulwark.model.bulwark import BulwarkEdge, save_checkpoint  # noqa: E402

DOC = Path(__file__).resolve().parents[1] / "docs" / "architecture.md"
EDGE = {"backbone": "test_efficientnet", "img_size": 64}


def contract_section() -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("## E1 ONNX contract")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


@pytest.fixture(scope="module")
def graph(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("onnx-metadata")
    torch.manual_seed(0)
    checkpoint = save_checkpoint(
        root / "e1.pt",
        BulwarkEdge(pretrained=False, **EDGE),
        {"step": 1, **EDGE, "supervised_labels": ["eyes_closed", "yawn"], "masked_labels": {"yawn": ["image"]}},
    )
    records = [
        ClipRecord(
            clip_id=f"dmd/{i}", dataset="dmd", subject_id=f"dmd:{i}", media_path=f"{i}.mp4", media_type="video",
            licence="t", commercial_ok=False, split="train", labels_gt={"eyes_closed": float(i % 2), "phone": 1.0},
            labels_teacher={"yawn": float(i % 2)},
        )
        for i in range(4)
    ]
    manifest = write_manifest(records, root / "manifest.parquet")
    return export_edge(
        checkpoint, root / "e1.onnx", frames=1, input_size=96, embeddings=True, manifest=manifest,
        temperature={"eyes_closed": 1.5}, thresholds={"eyes_closed": 0.4},
        calibration={"media": "image", "pooled": [], "no_val_clips": ["yawn"]},
    )


def metadata_table() -> str:
    """The rows of the section's `bulwark_contract` entry table."""
    return "\n".join(line for line in contract_section().splitlines() if line.startswith("| `"))


def test_every_contract_entry_is_in_the_metadata_table(graph: Path) -> None:
    terms = read_contract(graph)
    assert {"label_status", "supervised_labels", "masked_labels", "temperature", "thresholds", "calibration"} <= set(terms)
    table = metadata_table()
    missing = sorted(key for key in terms if f"`{key}`" not in table)
    assert not missing, f"the metadata table in docs/architecture.md does not name {missing}"


def test_every_single_key_metadata_entry_is_documented(graph: Path) -> None:
    props = {p.key for p in onnx.load(str(graph), load_external_data=False).metadata_props}
    assert CONTRACT_KEY in props and "thresholds" in props
    section = contract_section()
    missing = sorted(key for key in props if f"`{key}`" not in section)
    assert not missing, f"docs/architecture.md (E1 ONNX contract) does not name {missing}"


def test_documented_outputs_include_the_embeddings_when_exported_with_them(graph: Path) -> None:
    # The section describes `tokens` and `state` as the last outputs unless --no-embeddings.
    # The fixture exports with embeddings=True; that `bulwark export` adds them by default
    # is checked in test_eval_integration.py.
    names = [output["name"] for output in read_contract(graph)["outputs"]]
    assert names[-2:] == ["tokens", "state"]
    assert re.search(r"Unless exported with\s+`--no-embeddings`", contract_section())
