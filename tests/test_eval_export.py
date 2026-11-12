from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from bulwark.data.shards import ShardWriter  # noqa: E402
from bulwark.data.taxonomy import default_taxonomy  # noqa: E402
from bulwark.eval.predict import predict_shards  # noqa: E402
from bulwark.export.onnx import benchmark, export_edge, read_contract  # noqa: E402
from bulwark.model.bulwark import BulwarkEdge  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A random-weight E1 trained at 64 px, exported for 4-frame windows of 96 px frames."""
    root = tmp_path_factory.mktemp("export")
    torch.manual_seed(0)
    model = BulwarkEdge(default_taxonomy(), pretrained=False, img_size=64)
    checkpoint = root / "e1.pt"
    torch.save({"model": model.state_dict(), "backbone": "efficientvit_b1.r224_in1k", "img_size": 64, "step": 1}, checkpoint)
    return checkpoint, export_edge(checkpoint, root / "e1.onnx", frames=4, input_size=96)


def test_export_records_contract_and_matches_pytorch(exported: tuple[Path, Path]) -> None:
    checkpoint, onnx_path = exported
    terms = read_contract(onnx_path)
    assert terms["input"]["shape"] == ["batch", 4, 96, 96, 3] and terms["input"]["dtype"] == "uint8"
    assert terms["model_img_size"] == 64  # taken from the checkpoint, not a default
    assert terms["parity"]["max_abs_logit_diff"] <= terms["parity"]["tolerance"]
    assert [o["name"] for o in terms["outputs"]] == list(default_taxonomy().labels)
    assert terms["checkpoint"]["img_size"] == 64 and len(terms["checkpoint"]["sha256"]) == 64


def test_benchmark_reports_its_method(exported: tuple[Path, Path]) -> None:
    result = benchmark(exported[1], warmup=1, runs=3, threads=1)
    assert result["ms_p50"] > 0 and result["ms_p95"] >= result["ms_p50"]
    assert result["threads"] == 1 and result["input"] == "1x4x96x96x3 uint8" and result["onnxruntime"]


def test_e1_and_onnx_predictions_agree_over_shards(exported: tuple[Path, Path], tmp_path: Path) -> None:
    checkpoint, onnx_path = exported
    rng = np.random.default_rng(0)
    with ShardWriter(tmp_path, prefix="t", clips_per_shard=10) as writer:
        writer.add("video/a", rng.integers(0, 256, (4, 96, 96, 3), dtype=np.uint8))
        writer.add("video/b", rng.integers(0, 256, (4, 96, 96, 3), dtype=np.uint8))
        writer.add("image/c", rng.integers(0, 256, (1, 96, 96, 3), dtype=np.uint8))
    wanted = {"video/a", "video/b", "image/c", "missing"}
    torch_preds, skipped = predict_shards("e1", checkpoint, tmp_path, wanted, device="cpu")
    assert sorted(torch_preds.clip_ids) == ["image/c", "video/a", "video/b"] and skipped["not_in_shards"] == 1
    onnx_preds, skipped = predict_shards("onnx", onnx_path, tmp_path, wanted)
    assert sorted(onnx_preds.clip_ids) == ["video/a", "video/b"] and skipped["frame_shape"] == 1
    shared = torch_preds.select(onnx_preds.clip_ids)
    for name, value in onnx_preds.logits.items():
        assert np.abs(value - shared.logits[name]).max() < 1e-3


def test_contract_states_sampling_domain_and_no_local_paths(exported: tuple[Path, Path]) -> None:
    terms = read_contract(exported[1])
    assert "round(2.0 * fps) // 16" in terms["temporal"]["training_clip"]
    assert "JPEG quality 90" in terms["training_domain"]
    assert terms["checkpoint"]["path"] == "e1.pt"  # outside the repository: the file name only


def test_failed_parity_leaves_nothing_at_out(exported: tuple[Path, Path], tmp_path: Path) -> None:
    checkpoint = exported[0]
    out = tmp_path / "e1.onnx"
    with pytest.raises(AssertionError, match="nothing was written"):
        export_edge(checkpoint, out, frames=4, input_size=96, tolerance=-1.0)
    assert list(tmp_path.iterdir()) == []
    wrong = np.zeros((1, 4, 80, 80, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="the graph takes"):
        export_edge(checkpoint, out, frames=4, input_size=96, parity_frames=wrong)
    assert list(tmp_path.iterdir()) == []


def test_benchmark_record_names_the_graph(exported: tuple[Path, Path], tmp_path: Path) -> None:
    import json
    import shutil

    from bulwark.eval.provenance import sha256_file
    from bulwark.export.onnx import write_benchmark

    graph = Path(shutil.copy(exported[1], tmp_path / "e1.onnx"))
    path = write_benchmark(graph, [benchmark(graph, warmup=1, runs=2, threads=1)])
    record = json.loads(path.read_text())
    assert path.name == "e1.onnx.bench.json" and record["onnx"] == {"path": "e1.onnx", "sha256": sha256_file(graph)}
    assert record["measured_utc"] and record["results"][0]["runs"] == 2
