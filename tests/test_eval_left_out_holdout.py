"""A manifest staged for Kaggle or Azure without the datasets not cleared for third-party
storage records what it left out, so a remote evaluation reports the UTA-RLDD holdout as
left out of the upload rather than as not ingested."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow.parquet as pq

from bulwark.data.manifest import (
    ClipRecord,
    left_out_datasets,
    read_manifest,
    read_manifest_metadata,
    write_manifest,
)
from bulwark.data.splits import SplitPolicy, policy_metadata
from bulwark.eval.report import holdout_groups, manifest_record

STAGE_DATA = Path(__file__).parents[1] / "infra" / "kaggle" / "stage_data.py"


def stage_data():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("stage_data_left_out", STAGE_DATA)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["stage_data_left_out"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def clip(dataset: str, i: int, split: str) -> ClipRecord:
    return ClipRecord(
        clip_id=f"{dataset}/{i}", dataset=dataset, subject_id=f"{dataset}:{i}", media_path=f"{i}.mp4",
        media_type="video", licence="t", commercial_ok=False, split=split, labels_gt={"eyes_closed": 0.0},
    )


def test_a_holdout_left_out_of_the_upload_is_not_reported_as_not_ingested(tmp_path: Path) -> None:
    records = [*(clip("dmd", i, "test") for i in range(3)), *(clip("uta_rldd", i, "holdout") for i in range(2))]
    source = write_manifest(records, tmp_path / "manifest.parquet", metadata=policy_metadata(SplitPolicy()))
    assert left_out_datasets(source) == {}

    target = tmp_path / "staged" / "manifest.parquet"
    target.parent.mkdir()
    written, kept, dropped = stage_data().stage_manifest(source, target, {"dmd"}, drop=True)
    assert written and kept == 3 and dropped == {"uta_rldd": 2}
    # Same columns and the same split policy; only the left-out record is added.
    assert pq.read_schema(target).names == pq.read_schema(source).names
    assert read_manifest_metadata(target).items() >= read_manifest_metadata(source).items()
    assert left_out_datasets(target) == {"uta_rldd": 2}
    assert manifest_record(target, kept)["left_out_datasets"] == {"uta_rldd": 2}
    assert "left_out_datasets" not in manifest_record(source, len(records))

    staged = holdout_groups(read_manifest(target), SplitPolicy(), left_out_datasets(target))["uta_rldd"]
    assert staged["clips"] == 0 and "left out of the upload" in staged["status"]
    assert "not ingested" not in staged["status"]
    # A manifest that never held the dataset still says so.
    local = holdout_groups([r for r in records if r.dataset == "dmd"], SplitPolicy())["uta_rldd"]
    assert local["status"] == "not available (not ingested)"
