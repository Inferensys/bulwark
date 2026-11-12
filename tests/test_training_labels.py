"""What a training run learns and from which data: label masks, the split check, release_only,
and best.pt selection on the pairs a run trained."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch

from bulwark.data import registry
from bulwark.data.manifest import ClipRecord, write_manifest
from bulwark.data.taxonomy import default_taxonomy
from bulwark.model.bulwark import load_payload
from bulwark.train.common import Checkpointer, ResumeError, load_config, load_records
from bulwark.train.labels import ALL_MEDIA, label_plan
from bulwark.train.provenance import (
    ProvenanceError,
    build_provenance,
    checkpoint_identity,
    dataset_summary,
    library_versions,
    require_releasable,
)
from bulwark.train.validation import validate


def clip(i: int, media: str, gt: dict[str, Any], teacher: dict[str, Any] | None = None, *, split: str = "train",
         dataset: str = "drive_and_act", subject: str | None = None) -> ClipRecord:
    return ClipRecord(
        clip_id=f"{dataset}/{i}", dataset=dataset, subject_id=subject or f"{dataset}:{split}{i}", media_path=f"{i}",
        media_type=media, licence="test", commercial_ok=False, split=split, num_frames=1 if media == "image" else 16,
        labels_gt=gt, labels_teacher=teacher or {},
    )


def test_label_plan_masks_one_class_pairs_and_named_labels() -> None:
    taxonomy = default_taxonomy()
    records = [
        clip(0, "video", {"drink": 1.0, "activity": "drinking"}),
        clip(1, "video", {"drink": 1.0, "activity": "attentive"}),
        clip(2, "image", {"drink": 0.0, "attention_direction": "forward"}),
        clip(3, "image", {"drink": 1.0, "attention_direction": "away"}),
        clip(4, "image", {}, {"face_visible": 1.0}),
        clip(5, "video", {"yawn": 1.0}, split="val"),  # other splits do not count
    ]
    plan = label_plan(records, taxonomy, ["attention_direction"])
    assert plan.is_masked("drink", "video") and not plan.is_masked("drink", "image")
    assert plan.is_masked("attention_direction", "image") and plan.masked[("attention_direction", ALL_MEDIA)] == "mask_labels"
    assert plan.is_masked("face_visible", "image")  # teacher targets of one class count too
    assert ("activity", "video") in plan.missing_classes and "smoking" in plan.missing_classes[("activity", "video")]
    assert plan.supervised == tuple(n for n in taxonomy.labels if n in {"drink", "activity"})
    targets = plan.targets(records[:5], taxonomy)
    assert targets.mask["drink"].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert targets.mask["attention_direction"].sum() == 0 and targets.mask["face_visible"].sum() == 0
    assert plan.sample_mask(records[:4], ["drink"])["drink"].tolist() == [0.0, 0.0, 1.0, 1.0]
    summary = plan.summary()
    assert summary["masked"]["drink:video"].startswith("only '1'") and summary["supervised"] == list(plan.supervised)
    masked = plan.masked_media()  # what every checkpoint records as masked_labels
    assert masked["drink"] == ["video"] and masked["attention_direction"] == ["image", "video"]
    with pytest.raises(ValueError, match="does not have"):
        label_plan(records, taxonomy, ["no_such_label"])


def _manifest(path: Path, records: list[ClipRecord]) -> Any:
    write_manifest(records, path)
    return load_config(None, manifest=str(path))


def test_load_records_refuses_a_manifest_that_breaks_the_split_policy(tmp_path: Path) -> None:
    shared = [clip(0, "image", {}, subject="dmd:1"), clip(1, "image", {}, split="val", subject="dmd:1")]
    with pytest.raises(ValueError, match="breaks the split policy.*bulwark split"):
        load_records(_manifest(tmp_path / "m.parquet", shared))
    excluded = [clip(0, "image", {}), clip(1, "image", {}, split="excluded")]
    records = load_records(_manifest(tmp_path / "ok.parquet", excluded))
    assert len(records) == 2  # loaded, but no trainer selects the "excluded" split


def test_release_only_keeps_datasets_cleared_for_published_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cleared = dataclasses.replace(registry.get_spec("dmd"), weights_release_ok=True)
    monkeypatch.setitem(registry.BY_NAME, "dmd", cleared)
    rows = [clip(0, "video", {}, dataset="dmd"), clip(1, "video", {}, dataset="yawdd")]  # yawdd: not determined
    config = _manifest(tmp_path / "m.parquet", rows)
    config.release_only = True
    records = load_records(config)
    assert [r.dataset for r in records.values()] == ["dmd"]
    assert dataset_summary(records)["dmd"]["weights_release_ok"] is True
    provenance = build_provenance("heads", config, records, {})
    assert provenance["release_only"] is True and provenance["split_salt"] and provenance["taxonomy_version"]
    assert "bulwark_version" in provenance and "cuda" in provenance["versions"]

    require_releasable(None, "init")
    frozen = {"path": "x", "has_provenance": True, "datasets": None, "with_logits": False, "init": None}
    require_releasable(frozen, "features")  # no Bulwark training behind it, even without a recorded flag
    require_releasable({"path": "x", "has_provenance": True, "release_only": True, "datasets": {"dmd": {}}}, "init")
    with pytest.raises(ProvenanceError, match=r"weights_release_ok.*yawdd"):
        datasets = {"dmd": {}, "yawdd": {}}
        require_releasable({"path": "x", "has_provenance": True, "release_only": True, "datasets": datasets}, "teacher")
    with pytest.raises(ProvenanceError, match="no provenance"):
        require_releasable({"path": "x", "has_provenance": False}, "init")


def test_release_only_follows_the_whole_chain_not_one_hop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cleared = dataclasses.replace(registry.get_spec("dmd"), weights_release_ok=True)
    monkeypatch.setitem(registry.BY_NAME, "dmd", cleared)

    def checkpoint(name: str, release_only: bool, dataset: str, init: dict | None = None) -> dict:
        path = tmp_path / f"{name}.pt"
        provenance = {"release_only": release_only, "datasets": {dataset: {}}, "inputs": {"init": init}}
        torch.save({"model": {}, "provenance": provenance}, path)
        return checkpoint_identity(path, load_payload(path))

    # Heads on research-only data, then a LoRA run on releasable data only that starts from
    # them: the LoRA checkpoint's own datasets look releasable, but its flag does not.
    heads = checkpoint("heads", False, "yawdd")
    lora = checkpoint("lora", False, "dmd", init=heads)
    with pytest.raises(ProvenanceError, match="not produced with release_only=true"):
        require_releasable(lora, "init")
    clean_heads = checkpoint("heads-release", True, "dmd")
    require_releasable(checkpoint("lora-release", True, "dmd", init=clean_heads), "init")


def test_release_only_never_resumes_a_last_pt_without_releasable_provenance(tmp_path: Path) -> None:
    torch.save({"model": {}, "step": 3}, tmp_path / "last.pt")  # written before provenance existed
    with pytest.raises(ProvenanceError, match=r"last\.pt .*records no provenance.*fresh=true"):
        Checkpointer(tmp_path, 15, {**_provenance(release_only=True), "release_only": True}).resume(allow_mismatch=True)
    # Without release_only, allow_mismatch may still continue it.
    assert Checkpointer(tmp_path, 15, _provenance()).resume(allow_mismatch=True)["step"] == 3


def test_best_pt_selection_leaves_out_the_val_rows_of_masked_pairs() -> None:
    taxonomy = default_taxonomy()
    # activity has both classes on video, but only "other" on images: masked there.
    train = [clip(i, "video", {"activity": ("texting", "eating")[i % 2]}) for i in range(4)]
    train += [clip(4 + i, "image", {"activity": "other"}) for i in range(4)]
    plan = label_plan(train, taxonomy)
    assert plan.is_masked("activity", "image") and "activity" in plan.supervised
    val = [clip(10 + i, "video", {"activity": ("texting", "eating")[i % 2]}, split="val") for i in range(12)]
    val += [clip(30 + i, "image", {"activity": ("other", "eating")[i % 2]}, split="val") for i in range(12)]
    outputs = [{name: torch.zeros(len(val), taxonomy[name].width) for name in taxonomy.labels}]
    masked = validate(outputs, val, taxonomy, plan.supervised, plan)
    assert masked.support["activity"] == {"clips": 12, "subjects": 12}  # the video rows only
    everything = validate(outputs, val, taxonomy, plan.supervised)
    assert everything.support["activity"]["clips"] == 24
    assert masked.metrics == everything.metrics  # the logged metrics still cover every row


def _provenance(**config: object) -> dict:
    return {"job": "lora", "config": {"lr": 1e-3, **config}, "manifest_sha256": "m", "inputs": {}}


def test_resume_treats_config_keys_added_since_last_pt_as_their_defaults(tmp_path: Path) -> None:
    Checkpointer(tmp_path, 15, _provenance()).save("last.pt", {"model": {}, "step": 1})  # before the keys existed
    assert Checkpointer(tmp_path, 15, _provenance(release_only=False, mask_labels=[])).resume()["step"] == 1
    with pytest.raises(ResumeError, match=r"config\.mask_labels"):
        Checkpointer(tmp_path, 15, _provenance(mask_labels=["face_visible"])).resume()
    with pytest.raises(ResumeError, match="release_only cannot be overridden"):
        Checkpointer(tmp_path, 15, _provenance(release_only=True)).resume(allow_mismatch=True)


def test_checkpointer_writes_its_fields_into_every_payload(tmp_path: Path) -> None:
    checkpointer = Checkpointer(tmp_path, 15, fields={"supervised_labels": ("yawn",), "img_size": 224})
    payload = torch.load(checkpointer.save("last.pt", {"model": {}, "step": 1}), weights_only=True)
    assert payload["supervised_labels"] == ["yawn"] and payload["img_size"] == 224


def test_library_versions_name_the_code_and_the_accelerator_stack() -> None:
    versions = library_versions()
    assert {"bulwark", "python", "torch", "cuda", "numpy"} <= set(versions)
