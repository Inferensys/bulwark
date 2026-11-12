from __future__ import annotations

from pathlib import Path

import pytest

from bulwark.data.manifest import ClipRecord, read_manifest, write_manifest
from bulwark.data.registry import DATASETS, datasets_markdown
from bulwark.data.sources import ADAPTERS
from bulwark.data.splits import SplitPolicy, assign_splits, check_disjoint
from bulwark.data.taxonomy import default_taxonomy


def record(clip: str, subject: str, dataset: str = "state_farm", **kwargs: object) -> ClipRecord:
    return ClipRecord(
        clip_id=clip,
        dataset=dataset,
        subject_id=f"{dataset}:{subject}",
        media_path=f"{clip}.jpg",
        media_type="image",
        licence="test",
        commercial_ok=False,
        **kwargs,  # type: ignore[arg-type]
    )


def test_taxonomy_validates_names_ranges_and_classes() -> None:
    taxonomy = default_taxonomy()
    assert taxonomy["activity"].width == len(taxonomy["activity"].classes)
    assert taxonomy["yawn"].width == 1
    taxonomy.validate({"yawn": 1.0, "activity": "texting", "head_yaw": -0.3})
    with pytest.raises(KeyError):
        taxonomy.validate({"not_a_label": 1})
    with pytest.raises(ValueError):
        taxonomy.validate({"activity": "juggling"})
    with pytest.raises(ValueError):
        taxonomy.validate({"perclos": 1.5})


def test_every_adapter_has_a_registry_entry_marked_as_adapter() -> None:
    by_name = {spec.name: spec for spec in DATASETS}
    for name in ADAPTERS:
        assert by_name[name].adapter, name
    assert {s.name for s in DATASETS if s.adapter} == set(ADAPTERS)
    assert "| Commercial |" in datasets_markdown()


def test_manifest_roundtrip_and_ground_truth_wins(tmp_path: Path) -> None:
    a = record(
        "a",
        "1",
        labels_gt={"activity": "texting"},
        labels_teacher={"activity": "attentive", "yawn": 0.0},
        crop=(1, 2, 3, 4),
    )
    path = write_manifest([a.validate()], tmp_path / "m.parquet")
    (loaded,) = read_manifest(path)
    assert loaded == a
    assert loaded.labels == {"activity": "texting", "yawn": 0.0}
    with pytest.raises(ValueError):
        write_manifest([a, a], tmp_path / "dup.parquet")


def test_subject_prefix_is_enforced() -> None:
    bad = record("a", "1")
    bad.subject_id = "1"
    with pytest.raises(ValueError):
        bad.validate()


def test_splits_are_subject_disjoint_stable_and_hold_out_datasets() -> None:
    records = [record(f"c{i}", str(i % 40)) for i in range(400)]
    records += [record(f"u{i}", str(i), dataset="uta_rldd") for i in range(10)]
    first = {r.clip_id: r.split for r in assign_splits(records)}
    check_disjoint(records)
    assert {first[f"u{i}"] for i in range(10)} == {"holdout"}
    assert {"train", "val", "test"} <= set(first.values())
    again = {
        r.clip_id: r.split
        for r in assign_splits([record(f"c{i}", str(i % 40)) for i in range(400)])
    }
    assert all(first[k] == again[k] for k in again)


def test_check_disjoint_catches_leaks() -> None:
    a, b = record("a", "1"), record("b", "1")
    a.split, b.split = "train", "test"
    with pytest.raises(AssertionError):
        check_disjoint([a, b])
    held = record("u", "1", dataset="uta_rldd")
    held.split = "train"
    with pytest.raises(AssertionError):
        check_disjoint([held], SplitPolicy())


def test_train_only_datasets_never_reach_eval_splits() -> None:
    records = assign_splits([record(f"c{i}", "x", dataset="cew") for i in range(50)])
    assert {r.split for r in records} == {"train"}


def test_cross_dataset_duplicates_keep_the_evaluable_copy(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from bulwark.data.dedup import drop_cross_dataset_duplicates

    rng = np.random.default_rng(0)
    photo = cv2.GaussianBlur((rng.random((96, 128)) * 255).astype(np.uint8), (0, 0), 6)
    other = cv2.GaussianBlur((rng.random((96, 128)) * 255).astype(np.uint8), (0, 0), 6)
    (tmp_path / "sf").mkdir()
    (tmp_path / "zen").mkdir()
    cv2.imwrite(str(tmp_path / "sf" / "img_1.jpg"), photo)
    cv2.imwrite(
        str(tmp_path / "zen" / "img_1.jpg"),
        cv2.resize(photo, (683, 512)),
        [cv2.IMWRITE_JPEG_QUALITY, 60],
    )  # upscaled re-encode
    cv2.imwrite(str(tmp_path / "zen" / "img_2.jpg"), other)
    sf = record("img_1", "p1")
    sf.media_path = "img_1.jpg"
    zen = [record(f"zen{i}", f"z{i}", dataset="zenodo_driver") for i in (1, 2)]
    zen[0].media_path, zen[1].media_path = "img_1.jpg", "img_2.jpg"
    kept, dropped = drop_cross_dataset_duplicates(
        [sf, *zen],
        {"state_farm": tmp_path / "sf", "zenodo_driver": tmp_path / "zen"},
        prefer_evaluable=["state_farm"],
    )
    assert dropped == 1 and {r.clip_id for r in kept} == {"img_1", "zen2"}


def test_near_duplicate_groups_use_hamming_distance() -> None:
    from bulwark.data.dedup import near_duplicate_groups

    base = 0x0123_4567_89AB_CDEF
    groups = near_duplicate_groups(
        [base, base ^ 0b101, base ^ 0xFFFF_0000_0000_0000, base ^ (0b1111111 << 20)]
    )
    assert sorted(map(sorted, groups)) == [[0, 1], [2], [3]]
    # Same hash, but a 16:9 photo is no copy of a square crop; 4:3 upscaled to 683x512 is.
    groups = near_duplicate_groups(
        [base, base, base ^ 0b1], aspects=[683 / 384, 1.0, 683 / 512 * 1.01]
    )
    assert sorted(map(sorted, groups)) == [[0], [1], [2]]
    assert near_duplicate_groups([base, base], aspects=[4 / 3, 683 / 512]) == [[0, 1]]
    # Two near-identical frames of one dataset, and a copy of the first in another: with
    # sources, only the cross-dataset pair is joined (no chain through the second frame).
    chain = [base, base ^ 0b111111, base ^ (0b111111 << 20)]  # 0-1: 6 bits, 0-2: 6, 1-2: 12
    assert sorted(map(sorted, near_duplicate_groups(chain))) == [[0, 1, 2]]
    groups = near_duplicate_groups(chain, sources=["auc_v2", "auc_v2", "state_farm"])
    assert sorted(map(sorted, groups)) == [[0, 2], [1]]


def test_dedup_drops_only_images_that_match_another_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bulwark.data import dedup

    base = 0x0123_4567_89AB_CDEF
    hashes = {"a1.jpg": base, "a2.jpg": base ^ 0b111111, "s1.jpg": base ^ (0b111111 << 20)}
    monkeypatch.setattr(dedup, "dhash", lambda path, crop=None: (hashes[path.name], 4 / 3))
    records = [record("a1", "train_drivers", "auc_v2"), record("a2", "train_drivers", "auc_v2"),
               record("s1", "p1")]  # fmt: skip
    roots = {"auc_v2": tmp_path, "state_farm": tmp_path}
    kept, dropped = dedup.drop_cross_dataset_duplicates(
        records, roots, prefer_evaluable=["state_farm", "auc_v2"]
    )
    # a1 is a copy of s1 and goes; a2 is only near a1, its own dataset's frame, and stays.
    assert dropped == 1 and {r.clip_id for r in kept} == {"a2", "s1"}
    assert next(r for r in kept if r.clip_id == "s1").meta["near_duplicate_of"] == ["auc_v2"]
    assert "near_duplicate_of" not in next(r for r in kept if r.clip_id == "a2").meta
