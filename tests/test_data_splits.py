"""Split policy: official splits, view holdouts, train-only sources and the leak checks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from bulwark.data.manifest import ClipRecord, read_manifest_metadata, write_manifest
from bulwark.data.removals import purge_subjects
from bulwark.data.splits import (
    SPLIT_POLICY_KEY,
    SplitPolicy,
    assign_splits,
    check_disjoint,
    hash_splits,
    policy_metadata,
    policy_record,
    subject_bucket,
    subject_folds,
)

OFFICIAL = {"vp1": "train", "vp2": "train", "vp3": "val", "vp4": "test"}


def bucket_split(bucket: float, policy: SplitPolicy) -> str:
    if bucket < policy.test_fraction:
        return "test"
    return "val" if bucket < policy.test_fraction + policy.val_fraction else "train"


def clip(
    clip_id: str, subject: str, dataset: str = "state_farm", view: str = "", **kwargs: object
) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        dataset=dataset,
        subject_id=f"{dataset}:{subject}",
        media_path=f"{clip_id}.mp4",
        media_type="video",
        licence="t",
        commercial_ok=False,
        view=view,
        **kwargs,  # type: ignore[arg-type]
    )


def drive_and_act(view: str) -> list[ClipRecord]:
    return [
        clip(
            f"{view}-{vp}",
            vp,
            dataset="drive_and_act",
            view=view,
            meta={"official_split": {"0": split}},
        )
        for vp, split in OFFICIAL.items()
    ]


def test_official_split_is_used_for_drive_and_act() -> None:
    records = assign_splits(drive_and_act("a_column_driver"))
    assert {r.subject_id.split(":")[1]: r.split for r in records} == OFFICIAL
    check_disjoint(records)


def test_missing_official_split_asks_for_a_reingest() -> None:
    with pytest.raises(ValueError, match="re-ingest"):
        assign_splits([clip("x", "vp1", dataset="drive_and_act", view="ceiling")])


def auc(split: str, n: int = 3) -> list[ClipRecord]:
    """AUC v2 images as its adapter writes them: one subject group per side of the
    authors' split by driver."""
    return [
        clip(
            f"auc-{split}-{i}",
            f"{split}_drivers",
            dataset="auc_v2",
            view=("cam1", "cam2")[i % 2],
            meta={"official_split": {"v2": split}},
        )
        for i in range(n)
    ]


def test_auc_driver_groups_follow_the_authors_split_with_no_val() -> None:
    others = [clip(f"s{i}", str(i)) for i in range(30)] + [
        clip(f"d{i}", f"g{i}", dataset="dmd") for i in range(14)
    ]
    alone = {r.clip_id: r.split for r in assign_splits(others)}
    records = assign_splits(auc("train") + auc("test") + others)
    check_disjoint(records)
    groups = {r.subject_id: r.split for r in records if r.dataset == "auc_v2"}
    assert groups == {"auc_v2:train_drivers": "train", "auc_v2:test_drivers": "test"}
    # Adding AUC moves no other dataset's subject.
    assert {r.clip_id: r.split for r in records if r.dataset != "auc_v2"} == alone
    # A group is not a person: k-fold selection keeps the train group in training.
    folds = subject_folds(records, k=5)
    without = [r for r in records if r.dataset != "auc_v2"]
    assert "auc_v2:train_drivers" not in folds and folds == subject_folds(without, k=5)
    with pytest.raises(ValueError, match="re-ingest"):
        assign_splits([clip("x", "train_drivers", dataset="auc_v2")])  # no official split


def test_view_holdout_keeps_only_unseen_test_subjects() -> None:
    records = assign_splits(drive_and_act("a_column_driver") + drive_and_act("inner_mirror"))
    mirror = {r.subject_id.split(":")[1]: r for r in records if r.view == "inner_mirror"}
    assert mirror["vp4"].split == "holdout"  # test driver: new camera and new person
    assert {mirror[vp].split for vp in ("vp1", "vp2", "vp3")} == {"excluded"}
    assert mirror["vp1"].meta["subject_split"] == "train"
    check_disjoint(records)


def test_hash_split_floors_give_every_large_dataset_val_and_test_subjects() -> None:
    policy = SplitPolicy()
    splits = Counter(hash_splits([f"ds:{i}" for i in range(12)], policy).values())
    assert splits["test"] >= policy.min_test_subjects and splits["val"] >= policy.min_val_subjects
    # Too small for floors: plain bucket thresholds, even with fewer test subjects than
    # the floor asks for.
    few = [f"ds:{i}" for i in range(3)]
    small = hash_splits(few, policy)
    assert small == {s: bucket_split(subject_bucket(s, policy.salt), policy) for s in few}
    assert Counter(small.values())["test"] < policy.min_test_subjects
    # Stable: the same subjects always land in the same splits.
    assert hash_splits([f"ds:{i}" for i in reversed(range(12))], policy) == hash_splits(
        [f"ds:{i}" for i in range(12)], policy
    )


@pytest.mark.parametrize("dataset", ["synthetic_wan", "sh17", "cew"])
def test_train_only_datasets(dataset: str) -> None:
    records = assign_splits([clip(f"{dataset}{i}", str(i), dataset=dataset) for i in range(60)])
    assert {r.split for r in records} == {"train"}


def test_excluded_datasets_are_neither_trained_on_nor_scored() -> None:
    # zenodo_driver is out of v0 by a maintainer decision (its data use agreement; see the registry).
    records = assign_splits(
        [clip(f"z{i}", str(i), dataset="zenodo_driver") for i in range(30)]
        + [clip(f"s{i}", str(i)) for i in range(30)]
    )
    assert {r.split for r in records if r.dataset == "zenodo_driver"} == {"excluded"}
    assert "excluded" not in {r.split for r in records if r.dataset == "state_farm"}
    check_disjoint(records)


def test_check_disjoint_enforces_the_whole_policy() -> None:
    mirror = clip("m", "vp1", dataset="drive_and_act", view="inner_mirror", split="train")
    with pytest.raises(AssertionError, match="Held-out view"):
        check_disjoint([mirror])

    seen = clip("a", "vp2", dataset="drive_and_act", view="ceiling", split="train")
    held = clip("b", "vp2", dataset="drive_and_act", view="inner_mirror", split="holdout")
    with pytest.raises(AssertionError, match="seen in train"):
        check_disjoint([seen, held])

    cew = clip("c", "x", dataset="cew", split="test")
    with pytest.raises(AssertionError, match="Train-only"):
        check_disjoint([cew])

    # A manifest split before the Zenodo exclusion (zenodo_driver was train-only then).
    zen = clip("z", "x", dataset="zenodo_driver", split="train")
    with pytest.raises(AssertionError, match="Excluded dataset"):
        check_disjoint([zen])
    check_disjoint([clip("z", "x", dataset="zenodo_driver", split="excluded")])

    removed = clip("r", "gA_9", dataset="dmd", split="train")
    policy = SplitPolicy(removed_subjects=frozenset({"dmd:gA_9"}))
    with pytest.raises(AssertionError, match="Removed subjects"):
        check_disjoint([removed], policy)
    check_disjoint([removed], SplitPolicy(removed_subjects=frozenset()))


def test_training_and_evaluation_refuse_a_manifest_split_before_the_zenodo_exclusion(
    tmp_path: Path,
) -> None:
    from bulwark.eval.report import check_splits, manifest_policy
    from bulwark.train.common import load_config, load_records

    # The policy such a manifest records has no `excluded_datasets` field yet; the current
    # exclusion applies anyway.
    old_policy = {k: v for k, v in policy_record(SplitPolicy()).items() if k != "excluded_datasets"}
    records = [
        clip("z", "x", dataset="zenodo_driver", split="train"),
        clip("d", "gA_1", dataset="dmd", split="train"),
    ]
    path = write_manifest(
        records,
        tmp_path / "manifest.parquet",
        metadata={SPLIT_POLICY_KEY: json.dumps(old_policy).encode()},
    )
    policy = manifest_policy(path)
    assert policy.excluded_datasets == ("zenodo_driver",)
    with pytest.raises(ValueError, match="Excluded dataset.*bulwark split"):
        check_splits(records, path, policy)
    with pytest.raises(ValueError, match="Excluded dataset.*bulwark split"):
        load_records(load_config(None, manifest=str(path)))


def test_the_manifest_records_its_split_policy(tmp_path: Path) -> None:
    from bulwark.eval.provenance import split_policy

    policy = SplitPolicy(salt="custom", removed_subjects=frozenset({"dmd:gA_9", "dmd:gB_2"}))
    records = assign_splits(
        drive_and_act("a_column_driver") + [clip("d", "gA_1", dataset="dmd")], policy
    )
    path = write_manifest(records, tmp_path / "manifest.parquet", metadata=policy_metadata(policy))
    recorded = split_policy(path)
    assert recorded["source"] == "manifest" and recorded["salt"] == "custom"
    assert recorded["removed_subjects"] == "2 subject ids"  # ids are not published
    assert "gA_9" not in read_manifest_metadata(path)[SPLIT_POLICY_KEY].decode()
    # Purging rewrites the manifest and keeps the policy.
    purge_subjects(["dmd:gA_1"], manifests=[path])
    assert json.loads(read_manifest_metadata(path)[SPLIT_POLICY_KEY])["salt"] == "custom"
    # A manifest written without metadata has none of its own.
    assert read_manifest_metadata(write_manifest(records, tmp_path / "plain.parquet")) == {}


def test_subject_folds_spread_each_dataset_over_every_fold() -> None:
    records = [
        clip(f"s{i}", str(i), dataset="dmd", split="train" if i % 4 else "val") for i in range(12)
    ]
    records += [clip(f"t{i}", f"t{i}", dataset="dmd", split="test") for i in range(3)]
    records += [clip("c", "c", dataset="cew", split="train")]  # train-only: no person ids
    records += [clip("m", "vp9", dataset="drive_and_act", view="inner_mirror", split="excluded")]
    folds = subject_folds(records, k=4)
    assert set(folds) == {f"dmd:{i}" for i in range(12)}  # train and val subjects only
    assert Counter(folds.values()) == {0: 3, 1: 3, 2: 3, 3: 3}
    assert subject_folds(list(reversed(records)), k=4) == folds  # order does not matter
    # One subject per fold and dataset: leave one subject out.
    assert sorted(subject_folds(records, k=12).values()) == list(range(12))
    with pytest.raises(ValueError, match="k >= 2"):
        subject_folds(records, k=1)
