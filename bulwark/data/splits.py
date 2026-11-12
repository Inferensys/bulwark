"""Subject-disjoint splits and held-out evaluation sets.

A subject never appears in two of train/val/test. Datasets that publish an official
subject split (Drive&Act split_0, AUC Distracted Driver v2's split by driver) use it, so
their test subjects are the official ones (the clip protocol can still differ; see the
adapter), but for one correction: AUC v2's adapter moves 83 camera-1 test images of a
driver who is also in train to train. AUC v2 publishes no driver ids, so its subjects are
two groups of drivers, one per side of its train/test split (`subject_group_datasets`): it
has no val subjects, and k-fold selection leaves its train group in training. Train-only
datasets (no person ids, or prompted labels) go to train whole, and excluded datasets
(`excluded_datasets`, a maintainer decision: the Zenodo driver set is out of v0) go to
"excluded", neither trained on nor scored. Every other dataset is split by a stable hash
of the subject id, with a floor on the number of val and test subjects for datasets large
enough to afford it; adding clips or datasets never moves an existing subject, as long as
a dataset's own subject list does not change.

Held-out sets measure generalisation:
  * holdout datasets (UTA-RLDD) are never trained on;
  * holdout views (Drive&Act's inner-mirror camera) record the same moments as the other
    cameras, so only the view's test-split subjects become "holdout" (new camera and new
    driver). The rest of the view is "excluded": neither trained on nor scored. Each
    view-holdout clip keeps its subject's split in `meta["subject_split"]`.

`policy_metadata` is the manifest metadata that records the policy (pass it to
`write_manifest(..., metadata=...)`), and `subject_folds` deals the train and val
subjects into folds for k-fold model selection, since a single val split holds very few
subjects of some datasets.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from bulwark.data.manifest import ClipRecord
from bulwark.data.removals import load_removed_subjects

# Manifest metadata key for the split policy (bulwark.eval.provenance reads it).
SPLIT_POLICY_KEY = b"split_policy"


def _removed_subjects() -> frozenset[str]:
    return frozenset(load_removed_subjects())


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    val_fraction: float = 0.1
    test_fraction: float = 0.15
    # Hash splits: each dataset with at least min_floor_subjects subjects gets at least
    # this many val and test subjects, taken in hash order.
    min_val_subjects: int = 2
    min_test_subjects: int = 3
    min_floor_subjects: int = 10
    # Datasets split by an official subject split: (dataset, protocol in meta["official_split"]).
    # Drive&Act's split_0, and AUC Distracted Driver v2's train/test split by driver ("v2").
    official_splits: tuple[tuple[str, str], ...] = (("drive_and_act", "0"), ("auc_v2", "v2"))
    # Datasets whose subject ids name a group of people, not one person. AUC v2 publishes a
    # train/test split by driver but no driver ids, so each side of its official split is
    # one group, assigned whole (no val). A group is not a person: k-fold selection leaves
    # it in training (`subject_folds`), and its test results rest on one group, so they get
    # no subject-level interval and are marked indicative (bulwark.eval.metrics). Cross-dataset
    # de-duplication (`bulwark split`) keeps a shared image in a dataset with person ids
    # rather than in one of these.
    subject_group_datasets: tuple[str, ...] = ("auc_v2",)
    # Whole datasets or dataset views kept out of training to measure generalisation.
    holdout_datasets: tuple[str, ...] = ("uta_rldd",)
    holdout_views: tuple[tuple[str, str], ...] = (("drive_and_act", "inner_mirror"),)
    # Datasets that publish no person ids (one person spans many samples, so any val/test
    # share would leak), or whose labels are prompted rather than annotated. Train only.
    train_only_datasets: tuple[str, ...] = ("zenodo_driver", "cew", "sh17", "synthetic_wan")
    # Whole datasets kept out of training and scoring by a maintainer decision, recorded in
    # the registry's `notes`/`terms` (zenodo_driver: its data use agreement forbids sharing
    # with third parties and asks to review publications 30 days ahead; excluded from v0).
    excluded_datasets: tuple[str, ...] = ("zenodo_driver",)
    salt: str = "bulwark-v1"
    # "<dataset>:<subject>" ids that must not appear in any manifest (removals.py).
    removed_subjects: frozenset[str] = field(default_factory=_removed_subjects)
    extra: dict[str, str] = field(default_factory=dict)


def policy_record(policy: SplitPolicy) -> dict[str, Any]:
    """The policy as JSON values. Removed subject ids are not published, only their count."""
    record: dict[str, Any] = {}
    for key, value in asdict(policy).items():
        record[key] = f"{len(value)} subject ids" if isinstance(value, (set, frozenset)) else value
    return record


def policy_metadata(policy: SplitPolicy) -> dict[bytes, bytes]:
    """Manifest metadata recording `policy`, for `write_manifest(..., metadata=...)`."""
    return {SPLIT_POLICY_KEY: json.dumps(policy_record(policy), sort_keys=True).encode()}


def subject_bucket(subject_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{subject_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def hash_splits(subjects: Iterable[str], policy: SplitPolicy) -> dict[str, str]:
    """train/val/test for one dataset's subjects, by hash bucket, with val/test floors."""
    ranked = sorted(set(subjects), key=lambda s: subject_bucket(s, policy.salt))
    buckets = [subject_bucket(s, policy.salt) for s in ranked]
    n_test = sum(b < policy.test_fraction for b in buckets)
    n_val = sum(b < policy.test_fraction + policy.val_fraction for b in buckets) - n_test
    if len(ranked) >= policy.min_floor_subjects:
        n_test = max(n_test, policy.min_test_subjects)
        n_val = max(n_val, policy.min_val_subjects)
    return {
        subject: "test" if rank < n_test else ("val" if rank < n_test + n_val else "train")
        for rank, subject in enumerate(ranked)
    }


def assign_splits(
    records: Iterable[ClipRecord], policy: SplitPolicy | None = None
) -> list[ClipRecord]:
    policy = policy or SplitPolicy()
    out = list(records)
    official = dict(policy.official_splits)
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for record in out:
        if (
            record.dataset not in policy.excluded_datasets
            and record.dataset not in policy.holdout_datasets
            and record.dataset not in policy.train_only_datasets
            and record.dataset not in official
        ):
            by_dataset[record.dataset].add(record.subject_id)
    hashed = {
        s: split
        for subjects in by_dataset.values()
        for s, split in hash_splits(subjects, policy).items()
    }

    for record in out:
        if record.dataset in policy.excluded_datasets:
            record.split = "excluded"
            continue
        if record.dataset in policy.holdout_datasets:
            record.split = "holdout"
            continue
        if record.dataset in policy.train_only_datasets:
            subject_split = "train"
        elif record.dataset in official:
            subject_split = _official_split(record, official[record.dataset])
        else:
            subject_split = hashed[record.subject_id]
        if (record.dataset, record.view) in policy.holdout_views:
            record.meta["subject_split"] = subject_split
            record.split = "holdout" if subject_split == "test" else "excluded"
        else:
            record.split = subject_split
    return out


def _official_split(record: ClipRecord, protocol: str) -> str:
    split = record.meta.get("official_split", {}).get(protocol)
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"{record.clip_id}: no official split_{protocol} assignment in meta; re-ingest "
            f"{record.dataset} with the current adapter"
        )
    return split


def check_disjoint(records: Sequence[ClipRecord], policy: SplitPolicy | None = None) -> None:
    """Raise if a manifest breaks the split policy:

    * a subject is in more than one of train/val/test;
    * a holdout dataset has clips outside holdout;
    * a holdout view has clips in train/val/test, or holdout clips of a subject that is
      in train or val;
    * an excluded dataset has clips outside excluded (`bulwark split` puts them there);
    * a train-only dataset has clips in val, test or holdout;
    * a removed subject (removed_subjects.yaml) is still present.
    """
    policy = policy or SplitPolicy()
    seen: dict[str, str] = {}
    for record in records:
        if record.split not in {"train", "val", "test"}:
            continue
        previous = seen.setdefault(record.subject_id, record.split)
        if previous != record.split:
            raise AssertionError(
                f"Subject {record.subject_id} is in both {previous} and {record.split}"
            )

    def fail(message: str, clips: list[str]) -> None:
        if clips:
            raise AssertionError(f"{message}: {clips[:5]} ({len(clips)} clips)")

    fail(
        "Held-out dataset clips outside holdout",
        [
            r.clip_id
            for r in records
            if r.dataset in policy.holdout_datasets and r.split != "holdout"
        ],
    )
    views = [r for r in records if (r.dataset, r.view) in policy.holdout_views]
    fail(
        "Held-out view clips in train/val/test",
        [r.clip_id for r in views if r.split not in {"holdout", "excluded"}],
    )
    fail(
        "Held-out view clips of subjects seen in train or val",
        [
            r.clip_id
            for r in views
            if r.split == "holdout" and seen.get(r.subject_id) in {"train", "val"}
        ],
    )
    fail(
        "Excluded dataset clips outside excluded (run `bulwark split` again)",
        [
            r.clip_id
            for r in records
            if r.dataset in policy.excluded_datasets and r.split != "excluded"
        ],
    )
    fail(
        "Train-only dataset clips outside train",
        [
            r.clip_id
            for r in records
            if r.dataset in policy.train_only_datasets and r.split in {"val", "test", "holdout"}
        ],
    )
    fail(
        "Removed subjects still in the manifest (delete them with "
        "bulwark.data.removals.purge_subjects)",
        [r.clip_id for r in records if r.subject_id in policy.removed_subjects],
    )


def subject_folds(
    records: Iterable[ClipRecord], k: int = 5, policy: SplitPolicy | None = None
) -> dict[str, int]:
    """Fold 0..k-1 of every train and val subject, for k-fold model selection.

    Each dataset's train and val subjects are pooled and dealt round-robin in hash order,
    so every fold holds about 1/k of each dataset's subjects, and a dataset with at least
    k subjects is in every fold. With k at least the largest dataset's subject count, a
    fold holds at most one subject per dataset (leave one subject out per dataset). Test,
    holdout and excluded clips are in no fold, and neither are train-only datasets, whose
    ids do not identify people, or subject-group datasets, whose ids name a group of
    people (AUC v2's train drivers); their clips always stay in training.
    """
    if k < 2:
        raise ValueError(f"k-fold selection needs k >= 2, got {k}")
    policy = policy or SplitPolicy()
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if (
            record.split in {"train", "val"}
            and record.dataset not in policy.train_only_datasets
            and record.dataset not in policy.subject_group_datasets
        ):
            by_dataset[record.dataset].add(record.subject_id)
    folds: dict[str, int] = {}
    for subjects in by_dataset.values():
        ranked = sorted(subjects, key=lambda s: subject_bucket(s, f"{policy.salt}:folds"))
        folds.update({subject: rank % k for rank, subject in enumerate(ranked)})
    return folds
