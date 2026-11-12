"""Label coverage report and single-class detection."""

from __future__ import annotations

import json

from bulwark.data.coverage import (
    coverage_markdown,
    label_coverage,
    missing_classes,
    single_class_labels,
)
from bulwark.data.manifest import ClipRecord


def clip(
    i: int,
    media_type: str,
    gt: dict[str, object],
    teacher: dict[str, object] | None = None,
    split: str = "train",
) -> ClipRecord:
    return ClipRecord(
        clip_id=f"c{i}",
        dataset="drive_and_act",
        subject_id="drive_and_act:vp1",
        media_path="x",
        media_type=media_type,
        licence="t",
        commercial_ok=False,
        split=split,
        labels_gt=gt,
        labels_teacher=teacher or {},
    )


def test_coverage_flags_labels_with_one_class() -> None:
    records = [
        clip(0, "video", {"drink": 1.0, "activity": "drinking"}),
        clip(1, "video", {"drink": 1.0, "activity": "attentive"}),
        clip(2, "image", {"drink": 1.0}),
        clip(3, "image", {}, {"drink": 0.0}),
    ]
    flagged = single_class_labels(records)
    assert ("drink", "video") in flagged and ("drink", "image") not in flagged
    assert ("activity", "video") not in flagged
    rows = {(r.label, r.media_type, r.source): r for r in label_coverage(records)}
    assert rows[("drink", "video", "gt")].classes == {"1": 2}
    assert rows[("drink", "image", "teacher")].classes == {"0": 1}
    json.dumps([r.classes for r in rows.values()])  # plain data, ready for a report
    assert "| drink | drive_and_act | train | video | gt | 2 | 1: 2 |" in coverage_markdown(
        list(rows.values())
    )


def test_missing_classes_catch_two_class_categoricals() -> None:
    records = [
        clip(0, "video", {"attention_direction": "away", "drink": 1.0}),
        clip(1, "video", {"attention_direction": "down_center"}, {"drink": 0.0}),
        clip(2, "image", {"attention_direction": "forward"}, split="val"),
    ]
    assert not single_class_labels(records)  # two classes each, so not flagged there
    assert missing_classes(records) == {
        ("attention_direction", "video"): ("forward", "left", "right", "up", "down")
    }
