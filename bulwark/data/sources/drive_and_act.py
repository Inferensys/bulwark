"""Drive&Act (Fraunhofer IOSB), mid-level activities in 3 s chunks.

    activities_3s/<view>/midlevel.chunks_90.split_0.{train,val,test}.csv
        participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id
    <view>/vp<n>/run<k>_<date-time>.<view>.mp4

The union of the split_0 train/val/test files covers every chunk once. The official
protocols (split_0, split_1, split_2) are subject splits; each record keeps its drivers'
assignment in `meta["official_split"]` (protocol -> train/val/test), and
`bulwark.data.splits` assigns subjects by split_0. Only the subject split follows the
official protocol. The published benchmark classifies the 3 s chunks themselves; here the
views are pooled, clips are shorter, activities shorter than one clip are dropped and the
inner-mirror view is kept out of training (its test drivers are a cross-view holdout), so
the numbers are not directly comparable.

The five NIR views and the Kinect colour and IR views are ingested. The Kinect depth view
has annotation files too, but depth maps are none of the taxonomy's modalities, so it is
skipped.

Frames are sampled with the stride rule every video adapter shares
(`base.clip_stride`), centred on the chunk. At 30 fps that is every 3rd frame (16 frames
cover about 1.5 s); the 15 fps kinect_color view gets stride 1 and covers about 1.07 s,
so its clips play about 1.5x slower than those of the NIR views. A chunk shorter than
one clip borrows frames from the rest of its annotation (the same activity); activities
shorter than one clip are skipped rather than read into the next activity.

Clip ids name the window actually read (`<video>@<start>s<stride>:<chunk_id>`), so shards,
teacher labels and feature caches built with another sampling never match a new manifest.

The cameras record the same moments, and every view's annotation file lists the same
(participant, run, annotation_id, chunk_id) keys with the same activity (checked
2026-10-29 on the split_0 files: all 10,294 keys are in all 8 views). Frame numbers
differ between views (the kinect_color view runs at 15 fps), so `meta["event_id"]` is
built from those keys, not from frames: clips of one chunk from different cameras share
it, and evaluation counts them as one event.

phone, drink and food are 1 only in the activities that involve them (`ACTIVITY_LABELS`).
Their 0 comes from the object-level annotations of the same view
(`activities_3s/<view>/objectlevel.chunks_90.csv`, whose frame numbers match the
mid-level ones): each annotation names the object the driver reaches for, holds, uses or
puts away. A phone annotation covers 99.97% of the frames of the phone activities, a
bottle one 99.9% of the drinking and bottle ones, and a food one 95.8% of the eating and
food-preparing ones (checked 2026-10-29 in every ingested view). A clip whose activity
leaves phone, drink or food unset gets 0 for it when no annotation of a phone, bottle or
food object (`OBJECT_LABELS`) overlaps the frames it reads; phone 0 also sets
phone_at_ear 0. A clip that one overlaps (fetching or placing the object, say) stays
unlabelled for it, and so does every clip of a run without object-level annotations.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import (
    VIDEO_SUFFIXES,
    SourceAdapter,
    clip_stride,
    iter_files,
    probe_video,
)

ACTIVITY_LABELS: dict[str, dict[str, Any]] = {
    "sitting_still": {"activity": "attentive"},
    "drinking": {"activity": "drinking", "drink": 1.0},
    "eating": {"activity": "eating", "food": 1.0},
    "preparing_food": {"activity": "eating", "food": 1.0},
    "interacting_with_phone": {"activity": "texting", "phone": 1.0, "phone_at_ear": 0.0},
    "talking_on_phone": {"activity": "phone_call", "phone": 1.0, "phone_at_ear": 1.0},
    "reading_magazine": {"activity": "reading"},
    "reading_newspaper": {"activity": "reading"},
    "working_on_laptop": {"activity": "interacting_object"},
    "writing": {"activity": "interacting_object"},
    "opening_laptop": {"activity": "interacting_object"},
    "closing_laptop": {"activity": "interacting_object"},
    "opening_bottle": {"activity": "interacting_object", "drink": 1.0},
    "closing_bottle": {"activity": "interacting_object", "drink": 1.0},
    "using_multimedia_display": {
        "activity": "adjusting_controls",
        "attention_direction": "down_center",
    },
    "pressing_automation_button": {"activity": "adjusting_controls"},
    "looking_or_moving_around": {"activity": "looking_away", "attention_direction": "away"},
    "fetching_an_object": {"activity": "reaching"},
    "placing_an_object": {"activity": "reaching"},
    "opening_backpack": {"activity": "interacting_object"},
    "putting_laptop_into_backpack": {"activity": "interacting_object"},
    "taking_laptop_from_backpack": {"activity": "interacting_object"},
    "putting_on_sunglasses": {"activity": "grooming"},
    "taking_off_sunglasses": {"activity": "grooming"},
    "putting_on_jacket": {"activity": "other"},
    "taking_off_jacket": {"activity": "other"},
    "fastening_seat_belt": {"activity": "other"},
    "unfastening_seat_belt": {"activity": "other"},
    "entering_car": {"activity": "other"},
    "exiting_car": {"activity": "other"},
    "opening_door_inside": {"activity": "other"},
    "closing_door_inside": {"activity": "other"},
    "opening_door_outside": {"activity": "other"},
    "closing_door_outside": {"activity": "other"},
}


OFFICIAL_SPLIT_FILE = re.compile(r"^midlevel\.chunks_90\.split_(\d+)\.(train|val|test)\.csv$")
SKIPPED_VIEWS = frozenset({"kinect_depth"})  # depth maps: no taxonomy modality
OBJECT_FILE = "objectlevel.chunks_90.csv"  # every object-level chunk of a view, all splits
# label -> the object whose object-level annotations rule it out where they are absent.
OBJECT_LABELS = {"phone": "phone", "drink": "bottle", "food": "food"}


class DriveAndActAdapter(SourceAdapter):
    name = "drive_and_act"
    annotation_glob = "activities_3s/*/midlevel.chunks_90.split_0.*.csv"

    def records(self) -> Iterator[ClipRecord]:
        videos = {_video_key(path): path for path in iter_files(self.root, VIDEO_SUFFIXES)}
        official = official_splits(self.root)
        rows_by_view: dict[str, list[dict[str, str]]] = {}
        for csv_path in sorted(self.root.glob(self.annotation_glob)):
            if csv_path.parent.name in SKIPPED_VIEWS:
                continue
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows_by_view.setdefault(csv_path.parent.name, []).extend(csv.DictReader(handle))
        seen: set[str] = set()
        for view, rows in rows_by_view.items():
            spans = _annotation_spans(rows)
            objects = _object_spans(self.root / "activities_3s" / view / OBJECT_FILE)
            for row in rows:
                video = videos.get(_key_from_file_id(row["file_id"]))
                # e.g. "looking_or_moving_around (e.g. searching)" -> "looking_or_moving_around"
                activity = row["activity"].split("(")[0].strip().lower()
                if video is None or activity not in ACTIVITY_LABELS:
                    continue
                record = self._chunk(
                    video,
                    view,
                    row,
                    spans[_annotation_key(row)],
                    ACTIVITY_LABELS[activity],
                    objects.get(row["file_id"].strip()),
                )
                if record is None or record.clip_id in seen:
                    continue
                seen.add(record.clip_id)
                if splits := official.get(row["participant_id"].strip()):
                    record.meta["official_split"] = dict(splits)
                yield record

    def _chunk(
        self,
        video: Path,
        view: str,
        row: dict[str, str],
        annotation: tuple[int, int],
        labels: dict[str, Any],
        objects: dict[str, list[tuple[int, int]]] | None,
    ) -> ClipRecord | None:
        fps = _fps(video)
        stride = clip_stride(fps, self.clip_seconds, self.clip_frames)
        need = (self.clip_frames - 1) * stride + 1
        first, last = annotation
        # Activities shorter than one clip (and the ceiling view's zero-length rows,
        # clamped to the last frame) would spill into the next activity.
        if last - first < need:
            return None
        chunk_start, chunk_end = int(row["frame_start"]), int(row["frame_end"])
        start = min(max((chunk_start + chunk_end) // 2 - need // 2, first), last - need)
        rel = self.rel(video)
        return ClipRecord(
            clip_id=f"{self.name}/{rel}@{start}s{stride}:{row['chunk_id']}",
            dataset=self.name,
            subject_id=self.subject(f"vp{row['participant_id'].strip()}"),
            media_path=rel,
            media_type="video",
            licence=self.spec.licence,
            commercial_ok=self.spec.commercial_ok,
            start_frame=start,
            num_frames=self.clip_frames,
            fps=fps,
            frame_stride=stride,
            view=view,
            labels_gt={
                **labels,
                **_absent_objects(objects, labels, start, start + need),
                "modality": "rgb" if view == "kinect_color" else "nir",
                "context": "vehicle_cabin",
            },
            meta={"native_activity": row["activity"], "event_id": _event_id(row)},
        )


def official_splits(root: Path) -> dict[str, dict[str, str]]:
    """{participant_id: {protocol: split}} from every official split file of every view."""
    found: dict[str, dict[str, str]] = {}
    for csv_path in sorted(root.glob("activities_3s/*/midlevel.chunks_90.split_*.csv")):
        match = OFFICIAL_SPLIT_FILE.match(csv_path.name)
        if match is None:
            continue
        protocol, split = match.groups()
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for participant in {row["participant_id"].strip() for row in csv.DictReader(handle)}:
                previous = found.setdefault(participant, {}).setdefault(protocol, split)
                if previous != split:
                    raise ValueError(
                        f"Drive&Act participant {participant} is in both {previous} and {split} "
                        f"of split_{protocol}"
                    )
    return found


def _event_id(row: dict[str, str]) -> str:
    """drive_and_act/vp<participant>/<run>/<annotation_id>/<chunk_id>, the same in every view."""
    run = row["file_id"].strip().strip("/").split("/")[-1].rsplit(".", 1)[0]  # drop ".ids_1"
    participant, annotation = row["participant_id"].strip(), row["annotation_id"].strip()
    return f"drive_and_act/vp{participant}/{run}/{annotation}/{row['chunk_id'].strip()}"


def _annotation_key(row: dict[str, str]) -> tuple[str, str]:
    return row["file_id"].strip(), row["annotation_id"].strip()


def _annotation_spans(rows: list[dict[str, str]]) -> dict[tuple[str, str], tuple[int, int]]:
    """[first, last) frame of each annotated activity, over all of its 3 s chunks."""
    spans: dict[tuple[str, str], tuple[int, int]] = {}
    for row in rows:
        start, end = int(row["frame_start"]), int(row["frame_end"])
        first, last = spans.get(_annotation_key(row), (start, end))
        spans[_annotation_key(row)] = (min(first, start), max(last, end))
    return spans


def _object_spans(path: Path) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """{file_id: {object: [first, last) of each of its annotations}} from a view's
    object-level file; {} when the view has none."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = {_annotation_key(row): row["object"].strip().lower() for row in rows}
    found: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for key, span in _annotation_spans(rows).items():
        found.setdefault(key[0], {}).setdefault(names[key], []).append(span)
    return found


def _absent_objects(
    objects: dict[str, list[tuple[int, int]]] | None, labels: dict[str, Any], start: int, end: int
) -> dict[str, float]:
    """0 for each object label that `labels` leaves unset and whose object no annotation
    of the run overlaps in [start, end); nothing for a run without object annotations."""
    if objects is None:
        return {}
    absent: dict[str, float] = {}
    for label, name in OBJECT_LABELS.items():
        if label not in labels and not any(first < end and start < last for first, last in objects.get(name, ())):
            absent[label] = 0.0
    if "phone" in absent and "phone_at_ear" not in labels:
        absent["phone_at_ear"] = 0.0
    return absent


def _video_key(path: Path) -> str:
    # "<view>/vp<n>/run<k>_...<view>.mp4" -> "vp<n>/run<k>_...<view>"
    return f"{path.parent.name}/{path.stem}"


def _key_from_file_id(file_id: str) -> str:
    parts = file_id.strip().strip("/").split("/")
    return "/".join(parts[-2:])


@lru_cache(maxsize=4096)
def _fps(path: Path) -> float:
    return probe_video(path)[0] or 30.0
