"""DMD - Driver Monitoring Dataset (Vicomtech), OpenLABEL annotations.

    dmd/gX/0/s1/gX_0_s1_<timestamp>_rgb_ann_distraction.json
    dmd/gX/0/s1/gX_0_s1_<timestamp>_rgb_face.mp4   (also _body, _hands, _mosaic)

Annotations are `openlabel.actions[*]` with a `type` such as
`driver_actions/texting_right` or `eyes_state/close`, each with `frame_intervals`, and
`openlabel.objects[*]` with a `type` such as `cellphone`, marked the same way. We rebuild
per-frame tracks and summarise them over sliding clip windows.

Annotation frames follow the face camera (`streams.face_camera...sync.frame_shift` is 0;
body is ~54 frames late, hands ~74), checked on the 2026 RGB release, where all 97
annotated sessions have a face video. Only the face video is used: the body and hands
videos are misaligned and the mosaic is a multi-camera collage. Labels the face camera
cannot see (hands on the wheel) are not attached to face clips.

phone and drink are 1 when the window's majority driver action involves them
(`DRIVER_ACTIONS`). Their 0 comes from the distraction files' own annotations:
`driver_actions` gives each frame one action (`unclassified` marks transitions), and the
`cellphone` and `bottle` objects (one track each) are annotated while the driver handles
them: they cover every texting, phone-call and drinking frame and, around those, picking
the object up and putting it back (checked 2026-10-29 on the 2026 RGB release). A window
whose every frame has a known action, none `unclassified`, gets phone 0 (and
phone_at_ear 0) when no frame is texting or a phone call and none has the cellphone
object, and drink 0 when no frame is drinking and none has the bottle. These 0s are added
only to windows kept for their other labels, so they add no clips. food is never set:
DMD has no eating action and annotates no food object, and food positives come only from
Drive&Act's cabin cameras, so a food 0 on every DMD clip would rest on the recording
protocol alone and teach that DMD's face camera never shows food.

Eye labels use the annotated eye states of the window, excluding `undefined`: `perclos`
is the fraction of those frames with the eyes `close`, and `eyes_closed` follows the
taxonomy's last-half rule (`label_rules.eyes_closed_from_mask`). DMD annotates eye states,
blinks and yawns, not alertness. A window whose eyes stay closed without a break for at
least MICROSLEEP_MIN_CLOSED_SECONDS gets alertness "microsleep" as a derived label
(`meta["derived_labels"]`, merged into `labels_teacher`), never as ground truth: the
threshold is this project's choice, since definitions of microsleep in the literature
differ. The closure is measured inside the window (48 frames, 1.6 s at 30 fps), so a
closure split across two windows can be missed by both.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bulwark.data.label_rules import eyes_closed_from_mask
from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import SourceAdapter, probe_video

DRIVER_ACTIONS: dict[str, dict[str, Any]] = {
    "safe_drive": {"activity": "attentive"},
    "texting_right": {"activity": "texting", "phone": 1.0, "phone_at_ear": 0.0},
    "texting_left": {"activity": "texting", "phone": 1.0, "phone_at_ear": 0.0},
    "phonecall_right": {"activity": "phone_call", "phone": 1.0, "phone_at_ear": 1.0},
    "phonecall_left": {"activity": "phone_call", "phone": 1.0, "phone_at_ear": 1.0},
    "radio": {"activity": "adjusting_controls"},
    "drinking": {"activity": "drinking", "drink": 1.0},
    "reach_side": {"activity": "reaching"},
    "reach_backseat": {"activity": "reaching"},
    "hair_and_makeup": {"activity": "grooming"},
    "talking_to_passenger": {"activity": "talking"},
    "change_gear": {"activity": "adjusting_controls"},
    "standstill_or_waiting": {"activity": "other"},
}
GAZE_ZONES = {
    "front": "forward",
    "left": "left",
    "left_mirror": "left",
    "right": "right",
    "right_mirror": "right",
    "front_right": "right",
    "center_mirror": "up",
    "infotainment": "down_center",
    "steering_wheel": "down",
}
HANDS = {"both": "both", "both_hands": "both", "only_right": "one", "only_left": "one", "none": "none"}
# label -> the object annotated while the driver handles it (track "object/<type>").
OBJECT_LABELS = {"phone": "cellphone", "drink": "bottle"}
FACE_VIDEO = "_rgb_face.mp4"
EYES_MIN_COVERAGE = 0.8  # share of a window that needs a defined eye state
MICROSLEEP_MIN_CLOSED_SECONDS = 1.0  # longer than a blink
log = logging.getLogger(__name__)


def load_tracks(path: Path) -> dict[str, dict[int, str]]:
    """Return {track: {frame: value}} from one OpenLABEL file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    openlabel = data.get("openlabel", data.get("vcd", {}))
    tracks: dict[str, dict[int, str]] = {}
    for action in openlabel.get("actions", {}).values():
        action_type = str(action.get("type", action.get("name", "")))
        track, _, value = action_type.partition("/")
        if not value:
            continue
        frames = tracks.setdefault(track.strip().lower(), {})
        for interval in action.get("frame_intervals", []):
            for frame in range(int(interval["frame_start"]), int(interval["frame_end"]) + 1):
                frames[frame] = value.strip().lower()
    # One track per object type: a phone and a bottle can share a frame.
    for obj in openlabel.get("objects", {}).values():
        kind = str(obj.get("type", "")).strip().lower()
        if kind not in OBJECT_LABELS.values():
            continue
        frames = tracks.setdefault(f"object/{kind}", {})
        for interval in obj.get("frame_intervals", []):
            for frame in range(int(interval["frame_start"]), int(interval["frame_end"]) + 1):
                frames[frame] = kind
    return tracks


def window_labels(
    tracks: dict[str, dict[int, str]], start: int, end: int, view: str = "face"
) -> dict[str, Any]:
    """Summarise per-frame tracks over [start, end). `view` drops what that camera can't see."""
    labels: dict[str, Any] = {}
    frames = range(start, end)

    def values(track: str) -> list[str]:
        column = tracks.get(track, {})
        return [column[f] for f in frames if f in column]

    def majority(track: str) -> str | None:
        seen = values(track)
        if len(seen) * 2 < len(frames):
            return None
        value, count = Counter(seen).most_common(1)[0]
        return value if count * 2 > len(seen) else None

    if (action := majority("driver_actions")) in DRIVER_ACTIONS:
        labels.update(DRIVER_ACTIONS[action])
    if (zone := majority("gaze_zone")) in GAZE_ZONES:
        labels["attention_direction"] = GAZE_ZONES[zone]
    if majority("gaze_on_road") == "not_looking_road" and "attention_direction" not in labels:
        labels["attention_direction"] = "away"
    # Distraction files call it hands_using_wheel; the hands files call it hands_on_wheel.
    # The face camera does not see the wheel.
    for track in ("hands_using_wheel", "hands_on_wheel") if view != "face" else ():
        if (hands := majority(track)) in HANDS:
            labels["hands_on_controls"] = HANDS[hands]
            break
    if "talking" in tracks and len(values("talking")) * 2 > len(frames):
        labels.setdefault("activity", "talking")
    eyes = [v == "close" for v in values("eyes_state") if v != "undefined"]  # window order
    if eyes and len(eyes) >= EYES_MIN_COVERAGE * len(frames):
        labels["perclos"] = round(sum(eyes) / len(eyes), 4)
        labels["eyes_closed"] = eyes_closed_from_mask(eyes)
    if "blinks" in tracks:
        labels["blink"] = float(bool(values("blinks")))
    if "yawning" in tracks:
        labels["yawn"] = float(bool(values("yawning")))
    return labels


def absent_object_labels(tracks: dict[str, dict[int, str]], start: int, end: int) -> dict[str, float]:
    """phone (with phone_at_ear) and drink 0 over [start, end), where the annotations rule
    them out: every frame has a known driver action, none of them one that sets the label,
    and no frame has the label's object."""
    frames = range(start, end)
    column = tracks.get("driver_actions", {})
    if not frames or not all(column.get(f) in DRIVER_ACTIONS for f in frames):  # a gap or `unclassified`
        return {}
    actions = [column[f] for f in frames]
    labels: dict[str, float] = {}
    for label, kind in OBJECT_LABELS.items():
        held = tracks.get(f"object/{kind}", {})
        if not any(DRIVER_ACTIONS[a].get(label) for a in actions) and not any(f in held for f in frames):
            labels[label] = 0.0
    if "phone" in labels:
        labels["phone_at_ear"] = 0.0
    return labels


def longest_closure(tracks: dict[str, dict[int, str]], start: int, end: int) -> int:
    """Longest run of consecutive frames in [start, end) annotated with the eyes `close`."""
    column = tracks.get("eyes_state", {})
    longest = run = 0
    for frame in range(start, end):
        run = run + 1 if column.get(frame) == "close" else 0
        longest = max(longest, run)
    return longest


class DmdAdapter(SourceAdapter):
    name = "dmd"

    def records(self) -> Iterator[ClipRecord]:
        # The gaze and hands annotations of one s6 recording are separate files on the same
        # video, so merge every file that belongs to a video before cutting clips.
        by_video: dict[Path, list[Path]] = {}
        skipped = 0
        for annotation in sorted(self.root.rglob("*_ann_*.json")):
            video = _find_video(annotation)
            if video is None:
                skipped += 1
            else:
                by_video.setdefault(video, []).append(annotation)
        if skipped:
            log.warning("dmd: %d annotation files have no face video and were skipped", skipped)
        for video, files in sorted(by_video.items()):
            tracks: dict[str, dict[int, str]] = {}
            for annotation in files:
                for track, frames in load_tracks(annotation).items():
                    tracks.setdefault(track, {}).update(frames)
            session = "+".join(a.stem.rsplit("_ann_", 1)[-1] for a in files)
            subject = "_".join(files[0].name.split("_")[:2])  # gX_0
            fps, total = probe_video(video)
            if fps <= 0:
                continue
            base = {"modality": "rgb", "context": "vehicle_cabin", "face_visible": 1.0}
            for record in self.video_windows(
                video, subject, base, end=total, hop_seconds=1.0, view="face", meta={"session": session}
            ):
                end = record.start_frame + record.num_frames * record.frame_stride
                labels = window_labels(tracks, record.start_frame, end, view="face")
                if not labels:
                    continue
                # Only on windows kept for their other labels, so the set of clips stays the same.
                for label, value in absent_object_labels(tracks, record.start_frame, end).items():
                    labels.setdefault(label, value)
                record.labels_gt.update(labels)
                closed = longest_closure(tracks, record.start_frame, end)
                if closed >= MICROSLEEP_MIN_CLOSED_SECONDS * fps:
                    record.meta["derived_labels"] = {"alertness": "microsleep"}
                    record.labels_teacher.update(record.meta["derived_labels"])
                yield record


def _find_video(annotation: Path) -> Path | None:
    candidate = annotation.with_name(annotation.name.split("_ann_")[0].removesuffix("_rgb") + FACE_VIDEO)
    return candidate if candidate.exists() else None
