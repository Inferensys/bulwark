"""NTHU Driver Drowsiness Detection dataset.

    Training Dataset/001/glasses/nonsleepyCombination.avi
    Training Dataset/001/glasses/001_glasses_nonsleepyCombination_drowsiness.txt
    ... _eye.txt  _head.txt  _mouth.txt

Each annotation file is one string with a digit per frame:
    drowsiness: 0 alert, 1 drowsy
    eye:        0 still, 1 sleepy eyes
    head:       0 still, 1 nodding, 2 looking aside
    mouth:      0 still, 1 yawning, 2 talking/laughing
Scenarios: glasses, noglasses, sunglasses, night_glasses, night_noglasses (night = IR).

The eye track marks "sleepy eyes" (heavy lids, slow blinks), not frame-level closure, so
it does not become `perclos` or `eyes_closed`; its window fraction is kept in
`meta["sleepy_eye_fraction"]`. The face teacher does not fill the gap: `bulwark label
merge` keeps only its open-eye values (`FACE_KEEP` in bulwark/label/run.py: eyes_closed
0, and its eye_openness and perclos only where it says the eyes are open), so NTHU-DDD
clips get no closed-eye target from any source.

The layout is checked against the evaluation release when it arrives; the parser is
tolerant of missing annotation files (that label is just left out).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import VIDEO_SUFFIXES, SourceAdapter, iter_files

KINDS = ("drowsiness", "eye", "head", "mouth")


def read_frame_string(path: Path) -> str:
    return "".join(ch for ch in path.read_text(encoding="utf-8", errors="ignore") if ch.isdigit())


def find_annotation(video: Path, kind: str) -> Path | None:
    """`<subject>_<scenario>_<video stem>_<kind>.txt` (or `<video stem>_<kind>.txt`) next
    to the video.

    The underscore before the stem matters: `*sleepyCombination*` would also match the
    `nonsleepyCombination` file in the same folder.
    """
    exact = video.parent / f"{video.stem}_{kind}.txt"
    matches = sorted(
        {*video.parent.glob(f"*_{video.stem}_{kind}.txt"), *([exact] if exact.is_file() else [])}
    )
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {kind} annotation for {video}: {[m.name for m in matches]}")
    return matches[0] if matches else None


def window_labels(annotations: dict[str, str], start: int, end: int) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    window = {kind: values[start:end] for kind, values in annotations.items()}
    if (drowsy := window.get("drowsiness")) and len(drowsy) == end - start:
        labels["alertness"] = "drowsy" if drowsy.count("1") * 2 > len(drowsy) else "alert"
    if (head := window.get("head")) and len(head) == end - start:
        labels["head_nod"] = float("1" in head)
        if head.count("2") * 2 > len(head):
            labels["attention_direction"] = "away"
    if (mouth := window.get("mouth")) and len(mouth) == end - start:
        labels["yawn"] = float("1" in mouth)
        if mouth.count("2") * 2 > len(mouth):
            labels["activity"] = "talking"
    return labels


def sleepy_eye_fraction(annotations: dict[str, str], start: int, end: int) -> float | None:
    eye = annotations.get("eye", "")[start:end]
    return round(eye.count("1") / len(eye), 4) if eye and len(eye) == end - start else None


class NthuDddAdapter(SourceAdapter):
    name = "nthu_ddd"

    def records(self) -> Iterator[ClipRecord]:
        for path in iter_files(self.root, VIDEO_SUFFIXES):
            scenario = path.parent.name.lower()
            subject = path.parent.parent.name
            annotations = {
                kind: read_frame_string(found)
                for kind in KINDS
                if (found := find_annotation(path, kind)) is not None
            }
            base: dict[str, Any] = {
                "modality": "nir" if "night" in scenario else "rgb",
                "low_light": float("night" in scenario),
                "eyeglasses": float(
                    scenario.endswith("glasses") and "no" not in scenario and "sun" not in scenario
                ),
                "sunglasses": float("sunglasses" in scenario),
                "context": "vehicle_cabin",
            }
            for record in self.video_windows(
                path, subject, base, hop_seconds=1.0, view=scenario, meta={"behaviour": path.stem}
            ):
                end = record.start_frame + record.num_frames * record.frame_stride
                record.labels_gt.update(window_labels(annotations, record.start_frame, end))
                if (
                    fraction := sleepy_eye_fraction(annotations, record.start_frame, end)
                ) is not None:
                    record.meta["sleepy_eye_fraction"] = fraction
                yield record
