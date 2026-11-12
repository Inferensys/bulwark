"""UTA Real-Life Drowsiness Dataset.

    Fold1_part1/01/0.mov    0 = alert, 5 = low vigilance, 10 = drowsy (self-reported)
    Fold1_part1/01/5.mp4
    Fold1_part1/01/10.MOV

Each video is ~10 minutes. The state is self-reported for the whole video, so clip
labels are noisy; we sample sparse windows instead of tiling every frame.

The class codes are not KSS values. The project page maps them to KSS ranges: 0 = the
first three levels (alert), 5 = levels 6-7 (low vigilance), 10 = levels 8-9 (drowsy).
`kss` is the midpoint of the range on the taxonomy's (KSS - 1) / 8 scale.
"""

from __future__ import annotations

from collections.abc import Iterator

from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import VIDEO_SUFFIXES, SourceAdapter, iter_files

STATE_LABELS = {
    "0": {"alertness": "alert", "kss": 0.125},  # KSS 1-3 -> 2
    "5": {"alertness": "low_vigilance", "kss": 0.6875},  # KSS 6-7 -> 6.5
    "10": {"alertness": "drowsy", "kss": 0.9375},  # KSS 8-9 -> 8.5
}


class UtaRlddAdapter(SourceAdapter):
    name = "uta_rldd"
    hop_seconds = 10.0

    def records(self) -> Iterator[ClipRecord]:
        for path in iter_files(self.root, VIDEO_SUFFIXES):
            state = path.stem
            if state not in STATE_LABELS:
                continue
            subject = path.parent.name
            fold = path.parent.parent.name
            labels = {**STATE_LABELS[state], "modality": "rgb", "context": "vehicle_cabin"}
            yield from self.video_windows(
                path,
                subject,
                labels,
                hop_seconds=self.hop_seconds,
                meta={"fold": fold, "label_quality": "video_level_self_report"},
            )
