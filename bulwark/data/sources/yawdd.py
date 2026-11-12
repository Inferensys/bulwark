"""YawDD: yawning detection dataset.

    YawDD dataset/Mirror/Female_mirror/1-FemaleNoGlasses-Normal.avi
    YawDD dataset/Mirror/Male_mirror Avi Videos/12-MaleGlasses-Yawning.avi
    YawDD dataset/Dash/Female/1-FemaleNoGlasses.avi

Mirror videos carry one behaviour per file (Normal, Talking, Yawning). A Yawning video
still contains non-yawning frames, so only Normal/Talking get clip-level labels from the
file name; Yawning windows are marked `yawn_video` in meta and get no clip-level `yawn`
(`bulwark label merge` drops the face teacher's yawn, FACE_KEEP in
bulwark/label/run.py). The file name is real ground truth at video level: a video-level
score (any clip of a Yawning video) can use it. Dash videos have no per-file label and
are teacher-labelled only.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import VIDEO_SUFFIXES, SourceAdapter, iter_files

NAME = re.compile(
    r"^(?P<subject>\d+)-(?P<sex>Female|Male)(?P<glasses>NoGlasses|Glasses|SunGlasses)"
    r"(?:-(?P<behaviour>Normal|Talking|Yawning))?",
    re.IGNORECASE,
)


class YawddAdapter(SourceAdapter):
    name = "yawdd"

    def records(self) -> Iterator[ClipRecord]:
        for path in iter_files(self.root, VIDEO_SUFFIXES):
            match = NAME.match(path.stem)
            if match is None:
                continue
            view = "dash" if "dash" in path.as_posix().lower() else "mirror"
            sex = match["sex"].lower()
            subject = f"{sex}{int(match['subject']):02d}"
            glasses = match["glasses"].lower()
            labels: dict[str, Any] = {
                "eyeglasses": float(glasses == "glasses"),
                "sunglasses": float(glasses == "sunglasses"),
                "modality": "rgb",
                "context": "vehicle_cabin",
            }
            meta: dict[str, Any] = {}
            behaviour = (match["behaviour"] or "").lower()
            if behaviour == "normal":
                labels.update({"activity": "attentive", "yawn": 0.0})
            elif behaviour == "talking":
                labels.update({"activity": "talking", "yawn": 0.0})
            elif behaviour == "yawning":
                meta["yawn_video"] = True
            yield from self.video_windows(
                path, subject, labels, hop_seconds=1.0, view=view, meta=meta
            )
