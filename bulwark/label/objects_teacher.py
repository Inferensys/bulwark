"""RF-DETR (Apache-2.0, COCO classes) as a teacher for phone / drink / food negatives.

Ultralytics YOLO is deliberately not used: its weights are AGPL-3.0.

A COCO detection says an object is in view, while the taxonomy's object labels are about
the person holding or using it: a cup in a holder, a bottle in a door pocket or a phone
on a mount is not `drink` or `phone`. So a detection never makes a positive. The teacher
only writes a label as 0 when nothing of that kind scores even `negative_threshold` in
any sampled frame. Ground truth wins where a dataset gives the label: Drive&Act and DMD
give their own 0s from their object annotations (`bulwark.data.sources`). Positives come
from ground truth and the VLM teacher. On clips whose ground truth is positive,
`bulwark.label.agreement` counts how often this teacher wrongly says 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

COCO_TO_LABEL = {
    "cell phone": "phone",
    "cup": "drink",
    "bottle": "drink",
    "wine glass": "drink",
    "sandwich": "food",
    "banana": "food",
    "apple": "food",
    "donut": "food",
    "pizza": "food",
    "hot dog": "food",
    "cake": "food",
}
OBJECT_LABELS = tuple(sorted(set(COCO_TO_LABEL.values())))
MODEL_SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}


def summarise_detections(
    frame_scores: Sequence[dict[str, float]], negative_threshold: float
) -> dict[str, float]:
    """Clip negatives from per-frame best scores ({label: max confidence}, absent = none):
    0.0 for each object label that no sampled frame detects at `negative_threshold`."""
    if not frame_scores:
        return {}
    return {
        label: 0.0
        for label in OBJECT_LABELS
        if max(frame.get(label, 0.0) for frame in frame_scores) < negative_threshold
    }


class ObjectsTeacher:
    def __init__(
        self,
        size: str = "medium",
        negative_threshold: float = 0.2,
        frames_per_clip: int = 4,
    ) -> None:
        if size not in MODEL_SIZES:
            raise ValueError(f"RF-DETR size must be one of {', '.join(MODEL_SIZES)}, got {size!r}")
        import rfdetr
        from rfdetr.util.coco_classes import COCO_CLASSES

        self.model = getattr(rfdetr, MODEL_SIZES[size])()
        self.classes = COCO_CLASSES
        self.negative_threshold = negative_threshold
        self.frames_per_clip = frames_per_clip

    def clip(self, frames: np.ndarray) -> dict[str, Any]:
        picks = np.linspace(0, len(frames) - 1, min(self.frames_per_clip, len(frames))).astype(int)
        frame_scores = []
        for index in picks:
            # Detect down to the negative threshold, so weak detections block a negative.
            detections = self.model.predict(frames[index], threshold=self.negative_threshold)
            confidence = getattr(detections, "confidence", None)
            if confidence is None:
                confidence = np.ones(len(detections.class_id))
            best: dict[str, float] = {}
            for class_id, score in zip(detections.class_id, confidence, strict=True):
                label = COCO_TO_LABEL.get(self.classes[int(class_id)])
                if label is not None:
                    best[label] = max(best.get(label, 0.0), float(score))
            frame_scores.append(best)
        return summarise_detections(frame_scores, self.negative_threshold)
