"""MediaPipe Face Landmarker (Apache-2.0) as a teacher for eye, mouth and head signals.

Uses the model's blendshapes (eyeBlinkLeft/Right, jawOpen) rather than raw landmark
ratios: they are already normalised per face and hold up better across head pose.

Head pose and `attention_direction` are in the camera frame; label merging keeps the
direction only for cameras that roughly face the person (bulwark/label/run.py). Events
(blink, head nod, yawn) need several frames and are not emitted for shorter inputs, such
as still images.

Pitch is positive when the head tips down. MediaPipe's metric space has y up and z
towards the camera, so a face turned down has its forward axis rotated towards -y, which
`euler_from_matrix` reads as positive pitch; on a demo clip of a driver's head dropping
it rose to about +38 degrees (37.1 on the clip's native 1280x720 frames, 38.0 on the
256-pixel letterboxed frames the label pipeline uses; checked 2026-11-10). Until 0.1.0
this module described the sign the other way round, and its head-nod and direction
rules followed that description, so the v0 models trained on head_nod labels from a
rule that fired when the head rose and came back, not on a nod (MODEL_CARD.md,
"Limitations"). v0 masks attention_direction, so the direction rule did not reach them.

Model: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bulwark.data.label_rules import eyes_closed_from_mask

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MIN_EVENT_FRAMES = 4  # blink, nod and yawn are events in time; a still image cannot show one


@dataclass(frozen=True, slots=True)
class FaceThresholds:
    closed: float = 0.5  # blink blendshape above this = eye closed
    yawn_jaw: float = 0.45  # jawOpen above this = mouth wide open
    yawn_fraction: float = 0.4  # share of the clip with the mouth wide open
    forward_yaw_deg: float = 20.0
    forward_pitch_deg: float = 15.0
    nod_pitch_deg: float = 15.0
    min_face_fraction: float = 0.5


@dataclass(slots=True)
class FrameFace:
    closure: float  # 0 open .. 1 closed
    jaw: float
    yaw: float  # degrees, positive = subject turns to their left
    pitch: float  # degrees, positive = head tipped down
    roll: float


def euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch, roll in degrees from MediaPipe's facial transformation matrix."""
    r = matrix[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def geometric_yaw_sign(
    nose_x: float, right_eye_outer_x: float, left_eye_outer_x: float, fallback: float
) -> float:
    """+1 when the subject turns to their own left, read from the image itself.

    Mesh point 33 is the subject's right eye corner (image left for a camera facing
    them), 263 their left. Turning to their left moves the nose tip toward image right.
    This avoids depending on the transformation matrix's axis convention. `yaw_sign=-1`
    on the teacher handles mirrored (selfie-view) footage.
    """
    span = left_eye_outer_x - right_eye_outer_x
    if abs(span) < 1e-6:
        return 1.0 if fallback >= 0 else -1.0
    offset = (nose_x - (right_eye_outer_x + left_eye_outer_x) / 2) / abs(span)
    if abs(offset) < 0.02:  # near frontal: the sign hardly matters, keep the matrix's
        return 1.0 if fallback >= 0 else -1.0
    return 1.0 if offset > 0 else -1.0


def summarise(
    faces: list[FrameFace | None], thresholds: FaceThresholds | None = None
) -> dict[str, Any]:
    """Turn per-frame teacher readings into clip-level labels (taxonomy names)."""
    thresholds = thresholds or FaceThresholds()
    total = len(faces)
    seen = [f for f in faces if f is not None]
    labels: dict[str, Any] = {
        "face_visible": float(len(seen) / max(1, total) >= thresholds.min_face_fraction)
    }
    if not seen or len(seen) / max(1, total) < thresholds.min_face_fraction:
        return labels

    closure = np.array([f.closure for f in seen])
    closed = closure > thresholds.closed
    jaw = np.array([f.jaw for f in seen])
    yaw = np.array([f.yaw for f in seen])
    pitch = np.array([f.pitch for f in seen])
    roll = np.array([f.roll for f in seen])

    labels.update(
        eye_openness=round(float(1.0 - closure.mean()), 4),
        perclos=round(float(closed.mean()), 4),
        eyes_closed=eyes_closed_from_mask(closed.tolist()),
        mouth_open=round(float(np.clip(jaw.mean(), 0, 1)), 4),
        head_yaw=round(float(np.clip(np.median(yaw) / 90.0, -1, 1)), 4),
        head_pitch=round(float(np.clip(np.median(pitch) / 90.0, -1, 1)), 4),
        head_roll=round(float(np.clip(np.median(roll) / 90.0, -1, 1)), 4),
    )
    if total >= MIN_EVENT_FRAMES:
        labels.update(
            blink=float(_has_blink(closed)),
            yawn=float((jaw > thresholds.yawn_jaw).mean() >= thresholds.yawn_fraction),
            head_nod=float(_has_nod(pitch, thresholds.nod_pitch_deg)),
        )
    if not labels["eyes_closed"]:
        labels["attention_direction"] = _direction(
            float(np.median(yaw)), float(np.median(pitch)), thresholds
        )
    return labels


def _has_blink(closed: np.ndarray) -> bool:
    """open -> closed (short) -> open inside the clip."""
    open_frames = np.flatnonzero(~closed)
    if len(open_frames) == 0:
        return False
    after = closed[open_frames[0] :]
    closed_after = np.flatnonzero(after)
    return bool(len(closed_after)) and bool((~after[closed_after[0] :]).any())


def _has_nod(pitch: np.ndarray, threshold: float) -> bool:
    """The head drops (pitch rises by more than `threshold` degrees) and then comes back
    up by more than half of `threshold` degrees, however far it dropped, inside the clip.
    It cannot tell nodding off from looking down and back up."""
    if len(pitch) < 4:
        return False
    low = int(pitch.argmax())  # the lowest head position: pitch is positive downwards
    return (
        0 < low < len(pitch) - 1
        and pitch[low] - pitch[:low].min() > threshold
        and pitch[low] - pitch[low:].min() > threshold * 0.5
    )


def _direction(yaw: float, pitch: float, t: FaceThresholds) -> str:
    if abs(yaw) <= t.forward_yaw_deg and abs(pitch) <= t.forward_pitch_deg:
        return "forward"
    if abs(yaw) >= abs(pitch) * 1.3:
        return "left" if yaw > 0 else "right"
    return "down" if pitch > 0 else "up"


class FaceTeacher:
    """Runs MediaPipe on RGB frames. Construct once per process (it is not picklable)."""

    def __init__(
        self, model_path: Path, thresholds: FaceThresholds | None = None, yaw_sign: float = 1.0
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing; download {FACE_MODEL_URL}")
        self._mp = mp
        self.thresholds = thresholds or FaceThresholds()
        self.yaw_sign = yaw_sign
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def frame(self, rgb: np.ndarray) -> FrameFace | None:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._detector.detect(image)
        if not result.face_landmarks or not result.face_blendshapes:
            return None
        shapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        yaw, pitch, roll = euler_from_matrix(np.asarray(result.facial_transformation_matrixes[0]))
        points = result.face_landmarks[0]
        yaw = abs(yaw) * geometric_yaw_sign(points[1].x, points[33].x, points[263].x, yaw)
        return FrameFace(
            closure=(shapes.get("eyeBlinkLeft", 0.0) + shapes.get("eyeBlinkRight", 0.0)) / 2,
            jaw=shapes.get("jawOpen", 0.0),
            yaw=yaw * self.yaw_sign,
            pitch=pitch,
            roll=roll,
        )

    def clip(self, frames: np.ndarray) -> dict[str, Any]:
        return summarise([self.frame(f) for f in frames], self.thresholds)
