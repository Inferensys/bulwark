"""Decode a ClipRecord into a fixed-size uint8 frame stack."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bulwark.data.manifest import ClipRecord


def letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping aspect ratio, pad to a square with black borders."""
    height, width = frame.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    out = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    out[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return out


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _apply_crop(frame: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y : y + h, x : x + w]


def read_image(path: Path, *, gray: bool = False) -> np.ndarray:
    """Decode an image as displayed: 8-bit BGR (or grayscale) with EXIF orientation applied.

    Every image read (frames, crop boxes, dedup hashes) goes through here, so crops and
    sizes are always in the same frame. `IMREAD_UNCHANGED` would skip the EXIF rotation.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return image


def load_clip(record: ClipRecord, root: Path, size: int = 256) -> np.ndarray:
    """Return frames as (T, size, size, 3) uint8 RGB. Images give T=1."""
    path = root / record.media_path
    if record.media_type == "image":
        image = read_image(path)
        return letterbox(_to_rgb(_apply_crop(image, record.crop)), size)[None]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, record.start_frame)
        frames: list[np.ndarray] = []
        index = 0
        last: np.ndarray | None = None
        while len(frames) < record.num_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % record.frame_stride == 0:
                last = letterbox(_to_rgb(_apply_crop(frame, record.crop)), size)
                frames.append(last)
            index += 1
    finally:
        capture.release()
    if last is None:
        raise ValueError(f"No frames decoded for {record.clip_id}")
    while len(frames) < record.num_frames:  # short tail: repeat the last frame
        frames.append(last)
    return np.stack(frames)
