from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import cv2

from bulwark.data.manifest import ClipRecord
from bulwark.data.registry import DatasetSpec, get_spec

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".m4v", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class SourceAdapter:
    """Turns one dataset's native layout into ClipRecords.

    Paths in records are relative to the dataset root so manifests stay portable
    between a local machine, Kaggle and Azure.
    """

    name: ClassVar[str]

    def __init__(self, root: Path, *, clip_frames: int = 16, clip_seconds: float = 2.0) -> None:
        self.root = root
        self.clip_frames = clip_frames
        self.clip_seconds = clip_seconds

    @property
    def spec(self) -> DatasetSpec:
        return get_spec(self.name)

    def records(self) -> Iterator[ClipRecord]:
        raise NotImplementedError

    # ---- helpers ----------------------------------------------------------------------

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def subject(self, raw: str | int) -> str:
        return f"{self.name}:{raw}"

    def image_record(
        self,
        path: Path,
        subject: str | int,
        labels: dict[str, Any],
        *,
        view: str = "",
        crop: tuple[int, int, int, int] | None = None,
        meta: dict[str, Any] | None = None,
        clip_suffix: str = "",
        licence: str | None = None,
        commercial_ok: bool | None = None,
    ) -> ClipRecord:
        """One image sample. `licence` / `commercial_ok` override the dataset's terms for
        sources whose terms differ per item (e.g. per-project licences)."""
        rel = self.rel(path)
        return ClipRecord(
            clip_id=f"{self.name}/{rel}{clip_suffix}",
            dataset=self.name,
            subject_id=self.subject(subject),
            media_path=rel,
            media_type="image",
            licence=self.spec.licence if licence is None else licence,
            commercial_ok=self.spec.commercial_ok if commercial_ok is None else commercial_ok,
            view=view,
            crop=crop,
            labels_gt=labels,
            meta=meta or {},
        )

    def video_windows(
        self,
        path: Path,
        subject: str | int,
        labels: dict[str, Any],
        *,
        start: int = 0,
        end: int | None = None,
        hop_seconds: float | None = None,
        view: str = "",
        meta: dict[str, Any] | None = None,
        clip_seconds: float | None = None,
        licence: str | None = None,
        commercial_ok: bool | None = None,
    ) -> Iterator[ClipRecord]:
        """Cut [start, end) of a video into clip windows of `clip_seconds` (the adapter's
        default unless given). `licence` / `commercial_ok` override the dataset's terms.

        Clip ids are `<dataset>/<path>@<start frame>` and do not record the frame stride
        or count. Changing `clip_stride` or `clip_frames` therefore needs the stride added
        to the id (as Drive&Act ids do) and the shards, teacher labels and feature caches
        rebuilt; otherwise old ones match the new manifest silently."""
        fps, total = probe_video(path)
        if fps <= 0 or total <= 0:
            return
        end = total if end is None else min(end, total)
        clip_seconds = clip_seconds or self.clip_seconds
        span = max(1, round(clip_seconds * fps))
        stride = clip_stride(fps, clip_seconds, self.clip_frames)
        hop = span if hop_seconds is None else max(1, round(hop_seconds * fps))
        rel = self.rel(path)
        frame = start
        while frame + span <= end:
            yield ClipRecord(
                clip_id=f"{self.name}/{rel}@{frame}",
                dataset=self.name,
                subject_id=self.subject(subject),
                media_path=rel,
                media_type="video",
                licence=self.spec.licence if licence is None else licence,
                commercial_ok=self.spec.commercial_ok if commercial_ok is None else commercial_ok,
                start_frame=frame,
                num_frames=self.clip_frames,
                fps=fps,
                frame_stride=stride,
                view=view,
                labels_gt=dict(labels),
                meta=dict(meta or {}),
            )
            frame += hop


def clip_stride(fps: float, clip_seconds: float, clip_frames: int) -> int:
    """Frame step for `clip_frames` frames from a `clip_seconds` window: the sampling rule
    every video adapter shares. It rounds down to whole frames, so the footage a clip covers
    depends on the frame rate: every 3rd frame at 30 fps (16 frames span about 1.5 s of a
    2 s window), every frame at 15 fps (about 1.07 s). Most clip ids do not carry the
    stride, so changing this rule means new ids and rebuilt shards (see `video_windows`)."""
    return max(1, max(1, round(clip_seconds * fps)) // clip_frames)


def probe_video(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0.0, 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return fps, total
    finally:
        capture.release()


def iter_files(root: Path, suffixes: set[str]) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith("."):
            yield path


def copy_fingerprint(root: Path, suffixes: set[str] = IMAGE_SUFFIXES) -> tuple[str, int]:
    """sha256 over the sorted `<file sha256>  <file name>` lines of the matching files under
    `root`, and how many there are. It uses names rather than paths, so a copy that puts
    the same files in other folders (a mirror, or the official archive) fingerprints the
    same, and a re-download can be checked against the copy that was trained on."""
    lines = sorted(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in iter_files(root, suffixes)
    )
    return hashlib.sha256("".join(f"{line}\n" for line in lines).encode()).hexdigest(), len(lines)
