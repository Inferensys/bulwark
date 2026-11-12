from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from rich.text import Text


def plain(output: str) -> str:
    """CLI output as plain words, to look for phrases in. Under GITHUB_ACTIONS, Typer styles
    --help with ANSI codes that split option names; without a terminal, Rich wraps at 80
    columns, so where a line breaks moves with the length of a temp path printed before it."""
    return " ".join(Text.from_ansi(output).plain.split())


def write_image(path: Path, size: tuple[int, int] = (64, 48), value: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return path


def write_video(path: Path, frames: int = 90, fps: float = 30.0, size: tuple[int, int] = (64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if path.suffix == ".avi" else "mp4v"))
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for index in range(frames):
        writer.write(np.full((size[1], size[0], 3), index % 255, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def make_image():  # type: ignore[no-untyped-def]
    return write_image


@pytest.fixture
def make_video():  # type: ignore[no-untyped-def]
    return write_video
