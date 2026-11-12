"""Image datasets with simple folder or YOLO layouts: CEW, Zenodo driver behaviour, SH17."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bulwark.data.clips import read_image
from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import IMAGE_SUFFIXES, SourceAdapter, iter_files


class CewAdapter(SourceAdapter):
    """Closed Eyes in the Wild: dataset_B_FacialImages/{ClosedFace,OpenFace}/*.jpg"""

    name = "cew"

    def records(self) -> Iterator[ClipRecord]:
        for path in iter_files(self.root, IMAGE_SUFFIXES):
            folder = path.parent.name.lower()
            if "face" not in folder:
                continue  # skip the eye-patch folders; faces match our input better
            closed = folder.startswith("closed")
            labels = {
                "eyes_closed": float(closed),
                "eye_openness": 0.0 if closed else 1.0,
                "face_visible": 1.0,
                "modality": "rgb",
            }
            # No identities are published, so every image is its own subject.
            yield self.image_record(path, path.stem, labels)


# Folder names in the Zenodo release, matched case-insensitively by keyword. The record
# describes "turning" as drivers changing direction by moving their head or whole body, in
# either direction, so it carries no attention_direction; turning is part of driving, so
# there is no secondary activity.
ZENODO_CLASSES: dict[str, dict[str, Any]] = {
    "safe": {"activity": "attentive", "phone": 0.0},
    "turning": {"activity": "attentive"},
    "texting": {"activity": "texting", "phone": 1.0, "phone_at_ear": 0.0},
    "talking": {"activity": "phone_call", "phone": 1.0, "phone_at_ear": 1.0},
    "other": {"activity": "other"},
}


class ZenodoDriverAdapter(SourceAdapter):
    """Multi-Class Driver Behavior Image Dataset: <class folder>/<image>.

    Ingested clips are all split "excluded": the maintainers left the set out of v0
    (`SplitPolicy.excluded_datasets`; see the registry's terms for its data use agreement).
    """

    name = "zenodo_driver"

    def records(self) -> Iterator[ClipRecord]:
        for path in iter_files(self.root, IMAGE_SUFFIXES):
            folder = path.parent.name.lower()
            labels = next((v for k, v in ZENODO_CLASSES.items() if k in folder), None)
            if labels is None:
                continue
            # No driver ids are published; group by folder+file prefix so near-duplicate
            # bursts from one session stay in the same split.
            subject = f"{folder}-{path.stem[:6]}"
            yield self.image_record(
                path,
                subject,
                {**labels, "context": "vehicle_cabin", "modality": "rgb"},
                meta={"native_class": path.parent.name},
            )


# Order from sh17.yaml in github.com/ahmadmughees/SH17dataset.
SH17_CLASSES = (
    "person",
    "ear",
    "ear-mufs",
    "face",
    "face-guard",
    "face-mask",
    "foot",
    "tool",
    "glasses",
    "gloves",
    "helmet",
    "hands",
    "head",
    "medical-suit",
    "shoes",
    "safety-suit",
    "safety-vest",
)
SH17_TO_LABEL = {
    "helmet": "helmet",
    "safety-vest": "safety_vest",
    "gloves": "gloves",
    "face-mask": "face_mask",
    "glasses": "eyeglasses",
}


class Sh17Adapter(SourceAdapter):
    """SH17 PPE: images/<name>.jpg + labels/<name>.txt (YOLO: cls cx cy w h, normalised).

    Emits one person-centred crop per annotated person; PPE boxes whose centre falls
    inside the person box become that crop's object labels. Box coordinates are in the
    displayed (EXIF-rotated) frame, the same frame `load_clip` and dedup decode.
    """

    name = "sh17"
    min_person_px = 96

    def records(self) -> Iterator[ClipRecord]:
        for label_file in sorted(self.root.rglob("labels/*.txt")):
            image = _sibling_image(label_file)
            if image is None:
                continue
            height, width = _image_size(image)
            boxes = _read_yolo(label_file, width, height)
            persons = [b for b in boxes if b[0] == "person"]
            for index, (_, px, py, pw, ph) in enumerate(persons):
                if min(pw, ph) < self.min_person_px:
                    continue
                labels: dict[str, Any] = dict.fromkeys(SH17_TO_LABEL.values(), 0.0)
                for cls, bx, by, bw, bh in boxes:
                    cx, cy = bx + bw / 2, by + bh / 2
                    if cls in SH17_TO_LABEL and px <= cx <= px + pw and py <= cy <= py + ph:
                        labels[SH17_TO_LABEL[cls]] = 1.0
                labels.update({"context": "workplace", "modality": "rgb"})
                crop = _expand((px, py, pw, ph), width, height)
                # No worker ids are published and Pexels series repeat the same people, so
                # SH17 is train-only (bulwark.data.splits) and the image stands in as subject.
                yield self.image_record(
                    image, image.stem, labels, crop=crop, clip_suffix=f"#person{index}"
                )


def _sibling_image(label_file: Path) -> Path | None:
    images_dir = label_file.parent.parent / "images"
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{label_file.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _image_size(path: Path) -> tuple[int, int]:
    height, width = read_image(path, gray=True).shape[:2]
    return int(height), int(width)


def _read_yolo(path: Path, width: int, height: int) -> list[tuple[str, int, int, int, int]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = SH17_CLASSES[int(parts[0])]
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append(
            (
                cls,
                int((cx - w / 2) * width),
                int((cy - h / 2) * height),
                int(w * width),
                int(h * height),
            )
        )
    return boxes


def _expand(
    box: tuple[int, int, int, int], width: int, height: int, pad: float = 0.15
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    side = int(max(w, h) * (1 + pad))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    return x0, y0, min(side, width - x0), min(side, height - y0)
