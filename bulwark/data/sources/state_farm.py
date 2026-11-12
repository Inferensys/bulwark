"""State Farm Distracted Driver Detection (Kaggle).

Layout after `kaggle competitions download` + unzip:
    driver_imgs_list.csv          subject,classname,img
    imgs/train/c0..c9/*.jpg
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from typing import Any

from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import SourceAdapter

CLASS_LABELS: dict[str, dict[str, Any]] = {
    "c0": {"activity": "attentive", "phone": 0.0},
    "c1": {"activity": "texting", "phone": 1.0, "phone_at_ear": 0.0},
    "c2": {"activity": "phone_call", "phone": 1.0, "phone_at_ear": 1.0},
    "c3": {"activity": "texting", "phone": 1.0, "phone_at_ear": 0.0},
    "c4": {"activity": "phone_call", "phone": 1.0, "phone_at_ear": 1.0},
    "c5": {"activity": "adjusting_controls", "attention_direction": "down_center"},
    "c6": {"activity": "drinking", "drink": 1.0},
    "c7": {"activity": "reaching"},
    "c8": {"activity": "grooming"},
    "c9": {"activity": "talking"},
}
COMMON = {"context": "vehicle_cabin", "modality": "rgb", "face_visible": 1.0}


class StateFarmAdapter(SourceAdapter):
    name = "state_farm"

    def records(self) -> Iterator[ClipRecord]:
        listing = self.root / "driver_imgs_list.csv"
        with listing.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = self.root / "imgs" / "train" / row["classname"] / row["img"]
                if not path.exists():
                    continue
                labels = {**COMMON, **CLASS_LABELS[row["classname"]]}
                yield self.image_record(path, row["subject"], labels, view="side")
