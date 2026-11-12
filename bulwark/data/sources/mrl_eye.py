"""MRL Eye Dataset: infrared eye crops with labels encoded in the file name.

    mrlEyes_2018_01/sNNNN/sNNNN_NNNNN_0_0_0_0_0_01.png
    subject _ image _ gender _ glasses _ eye_state _ reflections _ lighting _ sensor

gender: 0 man, 1 woman. eye_state: 0 closed, 1 open. glasses: 0 no, 1 yes.
lighting: 0 bad, 1 good. sensor: 01 RealSense, 02 IDS, 03 Aptina. (Codes from the
official page, read through the Internet Archive's 2026-05-16 snapshot.) The page counts
33 men and 4 women; the file names of the ingested copy code 32 subjects as men and 5 as
women (each subject has one code). `meta["gender"]` follows the file names.

Labels come from file names only, so mirrors that regroup the files into other folders
work too. The registry describes the copy ingested for v0 (`source_copy`) and records its
image fingerprint (`source_copy_fingerprint`, see `base.copy_fingerprint`). Records of a
copy with that fingerprint carry that description in `meta["source_copy"]`; records of
any other copy carry the copy's own `meta["copy_fingerprint"]` and
`meta["copy_image_count"]` instead, so a manifest never credits another copy to the v0
source. A copy whose image count differs from the official release gets a warning.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import IMAGE_SUFFIXES, SourceAdapter, copy_fingerprint, iter_files

SENSORS = {"01": "realsense", "02": "ids", "03": "aptina"}
GENDERS = {"0": "male", "1": "female"}
OFFICIAL_IMAGE_COUNT = 84_898


class MrlEyeAdapter(SourceAdapter):
    name = "mrl_eye"

    def records(self) -> Iterator[ClipRecord]:
        parsed = skipped = 0
        meta: dict[str, Any] = {"crop_kind": "eye", **self.copy_meta()}
        for path in iter_files(self.root, IMAGE_SUFFIXES):
            parts = path.stem.split("_")
            if len(parts) != 8 or not parts[0].startswith("s"):
                skipped += 1
                continue
            parsed += 1
            subject, _, gender, glasses, eye_state, reflections, lighting, sensor = parts
            labels = {
                "eyes_closed": 0.0 if eye_state == "1" else 1.0,
                "eye_openness": 1.0 if eye_state == "1" else 0.0,
                "eyeglasses": float(glasses == "1"),
                "modality": "nir",
                "low_light": float(lighting == "0"),
            }
            yield self.image_record(
                path,
                subject,
                labels,
                view=SENSORS.get(sensor, sensor),
                meta={
                    **meta,
                    "reflections": int(reflections),
                    "gender": GENDERS.get(gender, gender),
                },
            )
        if skipped or parsed != OFFICIAL_IMAGE_COUNT:
            warnings.warn(
                f"MRL Eye: parsed {parsed} images and skipped {skipped} unparseable names; the "
                f"official release has {OFFICIAL_IMAGE_COUNT}. Check which copy this is.",
                stacklevel=2,
            )

    def copy_meta(self) -> dict[str, Any]:
        """The registry's `source_copy` if this copy has the fingerprint recorded with it,
        else this copy's own fingerprint and image count."""
        fingerprint, count = copy_fingerprint(self.root, IMAGE_SUFFIXES)
        if self.spec.source_copy and fingerprint == self.spec.source_copy_fingerprint:
            return {"source_copy": self.spec.source_copy}
        return {"copy_fingerprint": fingerprint, "copy_image_count": count}
