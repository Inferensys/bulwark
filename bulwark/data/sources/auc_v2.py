"""AUC Distracted Driver Dataset v2 (Machine Intelligence group, American University in Cairo).

The root is the v2 folder of the authors' archive (its name has a space after the last
underscore):

    v2_cam1_cam2_ split_by_driver/
        Camera 1/{train,test}/c0..c9/*.jpg     10,555 train / 1,123 test, 1920x1080
        Camera 2/{train,test}/c0..c9/*.jpg      2,000 train /   800 test, 640x480
        skin_nonskin_pixels/                    skin-colour pixels; not used

14,478 images in all, the total the dataset page gives (counted 2026-10-29 in the unpacked
archive and in its zip listing). Each image is one record (a one-frame clip), view "cam1"
or "cam2".

The class folders are State Farm's c0..c9 in the same order (`state_farm.CLASS_LABELS`):
checked 2026-10-29 against the dataset page's per-class counts (camera 1 and 2, train and
test: c1 1,718 is "Text Right", c7 1,034 "Reaching Behind", c8 1,044 "Hair or Makeup") and
against the v1 class folders. The authors' v1 split CSVs number the classes in another
order (Text Left 1, Hair & Makeup 7, Reach Behind 8); they are not used.

v1 is ignored: the archive's `v1_cam1_no_split/` folder (beside this root) and the v1
split CSVs. Every one of the 11,678 v2 camera-1 images is a byte-identical copy of a v1
image (checked 2026-10-29), and v1 was split at random, not by driver, so using both
would put copies of one image on both sides of a split.

Subjects and splits. The authors split train and test by driver (44 drivers), but publish
no per-image driver id. Each side is therefore one subject group, `auc_v2:train_drivers`
and `auc_v2:test_drivers`, and `meta["official_split"]["v2"]` records the side, which
`bulwark.data.splits` follows (`SplitPolicy.official_splits` and `subject_group_datasets`).
One driver is on both sides of the authors' split: camera 1's test/c0 folder ends with 83
images of her (322.jpg to 415.jpg), and her other classes open the train folders (c1/200.jpg,
c2/142.jpg, ..., c9/188.jpg; 260 images). Within each class folder the file numbers run in
recording order, driver after driver, and the authors' cut in c0 falls one driver later
than in c1-c9. Checked 2026-10-29 by eye and by thumbnail distance: each of the 83 is
closer to her train images than to any other train image, and no other test image is.
Those 83 images go to train (`MOVED_TO_TRAIN`; `meta["authors_split"]` keeps "test"):
camera 1 has 10,638 train and 1,040 test images here. No other driver was found on both
sides: the other camera-1 test drivers (three) and the camera-2 test drivers (four) had
no close match in train. That check compares thumbnails within one camera, so it does not
rule out a driver recorded again in other clothes, at another time of day or in another
car (the dataset page says some drivers took part in more than one recording session),
nor a camera-2 test driver who is among camera 1's train drivers, since nothing links the
two cameras' drivers. The train drivers cannot be divided without ids, so AUC gives no
val clips, and k-fold model selection leaves its train group in training. Its test
results come from one driver group: one "subject", so there is no subject-level bootstrap
interval and every AUC metric is marked indicative (`bulwark.eval.metrics`).
The two cameras are scored as separate clips; nothing links an image of one camera to an
image of the other.

Removal requests. Without driver ids, one AUC driver cannot be removed on their own:
`removed_subjects.yaml` can list only a whole group (`auc_v2:train_drivers` or
`auc_v2:test_drivers`, every image on that side), and `bulwark purge --clips` deletes
clips without keeping them out of the next `bulwark ingest`. The data card says so.

Labels. activity follows the class (State Farm's map). The class also gives the labels it
proves: phone 1 for texting and phone calls (c1-c4), with phone_at_ear 1 for the calls
(c2, c4) and 0 for texting (c1, c3), whose phone is in the hand; and drink 1 for drinking
(c6). It gives no 0 for phone, phone_at_ear or drink: a class names what the driver does,
not what they hold, and drivers hold a phone in other classes. Checked 2026-10-30 by eye
in camera 1's train folders: a driver holds a phone in every sampled frame of
c7/619.jpg-663.jpg (reaching behind), another in c9/188.jpg, 193 and 198 (talking to the
passenger) and another in c0/2506.jpg-2520.jpg (safe driving). The taxonomy defines these
labels as holding or using the object, so a 0 needs an object annotation, which AUC does
not have (DMD and Drive&Act give their 0s from theirs); the objects teacher can add 0s.
Until something does, phone and drink have only positives among AUC's images, and
training masks a label that its image targets give one class of (`bulwark.train.labels`).
food is never set: AUC has no eating class, and food positives come only from Drive&Act's
cabin cameras, so a food 0 on every AUC image would rest on the recording protocol alone
and teach that these side views never show food (the reason DMD sets none).
face_visible and attention_direction are not set either. The side cameras show the face
in profile in most classes but not in every image (reaching behind turns it away, and a
hand can cover it in hair and makeup), and no class fixes where the driver looks
(adjusting the radio does not require looking at it); both labels are masked in the v0
training configs anyway.

meta records the split used, the camera folder and the native class.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from bulwark.data.manifest import ClipRecord
from bulwark.data.sources.base import IMAGE_SUFFIXES, SourceAdapter, iter_files
from bulwark.data.sources.state_farm import CLASS_LABELS as STATE_FARM_CLASSES

# Camera folder -> view.
CAMERAS = {"Camera 1": "cam1", "Camera 2": "cam2"}
SPLITS = ("train", "test")
# Protocol name in meta["official_split"] (bulwark.data.splits): the authors' split, but
# for the images in MOVED_TO_TRAIN.
OFFICIAL_PROTOCOL = "v2"
# The phone classes -> phone_at_ear: the calls hold the phone at the ear, texting in the hand.
PHONE_CLASSES = {"c1": 0.0, "c2": 1.0, "c3": 0.0, "c4": 1.0}
DRINK_CLASS = "c6"


def _class_labels(cls: str) -> dict[str, Any]:
    """What the class proves: the activity and, for the phone and drinking classes, the
    object in use. No 0s: drivers in other classes may hold a phone or a drink."""
    labels: dict[str, Any] = {"activity": STATE_FARM_CLASSES[cls]["activity"]}
    if cls in PHONE_CLASSES:
        labels.update(phone=1.0, phone_at_ear=PHONE_CLASSES[cls])
    if cls == DRINK_CLASS:
        labels["drink"] = 1.0
    return labels


CLASS_LABELS: dict[str, dict[str, Any]] = {cls: _class_labels(cls) for cls in STATE_FARM_CLASSES}
COMMON = {"context": "vehicle_cabin", "modality": "rgb"}
# (camera, split, class) -> first and last file number of authors' test images moved to
# train: a train driver's c0 images (see the module docstring).
MOVED_TO_TRAIN = {("Camera 1", "test", "c0"): (322, 415)}


def _moved(camera: str, split: str, cls: str, stem: str) -> bool:
    span = MOVED_TO_TRAIN.get((camera, split, cls))
    return span is not None and stem.isdigit() and span[0] <= int(stem) <= span[1]


class AucV2Adapter(SourceAdapter):
    name = "auc_v2"

    def records(self) -> Iterator[ClipRecord]:
        if not any((self.root / camera).is_dir() for camera in CAMERAS):
            raise FileNotFoundError(
                f"{self.root} has no 'Camera 1' or 'Camera 2' folder; the root is the archive's "
                "'v2_cam1_cam2_ split_by_driver' folder (configs/data.yaml)"
            )
        for camera, view in CAMERAS.items():
            for split in SPLITS:
                for cls, labels in CLASS_LABELS.items():
                    folder = self.root / camera / split / cls
                    if not folder.is_dir():
                        continue
                    for path in iter_files(folder, IMAGE_SUFFIXES):
                        side = "train" if _moved(camera, split, cls, path.stem) else split
                        meta: dict[str, Any] = {
                            "official_split": {OFFICIAL_PROTOCOL: side},
                            "camera": camera,
                            "native_class": cls,
                        }
                        if side != split:
                            meta["authors_split"] = split
                        yield self.image_record(
                            path, f"{side}_drivers", {**COMMON, **labels}, view=view, meta=meta
                        )
