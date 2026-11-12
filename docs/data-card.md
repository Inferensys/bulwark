# Data card: Bulwark v0 training data

What the v0 models are trained and evaluated on, how the clips and labels are made, and
how personal data is handled. Its counts are those of the v0 release manifest (183,355
clips, sha256 `1d6175123358...`), the one B1 v0 was trained on. Licences,
access routes, the terms relied on and citations for every dataset are in
[datasets.md](datasets.md), which is generated from the registry
(`bulwark/data/registry.py`).

No dataset files, frames, manifests or labels are in this repository or in any weights
release. Most source datasets forbid redistribution, and their videos show real people.

## Sources

v0 uses only datasets licensed for research or non-commercial use: Drive&Act, the DMD
face camera, MRL Eye and AUC Distracted Driver v2. No commercially licensed data is
ingested yet.

| Dataset | Terms (summary) | What it contributes | People | State |
|---|---|---|---|---|
| Drive&Act | research use only | video of drivers in a car from several synchronised cameras (near-infrared views and Kinect colour); 34 mid-level activities mapped onto Bulwark's activity, attention and object labels | 15 drivers | ingested |
| DMD | CC BY-NC-ND 4.0, academic use | face-camera RGB video: eye state, blinks, yawns, gaze region, phone use, drinking; a derived microsleep label (below) | 14 subjects (17,658 clips) | ingested; body and hands cameras not ingested |
| MRL Eye | no licence stated; treated as research only | 84,898 near-infrared eye crops: eye state, glasses, lighting | 37 people | ingested from a Kaggle mirror whose images match the official archive ([datasets.md](datasets.md) records the copy and its fingerprint); the original authors' terms govern |
| UTA-RLDD | no licence stated; treated as research only | self-reported alertness and KSS over long videos | 60 participants | not ingested; reserved as the cross-dataset holdout |
| Multi-Class Driver Behavior (Zenodo 14908802) | labelled CC BY 4.0, but ships a data use agreement (below); appears to contain re-encoded State Farm images | 7,276 images in 5 classes (safe driving, turning, texting, talking on the phone, other): activity and phone labels | no person ids | excluded from v0 |
| AUC Distracted Driver v2 | signed licence agreement: non-commercial; use limited to developing distracted-driver detection without the MI-AUC group's further consent; no transfer to third parties without the group's prior authorization | 14,478 still RGB images from two cabin cameras (camera 1: 10,638 train / 1,040 test; camera 2: 2,000 train / 800 test) in State Farm's 10 classes: activity; phone, phone at the ear and drinking where the class shows them in use | 44 drivers; no per-image driver ids | ingested; train and test only (no val) |

**The Zenodo set is excluded from v0** (a maintainer decision, recorded in
`SplitPolicy.excluded_datasets` and in the registry). The record ships a `DATA USE
AGREEMENT.docx` that forbids sharing the dataset with third parties without written
permission and asks for a copy of any proposed publication 30 days before submission,
and no signed copy is held. If it is ingested, `bulwark split` puts its clips in split
`excluded`, so they are neither trained on nor scored, and its registry entry has
`weights_release_ok`, `kaggle_storage_ok` and `azure_storage_ok` set to `False`.
Training and evaluation refuse a manifest with its clips in any other split (the split
check, `check_disjoint`).

The registry lists further datasets with adapters (NTHU-DDD, YawDD, State Farm, CEW, SH17
and generated synthetic clips) that v0 does not use yet.

## Clips

- **Video.** Every dataset video adapter cuts clips the same way: 16 frames at a stride
  of `max(1, round(2 s x fps) // 16)` source frames (`bulwark.data.sources.base.clip_stride`).
  At 30 fps that is every 3rd frame, about 1.5 s of video; Drive&Act's Kinect colour view
  runs at 15 fps, so its clips take every frame and cover about 1.07 s, and play about
  1.5 times slower than those of its 30 fps NIR views. Drive&Act activities shorter than
  one clip are dropped.
- **Images.** Image datasets (MRL Eye and AUC Distracted Driver v2 in v0) give
  one-frame clips.
- **Storage.** Frames are letterboxed to 256 x 256 (aspect ratio kept, black padding) and
  stored as JPEG at quality 90 in tar shards.
- **Duplicates.** `bulwark split` drops images that appear in more than one dataset,
  keeping the copy in a dataset that can be evaluated (one with person ids before AUC
  Distracted Driver v2's driver groups), so a duplicate cannot sit in one dataset's
  training data and another's test set. An image is compared only with other datasets'
  images: consecutive frames of one dataset are near-identical, and chaining them would
  let one cross-dataset match drop thousands of images.

## Splits

Splits are subject-disjoint (details in [evaluation.md](evaluation.md#splits)). In the
v0 release manifest, the people per split are as below (the model card gives the clips
per split):

| Dataset | Train | Val | Test | Holdout |
|---|---:|---:|---:|---|
| Drive&Act (official `split_0`) | 10 | 2 | 3 | inner-mirror view of the 3 test drivers; the rest of that view is excluded |
| DMD (subject hash) | 9 | 2 | 3 | none |
| MRL Eye (subject hash) | 31 | 3 | 3 | none |
| AUC Distracted Driver v2 (authors' split by driver, corrected) | 1 driver group | 0 | 1 driver group | none |

**AUC Distracted Driver v2** keeps its authors' train/test split, which is by driver,
with one correction: a camera-1 driver is on both sides of it (83 of her safe-driving
images, test `c0/322.jpg` to `c0/415.jpg`, are in the authors' test folder and her other
classes in train), so those 83 images are moved to train (10,638 train / 1,040 test for
camera 1, against the authors' 10,555 / 1,123; checked 2026-10-29, see the adapter). No
other driver was found on both sides, but that check compared images within each camera.
It would not find a driver recorded again in other clothes, at another time of day or in
another car (the dataset page says some drivers took part in more than one recording
session), nor a camera-2 test driver who is also among camera 1's train drivers, since
nothing links the two cameras' drivers. The dataset publishes no per-image driver id.
Each side is therefore one subject group, `auc_v2:train_drivers` and
`auc_v2:test_drivers` (`SplitPolicy.official_splits` and `subject_group_datasets`). The
train drivers cannot be divided without ids, so AUC gives no val clips, and k-fold
selection keeps its train group in training. Its test results come from one driver
group: one subject in the report, so there is no subject-level bootstrap interval, and
every AUC metric is marked indicative (fewer than 5 subjects). The archive also holds a
version 1 (camera 1 only, split at random); it is not used, because every v2 camera-1
image is a byte-identical copy of a v1 image (checked 2026-10-29), so using both would
put copies of one image in train and test.

The Zenodo set, if it is ingested, is entirely in split `excluded`. Model selection in
the fine-tune and distillation stages (`best.pt`) and calibration therefore rest on few
people; the training-support figures in each checkpoint and report say how many.
Thresholds and temperatures are fitted per media type, on the val clips of one media type
at a time, so a fit for one media type rests on fewer people still: each entry of
`calibration.json` counts its val clips, positives and subjects, and the report's
thresholds table shows them. Below 10 val clips of each class nothing is fitted (the
floor counts clips, not people). The
heads stage picks its `best.pt` by 5-fold cross-validation over the train and val
subjects instead (`select_folds: 5` in the shipped heads configs; see
[training.md](training.md#2-heads-on-frozen-features)).

## Labels

Each clip carries only the labels its dataset annotates (`labels_gt`), mapped onto the
taxonomy in `bulwark/data/taxonomy.yaml`, plus teacher labels (`labels_teacher`) on labels
the dataset does not annotate. Ground truth always wins, teacher labels count half in the
loss, and only ground truth is scored. The [model card](../MODEL_CARD.md#labels-and-supervision)
lists, per label, the datasets and teachers that supervise it.

**Definitions worth knowing.**

- `eyes_closed`: the eyes are closed in more than half of the frames in the last half of
  the clip.
- `perclos`: the fraction of the clip's frames with the eyes closed. This is a clip-level
  closure fraction over 1 to 2 seconds. PERCLOS as used in the drowsiness literature
  averages eye closure over a rolling window of a minute or more; compute it downstream
  from consecutive clips.
- Object labels are about the person: `phone`, `food` and `drink` mean the person holds
  or uses the object during the clip (a cup in a holder is not `drink`); `eyeglasses`,
  `sunglasses`, `face_mask`, `seatbelt`, `helmet`, `safety_vest` and `gloves` mean the
  person wears it.

**Phone, drink and food from the datasets.** Positives come from the activities that
involve the object: Drive&Act's mid-level activities, and DMD's driver actions (the
action on most of the window's frames). Negatives come from the datasets' object
annotations, which mark an object while the driver reaches for it, holds or uses it,
and puts it away:

- Drive&Act: a clip whose activity does not set phone, drink or food gets 0 for it when
  no object-level annotation (`objectlevel.chunks_90.csv` of the same camera) of a
  phone, bottle or food overlaps the frames it reads. A clip in which the driver fetches
  or places that object stays unlabelled for it.
- DMD distraction sessions: a clip whose every frame has a driver action, none of them
  `unclassified` (a transition), gets phone 0 when no frame is texting or a phone call
  and none has the `cellphone` object, and drink 0 when no frame is drinking and none
  has the `bottle` object. DMD gives no food label: it has no eating action or food
  object, and food positives come only from Drive&Act, so a food 0 on every DMD clip
  would teach that DMD's face camera never shows food.
- A phone 0 also sets `phone_at_ear` 0.
- AUC Distracted Driver v2 has no object annotations, so it gives only the positives
  its classes show: phone 1 for texting and phone calls, with `phone_at_ear` 1 for calls
  and 0 for texting (the phone is in the hand), and drink 1 for drinking. It gives no
  phone, `phone_at_ear` or drink 0: a class names what the driver does, not what they
  hold, and drivers hold a phone in other classes (checked 2026-10-30: one while reaching
  behind, another while talking to the passenger, another while driving safely). Until a
  teacher adds 0s, phone and drink have only positives among the training images (4,661
  phone and 933 drink in v0), so training masks them on images (one class). AUC gives no
  food label, for the same reason as DMD: it has no eating class. It sets neither
  `face_visible` nor `attention_direction`: its side cameras do not show the face in
  every image of a class (reaching behind turns it away), and no class fixes where the
  driver looks (adjusting the radio does not require looking at it).

These rules only add 0s to clips that were already cut; they change no clip. Running the
adapters over the v0 video clips on 2026-10-29 gave these train targets (AUC Distracted
Driver v2's images add only the positives above):

| Label | Positives | Negatives |
|---|---:|---:|
| phone | 5,580 (DMD 2,403, Drive&Act 3,177) | 37,233 (DMD 4,461, Drive&Act 32,772) |
| drink | 1,700 (DMD 250, Drive&Act 1,450) | 41,268 (DMD 6,747, Drive&Act 34,521) |
| food | 3,166 (Drive&Act) | 32,615 (Drive&Act) |

`bulwark ingest` of DMD and Drive&Act writes these 0s, and the v0 manifest has them.
They change no clip, so shards, teacher label files and cached features do not depend
on them.

**Teachers.**

- The face teacher (MediaPipe Face Landmarker) labels eye, mouth and head signals wherever
  it finds a face. Its output is not used on eye crops, and its attention direction,
  measured in the camera frame, and a missed face (`face_visible` 0) are kept only for
  cameras that roughly face the person. Of its `eyes_closed`, `blink` and `yawn`, only
  `eyes_closed` 0 (eyes open) is kept; its `blink` and `yawn` are dropped, 0 and 1
  alike, so it gives neither label a value (see "Teacher label quality" below); a
  synthetic clip's prompted labels still can. Where it says the eyes are closed, its
  `eye_openness` and `perclos` for that clip are dropped too, unless the clip's
  ground-truth `eyes_closed` confirms it; where
  ground truth contradicts its `eyes_closed`, they are dropped whatever it says. These
  were measured on DMD's face camera, so on cameras that do not face the person (all of
  Drive&Act's) its `eyes_closed`, `eye_openness` and `perclos` are all dropped. Its
  other labels (`mouth_open`, head pose, `head_nod`, `face_visible` 1) are not filtered
  by these measurements.
- The objects teacher (RF-DETR) writes negatives only. A COCO detection shows that an
  object is in view, not that the person is using it, so a detection never becomes a
  positive. The teacher writes 0 for phone, drink or food when nothing of that kind scores
  0.2 or more in any of 4 sampled frames. Where Drive&Act and DMD give these labels from
  their object annotations (above), their ground truth wins.
- The VLM teacher (Qwen3-VL) labels activity, context, attention and alertness on clips
  without a ground-truth activity, and object flags as positives only.
- Neither the objects teacher nor the VLM teacher runs on eye crops; their output there
  would have no hands or scene behind it.
- **Derived label.** DMD annotates eye states, blinks and yawns, not alertness. A DMD
  clip in which the eyes stay closed without a break for 1 s or more (within the clip's
  window of 1.6 s at 30 fps) gets alertness `microsleep`. The threshold is this project's
  choice, so the label is kept at teacher level (`meta["derived_labels"]`, merged into
  `labels_teacher` above the model teachers), never as ground truth. A closure split
  across two windows can be missed by both.

v0 therefore has no ground-truth alertness for training: NTHU-DDD is not ingested and
UTA-RLDD is the holdout.

**Coverage.** `bulwark coverage` (`bulwark.data.coverage`) flags labels that a split can
teach only one class of, and classes with no training target, per media type; with
`--markdown <file>` it writes the counts of ground-truth and teacher labels per label,
dataset, split and media type. Training jobs run the same check and mask the one-class
pairs. On the v0 release manifest (run 2026-10-31), it flags `alertness` on video (only
`microsleep`, from the DMD rule), `context` (only `vehicle_cabin`), `face_visible`
(only 1), `phone` and `drink` on images (only 1, from AUC Distracted Driver v2), and the
classes with no training target: `smoking` (activity, both media types), `looking_away`,
`eating`, `reading`, `interacting_object` and `other` (activity on images), and
`thermal` (modality). Its counts for the train split, whose labels the models learn
from, are below; `bulwark coverage --markdown` also gives val, test, holdout and
excluded:

| Label | Dataset | Media | Source | Train clips | Classes |
|---|---|---|---|---:|---|
| `activity` | AUC Distracted Driver v2 | image | ground truth | 12,638 | adjusting_controls: 953, attentive: 2723, drinking: 933, grooming: 898, phone_call: 2212, reaching: 891, talking: 1579, texting: 2449 |
| `activity` | DMD | video | ground truth | 7,308 | adjusting_controls: 240, attentive: 3071, drinking: 250, grooming: 283, other: 12, phone_call: 1763, reaching: 868, talking: 181, texting: 640 |
| `activity` | Drive&Act | video | ground truth | 36,732 | adjusting_controls: 767, attentive: 10745, drinking: 676, eating: 3166, grooming: 542, interacting_object: 4406, looking_away: 429, other: 2970, phone_call: 1267, reaching: 5426, reading: 4428, texting: 1910 |
| `alertness` | DMD | video | teacher | 107 | microsleep: 107 |
| `attention_direction` | DMD | video | ground truth | 2,062 | away: 1329, down: 89, down_center: 75, forward: 129, left: 141, right: 225, up: 74 |
| `attention_direction` | DMD | video | teacher | 8,515 | down: 2170, forward: 5494, left: 414, right: 436, up: 1 |
| `attention_direction` | Drive&Act | video | ground truth | 977 | away: 429, down_center: 548 |
| `blink` | DMD | video | ground truth | 3,766 | 0: 1200, 1: 2566 |
| `context` | AUC Distracted Driver v2 | image | ground truth | 12,638 | vehicle_cabin: 12638 |
| `context` | DMD | video | ground truth | 11,119 | vehicle_cabin: 11119 |
| `context` | Drive&Act | video | ground truth | 36,732 | vehicle_cabin: 36732 |
| `drink` | AUC Distracted Driver v2 | image | ground truth | 933 | 1: 933 |
| `drink` | DMD | video | ground truth | 6,997 | 0: 6747, 1: 250 |
| `drink` | Drive&Act | video | ground truth | 35,971 | 0: 34521, 1: 1450 |
| `eye_openness` | DMD | video | teacher | 10,078 | - |
| `eye_openness` | MRL Eye | image | ground truth | 70,989 | - |
| `eyeglasses` | MRL Eye | image | ground truth | 70,989 | 0: 51917, 1: 19072 |
| `eyes_closed` | DMD | video | ground truth | 1,825 | 0: 1660, 1: 165 |
| `eyes_closed` | DMD | video | teacher | 8,344 | 0: 8344 |
| `eyes_closed` | MRL Eye | image | ground truth | 70,989 | 0: 33692, 1: 37297 |
| `face_visible` | AUC Distracted Driver v2 | image | teacher | 2,880 | 1: 2880 |
| `face_visible` | DMD | video | ground truth | 11,119 | 1: 11119 |
| `face_visible` | Drive&Act | video | teacher | 8,459 | 1: 8459 |
| `food` | Drive&Act | video | ground truth | 35,781 | 0: 32615, 1: 3166 |
| `head_nod` | DMD | video | teacher | 10,879 | 0: 10689, 1: 190 |
| `head_nod` | Drive&Act | video | teacher | 8,459 | 0: 8015, 1: 444 |
| `head_pitch` | AUC Distracted Driver v2 | image | teacher | 2,880 | - |
| `head_pitch` | DMD | video | teacher | 10,879 | - |
| `head_pitch` | Drive&Act | video | teacher | 8,459 | - |
| `head_roll` | AUC Distracted Driver v2 | image | teacher | 2,880 | - |
| `head_roll` | DMD | video | teacher | 10,879 | - |
| `head_roll` | Drive&Act | video | teacher | 8,459 | - |
| `head_yaw` | AUC Distracted Driver v2 | image | teacher | 2,880 | - |
| `head_yaw` | DMD | video | teacher | 10,879 | - |
| `head_yaw` | Drive&Act | video | teacher | 8,459 | - |
| `low_light` | MRL Eye | image | ground truth | 70,989 | 0: 24923, 1: 46066 |
| `modality` | AUC Distracted Driver v2 | image | ground truth | 12,638 | rgb: 12638 |
| `modality` | DMD | video | ground truth | 11,119 | rgb: 11119 |
| `modality` | Drive&Act | video | ground truth | 36,732 | nir: 30495, rgb: 6237 |
| `modality` | MRL Eye | image | ground truth | 70,989 | nir: 70989 |
| `mouth_open` | AUC Distracted Driver v2 | image | teacher | 2,880 | - |
| `mouth_open` | DMD | video | teacher | 10,879 | - |
| `mouth_open` | Drive&Act | video | teacher | 8,459 | - |
| `perclos` | DMD | video | ground truth | 1,825 | - |
| `perclos` | DMD | video | teacher | 8,344 | - |
| `phone` | AUC Distracted Driver v2 | image | ground truth | 4,661 | 1: 4661 |
| `phone` | DMD | video | ground truth | 6,864 | 0: 4461, 1: 2403 |
| `phone` | Drive&Act | video | ground truth | 35,949 | 0: 32772, 1: 3177 |
| `phone_at_ear` | AUC Distracted Driver v2 | image | ground truth | 4,661 | 0: 2449, 1: 2212 |
| `phone_at_ear` | DMD | video | ground truth | 6,864 | 0: 5101, 1: 1763 |
| `phone_at_ear` | Drive&Act | video | ground truth | 35,949 | 0: 34682, 1: 1267 |
| `yawn` | DMD | video | ground truth | 1,833 | 0: 1472, 1: 361 |

Teacher rows come from the face teacher, and `alertness` from the DMD rule; v0 ran no
other teacher. The `head_nod` and `attention_direction` teacher counts are v0's labels,
from face-teacher rules that had the pitch sign reversed: v0's `head_nod` fired on the
head rising and coming back, and v0's `attention_direction` has `down` and `up`
swapped (fixed in 0.1.0; see the model card's
[Limitations](../MODEL_CARD.md#limitations)).

**Teacher agreement.** `bulwark label agreement` (`bulwark.label.agreement`) compares each
teacher with ground truth on clips that have both (precision, npv, recall and F1 for
binary labels, accuracy for categorical labels, MAE for regression labels, per label and
dataset). Precision is the share of the teacher's positives that ground truth confirms,
and npv (negative predictive value) the share of its negatives that ground truth
confirms. `--split` limits it to some splits (repeatable or comma-separated; every split
by default; a `--split` that names no split is an error), and `--out <file>` also writes
the table as JSON, `{"splits": [...] or "all", "report": {...}}`, so a train-only table
can be told from an all-split one. The command leaves out
datasets the split policy excludes (`zenodo_driver`), so their ground truth is never
scored or published; the library function `teacher_agreement` scores whatever records it
is given. v0 ran only the face teacher, so its table is the only one. On the train split of
the v0 release manifest (`bulwark label agreement --split train`, run 2026-10-31):

| Label | Dataset | Clips with both | Agreement |
|---|---|---:|---|
| `attention_direction` | DMD | 1,434 | accuracy 0.239 |
| `attention_direction` | Drive&Act | 227 | accuracy 0.000 |
| `eyes_closed` | DMD | 1,825 | F1 0.764 (precision 0.668, npv 0.989, recall 0.891) |
| `perclos` | DMD | 1,825 | MAE 0.113 |
| `blink` | DMD | 3,713 | F1 0.771 (precision 0.973, npv 0.551, recall 0.638) |
| `yawn` | DMD | 1,830 | F1 0.301 (precision 1.000, npv 0.832, recall 0.177) |
| `face_visible` | DMD | 11,119 | F1 0.989 (precision 1.000, recall 0.978; ground truth has no negatives) |

The `attention_direction` rows score v0's teacher labels, whose `down` and `up` are
swapped by the reversed pitch sign (see the note under the coverage table and the model
card's [Limitations](../MODEL_CARD.md#limitations)).

Drive&Act's attention ground truth has only the classes `away` and `down_center`, which
the face teacher does not give, so its accuracy there is 0. `attention_direction` and
`face_visible` are masked in every v0 training config, so neither is trained.

**Teacher label quality.** The face teacher's raw output against DMD's own labels, on DMD
face-camera clips of the train split where the teacher found a face (measured
2026-10-29; `bulwark label agreement --split train` reproduces these counts):

| Label | Teacher says | Ground truth agrees | Disagrees | Right | Kept |
|---|---|---|---|---|---|
| `eyes_closed` | 1 (closed) | 147 | 73 | 66.8% | no |
| `eyes_closed` | 0 (open) | 1587 | 18 | 98.9% | yes |
| `blink` | 1 | 1621 | 45 | 97.3% | no (one-sided, below) |
| `blink` | 0 | 1127 | 920 | 55.1% | no |
| `yawn` | 1 | 64 | 0 | 100% | no (one-sided, below) |
| `yawn` | 0 | 1469 | 297 | 83.2% | no |

`bulwark label merge` keeps only the values listed in `FACE_KEEP`
(`bulwark/label/run.py`), and applies them to every dataset on cameras that face the
person. A value was chosen once, when both of these held: it is right at least 90% of
the time in this table, and the class that keeping only that value leaves out is rare
in the clips it labels, so the one-sided labels it then gives there are close to the
truth (below). The merge does not
measure accuracy as it runs, so a change in the teacher's accuracy changes nothing until
`FACE_KEEP` is edited. Only the train split was used to choose this, so val and test
ground truth never shape training labels; on the val and test splits the same values
pass and fail the 90% rule. The teacher reads looking down (texting,
reading) as closed eyes; its missed blinks are short blinks that fall between sampled
frames, and its missed yawns are covered or partial yawns. Its `perclos` is off from
ground truth by 0.113 on average (MAE), 0.103 where it says the eyes are open and 0.192
where it says they are closed (`bulwark label agreement` reports only the overall MAE;
the split groups the same 1,825 pairs, 1,605 and 220, by the teacher's own
`eyes_closed` on each clip), so where its `eyes_closed` is dropped, its `perclos` and
`eye_openness` (from the same eye-closure measure) are dropped with it. A clip's own
ground truth overrides this: where its `eyes_closed` confirms the teacher's closed eyes
(147 train clips) they are kept, and where it contradicts the teacher's open eyes (18)
they are dropped. No dataset gives video `eye_openness` ground truth, so these 147 are
the only video `eye_openness` targets from closed eyes that ground truth backs. Even so,
the teacher's video
`eye_openness` targets lean open: on the v0 train split 211 of 10,078 are below 0.5,
against 649 of 10,879 on the same camera without the rule, and its `perclos` targets
never exceed 0.75 (148 of 8,344 above 0.5).

Keeping one value of a label makes the face teacher's supervision one-sided on clips
without ground truth for that label. That is close to the truth only where the class
left out is rare in those clips: it is for `eyes_closed`, not for `blink`, and for
`yawn` it would matter on frontal datasets added later rather than on v0. The counts
below are for the v0 train split, video, cameras that face the person and clips without
ground truth for the label, which are all DMD face-camera clips:

- `eyes_closed`: the teacher calls 710 of the 9,054 clips it labels closed, 481 of them
  clips that DMD marks as texting or not looking at the road. Closed eyes, the class its
  "open" leaves out, are rare in these clips, so its 8,344 "open" labels, negatives
  only, are close to the truth, and they are kept.
- `blink`: "no blink", the class its blinks leave out, is common: the teacher says no
  blink on 4,127 of the 7,166 clips it labels and finds one on the other 3,039, and its
  "no blink" is right only 55.1% of the time where ground truth exists. Keeping only its
  blinks would give the blink output 3,039 positives and no negatives from the teacher,
  pulling it towards "blink" on every such clip, so neither value is kept.
- `yawn`: the teacher finds no yawn in the 9,049 clips it labels, so on v0 keeping only
  its yawns would add no rows, and dropping them changes nothing. "No yawn", the class
  they leave out, is common, so on frontal datasets added later (YawDD's dash camera,
  which has no yawn ground truth) its yawns alone would give positives-only rows; they
  are dropped for that reason, and its "no yawn" fails the 90% rule.

The face teacher therefore gives `blink` and `yawn` no value. A synthetic clip's
prompted labels still can (`scripts/synth_wan.py` prompts `yawn` 1 in one scene,
positives only), but v0 has no synthetic clips, so in v0 `blink` and `yawn` train on
DMD's ground truth only: on the v0 train split, `blink` 2,566 positives and 1,200
negatives, and `yawn` 361 positives and 1,472 negatives. Without the rule, and counting
Drive&Act's cabin cameras, the teacher would give `blink` 10,224 negatives and 5,401
positives, `yawn` 17,505 negatives and 3 positives, and `eyes_closed` 15,732 negatives
and 1,781 positives. Its `eyes_closed` rows train at half weight with no class
reweighting, so they pull `eyes_closed` towards 0; check that output's calibration on
val after training with them, in its video entry of `calibration.json` and the report's
thresholds table (the video and image fits are separate).

`bulwark label merge` applies the rule each time it rebuilds `labels_teacher`, and the
v0 manifest (`data/manifest.parquet`) is merged with it.

**Label quality.** Some labels are self-reported (UTA-RLDD alertness and KSS), some are
mapped from each dataset's own classes onto Bulwark's taxonomy, one is derived by a rule
(DMD microsleep), and teacher labels carry their models' errors; the face teacher is
built for front-facing video, and cannot tell looking down from closed eyes.

## Personal data and removal requests

The source datasets show real people's faces, so the data is personal data, and some
datasets restrict publishing it further:

- UTA-RLDD's page says the subjects' identity cannot be revealed and that only 36 of its
  60 participants allowed their faces to be published (listed in its "Image Publishable"
  sheet). Never publish frames or crops that could identify a UTA-RLDD subject.
- YawDD and NTHU-DDD restrict publishing images of their subjects.
- Bulwark's documentation and releases contain no frames from any dataset.

**Where copies live.** Data is kept on the machine that prepares it. The registry clears
each third-party storage provider separately. Shards go to a private Kaggle dataset only
for datasets whose registry entry has `kaggle_storage_ok=True`, or that the operator
names explicitly ([training.md](training.md#running-on-kaggle)). For Drive&Act, DMD and
MRL Eye the maintainers approved private Kaggle datasets on 2026-10-29, for training on
Kaggle's GPUs only; the Zenodo set may not be uploaded. The same day they approved one for
AUC Distracted Driver v2, whose licence agreement forbids transferring the data to third
parties without the MI-AUC group's prior authorization. The group has not authorized it.
AUC may not be kept on Azure Blob. Azure jobs read shards from private Blob storage in the
operator's own Azure subscription; `infra/azure/upload.sh` uploads only datasets with
`azure_storage_ok=True`, or that the operator names. No dataset is cleared for Azure Blob
yet: that is decided when an Azure job is needed.

**Removing a person.** People who appear in a training dataset, and dataset owners, can
ask Inferensys to remove data by writing to hello@inferensys.com; please do not post
personal data in public issues. DMD
participants can also ask Vicomtech, through the contact on the DMD website
(https://dmd.vicomtech.org/). A removal is applied as follows:

1. Add the subject to `bulwark/data/removed_subjects.yaml` as `<dataset>:<subject>`, with
   the date and source of the request (no names). `bulwark ingest` and `bulwark split`
   then leave the subject's clips out, and training and evaluation refuse a manifest that
   still holds them (the split leak check).
2. Run `bulwark purge <dataset>:<subject>` (or `bulwark purge` for every listed subject).
   It deletes their clips from the per-dataset and merged manifests, the shards in
   `data/shards` and `data/shards-sample`, the teacher labels and every feature cache
   under `data/features`, and logs the purge in `data/purges.jsonl`. It refuses a
   subject not listed in step 1. Saved evaluation predictions (`runs/*/predictions`) and
   reports also hold the clips' ids and outputs: delete them, or evaluate again.
3. Run `scripts/purge_cloud_copies.sh <dataset>` on the same machine. It refuses while a
   local manifest still holds a listed subject. In Azure Blob, for a dataset cleared for
   Blob (`azure_storage_ok`), it uploads the purged shards over their copies and only
   then deletes the dataset's shard blobs that no local shard replaces; for a dataset that
   is not cleared for Blob, it uploads no shard and deletes every shard blob of it. It
   deletes the dataset's feature blobs for the cache job to recompute, replaces the
   manifest after the same clearance check as `infra/azure/upload.sh` (or, with
   `DROP_UNCLEARED=1`, deletes it when no dataset in the local manifest is cleared for
   Blob), and deletes the teacher-label blobs (jobs read labels through the manifest); it
   warns when blob soft delete or versioning would keep old copies. Kaggle keeps every
   version of a dataset, so `KAGGLE_SLUGS="<slug> ..."` deletes the Kaggle datasets that
   held the dataset's shards or the manifest; the script prints how to recreate them with
   `infra/kaggle/push_data.sh`, with `SHARD_GLOB` listing every source each one held.
   Delete kernels whose outputs hold copies (a cache job's features) with `kaggle kernels
   delete`. Each run appends the date, the dataset, the number of subjects and what was
   done to `docs/purge-log.md`, without subject ids. `DRY_RUN=1` shows what would
   happen.
4. Retrain. A removal applies from the next retrain onward; weights that were already
   released are not retrained.

AUC Distracted Driver v2 cannot honour a removal request for one driver.
It publishes no driver ids, so its subject ids name whole driver groups: the only valid
entry in step 1 is `auc_v2:train_drivers` or `auc_v2:test_drivers`, which removes every
image on that side of its split. `bulwark purge --clips` can delete one person's images by
clip id, but records nothing that `bulwark ingest` or `bulwark split` honour, so the next
ingest brings them back. Until a per-image removal list that ingest honours exists, a
request from an AUC participant can only be met by removing that person's whole group or
the whole dataset.

Before each release, compare each dataset's subject list with its provider's current
release, so that subjects the provider removed are removed here too. AUC has no subject
list to compare: compare its image list with the authors' current archive instead.

## Release decisions

- **Publishing weights at all.** The terms of the v0 training sources (Drive&Act, DMD,
  MRL Eye and AUC Distracted Driver v2) do not settle whether weights trained on them may
  be published. On 2026-10-30 Inferensys decided to release the v0 weights for
  non-commercial research only, under CC BY-NC 4.0, on each dataset's research terms.
  That is Inferensys's decision, not a statement by the datasets' owners. Users of the
  weights must also follow each dataset's terms, and if a dataset's owners object, the
  weights trained on it are withdrawn and the models retrained without it. The registry
  records the decision for these four datasets as `weights_release_ok=True` with the
  decision as `weights_release_basis`; `bulwark release-check` prints that basis and
  refuses weights trained on any dataset that is not cleared. B1 v0 and E1 v0 are
  published, E1 v0 as a research preview that scores well below B1 (see
  [E1 v0 test results](evaluation.md#e1-v0-test-results) and
  [RELEASING.md](../RELEASING.md)).
- **DMD and Drive&Act.** DMD may be used for academic purposes only (CC BY-NC-ND 4.0) and
  Drive&Act for research only, and neither says anything explicit about trained models.
  Bulwark is developed by a company, Inferensys, which trains on both internally for
  non-commercial research. The terms do not settle whether that use, and a research-only
  release of weights trained on them (for DMD, including whether such weights count as
  adapted material under ND), are allowed; releasing the v0 weights rests on the decision
  above.
- **MRL Eye.** The live site was unreachable when checked, so its terms were read from
  the Internet Archive's copy of the official page, which states no licence or terms of
  use and refers users to the paper cited in [datasets.md](datasets.md#citations). The
  copy used came from a Kaggle mirror whose images match the official archive
  ([datasets.md](datasets.md) records the comparison). Releasing the v0 weights rests on
  the decision above.
- **AUC Distracted Driver v2.** Its licence agreement forbids commercial use, limits use
  without the MI-AUC group's further consent to developing distracted-driver detection,
  and says nothing about trained models. Releasing the v0 weights trained on it for
  non-commercial research rests on the decision above. The maintainers approved keeping a
  private Kaggle copy (see "Where copies live"); the group has not authorized it.

Decided: the Zenodo set is excluded from v0 (see [Sources](#sources)). Using it in a
later release would first need Daffodil International University's written permission
for any third-party storage, the publication review its data use agreement asks for, and
a reading of the State Farm competition rules for the images it appears to share with
State Farm.

[RELEASING.md](../RELEASING.md) lists these checks.

## Known gaps

- Few people: most labelled video comes from 29 people in two datasets (15 Drive&Act
  drivers and 14 DMD subjects).
- No demographic fields apart from MRL Eye's gender code (`meta["gender"]`, from its file
  names), so performance per group cannot be measured on video.
- Mostly in-car data, much of it near-infrared; no thermal data; no real forklift, crane,
  factory or patient footage.
