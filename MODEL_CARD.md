# Model card: Bulwark-B1 and Bulwark-E1 (v0)

> **Status.** This is the full card of the models of the first release, Bulwark-B1 v0 and
> Bulwark-E1 v0. Both are trained and evaluated (indicative results under
> [Evaluation](#evaluation)), and each is published in its own Hugging Face model
> repository: B1 at https://huggingface.co/Inferensys/bulwark-b1-driver-monitoring and E1
> at https://huggingface.co/Inferensys/bulwark-e1-driver-monitoring-onnx.
>
> **Which model to use.** Use Bulwark-B1 for research. Bulwark-E1 v0, the smaller ONNX
> student for CPU, is a research preview that scores well below B1. It does not detect
> closed eyes in driver video: on DMD's test people it scores at chance. Its
> phone, eating and drinking detection falls to chance or below on a camera view it was
> not trained on, and its activity results are weak. It does well on close-up still
> images of eyes (MRL Eye) and on phone use from the camera views it was trained on. It is
> released for trying CPU and ONNX deployment and Bulwark's output format, not for its
> accuracy, and E1 will be retrained with a different recipe (see
> [docs/evaluation.md](docs/evaluation.md#e1-v0-test-results)).
>
> The README of each repository is a shorter card for that model, generated from
> `release/hf/` with its own results, and each repository also ships this card as
> MODEL_CARD.md. A repository is opened only when every artifact in it clears
> `bulwark release-check` and Inferensys has reviewed its private page (see
> [RELEASING.md](RELEASING.md)). It is then a gated repository with automatic
> approval: the README is public, and downloads need an access request (name, company or
> affiliation, intended use, acceptance of CC BY-NC 4.0 and the training datasets' terms,
> and whether Inferensys may contact the requester about commercially licensed models),
> which is approved as soon as it is sent. Files other than the README and LICENSE open
> only after that, so the README itself states the licence, the training datasets' terms,
> the out-of-scope uses and the contact for removal requests, and links to the
> documentation on GitHub.

## Model details

| Item | Detail |
|---|---|
| Developer | Inferensys |
| Models | **Bulwark-B1** (base) and **Bulwark-E1** (edge student, a research preview that scores well below B1), both released |
| Model release | v0 (the training configs write to `checkpoints/*-v0`; B1 v0's run wrote to `checkpoints/b1-lora-auc`, E1 v0's to `checkpoints/e1-v0`, see [Evaluation](#evaluation)) |
| Weights | each in its own Hugging Face model repository, gated with automatic approval (see [Licence](#licence)): B1 at https://huggingface.co/Inferensys/bulwark-b1-driver-monitoring (safetensors), E1 at https://huggingface.co/Inferensys/bulwark-e1-driver-monitoring-onnx (a 16-frame ONNX graph for video and a 1-frame graph for still images) |
| Code | `inferensys-bulwark` 0.1.0 (`bulwark.__version__`), Apache-2.0 (vendored Meta code: MIT) |
| Label taxonomy | version 1 (`bulwark/data/taxonomy.yaml`): 34 outputs in 4 heads |
| Task | multi-task prediction over a short clip of one person on camera: eye, head and mouth signals, objects, alertness and activity, and capture conditions |
| Weights licence | CC BY-NC 4.0, plus the terms of the training datasets (see [Licence](#licence)) |

**Bulwark-B1.** Meta's V-JEPA 2.1 ViT-B/16 video encoder (frozen, about 86.8M
parameters) with rank-16 LoRA adapters on the `qkv`, `proj`, `fc1` and `fc2` layers of
its last 4 blocks (16 layers) and its final norm trained (about 0.79M trainable
parameters), pooled into 25 summary tokens, then a 2-layer transformer head (about 6.8M
parameters); the fine-tune trains 7.56M parameters in all. Input: 16 RGB frames at
256 x 256, or a single image.

**Bulwark-E1.** EfficientViT-B1 (timm
`efficientvit_b1.r224_in1k`) applied to each frame at 224 px, then a 2-layer transformer
head over the 16 frame tokens; about 9.5M parameters in total. Trained on ground-truth
labels plus B1's soft outputs, and exported to ONNX. Input: 16 letterboxed RGB uint8
frames at 256 x 256 (resized inside the graph), or one frame in the graph for still
images; the full input and output contract is in
[docs/architecture.md](docs/architecture.md#e1-onnx-contract). E1 v0 is a research preview
that scores well below B1 (see [Evaluation](#evaluation)).

Both models output raw logits per label: probabilities after a sigmoid (binary labels)
or softmax (categorical labels), and values in the label's range after a scaled sigmoid
(regression labels). Each also gives a state embedding (the head's CLS vector, 512-d for
B1 and 256-d for E1) for training small task adapters. Every checkpoint lists the outputs
it was trained on (`supervised_labels`); the others are untrained (see
[Labels and supervision](#labels-and-supervision)). Outputs are uncalibrated unless
per-label temperatures from the evaluation's `calibration.json` for the same weights are
applied (`bulwark.inference.load(..., temperature=...)`) or recorded with the model
(`bulwark release --calibration` for a release directory, `bulwark export --calibration`
for the ONNX graph), which `load` then applies. Temperatures and decision thresholds are
fitted per media type, since the model's confidence differs between still images and
video clips: images get those fitted on image val clips and video those fitted on video
val clips, and a graph exported for one media type carries that media type's.

B1 training checkpoints contain only the LoRA weights, the encoder's final output norm and
the heads; using them needs Meta's V-JEPA 2.1 checkpoint, which
`scripts/fetch_pretrained.sh` downloads. A release directory (`bulwark release` or
`examples/make_release.py`) holds the full weights in `model.safetensors` and a
`config.json` with the advertised labels, the weights licence, `commercial_only`, the
training provenance (names and hashes, no local paths) and the release check; a B1
release also holds `LICENSE-vjepa2`, Meta's licence notice for the encoder weights it
contains.

### Input contract

| Item | Detail |
|---|---|
| Frames | uint8 RGB, (batch, 16, 256, 256, 3), in time order, channels last |
| Resize | each frame letterboxed to 256 x 256: aspect ratio kept, shrunk with `cv2.INTER_AREA` (enlarged with `INTER_LINEAR`), centred on black padding, as `bulwark.data.clips.letterbox` does; no stretching or centre crop. E1 resizes to 224 x 224 inside the model, in PyTorch and ONNX alike. |
| Normalisation | inside the model: scaling to [0, 1] and ImageNet mean and std |
| Temporal | sample frames as in training: 16 frames, every k-th source frame with `k = max(1, round(2 x fps) // 16)`, so every 3rd frame at 30 fps (about 1.5 s) and every frame at 15 fps (about 1.07 s). 16 consecutive frames of a 30 fps camera span about 0.5 s, a third of a training clip. Image datasets were trained as single frames. An ONNX graph takes the frame count it was exported with (16 by default); export one with `--frames 1` for still images. |
| Training domain | frames were letterboxed as above and stored as JPEG at quality 90, so the models saw JPEG-compressed frames |
| Outputs | one raw logit tensor per label, in taxonomy order: (batch,) for binary and regression labels, (batch, C) for categorical labels; an ONNX graph from `bulwark export` also returns `tokens` and `state` (not with `--no-embeddings`) |
| Windows | every call encodes all 16 frames; a sliding window costs one full call per hop |

The exported ONNX graph stores this contract in its metadata, with the weights licence, a
link to E1's model repository, the training record and the
trained outputs (`bulwark.export.onnx.read_contract`);
[docs/architecture.md](docs/architecture.md#e1-onnx-contract) lists the entries and gives
a decoding example.

## Licence

**Weights.** The B1 v0 and E1 v0 weights are licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
(legal code: [LICENSES/CC-BY-NC-4.0.txt](LICENSES/CC-BY-NC-4.0.txt)), and their use is
also subject to the terms of every dataset they were trained on (listed in
[Training data](#training-data)). Several of those datasets are licensed for
non-commercial or academic research only. In practice:

- **Allowed:** non-commercial research and teaching, including benchmarking, with attribution to
  Inferensys and Bulwark, a link to the licence, and citations of the training datasets.
- **Not allowed:** any commercial use, including use in a product or service, product or
  sales demos, pilots, proofs of concept and customer evaluations.
- **Commercial use** needs weights trained only on commercially licensed data (the
  commercial-licence track in [docs/training.md](docs/training.md#commercial-licence-track))
  under a separate licence. Contact Inferensys at https://inferensys.com/contact or
  hello@inferensys.com.

The non-commercial terms cover every use of the v0 weights, including computing
embeddings with them and training adapters on those embeddings.

**Release status.** Whether a model trained on a dataset may be published at all depends
on that dataset's terms, and the terms of the v0 training sources do not settle it. DMD's
terms allow academic use only (CC BY-NC-ND 4.0) and Drive&Act's research use only, and
neither says anything explicit about trained models; MRL Eye states no licence. AUC
Distracted Driver v2 is licensed under a signed agreement that forbids commercial use
(including testing commercial systems), limits use without the MI-AUC group's further
consent to developing distracted-driver detection, forbids transferring the data to third
parties without the group's prior authorization, and does not mention trained models.
Bulwark is developed by a company, Inferensys, which trains on these datasets internally
for non-commercial research. On 2026-10-30 Inferensys decided to release the v0 weights for non-commercial
research only, under CC BY-NC 4.0, on each dataset's research terms. This is
Inferensys's decision, not a statement by the datasets' owners (Vicomtech for DMD,
Fraunhofer IOSB for Drive&Act, the MRL Eye authors, and the MI-AUC group for AUC
Distracted Driver v2). Users of the weights must also follow each dataset's terms. If a
dataset's owners object, the weights trained on that dataset are withdrawn and the models
retrained without it. The dataset registry records the decision for each of these
datasets as `weights_release_ok` with its `weights_release_basis`, and
`bulwark release-check` prints that basis with its verdict. Weights are published only
once every dataset in the release, including the datasets of the models a release
learned from, is cleared and the release is complete (see [RELEASING.md](RELEASING.md)
and the [data card](docs/data-card.md#release-decisions)).

**Code.** The source code is Apache-2.0 ([LICENSE](LICENSE)), except Meta's V-JEPA 2.1
encoder code vendored in `bulwark/model/vjepa21`, which is under the MIT License (the
`LICENSE` file in that folder). The code licences do not cover the weights or any
dataset.

**Upstream components.** E1 checkpoints are modified from timm's EfficientViT-B1
ImageNet-1k weights (Apache-2.0); the terms of that ImageNet-1k pretraining need checking
before any commercially licensed model is built on those weights. B1 training
checkpoints include only the encoder's final output norm of Meta's V-JEPA 2.1
checkpoint, fine-tuned. A B1 release contains the full encoder
weights, derived from that checkpoint with the LoRA adapters merged in; the V-JEPA 2
repository that publishes them is under the MIT License (Copyright (c) Meta Platforms,
Inc. and affiliates), and the release ships that notice as `LICENSE-vjepa2`. See
[NOTICE](NOTICE). On Hugging Face, B1's card names no base model, because B1's V-JEPA 2.1
encoder comes from Meta's own download, which has no Hugging Face repository. E1's card
names `timm/efficientvit_b1.r224_in1k`, E1's initialisation only, as its base model.

## Intended use

- Research on monitoring a person's physical state on camera: drowsiness, distraction,
  attention, eye and head behaviour, and risky actions such as phone use.
- A research base model: frozen features and embeddings for training small task
  adapters, and a reference for building and evaluating such systems.
- Outputs are signals for decision support, for example an alert to the person being
  monitored or an input to a safety system that has its own logic and validation.
- Bulwark-E1 v0 is for trying CPU and ONNX deployment and Bulwark's output format. Its
  accuracy is well below B1's: use B1 for research.

## Out-of-scope uses

- Any commercial use of v0 weights (see [Licence](#licence)).
- Employment, disciplinary, performance, insurance or other decisions about individuals,
  including scoring or ranking workers.
- Identifying, recognising or tracking people. The models have no identity outputs, and
  their embeddings must not be used for re-identification.
- Inferring emotions, intentions or other inner states. The labels describe physical
  states (eyes closing, yawning, head pose) and visible actions.
- Covert monitoring, or monitoring people without the notice and legal basis the law
  requires.
- Use as a certified safety component, a type-approved vehicle system, a medical device
  or a diagnostic tool. Bulwark has not been validated or certified for any of these.

## Regulatory context

This is general information for readers building on Bulwark, not legal advice. The base
model has not been validated for any of the product settings below; whoever builds and
deploys a product is responsible for its conformity.

| Setting | Rules that may apply in the EU |
|---|---|
| Driver monitoring in vehicles | Driver drowsiness and distraction warning systems are covered by vehicle type-approval rules (General Safety Regulation (EU) 2019/2144). GDPR applies to the video. |
| Workers (professional drivers, forklift and crane operators, factory staff) | AI Act Annex III, point 4: systems used to monitor and evaluate the behaviour of workers are high-risk. Article 5(1)(f) prohibits inferring emotions in the workplace except for medical or safety reasons; Recital 18 states that physical states such as fatigue are not emotions. GDPR (including a data protection impact assessment under Article 35) and national employment and works-council rules apply. |
| Patients | Software intended to monitor a medical condition may be a medical device under Regulation (EU) 2017/745 (MDR). Health data is special-category data under GDPR Article 9. |

Face video is personal data. Under GDPR it becomes biometric (special-category) data when
it is processed to identify someone. The AI Act defines biometric data more widely
(Article 3(34)): data from the technical processing of a person's physical or behavioural
features, such as facial images, can count as biometric data even without
identification, so the face, eye and head features a model like Bulwark extracts may
fall under it. Bulwark has no identity outputs; do not combine its outputs with
identification.

## Training data

v0 is trained only on datasets licensed for research or non-commercial use. No
commercially licensed data is ingested yet. Every clip in the manifest records its
dataset, licence and `commercial_ok` flag. Licences, the terms relied on, access routes
and citations for all datasets are in [docs/datasets.md](docs/datasets.md). The
[data card](docs/data-card.md) describes the v0 data in detail: how clips are cut, the
people per split, label definitions, the teachers, personal data and removal requests,
and the release decisions.

Sources with an adapter, and whether v0 uses them (the clip and people counts of the v0
release manifest follow the table):

| Dataset | Licence (summary) | Used for | State |
|---|---|---|---|
| Drive&Act | research use only | train / val / test (official `split_0`: 10 / 2 / 3 drivers); inner-mirror camera of the 3 test drivers held out | ingested |
| MRL Eye | no licence stated, treated as research only | train / val / test (31 / 3 / 3 people) | ingested, from a Kaggle mirror identical to the official archive (see docs/datasets.md) |
| DMD | CC BY-NC-ND 4.0, academic use | train / val / test (9 / 2 / 3 subjects) | ingested (face camera, RGB); body and hands cameras not ingested |
| UTA-RLDD | no licence stated, treated as research only | cross-dataset holdout | not ingested |
| Multi-Class Driver Behavior (Zenodo 14908802) | labelled CC BY 4.0, but its data use agreement forbids sharing the data with third parties and asks to review publications 30 days before submission; appears to contain re-encoded State Farm images | none: excluded from v0 (`SplitPolicy.excluded_datasets`); its clips, if ingested, are in split `excluded` | excluded |
| NTHU-DDD | EULA, academic use | train / val / test | not ingested |
| YawDD | research use; publication of images restricted per subject | train / val / test | not ingested |
| State Farm | Kaggle competition rules (not yet reviewed) | train / val / test | not ingested |
| AUC Distracted Driver v2 | signed licence agreement: non-commercial; use limited to developing distracted-driver detection without the MI-AUC group's further consent; no transfer to third parties without the group's prior authorization | train / test (its authors' split by driver, corrected: 83 camera-1 test images of a driver who is also in train are moved to train; no per-image driver ids, so one driver group each and no val) | ingested (both cameras, 14,478 images) |
| CEW | research use by a researcher only; no commercial use or redistribution | train only | not ingested |
| SH17 | CC BY-NC-SA 4.0 (ShareAlike) | train only | not ingested |
| Synthetic clips (Wan 2.2) | generated with an Apache-2.0 model; prompt labels used as teacher labels | train only | none generated |

v0 is trained on the four ingested sources: Drive&Act, MRL Eye, the DMD face camera and
AUC Distracted Driver v2. The v0 release manifest, the one B1 v0 and E1 v0 were trained on
(sha256 `1d6175123358...`), holds 183,355 clips, 131,478 of them train. Clips (people) per split:

| Dataset | Train | Val | Test | Holdout | Excluded |
|---|---:|---:|---:|---:|---:|
| Drive&Act | 36,732 (10) | 7,908 (2) | 12,231 (3) | 2,032 (3; inner-mirror camera) | 7,418 (12; inner-mirror camera) |
| DMD, face camera | 11,119 (9) | 2,222 (2) | 4,317 (3) | 0 | 0 |
| MRL Eye | 70,989 (31) | 8,395 (3) | 5,514 (3) | 0 | 0 |
| AUC Distracted Driver v2 | 12,638 (1 driver group) | 0 | 1,840 (1 driver group) | 0 | 0 |

Drive&Act, DMD and MRL Eye shards may be kept in private Kaggle datasets for training
(approved by the maintainers on 2026-10-29, recorded in the registry as
`kaggle_storage_ok`); Azure Blob is not yet approved for any dataset (`azure_storage_ok`),
and the Zenodo set may be kept on neither. The maintainers also approved a private Kaggle
copy of AUC Distracted Driver v2 that day, although its licence agreement forbids any
transfer to third parties without the MI-AUC group's prior authorization; the group has
not given it.

**Teacher labels.** v0's teacher labels come from the face teacher and one rule only:
the objects and VLM teachers below were not run for v0, so they gave it no labels. A
MediaPipe face model labels eye, mouth and head signals wherever it finds a face. RF-DETR
writes only negatives for phone, drink and food (0 when nothing of that kind is detected
even at low confidence), because a detection shows an object in view, not the person
using it. Qwen3-VL labels activity, context, attention and
alertness on clips without a ground-truth activity, and object flags as positives only.
One label is derived by a rule from a dataset's own annotations rather than annotated:
DMD clips in which the eyes stay closed without a break for 1 s or more get alertness
`microsleep`, at teacher level, since the threshold is this project's choice. Teacher
and derived labels have half the weight of ground truth in the loss and are never used
for evaluation. The face teacher was compared with DMD's own labels on the train split
(2026-10-29; see the [data card](docs/data-card.md#labels)). Of its eyes-closed, blink
and yawn labels only "eyes open" becomes a label, and only on cameras that face the
person: ground truth confirmed it 98.9% of the time, and closed eyes are rare in the
clips it labels. Its blink and yawn labels are not used at all: its "no blink" and "no
yawn" were right only 55.1% and 83.2% of the time, and its blinks alone would give the
blink output positives only on clips without ground truth, where "no blink" is common
(it says no blink on 4,127 of the 7,166 DMD train clips it labels there). It finds no
yawn in those clips, so on v0 dropping its yawns changes nothing; they are dropped so
that frontal datasets added later do not get positives-only yawn labels from it. A
synthetic clip's prompted labels can still give `yawn` (positives only), but v0 has no
synthetic clips, so in v0 the blink and yawn outputs learn from DMD's ground truth only.
Its other labels (mouth open, head pose, head nod, face visible and, on cameras that face
the person, attention direction) are not filtered this way. Its "eyes closed" is not used, nor its eye openness and
PERCLOS on clips where it says the eyes are closed, unless the clip's ground truth
confirms the eyes were closed; where ground truth contradicts its eyes-closed reading,
those two go as well. The kept "eyes open" gives the eyes-closed output negatives only on
clips without ground truth. `bulwark label merge` applies this rule. Re-merging a copy of
B1 v0's training manifest with the current rules gives the same file byte for byte
(checked 2026-10-31, as RELEASING.md describes), so B1 v0 learned from the labels these
rules give. Since v0 ran no other teacher, the face teacher is the only one to compare
with ground truth; the data card gives its full `bulwark label agreement` table, which
leaves out datasets the split policy excludes (`zenodo_driver`), so their ground truth is
never scored or published. See [docs/architecture.md](docs/architecture.md#teacher-labels).

**Upstream pretraining.** The V-JEPA 2.1 ViT-B/16 encoder was pretrained by Meta; the
V-JEPA 2 repository's configs for this model list Kinetics-710, Something-Something v2,
HowTo100M and ImageNet-1k. E1 starts from EfficientViT-B1 weights trained on
ImageNet-1k. Bulwark's `commercial_only` filter does not apply to this data. Teacher
labels (MediaPipe, RF-DETR, Qwen3-VL) and synthetic clips (Wan 2.2) also come from models
trained on their own data under their own terms (see [NOTICE](NOTICE)); `commercial_only`
does not filter those either.

**No dataset frames are reproduced** in this repository, its documentation or its
releases. Some sources restrict publishing images of their subjects: YawDD and NTHU-DDD,
and UTA-RLDD, whose subjects' identity must not be revealed (only 36 of its 60
participants allowed their faces to be published). Never publish example frames that
could identify a UTA-RLDD subject.

## Labels and supervision

An output is only meaningful if a training source supervises it. The table lists, per
label, the datasets whose adapters give ground truth (the `provides` field of each entry
in `bulwark/data/registry.py`) and the teachers that label it.

| Head | Label | Ground truth from | Teacher |
|---|---|---|---|
| signals | eye_openness | MRL Eye, CEW | face |
| | eyes_closed | MRL Eye, CEW, DMD | face |
| | perclos | DMD | face |
| | blink | DMD | none (the face teacher's is dropped) |
| | mouth_open | none | face |
| | yawn | DMD, NTHU-DDD, YawDD | synthetic prompts (positives only; the face teacher's is dropped) |
| | head_yaw, head_pitch, head_roll | none | face |
| | gaze_yaw, gaze_pitch | **none** | none |
| | head_nod | NTHU-DDD | face |
| | attention_direction | Drive&Act, DMD, NTHU-DDD, State Farm | face (frontal views), vlm |
| | hands_on_controls | none (DMD's body and hands cameras annotate it; not ingested) | none |
| objects | phone, drink | Drive&Act, DMD, State Farm, AUC v2, Zenodo (phone) | objects, vlm |
| | phone_at_ear | Drive&Act, DMD, State Farm, AUC v2, Zenodo | none |
| | food | Drive&Act | objects, vlm |
| | eyeglasses | MRL Eye, NTHU-DDD, YawDD, SH17 | vlm |
| | sunglasses | NTHU-DDD, YawDD | vlm |
| | face_mask, helmet, safety_vest | SH17 | vlm |
| | gloves | SH17 | none |
| | cigarette, seatbelt | none | vlm |
| state | alertness | NTHU-DDD, UTA-RLDD (holdout) | vlm, synthetic prompts, DMD eye-closure rule (microsleep only; derived) |
| | kss | UTA-RLDD (holdout only) | none |
| | activity | Drive&Act, DMD, State Farm, AUC v2, Zenodo, NTHU-DDD, YawDD | vlm |
| condition | modality | every adapter except synthetic clips (RGB and NIR classes only; **no thermal data**) | none |
| | low_light | MRL Eye, NTHU-DDD | none |
| | face_visible | CEW, DMD, State Farm | face |
| | face_occluded | **none** | none |
| | context | most adapters | vlm |

The objects teacher gives only negatives (0) and the VLM teacher's object flags only
positives (1). The face teacher gives `eyes_closed` only negatives (0), and only on
cameras that face the person; its `blink` and `yawn` are not used, so it gives those two
nothing (synthetic prompts can still give `yawn` positives); see
[Training data](#training-data). Zenodo appears in the table
because its adapter writes those labels; v0 excludes it.

What this means for v0, with its four datasets (Drive&Act, the DMD face camera, MRL Eye
and AUC Distracted Driver v2):

- **Never trained:** `gaze_yaw`, `gaze_pitch`, `face_occluded` and the `thermal` class of
  `modality` have no source at all. Ignore these outputs.
- **Untrained in v0:** `hands_on_controls` and `gloves`. DMD, whose body and hands
  cameras annotate `hands_on_controls`, is ingested from the face camera, which does not
  see the wheel, so its clips carry no such label; SH17, the only source of `gloves`, is
  not ingested, and no teacher labels it. Ignore these outputs.
- **Masked in v0:** `attention_direction` and `face_visible`. Every v0 training config
  lists them in `mask_labels`: Drive&Act's attention targets never include `forward`,
  and on cameras that do not face the person the face teacher keeps only
  `face_visible` = 1. They leave the loss and are not distilled. Ignore these outputs.
- **Blink and yawn from DMD only:** their only training labels in v0 are DMD's own, from
  its face camera (3,766 `blink` and 1,833 `yawn` train clips); Drive&Act, MRL Eye and
  AUC Distracted Driver v2 give them none, and the face teacher gives them no value.
- **Teacher-only, not evaluated:** `mouth_open`, `head_yaw`, `head_pitch` and `head_roll`,
  which no dataset adapter gives ground truth, and `head_nod`, whose only ground-truth
  source, NTHU-DDD, is not ingested (all from the face teacher). Evaluation scores ground
  truth only, so their quality is unmeasured. `head_pitch` is positive when the head
  tips down relative to the camera. The v0 `head_nod` labels are wrong: the teacher's nod
  rule had that sign reversed (see [Limitations](#limitations)).
- **Alertness has no ground truth in v0 training:** NTHU-DDD is not ingested and
  UTA-RLDD is the holdout. Its training labels come only from the VLM teacher, synthetic
  prompts (none generated) and the DMD rule, which gives one class, `microsleep`;
  without VLM labels it is masked as a one-class label. The evaluation report scores only
  labels with ground-truth training data, so it does not score alertness, not even on
  UTA-RLDD.
- **Untrained in effect:** `cigarette` and `seatbelt` have no ground truth at all, and
  `sunglasses`, `face_mask`, `helmet` and `safety_vest` have it only from datasets that
  are not ingested (NTHU-DDD and YawDD for `sunglasses`, SH17 for the others). The VLM
  teacher gives them positives only (nothing at all if it was not run for the release),
  so a model cannot learn when they are absent. Ignore these outputs.
- **Evaluation only:** `kss` comes only from UTA-RLDD, which is the held-out set, so it
  gets no training labels.
- **Clip-level PERCLOS:** `perclos` is the fraction of one clip's frames (1 to 2 seconds)
  with the eyes closed. PERCLOS as used in the drowsiness literature averages eye closure
  over a minute or more; compute it from consecutive clips.
- **Only as good as the ingested data:** a release that ingests more datasets (NTHU-DDD,
  YawDD or SH17, say) changes the statuses above, which are those of v0's four datasets.
- **Phone, drink and food:** on video, positives come from the Drive&Act and DMD
  activities that involve the object, and negatives from the datasets' object
  annotations (Drive&Act's object-level annotations; DMD's `cellphone` and `bottle`
  objects with its per-frame driver actions; DMD gives no `food` label); see the
  [data card](docs/data-card.md#labels). Clips that mostly show the driver fetching or
  putting away the object have no label for it, while a DMD clip that mostly shows the
  object in use is a positive even when it includes picking it up or putting it back.
  Positives are few: 5,580 phone, 1,700 drink and 3,166 food targets on the v0 train
  split's video clips against 37,233, 41,268 and 32,615 negatives, and the loss does not
  reweight classes. On images, AUC Distracted Driver v2 gives phone and drink positives
  only (4,661 and 933 train images; it has no object annotations), so training masks both
  on images and those two outputs are untrained there; its `phone_at_ear` has both
  classes (calls 1, texting 0).
- **One-sided labels:** some teachers give one class only: the objects teacher
  negatives, the VLM teacher's object flags positives, and the face teacher
  `eyes_closed` negatives (it gives `blink` and `yawn` no value). Every training job
  checks its training clips (ground truth and teacher labels together) per label and
  media type, masks each pair with one class only, prints what it masks and records the
  outputs it trained as `supervised_labels` (and the masked pairs as `masked_labels`) in
  each checkpoint. `bulwark coverage` runs the same check on a manifest beforehand. A
  pair where another source gives the other class (DMD's ground truth, for the face
  teacher's `eyes_closed`) is not masked, but the one-sided teacher rows still pull its
  head towards their class.

The table gives each output's status in B1 v0, from its checkpoint's `supervised_labels`
and `masked_labels` and its evaluation report's training support (train clips with each
kind of label; a clip can have both). The data card gives the clips per label, dataset, split
and media type from `bulwark coverage`. E1 v0 trained and masked the same outputs as B1
v0, as its checkpoint and graphs record (`supervised_labels`, `masked_labels`).

| Output | Status in B1 v0 | Train clips with ground truth | Train clips with teacher labels |
|---|---|---:|---:|
| `eye_openness` | trained and evaluated | 70,989 | 10,078 |
| `eyes_closed` | trained and evaluated | 72,814 | 8,344 |
| `perclos` | trained and evaluated | 1,825 | 8,344 |
| `blink` | trained and evaluated | 3,766 | 0 |
| `mouth_open` | trained on teacher labels only; not evaluated | 0 | 22,218 |
| `yawn` | trained and evaluated | 1,833 | 0 |
| `head_yaw` | trained on teacher labels only; not evaluated | 0 | 22,218 |
| `head_pitch` | trained on teacher labels only; not evaluated | 0 | 22,218 |
| `head_roll` | trained on teacher labels only; not evaluated | 0 | 22,218 |
| `gaze_yaw` | not trained: no training targets | 0 | 0 |
| `gaze_pitch` | not trained: no training targets | 0 | 0 |
| `head_nod` | trained on teacher labels only, from a reversed rule (see Limitations); do not use | 0 | 19,338 |
| `attention_direction` | not trained: masked on images and video | 3,039 | 8,515 |
| `hands_on_controls` | not trained: no training targets | 0 | 0 |
| `phone` | trained and evaluated; masked on images | 47,474 | 0 |
| `phone_at_ear` | trained and evaluated | 47,474 | 0 |
| `food` | trained and evaluated | 35,781 | 0 |
| `drink` | trained and evaluated; masked on images | 43,901 | 0 |
| `cigarette` | not trained: no training targets | 0 | 0 |
| `eyeglasses` | trained and evaluated | 70,989 | 0 |
| `sunglasses` | not trained: no training targets | 0 | 0 |
| `face_mask` | not trained: no training targets | 0 | 0 |
| `seatbelt` | not trained: no training targets | 0 | 0 |
| `helmet` | not trained: no training targets | 0 | 0 |
| `safety_vest` | not trained: no training targets | 0 | 0 |
| `gloves` | not trained: no training targets | 0 | 0 |
| `alertness` | not trained: masked on video | 0 | 107 |
| `kss` | not trained: no training targets | 0 | 0 |
| `activity` | trained and evaluated | 56,678 | 0 |
| `modality` | trained; scored as a sanity check only | 131,478 | 0 |
| `low_light` | trained; scored as a sanity check only | 70,989 | 0 |
| `face_visible` | not trained: masked on images and video | 11,119 | 11,339 |
| `face_occluded` | not trained: no training targets | 0 | 0 |
| `context` | not trained: masked on images and video | 60,489 | 0 |

## Evaluation

Protocol: subject-disjoint train/val/test splits, with Drive&Act's official `split_0`
drivers, checked for leaks before scoring; the UTA-RLDD cross-dataset holdout once it is
ingested; the Drive&Act inner-mirror camera held out and scored on the 3 test drivers
only; each holdout scored on its own; ground-truth labels only, and only labels with
ground-truth training data; a constant-predictor baseline, counts of clips, subjects and
independent events (multi-camera events fused into one prediction), 95% subject-level
bootstrap intervals, per-dataset and per-condition slices, and calibration. Scores use
Bulwark's own clips and taxonomy, so they are not comparable with the datasets' published
benchmarks. The report scores only labels that have ground truth in the training split
and that the checkpoint records as trained (`supervised_labels`), so masked outputs such
as `attention_direction` in v0 are listed as not scored. Details in
[docs/evaluation.md](docs/evaluation.md).

The B1 fine-tune and E1 distillation select a `best.pt` on a small val split (2 Drive&Act
drivers, 2 DMD subjects and 3 MRL Eye people; AUC Distracted Driver v2 has no val clips).
The heads stage picks its `best.pt` by 5-fold cross-validation over the train and val
subjects instead (`select_folds: 5` in the shipped heads configs). Each records how many
clips and subjects each label's score rests on (`select.support`). Where that is too few,
the run's `final.pt` (fixed budget, no selection) is the alternative. B1 v0 is the
`best.pt` of the LoRA fine-tune on the four v0 datasets (run directory
`checkpoints/b1-lora-auc`, sha256 starting `de184992f7ed`), selected on val with
a score of 0.9328 over 13 labels at step 6,162, the last of its three epoch-end passes;
the fine-tune started from the `best.pt` of a heads run on the same data, picked by 5-fold
cross-validation. E1 v0 was distilled from B1 v0's outputs and selected the same way, with
a score of 0.7846 at step 16,432 (run directory `checkpoints/e1-v0`, sha256 starting
`065b89dc5eda`; [docs/evaluation.md](docs/evaluation.md#e1-first-distillation)). Its two
graphs were exported from that `best.pt`, and record it and how it was selected
(`bulwark_contract.checkpoint`).

**B1 v0 results.** All indicative: each dataset's test split holds 3 subjects (AUC
Distracted Driver v2's, one driver group), so no result has a confidence interval, and
none is a measure of general performance. On the Drive&Act inner-mirror camera, held out
from training, B1 v0 scores AUROC 0.845 for `phone_at_ear`, 0.808 for `phone`, 0.911 for
`food` and 0.911 for `drink`, and activity macro-F1 0.290. On test, activity macro-F1 is
0.772 on Drive&Act, 0.586 on DMD and 0.794 on AUC Distracted Driver v2; `eyes_closed`
AUROC 0.902 on DMD and 0.998 on MRL Eye; DMD `blink` 0.748 and `yawn` 0.904; `phone` 0.996
on Drive&Act and 0.981 on DMD; MRL Eye `eye_openness` MAE 0.033. Known regression: against
the first full run, trained without AUC Distracted Driver v2, DMD activity macro-F1 falls
from 0.701 to 0.586 (for each of DMD's 3 test subjects, most in `talking` and
`grooming`), and DMD `blink` AUROC from 0.795 to 0.748, while the holdout scores for
`phone_at_ear`, `food` and `phone` rise. The tables, with clip and subject counts, the
comparison and why B1 v0 is E1's teacher are in
[docs/evaluation.md](docs/evaluation.md#results). B1 v0's generated `report.html` and
`metrics.json` are in `evaluation/b1/` of its Hugging Face repository, which opens once
access is granted. Numbers from `--limit` smoke runs are pipeline checks and are never
quoted as results.

**E1 v0 results.** All indicative, on the same test clips and holdout as B1 v0, scored by
`bulwark evaluate --model e1` with B1 v0's predictions as the reference. E1 v0 scores well
below B1 v0 on most outputs (B1 v0's score in brackets). On DMD's driver video,
`eyes_closed` AUROC is 0.497 (0.902), chance level, `blink` 0.589 (0.748) and `yawn`
0.753 (0.904). Activity macro-F1 is 0.227 on Drive&Act (0.772), 0.283 on DMD (0.586) and
0.284 on AUC Distracted Driver v2 (0.794). `phone` AUROC is 0.850 on Drive&Act (0.996) and
0.905 on DMD (0.981); Drive&Act `food` 0.763 (0.981) and `drink` 0.909 (0.996). These
Drive&Act scores are per event, the mean of several cameras' probabilities; with each clip
scored on its own, from one camera, `phone` is 0.729 (0.942), `food` 0.665 (0.928) and
`drink` 0.789 (0.975). On MRL
Eye's still eye images it does as well as B1 v0: `eyes_closed` AUROC 0.998 (0.998) and
`eye_openness` MAE 0.017 (0.033). On the Drive&Act inner-mirror camera, held out from
training and scored per clip, it falls to chance or below: AUROC 0.547 for `phone` (0.808), 0.501 for
`phone_at_ear` (0.845), 0.495 for `food` (0.911) and 0.404 for `drink` (0.911), and
activity macro-F1 0.085 (0.290). The tables, with clip and subject counts, are in
[docs/evaluation.md](docs/evaluation.md#e1-v0-test-results). E1 v0's generated
`report.html` and `metrics.json` are in `evaluation/e1/` of its Hugging Face repository,
with B1 v0's in `evaluation/b1/`. The two graphs were also scored on the same splits
with `bulwark evaluate --model onnx` on CPU, with the checkpoint's predictions as the
reference. Each graph scores only clips of its own frame count: the video graph scored the
28,710 video clips (DMD and Drive&Act) and the image graph the 15,749 image clips (MRL Eye
and AUC Distracted Driver v2). On every clip, the video graph's label logits differ from
the checkpoint's by at most 2.3e-03 and the image graph's by at most 2.6e-04.
Every yes/no output scored is on the same side of 0.5 as the checkpoint's, the top
activity class is the same on every clip, and every test and holdout score above differs
from the checkpoint's by at most 6.2e-06 (`reference` in each evaluation's
`metrics.json`). The graphs' scores are therefore the checkpoint's
([docs/evaluation.md](docs/evaluation.md#e1-v0-test-results)).

**E1 latency.** Measured with the `bulwark export` benchmark on 2026-11-10, on an Apple
M1 Max (10 cores: 8 performance, 2 efficiency) with onnxruntime 1.31.1, 100 timed calls
after 10 warm-up calls, one window per call (`<graph>.bench.json`, shipped with the
graphs). The video graph (input 1 x 16 x 256 x 256 x 3, uint8) takes p50 365 ms and p95
372 ms per call on one thread, and p50 203 ms and p95 287 ms on all 10 cores. The image
graph (1 x 1 x 256 x 256 x 3) takes p50 23 ms and p95 24 ms on one thread, and p50 18 ms
and p95 25 ms on all 10 cores. These cover the model only: decoding and letterboxing are
not included. Conditions: macOS 26.5.1 on mains power, with no other Bulwark job running
and the usual desktop apps open; just before the measurement the 1-minute load average
was 4.26 and the CPU 88% idle. Times on other machines will differ.

## Limitations

- **Few people.** Most labelled video comes from a few datasets with few subjects
  (Drive&Act has 15 drivers and the ingested DMD video 14; MRL Eye has 37 people), and
  each test split holds only 3 of them. Expect weaker results on people, cameras and
  mounting positions unlike those in training.
- **Little subgroup evaluation.** The only demographic attribute in the data is a
  binary gender code in MRL Eye's file names (eye crops of 37 people), which the
  evaluation report uses as a slice; eyewear is a slice where a dataset labels it. No
  video dataset records age, sex or skin tone, so performance across them cannot be
  measured yet.
- **Narrow conditions.** Mostly in-car data, much of it near-infrared. No thermal data.
  No real forklift, crane, factory or patient footage; synthetic clips for those settings
  are optional and prompt-labelled.
- **Noisy labels.** Some labels are self-reported (UTA-RLDD alertness), some come from
  teacher models or from a rule of this project's (DMD microsleep), and some are mapped
  from each dataset's own classes onto Bulwark's shared taxonomy. The face teacher is
  built for front-facing video and degrades for faces turned far from the camera. It
  cannot tell looking down (texting, reading) from closed eyes, so its closed-eye calls
  are not used as labels: where `eyes_closed` comes from the teacher it is only ever 0.
  Its blink and yawn calls are not used at all, so `blink` and `yawn` learn only from
  ground truth and synthetic prompts; in v0 (no synthetic clips) that is
  DMD's ground truth, from one camera of one dataset. Its `eye_openness` and `perclos`
  are also dropped where it says the eyes are closed (kept only where ground truth
  confirms that), so on video they lean open: on the v0 train split, 211 of its 10,078
  video `eye_openness` targets are below 0.5 (649 of 10,879 on the same camera without
  the rule), and its `perclos` targets never exceed 0.75. No dataset gives video
  `eye_openness` ground truth, so evaluation cannot show an eye-openness output that
  reads open on closed-eye video; read video `eye_openness` with this in mind.
- **Short clips.** A clip covers about 1.5 to 1.9 seconds at 25 or 30 fps, and about
  1.07 seconds for 15 fps sources such as Drive&Act's Kinect colour view. Drowsiness
  develops over minutes, and measures such as PERCLOS are defined over a minute or more;
  any product decision should aggregate outputs over consecutive clips.
- **`head_nod` is wrong in v0.** The face teacher's code described head pitch as
  positive when looking up, and its nod rule looked for the pitch falling and coming
  back. MediaPipe's pitch is positive when the head tips down (on a demo clip it rose
  to about +38 degrees as the driver's head dropped), so the rule fired on the head
  rising and coming back, and missed most real nods. The v0 models learned `head_nod`
  from those labels (634 positives in 19,338 video clips), so ignore it. `head_pitch`
  itself is unaffected: it is the teacher's angle, positive downwards relative to the
  camera. The sign error is fixed in the code from 0.1.0, which changes `head_nod` labels
  for the next training run. The rule still cannot tell nodding off from looking down
  (at a phone or the radio, say), and no ground truth has checked it: its only
  ground-truth source, NTHU-DDD, is not ingested. The teacher's `attention_direction`
  rule had up and down swapped the same way; v0 masks that label, so it did not reach
  the models.
- **Untrained outputs** return arbitrary values (see
  [Labels and supervision](#labels-and-supervision)).
- **Bulwark-E1 v0 is weaker than B1.** It scores well below B1 v0 on most outputs. It
  does not detect closed eyes in driver video, its blink, yawn and activity results are
  weak, and on a camera view that training never used its phone, eating and drinking
  detection falls to chance or below. It scores much better on the people it was trained
  on than on new people. Use B1 for research.
- **Not calibrated for real settings.** Probabilities reflect the training mix of
  datasets, not the base rates of any real setting; research on other data needs
  thresholds set on that data.

## Responsible use

- In research, check results on your own population, cameras and conditions, and report
  them per subgroup where the law allows collecting the attributes. A model for a
  product, which these weights are not, needs the same before any deployment.
- Keep a human in the loop for any consequence for a person, and give people a way to
  contest it.
- Prefer on-device processing. Do not store face video by default; if you must,
  minimise and protect it.
- Tell the people being monitored, and get their consent where the law requires it.
- Do not use Bulwark for the uses listed in [Out-of-scope uses](#out-of-scope-uses).

## Versioning and provenance

| Identifier | Meaning |
|---|---|
| package version (0.1.0) | the code (`inferensys-bulwark`) |
| model release (v0) | the models trained on the v0 data: Bulwark-B1 v0 and Bulwark-E1 v0, both released (E1 v0 as a research preview; E1 will be retrained); the training configs write their checkpoints to `checkpoints/*-v0` (B1 v0's run wrote to `checkpoints/b1-lora-auc`) |
| taxonomy version (1) | the label set and output layout |
| split salt (`bulwark-v1`) | the frozen subject-to-split assignment; never changed. A person keeps their split as long as their dataset's own list of subjects does not change. |

Each checkpoint records its provenance: the training config, `commercial_only` and
`release_only`, the datasets used with their licences, `weights_release_ok` and clips per
split, a hash of the manifest, the split salt and taxonomy version, its inputs (init
checkpoint, teacher or features, encoder weights) by hash with the datasets of every
model up the chain behind them (`upstream_datasets`) and the inputs up that chain whose
training data is not recorded (`unrecorded_inputs`), the outputs it trained and
masked, the package version (`bulwark`), the code commit of every training session
(`sessions`) and the Python, PyTorch, CUDA, NumPy, timm and onnx versions and GPU. Kaggle
jobs also write `environment.txt`, and Azure jobs `jobs/<name>.environment.txt`, with the
installed package versions and the GPU and driver. B1 v0 was trained with `bulwark` 0.1.0
at commit `f157c05` (no local changes), Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8,
NumPy 2.1.3, timm 1.0.30 and onnx 1.23.2, on a Tesla T4 (its `provenance.json`, which its
repository ships). E1 v0 was trained with `bulwark` 0.1.0 at commit `5338f65` in both of
its sessions (no local changes), with the same Python, PyTorch, CUDA, NumPy, timm and onnx
versions, on a Tesla T4 (its `provenance.json`, which E1's repository ships).

The Bulwark commits that the v0 records name belong to Bulwark's development before its
first public release: for B1 v0, `f157c05` for training, `efe4bab` for evaluation and
those in [Measured for v0](docs/training.md#measured-for-v0); for E1 v0, `5338f65` for
training (its `provenance.json` and its graphs' `trained_with_commit`), `fafa16d` for
its evaluation and its ONNX export (its evaluation report and the graphs' `exporter`
entry) and `669d7a8` for its graphs' latency (`git` in each `<graph>.bench.json`). That
history is not published: the public repository starts at the tag `v0.1.0`.
The package's version was already 0.1.0 during that development, so `bulwark` 0.1.0 in
these records names that earlier code, not the tag.

Run with the released 0.1.0 code (tag `v0.1.0`) and with `efe4bab`, in fp32 on a CPU, B1
v0 gave the same outputs bit for bit: through `bulwark predict` on 5 video
clips (35 windows), and through the evaluation's prediction code on 24 val and test clips
(6 each from Drive&Act, DMD, MRL Eye and AUC Distracted Driver v2). Since `f157c05`, B1's
training configs hold the same values; only their comments changed. Meta's V-JEPA 2.1
encoder code is now vendored in `bulwark/model/vjepa21` from the same pinned commit
instead of read from a clone. One forward and backward pass in training mode, with B1
v0's LoRA and head weights and a test loss over all its outputs, gives the same loss and
gradients, bit for bit, with either copy (fp32, CPU). Jobs now list the shards 16 at a
time, which changes only setup time. The registry now clears the four v0 datasets for
release, which changes a run's data only with `release_only: true` (B1 v0 and E1 v0
trained with it false).

E1 v0's graphs were exported, and its checkpoint evaluated, at `fafa16d`. Between that
commit and the tag, the package changed only in `bulwark/train/stream.py`, where a forked
DataLoader worker now opens its own shard files. The graphs' own evaluations ran at
`539c637` (video graph) and `669d7a8` (image graph), and their latency was measured at
`669d7a8`; the package at both commits is the tag's. The graphs run in ONNX Runtime without Bulwark's code.

## Contact and removal requests

Commercially licensed models: https://inferensys.com/contact or hello@inferensys.com.
Questions and problems: GitHub issues on https://github.com/Inferensys/bulwark, or the
Community tab of either model's repository on Hugging Face. Security issues: see
[SECURITY.md](SECURITY.md).

Dataset owners (also to object to the release of weights trained on their dataset; see
Release status under [Licence](#licence)), and people who appear in a training dataset and
want their data removed, can write to Inferensys privately at hello@inferensys.com;
please do not post personal data in public issues. DMD
participants can also contact Vicomtech through the DMD website
(https://dmd.vicomtech.org/). Removed subjects are listed in
`bulwark/data/removed_subjects.yaml` by their dataset subject id (no names), with the
date and source of the request. `bulwark ingest` and `bulwark split` leave them out,
`bulwark purge` deletes their clips from local manifests, shards, teacher labels and
feature caches, and training and evaluation refuse a manifest that still holds them.
`scripts/purge_cloud_copies.sh <dataset>` then carries the purge over to the cloud
copies: it replaces the Azure Blob copies with the purged local data (deleting the shard
blobs of a dataset not cleared for Azure Blob, and the teacher-label blobs), and
the Kaggle datasets that held the dataset's shards or the manifest are deleted (with
their old versions) and created again. Each purge is logged in `docs/purge-log.md`,
without subject ids. A removal applies from the next retrain onward; weights that were
already released are not retrained. The
[data card](docs/data-card.md#personal-data-and-removal-requests) gives the steps.

## Datasets, models and acknowledgements

The training datasets are listed in [Training data](#training-data), with the citation
each one asks for in [docs/datasets.md](docs/datasets.md#citations). Upstream models and
their licences (details in [NOTICE](NOTICE)):

| Model | Role | Licence |
|---|---|---|
| V-JEPA 2 code and V-JEPA 2.1 ViT-B/16 weights (Meta) | B1 encoder | MIT, Copyright (c) Meta Platforms, Inc. and affiliates |
| EfficientViT-B1 (MIT Han Lab), weights through timm | E1 backbone initialisation | Apache-2.0 |
| MediaPipe Face Landmarker (Google) | face teacher | Apache-2.0 |
| RF-DETR (Roboflow) | objects teacher (optional) | Apache-2.0 |
| Qwen3-VL-4B-Instruct (Qwen team) | VLM teacher (optional) | Apache-2.0 |
| Wan2.2-TI2V-5B (Wan-AI) | synthetic clips (optional) | Apache-2.0 |

We thank the authors of these models and datasets.

## Citation

Cite Bulwark with [CITATION.cff](CITATION.cff), and cite the upstream models
([docs/references.md](docs/references.md)) and every training dataset
([docs/datasets.md](docs/datasets.md#citations)).
