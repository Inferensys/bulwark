# Architecture

Bulwark has two models that share one label taxonomy and one head design:

- **Bulwark-B1** (base): a pretrained video encoder, a fixed set of summary tokens and
  multi-task heads. Only LoRA adapters in the top encoder blocks and the heads are trained.
- **Bulwark-E1** (edge): a per-frame image backbone and the same kind of heads, distilled
  from B1 and exported to a single ONNX graph.

The code is in `bulwark/model/`. Shapes below are for the default configuration; the
parameter counts are what this code instantiates.

```
                     Bulwark-B1
frames ─────▶ V-JEPA 2.1 ViT-B/16 ─────▶ token grid ─────▶ 25 summary ─────▶ heads ──▶ 34 outputs
(B,16,256,256,3)  frozen, LoRA on       (B,8,16,16,768)    tokens            (512-d     (65 logits)
uint8 RGB         the top 4 blocks                        (B,25,768)         CLS)

                     Bulwark-E1
frames ─────▶ EfficientViT-B1 per frame ─────▶ 16 frame tokens ─────▶ heads ──▶ 34 outputs
(B,16,256,256,3)  at 224 px (timm,             (B,16,1600)           (256-d     (65 logits)
uint8 RGB         ImageNet-1k init)                                  CLS)
```

## Inputs

A training sample is a clip: 16 RGB frames for video sources, or a single frame (T=1)
for image sources. Frames are letterboxed to a square (aspect ratio kept, black padding,
`bulwark.data.clips.letterbox`) and stored at 256 px as JPEG in tar shards. Frames enter
both models as uint8 RGB in (B, T, H, W, 3) layout; scaling to [0, 1] and ImageNet
mean/std normalisation happen inside the model.

Every dataset video adapter samples clips the same way
(`bulwark.data.sources.base.clip_stride`): 16 frames at a stride of
`max(1, round(2 s x fps) // 16)` source frames. That is every 3rd frame at 25 or 30 fps,
so a clip covers about 1.5 to 1.9 seconds, and every frame at 15 fps, as in Drive&Act's
Kinect colour view, so a clip covers about 1.07 seconds. Generated synthetic clips apply
the same rule to a window as long as the whole generated video.
Frames are stored as JPEG at quality 90, so the models are trained on JPEG-compressed
frames.

## Bulwark-B1

**Encoder.** Meta's V-JEPA 2.1 ViT-B/16 (`vjepa2_1_vitb_dist_vitG_384.pt`), built from
Meta's V-JEPA 2 code at a pinned commit, vendored in `bulwark/model/vjepa21`
(`bulwark/model/backbone.py`): 12 blocks,
width 768, 16 px patches, tubelets of 2 frames, RoPE, run at 256 px. About 86.8M
parameters. A 16-frame clip gives a token grid of (8, 16, 16, 768); a single image goes
through the encoder's image path and gives (1, 16, 16, 768).

**LoRA.** The encoder is frozen. Rank-16 LoRA (alpha 32) is added to the `qkv`, `proj`,
`fc1` and `fc2` linear layers of the last 4 blocks (16 layers), and the encoder's final
output norm is trained too: about 0.79M trainable encoder parameters
(`bulwark/model/lora.py`). With the heads, the fine-tune trains about 7.56M parameters
(it prints `LoRA layers: 16, trainable params: 7.56M`). With `lora_blocks=0` the encoder
is fully frozen, which is how cached features are computed.

**Summary tokens.** The token grid is reduced to 25 tokens (`bulwark/model/pooling.py`):

| Tokens | Count | What they keep |
|---|---:|---|
| global | 1 | mean over time and space |
| temporal | 8 | mean per time step (blinks, nods and yawns need timing) |
| spatial | 16 | 4 x 4 grid of spatial means over time (hands vs. face vs. lap) |

Images have one time step, which is repeated to fill the 8 temporal tokens. These 25
tokens are what `bulwark train cache` stores, so heads trained on cached features and
heads trained end to end see the same input.

**Heads.** `BulwarkHeads` (`bulwark/model/heads.py`), about 6.8M parameters:

1. per-position standardisation of the input tokens, with statistics fitted on training
   tokens (the encoder's summary tokens share a large constant component);
2. LayerNorm and a linear projection to width 512;
3. a learned CLS token and learned position embeddings;
4. a 2-layer pre-norm transformer encoder (8 attention heads, feed-forward 4x, dropout 0.1);
5. a final LayerNorm; the CLS output is the clip's **state embedding** (512-d);
6. one linear layer that produces all 65 logits, split into the 34 taxonomy outputs.

## Bulwark-E1

`BulwarkEdge` (`bulwark/model/bulwark.py`) resizes each frame to 224 px, runs timm's
`efficientvit_b1.r224_in1k` (EfficientViT-B1 from MIT Han Lab, ImageNet-1k weights) on it
and takes its pooled 1600-d feature as one token per frame. The heads are the same design as
B1's with width 256 and 4 attention heads, and their transformer does the temporal
modelling over the 16 frame tokens. About 9.5M parameters in total (7.5M backbone, 2.0M
heads). The state embedding is 256-d.

## Outputs

Every label in `bulwark/data/taxonomy.yaml` gets one output. The heads return raw logits;
`bulwark.model.heads.activate` turns them into values:

| Kind | Logits | Decoding |
|---|---|---|
| binary | (B,) | `sigmoid(x)`, a probability |
| categorical | (B, C), classes in taxonomy order | `softmax(x)` over C |
| regression | (B,) | `low + (high - low) * sigmoid(x)`, with `[low, high]` the label's `range` |

The taxonomy has a version number (`version: 1`). Adding, removing or reordering labels
or classes changes the output layout and needs a new taxonomy version.

Not every output is trained in every release: an output whose label has no training
data, or that a training run masked, returns arbitrary values. Every checkpoint lists the
outputs it was trained on (`supervised_labels`), and the model card lists them for a
release.

## Training objective

`bulwark/model/losses.py`:

- **Sparse targets.** Each clip carries only the labels its dataset (or a teacher model)
  provides. For each label, the loss averages over the clips that have it and ignores the
  rest: cross-entropy for categorical labels, binary cross-entropy on logits for binary
  labels, and smooth L1 (beta 0.1) on the decoded value for regression labels.
- **Masked labels.** A head that only ever sees one class learns a constant. Before
  training, each run masks every label and media type pair whose training targets have
  one class, plus the labels in `mask_labels` (`bulwark/train/labels.py`); masked targets
  leave the loss. See [training.md](training.md#labels-a-run-trains).
- **Ground truth and teacher labels.** Ground-truth labels have weight 1.0 and teacher
  labels weight 0.5 inside that average.
- **Uncertainty weighting.** Per-label losses are combined with learned homoscedastic
  uncertainty weights (Kendall et al., 2018): `0.5 * exp(-s) * L + 0.5 * w * s` per label,
  with `s = log(sigma^2)` learned and clamped to [-4, 4], and `w` the mean weight of the
  label's targets in the batch (1.0 when all are ground truth, 0.5 when all are teacher
  labels). Without `w`, the learned `s` would cancel the teacher weight; with it, a batch
  of teacher labels counts half as much as the same batch labelled as ground truth.
- **Dataset balance.** Each dataset's share of an epoch grows with the square root of its
  size rather than linearly (`dataset_power: 0.5`), both when training heads on cached
  features and when streaming shards for the fine-tune and distillation, where datasets
  are interleaved within each epoch.

## Teacher labels

Teachers add labels that the source datasets do not annotate (`bulwark/label/`):

| Teacher | Model | Labels |
|---|---|---|
| `face` | MediaPipe Face Landmarker (blendshapes and head pose) | eye openness and closure, PERCLOS, mouth opening, head yaw/pitch/roll, head nod, face visible, attention direction (it also writes blink and yawn, which the merge drops) |
| `objects` | RF-DETR (COCO classes) | negatives only for phone, drink and food |
| `vlm` | Qwen3-VL-4B-Instruct through MLX, on clips without a ground-truth activity | context, activity, attention direction, alertness, and positives for a set of object flags |

The face teacher's v0 head nod and attention direction labels came from rules that had
the pitch sign reversed (fixed in 0.1.0), so v0's `head_nod` output is wrong, and v0
masks `attention_direction` (model card, [Limitations](../MODEL_CARD.md#limitations)).
Head pitch is positive when the head tips down relative to the camera.

Some adapters also write derived labels (`meta["derived_labels"]`): values computed by a
rule of this project's from the dataset's own annotations, which therefore count as
teacher labels, not ground truth. DMD is the only one so far: a clip in which the eyes
stay closed without a break for 1 s or more gets alertness `microsleep`.

The object labels are about the person using the object (a cup in a holder is not
`drink`), while a detector only sees that an object is in view. So the objects teacher
never writes a positive: it writes 0 for phone, drink or food only when nothing of that
kind scores 0.2 or more in any of 4 sampled frames. Ground truth wins where a dataset
gives these labels; Drive&Act and DMD give their own 0s from their object annotations
([data card](data-card.md#labels)). The VLM teacher keeps objects as positives only,
since leaving an object out of a short list is no evidence that it is absent.

Teacher outputs are stored per clip in `data/labels/<teacher>.jsonl` and merged into the
manifest's `labels_teacher` field. Merging rules (`bulwark/label/run.py`):

- ground truth always wins, and teacher output never becomes ground truth;
- when the other sources disagree on a label, derived labels beat face, which beats
  objects, which beats vlm, which beats the prompted labels of synthetic clips;
- face, objects and VLM teacher output is dropped for eye crops, which hold no face,
  hands or scene (`bulwark label face`, `objects` and `vlm` skip them); the face
  teacher's attention direction (measured in the camera frame) and a missed face
  (`face_visible` 0) are kept only for cameras that roughly face the person
  (`FRONTAL_FACE_VIEWS`: DMD's face camera, NTHU-DDD, UTA-RLDD and YawDD's dash camera);
- of the face teacher's `eyes_closed`, `blink` and `yawn`, only `eyes_closed` 0 is kept
  (`FACE_KEEP`). A value is kept when ground truth confirmed it at least 90% of the time
  on the train split and the class that keeping only that value leaves out is rare in
  the clips it labels, so the one-sided labels it then gives there are close to the
  truth. `blink` and `yawn` keep no value: their "no" calls fall short of 90%, and their
  "yes" calls alone would leave out "no", which is common in those clips, and pull those
  outputs towards 1 (numbers in the [data card](data-card.md#labels)). The face teacher
  therefore labels neither; a synthetic clip's prompted labels still can. Its
  `eye_openness` and `perclos`, from the same eye-closure measure, follow its
  `eyes_closed`: on a clip with ground-truth `eyes_closed` they stay only when the
  teacher agrees with it, elsewhere they go where its `eyes_closed` is dropped. All of
  these were measured on DMD's face camera, so they are kept only for cameras in
  `FRONTAL_FACE_VIEWS`. On clips without ground truth the teacher gives `eyes_closed`
  negatives only;
- event labels (blink, head nod, yawn) are dropped for still images;
- teacher keys that start with `_` (such as the VLM caption) go to the clip's metadata.

`bulwark label agreement` (`bulwark.label.agreement`) measures how well each teacher
agrees with ground truth on clips that have both: precision, npv (the share of the
teacher's negatives that ground truth confirms), recall and F1 for binary labels,
accuracy for categorical labels and MAE for regression labels, per label and dataset,
over every split or those given with `--split`. The command leaves out datasets the
split policy excludes (`zenodo_driver`), so their ground truth is never scored or
published; the library function `teacher_agreement` scores whatever records it is given.

Evaluation scores ground-truth labels only (see [evaluation.md](evaluation.md)).

## Distillation (B1 to E1)

B1's logits are computed once over the shards (`bulwark train cache` with `init` set to
the fine-tuned B1 checkpoint) and stored next to the summary tokens, so the teacher never
has to share the GPU with the student. E1 is then trained on the same shards with:

```
loss = multi-task loss on the clip's labels + distill_weight * soft-target loss
```

The soft-target loss matches B1's outputs: KL divergence at temperature 2 for
categorical labels, binary cross-entropy against `sigmoid(teacher)` for binary labels and
smooth L1 between the sigmoids for regression labels, averaged over labels. It covers
only the outputs B1 was trained on (the `supervised_labels` its features directory
records), without the pairs the distillation run masks itself.

B1's logits were cached on full, unaugmented clips, so the student's augmentation is
photometric only (gamma, contrast, brightness, grayscale), without the crop the fine-tune
uses. No stage mirrors frames.

## Checkpoints

- B1 checkpoints from `bulwark train` store only what was trained (LoRA weights, the
  encoder's final output norm and the heads). Loading one needs Meta's V-JEPA 2.1
  checkpoint, fetched by `scripts/fetch_pretrained.sh`.
- E1 and heads checkpoints store the full model.
- A release written by `bulwark release` (or `bulwark.inference.save_release`,
  `examples/make_release.py`) stores the full model in `model.safetensors`, with LoRA
  merged into the encoder, and a `config.json`; it loads without the pretrained V-JEPA
  file. A B1 release therefore contains weights derived from Meta's V-JEPA 2.1 checkpoint
  and ships Meta's licence notice as `LICENSE-vjepa2` (see
  [inference-and-adapters.md](inference-and-adapters.md)).
- Every checkpoint also records its provenance: the training config, `commercial_only`
  and `release_only`, the datasets used, a hash of the manifest, its inputs with the
  datasets of every model up the chain behind them (`upstream_datasets`) and the inputs
  up that chain whose training data is not recorded (`unrecorded_inputs`), the outputs it
  was trained on (`supervised_labels`), the code version of every training session and
  the library versions. Every cached-features directory records in `meta.json` the
  encoder weights, the init checkpoint if any (its hash, flags, datasets,
  `upstream_datasets` and `unrecorded_inputs`, manifest hash and commit, and its `supervised_labels` and
  `masked_labels`), the precision, `commercial_only` and `release_only`, the package
  version (`bulwark_version`), `taxonomy_version`, `split_salt`, the code and library
  versions and a fingerprint of every shard. A run refuses to resume a `last.pt` from a
  different run, and a `commercial_only` or `release_only` run refuses any upstream
  checkpoint or features that do not record the same flag (see
  [training.md](training.md#how-runs-behave)).

Checkpoints are PyTorch pickles. Bulwark reads them with `torch.load(weights_only=True)`,
which restricts unpickling to tensors and plain Python types (PyTorch 2.6 or later, as
the package requires); load only checkpoints from sources you trust (see
[SECURITY.md](../SECURITY.md)).

## E1 ONNX contract

`bulwark export --checkpoint <E1 checkpoint> --out bulwark-e1.onnx` writes one graph
(opset 17), checks its outputs against the PyTorch model, and stores this contract in the
file's metadata (read it with `bulwark.export.onnx.read_contract`).

**Input** `frames`

| Property | Value |
|---|---|
| dtype | uint8, values 0 to 255 (no scaling or normalisation by the caller) |
| shape | (batch, T, S, S, 3); batch is the only dynamic axis. T and S are fixed at export: T is 16 by default (`--frames`), and S is 256 by default (`--input-size`), the size of the training shards. Read both from the graph's input shape. |
| time axis | fixed: the exporter records the attention's sequence length as a constant, so a graph exported at one T fails at another. Image datasets were trained as single frames; for still images, export a graph with `--frames 1`. |
| layout | frames in time order, channels last |
| colour | RGB (OpenCV decodes BGR: convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`) |
| resize | letterbox each frame to S x S: keep the aspect ratio, shrink with `cv2.INTER_AREA` (enlarge with `INTER_LINEAR`) and centre on black padding, as `bulwark.data.clips.letterbox` does. Do not stretch or centre-crop. |
| timing | 16 frames sampled as in training: every k-th source frame with `k = max(1, round(2 x fps) // 16)`, so every 3rd frame of a 30 fps stream (about 1.5 s) and every frame of a 15 fps stream (about 1.07 s). 16 consecutive frames of a 30 fps camera span about 0.5 s, a third of a training clip at that rate. |
| training domain | frames letterboxed as above and stored as JPEG at quality 90, so the model saw JPEG-compressed frames |

Resizing to the student's 224 x 224 and ImageNet normalisation happen inside the graph.
Every call encodes all T frames; a sliding window costs one full call per hop.
`bulwark export` also writes the graph's CPU latency to `<graph>.bench.json` (see
[training.md](training.md#speed-and-latency)).

**Outputs.** One float32 tensor per taxonomy label, named after the label and in
taxonomy order: (batch,) for binary and regression labels, (batch, C) for categorical
labels. All are raw logits; decode them as in [Outputs](#outputs). Unless exported with
`--no-embeddings`, the graph has two more outputs after them: `tokens` (batch, T, 1600),
the per-frame features the heads read, and `state` (batch, 256), the state embedding.

**Metadata.** `bulwark_contract` holds the whole record as JSON, which `read_contract`
returns:

| Entry | Contents |
|---|---|
| `input`, `outputs`, `activation`, `model_img_size` | the input (with `preprocessing` and `time_axis`) and every output with its kind, shape, classes or range, and how to decode it |
| `window`, `temporal`, `training_domain`, `taxonomy_version` | the timing and training-domain notes above, and the taxonomy the outputs follow |
| `parity` | the largest difference from PyTorch on the parity batch (real clips with `--shards`), with the tolerance |
| `checkpoint` | the checkpoint exported, its sha256 and its recorded provenance |
| `exporter` | PyTorch version, opset, code commit and package version |
| `licence`, `model_card`, `model_release` | the weights licence (the research licence unless `--weights-licence` names another; one that does not restrict use to non-commercial research needs a checkpoint that `bulwark release-check --commercial` clears), a link to the model card, which is E1's Hugging Face model repository (`bulwark.export.release.model_card_url("e1")`), and the model release (v0) |
| `training` | `commercial_only`, the code commit and whether it had local changes, the distillation teacher's init checkpoint hash, and every training dataset with its licence, `commercial_ok`, `weights_release_ok`, `weights_release_basis` (on what the registry's clearance rests: the terms, the owners' written permission or the maintainers' own decision), `in_registry` and `via` (`training` for the run's own clips; `teacher`, `init` or `features` for the data of an upstream model) |
| `release_check` | the result of the release check: whether the weights are cleared for publication, the problems found, the datasets and the upstream inputs whose data is not recorded (`unrecorded_inputs`) |
| `label_status` | with `--manifest <training manifest>`: the outputs the checkpoint trained with ground truth (`ground_truth`) or only teacher labels (`teacher_only`) in that manifest's training split, those with no targets (`untrained`), those with targets that the checkpoint did not train (`not_trained_by_checkpoint`), and whether the manifest is the one the checkpoint was trained with |
| `supervised_labels` | the outputs the checkpoint trained, as it records them; only for a checkpoint that records none, with `--manifest`, the manifest's outputs with training targets, masked or not (`supervised_labels_from` says which) |
| `masked_labels` | the checkpoint's record of the labels training masked on a media type ({label: media types}), whose outputs are withheld on that media |
| `temperature` | per-label temperatures, one per label: from `--calibration <calibration.json>` (refused when fitted on other weights) those fitted for the graph's media type (video for T > 1, image for `--frames 1`; the fit on all val clips for a file without per-media-type entries), else those the checkpoint records |
| `thresholds` | with `--calibration`: the val-tuned decision thresholds of binary labels from that file for the graph's media type, one per label, on the uncalibrated probability as in `calibration.json` |
| `calibration` | with `--calibration`: the media type the `temperature` and `thresholds` were taken for (`media`: `video` or `image`), the labels whose values were fitted on all val clips because the file has no per-media-type entries (`pooled`), and the labels whose val clips held none of that media type, so the graph carries no threshold or temperature for them (`no_val_clips`) |

The other metadata entries repeat the main fields for tools that read single keys:
`frames`, `input_size`, `img_size`, `license`, `model_card`, `model_release`,
`bulwark_version`, `commercial_only`, `training_datasets`, `trained_with_commit`,
`teacher_init_sha256`, `release_cleared`, and, when known, `ground_truth`,
`teacher_only`, `untrained`, `not_trained_by_checkpoint`, `supervised_labels`,
`masked_labels`, `temperature` and `thresholds`. Text values are stored as they are and
others as JSON; entries without a value are left out. `calibration` is in
`bulwark_contract` only.

Request the outputs by name, so the same code also reads a graph with embeddings. For
example, with `onnxruntime`:

```python
import cv2
import numpy as np
import onnxruntime as ort

from bulwark.data.clips import letterbox
from bulwark.data.taxonomy import default_taxonomy

taxonomy = default_taxonomy()
session = ort.InferenceSession("bulwark-e1.onnx", providers=["CPUExecutionProvider"])
size = session.get_inputs()[0].shape[2]  # S, fixed at export
names = list(taxonomy.labels)  # the logit outputs, in taxonomy order


def prepare(bgr_frames: list[np.ndarray]) -> np.ndarray:
    """T OpenCV (BGR) frames -> (1, T, S, S, 3) uint8 RGB."""
    rgb = [letterbox(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), size) for f in bgr_frames]
    return np.stack(rgb)[None]


def decode(name: str, logits: np.ndarray) -> np.ndarray:
    spec = taxonomy[name]
    if spec.kind == "categorical":
        e = np.exp(logits - logits.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    p = 1.0 / (1.0 + np.exp(-logits))
    if spec.kind == "binary":
        return p
    low, high = spec.value_range
    return low + (high - low) * p


def predict(bgr_frames: list[np.ndarray]) -> dict[str, np.ndarray]:
    raw = session.run(names, {"frames": prepare(bgr_frames)})
    return {name: decode(name, value) for name, value in zip(names, raw)}
```

The Python API in `bulwark.inference` wraps the same steps for B1, E1 and ONNX models,
including cutting a video into windows and applying temperatures; see
[inference-and-adapters.md](inference-and-adapters.md).
