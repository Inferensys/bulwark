# Inference and adapters

Two modules make up the public API for using Bulwark rather than training it:

- `bulwark.inference` loads a trained model (B1, E1 or an exported ONNX graph), runs it on
  clips or videos, and returns the taxonomy outputs and embeddings.
- `bulwark.adapters` trains a small classifier for a label set of your own on Bulwark
  embeddings, saves it and loads it back.

Runnable scripts are in [examples/](../examples/) (see
[examples/README.md](../examples/README.md)). Two commands wrap the same functions:
`bulwark predict MODEL VIDEO` writes the timeline of `predict_video` (or one entry for an
image), and `bulwark release CHECKPOINT OUT` writes a release with `save_release`. The
tests in `tests/test_api_*.py` cover the same paths with a stand-in encoder, so they need
no weights.

> **Weights licence.** The B1 v0 and E1 v0 weights are released. Both are trained on
> datasets whose terms do not allow commercial use (Drive&Act, DMD, MRL Eye and AUC
> Distracted Driver v2), so their weights are for non-commercial research use only. That applies
> to every use of those weights, including computing embeddings for an adapter. Commercial use needs weights from the
> commercially clean retrain (`commercial_only=true`).

## Loading a model

```python
from bulwark.inference import load

model = load("checkpoints/b1-lora-v0/best.pt")   # device="auto" picks CUDA, then MPS, then CPU
```

`load` detects the format from the path and the checkpoint's keys:

| Path | `model.kind` | Notes |
|---|---|---|
| B1 checkpoint from `bulwark train lora` (`best.pt`, `last.pt` or `final.pt`) | `b1` | LoRA layers are rebuilt from the checkpoint's keys, then merged into the encoder. The frozen V-JEPA 2.1 weights come from `checkpoints/pretrained/` (`scripts/fetch_pretrained.sh`) or `backbone_checkpoint=`. |
| Heads checkpoint from `bulwark train heads` (`best.pt`, `last.pt` or `final.pt`) | `b1-frozen` | The frozen V-JEPA encoder plus the trained heads. |
| E1 checkpoint from `bulwark train distill` (`best.pt`, `last.pt` or `final.pt`) | `e1` | Backbone name and image size are read from the checkpoint. |
| `*.onnx` from `bulwark export` | `e1-onnx` | Runs with onnxruntime (`pip install -e ".[export]"`); `providers=` picks the execution provider. A graph whose `bulwark_contract` records another taxonomy version, or an output whose kind, classes or range differ from the current taxonomy, is refused. |
| Release directory from `save_release` or `bulwark release` | as saved | Full weights in safetensors; no pretrained V-JEPA file needed. |
| Hugging Face repository id, such as `Inferensys/bulwark-b1-driver-monitoring` | as released | Used when no local path has that name (`bulwark.inference.resolve_model`). huggingface_hub (an optional dependency) downloads B1's config.json and model.safetensors, which load as a release directory, or E1's 16-frame graph (`Inferensys/bulwark-e1-driver-monitoring-onnx`); `<id>/bulwark-e1-image.onnx` gives E1's graph for still images. A gated repository needs an approved access request and `hf auth login`. `bulwark predict` takes the id too. |

The v0 configs write these runs to `checkpoints/heads-v0/`, `checkpoints/b1-lora-v0/` and
`checkpoints/e1-v0/`; `best.pt` is the checkpoint chosen on the val split (for the heads
run, whose config sets `select_folds: 5`, by 5-fold cross-validation over the train and
val subjects).
Checkpoints are read with `torch.load(..., weights_only=True)`, which restricts
unpickling to tensors and plain Python types (the package needs PyTorch 2.6 or later,
which fixed a known bypass of that restriction). A `.pt` file is still a pickle: load
checkpoints only from sources you trust, and verify the checksums published with a
release (see [SECURITY.md](../SECURITY.md)). Entries that describe the model (step, and the
provenance the training jobs record) are kept in `model.metadata`; optimiser and loss
state are dropped.

Other options: `precision="fp16"` or `"bf16"` runs under autocast (useful on a T4),
`labels=[...]` selects the outputs in place of the labels the model records (see Outputs;
a label outside those comes from a head that may have had no training targets),
`temperature=` calibrates them, and `TorchModel.batch_size` (default 8) bounds memory for
large batches.

## Inputs

| | |
|---|---|
| Pixels | `uint8` RGB. Frames read with OpenCV are BGR and must be converted. |
| Shape | `(H, W, 3)` image, `(T, H, W, 3)` clip or `(B, T, H, W, 3)` batch. |
| Size | Frames are letterboxed (aspect kept, black padding) to 256 x 256 px, as the training shards were, for every model (`model.input_size`). E1 resizes them to its own image size (224 px for EfficientViT-B1) inside the model, as it did in distillation; its ONNX graph does the same, so PyTorch and ONNX E1 see the same pixels. |
| Clip length | 16 frames for video, 1 frame for an image. The dataset adapters cut video clips as 16 frames at a fixed stride over about 2 s, so that is what the model has seen. Other lengths run in PyTorch but are outside training. An exported ONNX graph has a fixed length (16 unless exported otherwise) and refuses anything else. |

Helpers produce inputs in this form:

- `read_image(path)` returns a 1-frame clip.
- `read_clip(path, start=12.0)` returns one 16-frame clip from a video, starting at 12 s.
- `video_windows(path, window_seconds=2.0, hop_seconds=1.0)` yields sliding clips with
  their start and end times, decoding the video as a stream.
- `as_clip_batch(frames, size)` is the normalisation every model applies internally.

## Outputs

```python
outputs = model.predict(frames)       # {label: np.ndarray}
rows = model.to_dicts(outputs)        # one JSON-ready dict per clip
```

Each label in `bulwark/data/taxonomy.yaml` gives one output, activated by its kind:

| Kind | Array | Value |
|---|---|---|
| binary | `(B,)` | sigmoid probability |
| categorical | `(B, C)` | softmax over the label's classes, in taxonomy order |
| regression | `(B,)` | a value inside the label's range (for example head yaw in [-1, 1]) |

Two caveats apply to every output.

1. The values are not calibrated probabilities unless temperatures are applied. With
   per-label temperatures, binary and categorical logits are divided by them before
   activation. Pass them to `load(..., temperature=...)` as a mapping label -> t, or as
   the `calibration.json` that `bulwark evaluate` fits on the validation split (see
   [Calibration files](#calibration-files)), whose temperatures are fitted per media
   type: `predict` applies those of the input's media type
   (`model.temperature_for("video")`, `model.temperature_for("image")`). Otherwise
   `load` uses temperatures the model records:
   `temperature` in a checkpoint or release, or in an ONNX graph's metadata (top level,
   inside its `bulwark_contract` JSON, or in that contract's record of the source
   checkpoint). The training jobs record none; `bulwark export --calibration` and
   `bulwark release --calibration` store the temperatures and val-tuned thresholds of a
   calibration.json in the graph or the release, and `load` reads both back. Without any
   of these the outputs are uncalibrated (`model.temperature_for(...)` is empty).
2. Every head produces a value, including heads whose label had no targets in the training
   data. Those values carry no information. When the model records the labels it was
   supervised on (`supervised_labels`: in a checkpoint or release; for an ONNX graph, in
   its contract's record of the source checkpoint, or else in the graph's own metadata),
   `predict` returns only those. Otherwise it returns every label, `model.labels_known` is
   False, and you should pass `labels=` with the ones the training data covered. The
   training manifest shows which those are: the keys that appear in `labels_gt` or
   `labels_teacher` of its training clips.

A label can also be supervised on one media type only. Training masks a label on images
or on video when its training targets there cover a single class, and on both when the
config's `mask_labels` names it (`bulwark.train.labels`); every heads, LoRA and
distillation checkpoint records the result as `masked_labels` ({label: [media types]}),
and a release or an exported graph carries it on (a graph reads it from its contract's
record of the checkpoint, else from its top-level `masked_labels` key, else from the
contract's own). A 1-frame clip counts as an image and a
longer one as video, as in training, and `predict` leaves out the labels masked on that
media type, even when `labels=` names them: `model.labels_for("video")` and
`model.labels_for("image")` list what it returns.

### Calibration files

`bulwark evaluate` writes `calibration.json` next to its report:

```json
{"meta": {"model": "e1", "checkpoint": {"path": "...", "sha256": "..."}, "fitted_on": "val", "...": "..."},
 "labels": {"eyes_closed": {"temperature": 1.4, "threshold": 0.8, "threshold_from": "val", "source": "val",
                            "n_val": 900, "n_val_positive": 400, "n_val_subjects": 30,
                            "media": {"image": {"temperature": 1.5, "threshold": 0.8, "threshold_from": "val", "...": "..."},
                                      "video": {"temperature": 1.2, "threshold": 0.1, "threshold_from": "val", "...": "..."}}},
            "eyeglasses": {"temperature": 1.1, "threshold": 0.6, "...": "...", "media": {"image": {"...": "..."}}},
            "phone": {"temperature": null, "threshold": 0.5, "threshold_from": "default", "source": "default", "media": {}}}}
```

(The numbers only show the format.) Each label's top-level entry is its fit on all val
clips. A clip model's confidence differs between still images and video clips, so binary
and categorical labels also have `media`: the same fit on the val clips of each media
type alone. `load(..., temperature="runs/eval/calibration.json")` reads it as follows:

- Each media type takes its own entries: `model.temperature_for(media_type)` and
  `model.thresholds_for(media_type)` hold them, and `predict` applies those of the input
  (a 1-frame clip is an image, a longer one video). A media entry applies even when no
  fit was made (val clips of one class only, or fewer than 10 of each class; its
  `not_fitted` says why): a null temperature or a default threshold means none on that
  media type, never the fit on all val clips, which is mostly a fit on the other media
  type. A label whose `media` has no entry for a media type (`eyeglasses` on video above:
  val held no video clips with its ground truth) gets neither there;
  `model.calibration.for_media(media_type).no_val_clips` names those labels (a release
  built from the file keeps the values only).
- An entry without `media` (calibration.json files written before the per-media-type
  entries) applies to every media type; `model.calibration.for_media(media_type).pooled`
  names the labels that took such an entry. `bulwark report` re-scores the saved
  predictions of an evaluation and writes the per-media-type entries
  ([evaluation.md](evaluation.md#running-it)).
- `model.temperature`, `model.raw_thresholds` and `model.thresholds` keep the fit on all
  val clips, as before, so they are not what `predict` applies when the file has
  per-media-type entries (reading `model.thresholds` then warns); use `temperature_for`
  and `thresholds_for`.
  `read_calibration(path)` returns the same as a `Calibration` (`for_media(media_type)`
  resolves one media type).
- Labels with a temperature are calibrated; labels whose temperature is null keep their
  trained outputs.
- Thresholds tuned on val (`threshold_from: "val"`, binary labels only) become
  `model.raw_thresholds` (those of one media type:
  `model.calibration.for_media(media_type).thresholds`). `bulwark evaluate` tunes them on
  the uncalibrated probability, so `model.thresholds_for(media_type)` converts them to the
  probabilities `predict` returns for that media type: a temperature t moves a threshold
  p to sigmoid(logit(p) / t), and a clip is flagged at the same logit either way.
  `model.thresholds` converts `model.raw_thresholds` with `model.temperature`, which goes
  with what `predict` returns only when the calibration is the same on every media type
  (a file without per-media-type entries, or an ONNX graph's own values). A threshold of
  exactly 0 or 1 is valid: `bulwark evaluate` picks it among the val probabilities, which
  saturate on a confident head, and no temperature moves it. Values outside [0, 1] are
  refused.
- Temperatures and thresholds belong to one set of weights. `meta.checkpoint.sha256`
  (and, for an evaluated ONNX graph, `exported_from_sha256`, the checkpoint it was
  exported from) must match the weights being loaded: the checkpoint's sha256, the
  checkpoint a release or ONNX graph was built from, or the graph file itself. A file
  fitted on other weights is refused; a file that records no checkpoint (including files
  written before calibration.json had a `meta` section) is applied with a warning.

A plain mapping (`temperature={"yawn": 1.4}`) is taken as given and not checked; it
replaces the temperatures on every media type and keeps any thresholds the model
records for each. `read_temperatures(path)` returns a file's temperatures fitted on all
val clips, without that check.

The commands check the file too: `bulwark predict --calibration`, `bulwark release
--calibration` and `bulwark export --calibration` refuse a calibration.json whose
`meta.checkpoint` names other weights than the model's checkpoint (predict and release
apply it with a warning when the model does not record its checkpoint's sha256). All three
take its temperatures and its val-tuned thresholds: predict lists those of the input's
media type in its JSON; release stores those of both media types in `config.json`; and
export stores those of the graph's media type in its metadata, one value per label (video
for a graph of more than one frame, image for `--frames 1`), records that choice in the
contract's `calibration` entry and says which labels had no entry of their own.

## Embeddings

```python
tokens = model.embed(frames)                  # kind="tokens"
vector = model.embed(frames, kind="pooled")
state = model.embed(frames, kind="state")
result = model.run(frames)                    # outputs, tokens and state from one pass
```

| Kind | B1 | E1 | ONNX |
|---|---|---|---|
| `tokens` | `(B, 25, 768)` summary tokens: 1 global, 8 temporal, 16 spatial (the same tokens `bulwark train cache` stores) | `(B, T, D)` one feature vector per frame (D = 1600 for EfficientViT-B1) | only if the graph has a `tokens` output |
| `pooled` | `(B, 768)` mean over the tokens (equal to the global token at 256 px) | `(B, D)` mean over frames | as `tokens` |
| `state` | `(B, 512)` the heads' state embedding | `(B, 256)` | only if the graph has a `state` output |

`tokens` and `pooled` describe the clip independently of the taxonomy heads; `state` is
the representation the heads read, so it is shaped by the training labels. A graph has the
`tokens` and `state` outputs unless it was exported with `bulwark export --no-embeddings`;
without them `embed` raises `NotImplementedError`. `bulwark export --manifest` adds the
label status of the training manifest to the graph's metadata (`ground_truth`,
`teacher_only`, `untrained` and `not_trained_by_checkpoint`). The graph's
`supervised_labels` and `masked_labels` are the checkpoint's own record, so it returns
the same outputs as the PyTorch checkpoint; only for a checkpoint that records no
`supervised_labels` does it list every output with a training target in the manifest,
masked or not. `--calibration` stores the temperatures and val-tuned thresholds of a
calibration.json for the graph's media type, so the graph's `model.temperature` is that
media type's.

## Video timelines

```python
from bulwark.inference import predict_video

for window in predict_video(model, "cabin.mp4", hop_seconds=1.0):
    print(window["start"], window["outputs"]["eyes_closed"])
```

Each window covers `window_seconds` (default 2 s) and windows start every `hop_seconds`,
so the default is one entry per second:

```json
{"start": 12.0, "end": 14.0,
 "outputs": {"eyes_closed": 0.03, "activity": {"attentive": 0.81, "texting": 0.02, "...": 0.0}},
 "adapters": {"unsafe_reach": {"no": 0.9, "yes": 0.1}}}
```

(The numbers above only show the format.) `outputs` holds `model.labels_for("video")`;
for an ONNX graph exported with `--frames 1`, whose windows are single frames,
`model.labels_for("image")`.
`adapters={"name": adapter}` adds each adapter's class probabilities, computed from the
same forward pass; an adapter trained on another model's embeddings is refused (see
Adapters). Windows are cut as the dataset adapters cut training clips: a window is kept
only when its whole span lies inside the video, so the last `end` never passes the
video's duration. A video shorter than one window gives one window, padded with its last
frame and ending where the video ends.

`predict_image(model, "face.jpg")` gives one entry of the same form for an image (start
and end 0), run as one 1-frame clip with `model.labels_for("image")`; an ONNX graph needs
`bulwark export --frames 1` for that. [examples/predict_video.py](../examples/predict_video.py)
writes the timeline to a file, and so does `bulwark predict MODEL VIDEO --out FILE`. The
command's file lists under `labels` the outputs returned for the input's media type
(named under `media`), with `labels_known`, the model's `masked_labels`, and the
`temperature` and val-tuned `thresholds` of the returned labels for that media type
(`temperature_for(media)` and `thresholds_for(media)`: from `--calibration`, or those the
model records);
when the video outputs include `perclos`, each window also gets `rolling` and the file
records `perclos_seconds` (see below).

### Rolling PERCLOS

The `perclos` output is a clip-level value: the fraction of the clip's frames with the
eyes closed. PERCLOS as the drowsiness literature uses it averages eye closure over a
minute or more. `RollingMean` computes that from consecutive windows:

```python
from bulwark.inference import RollingMean, predict_video

perclos = RollingMean("perclos", seconds=60)
for window in predict_video(model, "cabin.mp4"):
    value = perclos.update(window)   # None until 60 s of consecutive windows
```

Each value is the mean clip `perclos` over the windows that start at most 60 s before the
current window ends. It is None until consecutive windows cover 60 s. A window that
starts after the previous one ended (missing video, or a hop longer than the window)
starts the average again, so it never spans time the model did not see. With the default
2 s windows every 1 s, every moment counts equally except the first and last second of
the span. The current window always counts, so a span shorter than one window gives that
window's own value. `bulwark predict` and `examples/predict_video.py` add it to each
window as `"rolling": {"perclos_60s": ...}` when the model's video outputs include
`perclos` (`--perclos-seconds`, default 60, changes the span, and 0 turns it off; both
skip rolling PERCLOS when the span is shorter than `--window`, or when `--hop` is longer
than `--window`). `perclos` is only meaningful when the model was trained on it: check
`model.labels_known`, `model.labels_for("video")` and the training data's label
coverage.

## Releases

`save_release(model, "releases/bulwark-b1-v0", weights_licence="...", labels=[...])`,
[examples/make_release.py](../examples/make_release.py) or `bulwark release` writes:

- `model.safetensors`: every weight, with LoRA already merged into the encoder.
- `config.json`: described below.
- `LICENSE-vjepa2` (B1 releases only): a B1 release contains the V-JEPA 2.1 encoder
  weights, which Meta publishes under the MIT licence, so their copyright and licence
  notice ships with it, copied from `bulwark/model/vjepa21/LICENSE` (the licence of
  Meta's encoder code, which the package vendors).

`weights_licence` states the terms of the weights. Unless the training provenance shows
commercially usable data only (`commercial_only` true, every recorded dataset
`commercial_ok`, including the datasets of upstream models, and no upstream input whose
training data is unrecorded), `save_release` refuses a licence that does not restrict use
to non-commercial research. The licence must name a non-commercial licence (for example
"CC BY-NC 4.0 (research use only)", "CC-BY-NC-SA-4.0" or "non-commercial") or say
"research use only" (or "research only"). It must also not mention commercial use other
than as non-commercial, nor name a licence that allows commercial use (MIT, Apache, BSD,
CC0, CC BY without NC, ...): "MIT (research use only)" and "Apache-2.0; research and
commercial use" are both refused. A checkpoint without provenance counts as not clean. The
v0 weights therefore cannot be released for commercial use.

`labels` are the outputs the release advertises, which should be the labels its training
data supervised. They default to the model's labels when the caller or the checkpoint
named them; when neither did, `save_release` refuses, so a release never lists heads that
were never trained.

`config.json` holds:

| Field | Content |
|---|---|
| `kind`, `architecture` | `b1`, `b1-frozen` or `e1`, and what is needed to rebuild the module (no local paths) |
| `taxonomy_version`, `outputs` | the taxonomy the heads were built for, and all heads in order |
| `labels`, `temperature` | the advertised outputs and their per-label temperatures without regard to media type (from a calibration.json, the fit on all val clips; empty when none were given), which loaders that predate `media_calibration` apply |
| `masked_labels` | {label: media types} whose training targets were masked, as the checkpoint records them (empty when it records none) |
| `thresholds` | val-tuned thresholds of binary labels on the uncalibrated probability, as in calibration.json and as `temperature` (empty when none were given); `model.thresholds` converts them |
| `media_calibration` | {media type: {`temperature`, `thresholds`}}: the values `load` applies to images and to video, resolved from the calibration.json as described in [Calibration files](#calibration-files) (empty without per-media-type values, and a release without this field applies `temperature` and `thresholds` to both) |
| `input` | the input contract: layout, frame size, frames and seconds per clip |
| `weights_licence` | the terms given to `save_release` |
| `commercial_only` | from the training provenance: true when the run was restricted to commercially usable datasets; null when the checkpoint records no provenance |
| `source` | the checkpoint's file name and sha256, its step, and from the training provenance the job, `commercial_only`, the datasets (licence, commercial use, clips per split), the manifest's sha256 and the code commit; `upstream_datasets`, the datasets of the models the run learned from (distillation teacher, init checkpoint, the model behind cached features) with how each reached the weights (`via`), and `unrecorded_inputs`, upstream inputs whose training data is not recorded (both null without provenance) |

`extra=` adds fields to `config.json` but cannot replace any of these. `bulwark release`
and `examples/make_release.py` add `model_card` (a link to the model card, which is the
Hugging Face model repository of the release's model, B1 or E1:
`bulwark.export.release.model_card_url(kind)`), `model_release`
and `release_check` (the verdict of `bulwark release-check`, which they run first, with
what each dataset's clearance rests on, `weights_release_basis`;
`bulwark.export.release.prepare_release`), default the licence to the research licence
(`bulwark.export.release.RESEARCH_LICENCE`), and refuse a licence that does not restrict
use to non-commercial research unless that check clears the weights for commercial
release.
Local paths in the checkpoint's metadata (the checkpoint itself, the manifest, shards,
features and init checkpoints) are not copied into `config.json`. `load("releases/...")`
refuses a release built for a different taxonomy version or output order.

## Adapters

The taxonomy heads cover the shared labels. A product that needs another label (a
site-specific unsafe action, a customer's own alert classes) can train an adapter on
embeddings of a few labelled clips instead of fine-tuning the encoder.

```python
from bulwark.adapters import Adapter, features_source, fit_adapter

adapter = fit_adapter(train_tokens, train_labels)       # (N, 25, 768) and N labels
adapter.save("adapters/unsafe_reach.pt")

adapter = Adapter.load("adapters/unsafe_reach.pt")
probs = adapter.predict_proba(model.embed(clips))       # (N, C), columns in adapter.config.classes
```

How `fit_adapter` trains:

- Inputs are `(N, D)` vectors or `(N, P, D)` tokens. Tokens are pooled with `pooling`:
  `mean` (default), `first` (token 0), or `flatten` (all tokens, P x D features). The
  pooling is stored and applied again at prediction time.
- Features are standardised with training statistics, and with `balanced=True` (default)
  each class counts equally in the loss.
- `hidden=0` (default) is multinomial logistic regression with an L2 penalty (`l2`),
  solved with full-batch L-BFGS. The problem is convex, so the result depends only on the
  data. `hidden>0` adds one hidden layer and trains with Adam for `steps` steps.
- Labels are compared as strings; `classes=` fixes their order.

The saved file holds tensors and plain values only (it loads with `weights_only=True`)
plus the config: classes, pooling, the embedding kind (`tokens`, `pooled` or `state`), a
free-text `source`, and `model`, a record of what produced the embeddings.

An adapter only fits the embeddings it was trained on. Tokens from the frozen encoder
(`b1-frozen`, `bulwark train cache` without `init`) differ from tokens of a LoRA-tuned B1,
and B1 and E1 embeddings are unrelated. To have this checked, pass `model=` when fitting:

```python
adapter = fit_adapter(model.embed(clips), labels, model=model.embedding_source("tokens"))
adapter = fit_adapter(tokens, labels, model=features_source("data/features/frozen"))
```

`embedding_source(kind)` is `{"family": "b1" | "e1", "weights": ...}`, where `weights` is
`"frozen"` for tokens of the frozen encoder and otherwise the sha256 of the checkpoint the
weights come from (for a release, of the checkpoint it was built from; for an ONNX graph,
as its export contract records it). `features_source` reads the same record from a
features directory's `meta.json` (`"frozen"` without `init`, the init checkpoint's sha256
with it). `adapter.check(model.embedding_source(kind))` raises on a mismatch, and
`predict_video` and `predict_image` run it for every adapter (`check_adapters=False`
skips it). Fields that
either side does not know are not compared, so an adapter fitted without `model=` is only
checked for its feature width. Retrain adapters when the base checkpoint changes.

### Where embeddings come from

- Your own clips: [examples/train_adapter.py](../examples/train_adapter.py)
  `MODEL clips.csv --out adapter.pt` reads a CSV of `path,label[,subject][,start]`, embeds
  one clip per row and fits the adapter. With a `subject` column it first scores the
  adapter on held-out subjects.
- The training pipeline: `load_cached_embeddings(features_dir, manifest, label, split=...)`
  reads tokens from a `bulwark train cache` directory with one label per clip. `label` is a
  taxonomy label, taken from the manifest's ground truth only (teacher labels are never
  targets here), or a mapping from clip id to your own label.

Score an adapter on subjects it has not seen. Clips of the same person, or the same video,
are strongly correlated, and a random clip split overstates accuracy.

### Few-shot curves

[examples/fewshot_curve.py](../examples/fewshot_curve.py) measures how an adapter improves with
labelled clips:

```bash
PYTHONPATH=. python examples/fewshot_curve.py \
    --features data/features/frozen --manifest data/manifest.parquet \
    --label activity --dataset drive_and_act --out runs/fewshot/activity.json
```

For each k in `--shots` it draws k training clips per class, fits an adapter, scores it on
the evaluation split, and repeats with new draws (`--repeats`). The output reports
balanced accuracy, macro-F1 and, for two classes, AUROC as mean, standard deviation,
minimum and maximum over repeats, next to a majority-class baseline, the adapter trained
on all training clips, and the number of clips and subjects on each side. For that last
adapter it adds a 95% interval from resampling evaluation subjects (`--bootstrap`, only
with at least 10 subjects); the spread over repeats reflects the training draws, not the
size of the evaluation set. Draws are made per clip, so a draw can repeat a subject or a
video.

The script stops instead of reporting when a subject appears in both the training and the
evaluation split, or when the evaluation split, after dropping classes absent from
training, holds fewer than two classes: a constant prediction would then score perfectly.

To measure adapters on a dataset no Bulwark model was trained on, use one split for both
sides. When `--train-split` equals `--eval-split`, that split's subjects are divided:
`--eval-subjects` of them (default: half), drawn with `--seed`, are scored and the rest
are the training pool, so training and evaluation still share no subject. UTA-RLDD, for example, is never trained on
and all of its clips are in the holdout split:

```bash
PYTHONPATH=. python examples/fewshot_curve.py \
    --features data/features/frozen --manifest data/manifest.parquet \
    --label alertness --dataset uta_rldd --train-split holdout --eval-split holdout \
    --eval-subjects 30 --out runs/fewshot/alertness-uta_rldd.json
```

With 30 evaluation subjects the subject-bootstrap interval is reported. An evaluation set
with fewer than 10 subjects, such as Drive&Act's test split (the dataset has 15 drivers in
all), gives the curve without that interval.

No few-shot curve on the full training data has been published yet. A smoke-sized
sample has only a handful of subjects per split, which is enough to check that the script
runs and far too few to compare models.
