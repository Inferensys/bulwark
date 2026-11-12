# Training

How to reproduce Bulwark from raw datasets to an exported E1 model. Every step is a
`bulwark` command, and the same command runs on a laptop, a Kaggle notebook or a cloud
VM; configs live in `configs/train/`. See [architecture.md](architecture.md) for what is
being trained and [evaluation.md](evaluation.md) for how it is scored.

## Stages and hardware

| Stage | Command | Hardware |
|---|---|---|
| Ingest, split, shard | `bulwark ingest <dataset>`, `bulwark split`, `bulwark shard` | CPU; disk-bound |
| Face teacher labels | `bulwark label face` | CPU |
| Object teacher labels (optional) | `bulwark label objects` | GPU or Apple Silicon |
| VLM teacher labels (optional) | `bulwark label vlm` | Apple Silicon (MLX); it generates text for every clip, so it is slow (not benchmarked): run it on a subset |
| Merge teacher labels | `bulwark label merge` | CPU |
| Frozen features | `bulwark train cache --config configs/train/cache.yaml` | GPU with 16 GB (fp16) recommended |
| Heads on frozen features | `bulwark train heads --config configs/train/heads.yaml` | Apple Silicon (MPS), CPU or any GPU |
| B1 LoRA fine-tune | `bulwark train lora --config configs/train/lora-kaggle.yaml` (or `lora.yaml`) | 16 GB GPU in fp16 (T4), or an A100 in bf16 |
| B1 outputs for distillation | `bulwark train cache --config configs/train/cache-b1.yaml` | GPU with 16 GB (fp16) |
| Distil E1 | `bulwark train distill --config configs/train/distill.yaml` | one GPU with 16 GB (fp16); not split across GPUs ([Multiple GPUs](#how-runs-behave)) |
| Evaluate | `bulwark evaluate --checkpoint ...` | heads: CPU; B1 and E1: a GPU is faster |
| Export E1 to ONNX | `bulwark export --checkpoint ...` | CPU |
| Synthetic clips (optional) | `scripts/synth_wan.py` | GPU with more than 16 GB |

Data preparation and teacher labelling run on your own machine: the Kaggle and Azure
scripts below run only `bulwark train` and `bulwark evaluate` jobs. The teachers need
extras: `label` for the face teacher, `detect` for the object teacher and `vlm` for the
VLM teacher.

On CUDA GPUs, `precision: auto` uses bf16 where the GPU supports it natively and fp16
otherwise; on Apple Silicon and CPU it trains in fp32. The T4 configs set fp16
explicitly because the T4 has no native bf16.

## 1. Data

```bash
scripts/fetch_pretrained.sh          # V-JEPA 2.1 ViT-B weights, MediaPipe face model, V-JEPA 2 code (pinned) for a parity test
bulwark datasets                     # every known dataset: licence, commercial use, size, adapter, local copy
```

Download each dataset you are licensed to use from its official page (links and access
routes in [datasets.md](datasets.md)) and unpack it under `data/raw/<name>/`, or point
`configs/data.yaml` at it. Then:

```bash
bulwark ingest <name> --dry-run      # check the adapter against the real layout; prints a sample clip
bulwark ingest <name>                # writes data/manifests/<name>.parquet
bulwark split                        # merge manifests, drop cross-dataset duplicates, assign splits
bulwark shard                        # 256 px JPEG frame stacks in tar shards under data/shards/
bulwark label face                   # teacher labels into data/labels/
bulwark label merge                  # fold teacher labels into the manifest
bulwark coverage                     # labels whose train targets have one class only, or miss classes
scripts/check_shards.py              # manifest and shards agree, dataset by dataset
```

`scripts/prepare_local.sh` runs the ingest, split, shard, face-teacher, merge and check
steps for every dataset that has a local root. To add one dataset to data that is already
prepared, rerun it, or run the steps for that dataset alone as its header lists them
(`bulwark shard --dataset <name>`, `bulwark label face --pattern '<name>-*.tar'`). Every
clip in the manifest records its dataset's licence and `commercial_ok` flag.

**Teacher labels.** `bulwark label merge` rebuilds each clip's `labels_teacher` from the
teacher files in `data/labels/` (rules in `bulwark/label/run.py`). Ground truth always
wins: a clip gets a teacher label only where it has no ground truth for that label, and
teacher labels count at half the weight of ground truth in the loss. Of the face
teacher's `eyes_closed`, `blink` and `yawn`, only `eyes_closed` 0 is kept (`FACE_KEEP`):
a value is kept when ground truth confirmed it at least 90% of the time on the train
split and the class that keeping only that value leaves out is rare in the clips it
labels, so the one-sided labels it then gives there are close to the truth. Its "eyes
open" passes both: it is right 98.9% of the time, and it leaves out closed eyes, which
are rare in the clips it labels (on the v0 train split, video clips without ground truth
on cameras that face the person, it says closed on 710 of 9,054). Its "blink" is right
97.3% of the time, but it leaves out "no blink", which is common (it says no blink on
4,127 of the 7,166 clips it labels there and finds a blink on the other 3,039), so
keeping only its blinks would pull the blink output towards "blink"; its "no blink" is
right only 55.1% of the time. Its "yawn" would leave out "no yawn" in the same way. On
v0 dropping it changes nothing, since the teacher finds no yawn in the 9,049 clips it
labels there; it is dropped so that frontal datasets added later (YawDD's dash camera,
which has no yawn ground truth) do not get positives-only yawn rows. The face teacher
therefore labels neither `blink` nor `yawn`. A synthetic clip's prompted labels still
can (`scripts/synth_wan.py` prompts `yawn` 1 in one scene); v0 has none, so in v0 both
train on DMD's ground truth only. Where the face teacher says the eyes are closed, the
clip also loses its teacher `eye_openness` and `perclos`, which come from the same
eye-closure measure, unless the clip's ground-truth `eyes_closed` confirms the closed
eyes; where ground truth contradicts the teacher's
`eyes_closed`, they go either way. The teacher's video `eye_openness` targets therefore
lean open (211 of 10,078 below 0.5 on the v0 train split), and no dataset gives video
`eye_openness` ground truth to check that output against. All of this was measured on
DMD's face camera, so on cameras outside `FRONTAL_FACE_VIEWS` (all of Drive&Act's) the
face teacher's `eyes_closed`, `eye_openness` and `perclos` are dropped as well. On clips
without ground truth the teacher's `eyes_closed` is one-sided: on the v0 train split it
gives 8,344 negatives and no positives (all DMD). Nothing reweights these rows, so they
pull `eyes_closed` towards 0; check that output's calibration on val after training with
them, in its video entry of `calibration.json` (thresholds and temperatures are fitted
per media type; see [evaluation.md](evaluation.md#metrics)). The measurements behind the rule are in the
[data card](data-card.md#labels); `bulwark label agreement --split train` reproduces the
per-value counts (precision and npv) and the overall `perclos` MAE; the MAE split by the
teacher's `eyes_closed` (0.103 open, 0.192 closed) groups those same `perclos` pairs by
the teacher's `eyes_closed` on each clip, which the command does not do. `bulwark label
merge` applies the rule each time it rebuilds `labels_teacher`, so a change to
`FACE_KEEP` or the other rules reaches a manifest only when it is merged again. After a
re-merge, cached frozen features are still valid (summary tokens do not depend on
labels), but a run does not resume a `last.pt` written with the previous manifest, whose
hash is part of the run fingerprint (see [Resume](#how-runs-behave)): train into a new
`out`, or pass `--set fresh=true`.

v0 trains on Drive&Act, the DMD face camera, MRL Eye and AUC Distracted Driver v2. The
Zenodo driver-behaviour set is excluded (`SplitPolicy.excluded_datasets` in
`bulwark/data/splits.py`): its data use agreement forbids sharing the data with third
parties and asks to review publications 30 days before submission. `bulwark split`
prints clips per dataset and split, and its `excluded` column counts the clips that are
neither trained on nor scored: every clip of an excluded dataset, and the Drive&Act
inner-mirror clips of drivers outside the test split. The split check that training and
evaluation run (`check_disjoint`) refuses a manifest with clips of an excluded dataset
outside `excluded`; `bulwark split` rebuilds it under the current policy.

`bulwark shard` shards `excluded` clips too (the Drive&Act mid-level probe uses every
camera view) and skips clips that fail to decode. Next to each dataset's shards it
writes `<dataset>.shards.json` with the frame size, the clips that failed and a
fingerprint of how the sharded clips are cut; a re-run adds only clips not yet sharded,
and refuses when the manifest now cuts already-sharded clips differently under the same
clip ids. `scripts/check_shards.py` reports, per dataset, the clips missing from the
shards and fails when more than `--max-missing` of them (default 0.005, that is 0.5%) are
missing for reasons other than a recorded decode failure, when shards hold clips the
manifest no longer has, when a shard has no index, or when clips were cut differently.

**Rebuilding a dataset's shards.** Shards are only ever added to, so clips whose frames
changed under the same clip id (after an adapter fix) keep their old frames until the
dataset is resharded: `RESHARD=<dataset> scripts/prepare_local.sh` moves its shards to
`data/shards-replaced/<UTC time>/`, drops its clips' teacher labels and shards it again.
With `SKIP_LABELS=1`, rerun `bulwark label face --pattern '<dataset>-*.tar'` and
`bulwark label merge` afterwards. Feature caches recompute those shards on their next
run. The cloud copies still hold the old frames: replace the Azure Blob copies with
`scripts/purge_cloud_copies.sh --reshard <dataset>`, and push a new version of each
Kaggle dataset that holds the dataset's shards, with `SHARD_GLOB` listing every source
that dataset holds.

## 2. Heads on frozen features

```bash
bulwark train cache --config configs/train/cache.yaml   # summary tokens -> data/features/frozen
bulwark train heads --config configs/train/heads.yaml   # -> checkpoints/heads-v0
```

Caching runs the frozen encoder once over every shard, so the heads can train on the
cached tokens without a CUDA GPU, on a CPU or Apple silicon (MPS). On Kaggle's two T4s,
the 170 shards without AUC Distracted Driver v2 took about 51 minutes of encoding in a
session of about 70, and AUC's 15 shards 98 s of encoding in a separate session of about
4 minutes ([Measured for v0](#measured-for-v0)); the run time of `heads.yaml` on the v0
data is not measured yet.
Frozen summary tokens do not depend on `taxonomy.yaml`, so editing labels never
invalidates them.

Every training job (`heads`, `lora`, `distill`) writes `last.pt`, `final.pt` and, when
the manifest has val clips in the shards or features, `best.pt` (a heads run with
`select_folds`, as the shipped heads configs set, writes `best.pt` whenever a fold could
be scored, val clips or not):

- `best.pt` has the best validation score on ground-truth labels only
  (`gt_mean_headline`): the mean over labels of AUROC for binary labels, macro-F1 for
  categorical labels and 1 - MAE / range width for regression labels. The score covers
  the labels the run trains, without the val rows of label and media-type pairs it masks
  (see [Labels a run trains](#labels-a-run-trains)). Labels with fewer than 10
  ground-truth val clips, or a single class, are left out of the mean, and teacher labels
  never count. `best.pt` records the score per label and, in `select.support`, how many
  clips and distinct subjects each label's score rests on.
- `final.pt` is the model after the fixed epoch or step budget, chosen without looking
  at val.

The heads job validates at the end of every epoch. The val split is small. With the
datasets used at the time of writing it holds 2 Drive&Act drivers (the official
`split_0` val drivers), 2 DMD subjects and 3 MRL Eye subjects. The shipped heads configs
(`heads.yaml` and `heads-clean.yaml`) therefore pick `best.pt` by 5-fold
cross-validation over the train and val subjects (`select_folds: 5`): on the v0 data the
five held-out folds together score 12 Drive&Act drivers, 11 DMD subjects and 34 MRL Eye
subjects. `--set select_folds=0` (with a new `out`, or `--fresh`) picks `best.pt` on the
val split instead. Either way, check `select.support` before trusting `best.pt`: where a
label's score rests on one or two people, start the fine-tune from `final.pt` instead
(`--set init=checkpoints/heads-v0/final.pt`).

With `select_folds: k`:

- Each dataset's train and val subjects are dealt into k folds
  (`bulwark.data.splits.subject_folds`); clips of train-only datasets
  (`SplitPolicy.train_only_datasets`) are in no fold and always stay in training.
- Before the main run, the job trains k extra heads models, each on the train and val
  clips outside one fold, with no checkpoints, and scores each on its held-out fold at
  every epoch end (`cv` events in `log.jsonl`). Each trains for the config's `epochs` on
  about (k - 1)/k of the train and val clips, so with 5 folds the job takes several times
  as long as with `select_folds: 0`.
- The main run still trains on the train split only. `best.pt` is its model after the
  chosen epoch's steps (`epoch` x steps per epoch), the epoch with the best mean score
  over the folds, and it is written even when the manifest has no val clips.
- `best.pt`'s `select` then also records `folds`, the chosen `epoch`, `fold_scores` (each
  fold's score at that epoch) and `epoch_scores` (the mean over folds per epoch), and its
  `support` sums, per label, the held-out clips and subjects of the folds that scored it.
  The evaluation report (`report.html`) states the score as a mean over folds rather
  than a val score.
- `last.pt` keeps the choice, so a resumed run does not repeat the cross-validation.

### Labels a run trains

Before training, every job checks the targets of its training clips (ground truth and
teacher labels together) per label and media type, since images and videos are batched
separately. A pair with only one class (context on video, where every clip is in a
vehicle cabin, for example) is masked: its targets leave
the loss, and distillation does not copy the teacher's outputs for it. `mask_labels`
masks whole labels on top of that; the heads, lora and distill configs, v0 and clean
alike, mask `attention_direction` (Drive&Act's targets never include `forward`) and
`face_visible` (on cameras that do not face the person, the face teacher keeps only
`face_visible` = 1). Classes that no training target covers are reported, not masked.
The job prints what it masks, and every checkpoint records the labels it trained as
`supervised_labels` and the masked pairs as `masked_labels`; `bulwark.inference.load`
returns only the supervised outputs by default. Run `bulwark coverage` on a manifest to
see the one-class labels before a job starts.

Phone, drink and food have both classes on video, with negatives from DMD's and
Drive&Act's object annotations ([data card](data-card.md#labels)), but few positives: on
the v0 train split's video clips 5,580 phone, 1,700 drink and 3,166 food positives
against 37,233, 41,268 and 32,615 negatives. On images, AUC Distracted Driver v2 gives
phone and drink positives only (4,661 and 933 train images), so jobs on the v0 data mask
both on images. The binary loss is plain cross-entropy with no positive weight, focal
term or class-balanced sampling; only datasets are balanced (`dataset_power`).

## 3. B1: LoRA fine-tune

```bash
bulwark train lora --config configs/train/lora-kaggle.yaml   # 16 GB GPU, fp16
bulwark train lora --config configs/train/lora.yaml          # A100, bf16
```

Both configs start from `checkpoints/heads-v0/best.pt`, use the same effective batch
(64) and write `last.pt`, `best.pt` and `final.pt` to `checkpoints/b1-lora-v0/`. LoRA
wraps the `qkv`, `proj`, `fc1` and `fc2` linear layers of the encoder's last 4 blocks,
and the encoder's final norm is trained as well; at start the job prints
`LoRA layers: 16, trainable params: 7.56M` (about 0.79M in the encoder and 6.77M in the
heads).

The fine-tune and distillation run a validation pass over every val clip in the shards
every `eval_every` optimizer steps and at each epoch end, and keep the best as `best.pt`.
The `lora*` and `distill*` configs set `eval_every: 0`, so they validate at epoch ends
only. With `lora-kaggle.yaml` on the v0 data a pass took about 6 minutes on Kaggle's
two T4s and an epoch about 67 minutes (6.6 and 71 in the first full run, without AUC
Distracted Driver v2; [Measured for v0](#measured-for-v0)), so the whole run fits in one
session and its three epoch-end passes choose `best.pt`. A distillation pass over the
same val clips, E1 on one T4, took 211 s.

Clips get light augmentation, the same for every frame of a clip (`AUGMENT_POLICY` in
`bulwark/train/stream.py`): a crop of 80 to 100% of the area, gamma, contrast and
brightness changes, and grayscale for 20% of clips. Frames are never mirrored, because
left and right carry meaning in labels such as attention direction and head yaw.

## 4. E1: distillation

First compute B1's outputs over the shards, into their own features directory, then
train the student, evaluate it and export it:

```bash
bulwark train cache --config configs/train/cache-b1.yaml     # B1 logits -> data/features/b1-lora-v0
bulwark train distill --config configs/train/distill.yaml    # -> checkpoints/e1-v0
bulwark evaluate --checkpoint checkpoints/e1-v0/best.pt --shards data/shards --out runs/eval-e1
bulwark export --checkpoint checkpoints/e1-v0/best.pt --out checkpoints/bulwark-e1.onnx \
  --shards data/shards --calibration runs/eval-e1/calibration.json
```

`cache-b1.yaml` reads the fine-tune's `best.pt`; set `init` to its `final.pt` to use the
fixed-budget model instead. On Kaggle's two T4s, caching the outputs of the first full
run's B1 (without AUC Distracted Driver v2) over its 170 shards took about 64 minutes of
encoding, and B1 v0's over the 185 v0 shards about 55. Distillation runs on one GPU
([Measured for v0](#measured-for-v0), [Multiple GPUs](#how-runs-behave)). B1 outputs
need their own `features` directory. Its `meta.json` records the B1 checkpoint that
produced it, the outputs that checkpoint was trained on (`supervised_labels`,
`masked_labels`) and the layout of the cached logits (`logit_layout_sha256`, a hash of
every label's name, head, kind, classes and range), so a retrained B1 cannot top up an
old directory. A clip whose tokens or logits are not finite (an fp16 overflow, say) is
left out of its shard's file and listed in `meta.json` under `nonfinite`; neither of
those two caches left any out.

The student trains on the clips that have teacher logits, and copies the teacher only
on the outputs the teacher was trained on, minus the pairs its own run masks. Its
augmentation is photometric only (`PHOTOMETRIC_POLICY`, the policy above without the
crop): the teacher's logits were cached on full clips, and a crop would show the student
a different view than the one its soft targets describe.

`bulwark export` checks the ONNX graph against the PyTorch model (on real clips with
`--shards`) and measures CPU latency with onnxruntime (`--no-bench` to skip); see
[Speed and latency](#speed-and-latency). `--calibration` stores the temperatures and
the val-tuned thresholds of a `calibration.json` fitted on the same weights, those of the
graph's media type (video for 16 frames, image for `--frames 1`; `--temperature` is the
same option, in `bulwark predict` and `bulwark release` too),
the graph has the `tokens` and `state` outputs that adapters read unless
`--no-embeddings` is given, and `--frames 1` exports a graph for still images. The input
and output contract of the ONNX file, and everything its metadata records, is in
[architecture.md](architecture.md#e1-onnx-contract).

## How runs behave

**Checkpoints.** The training jobs (`heads`, `lora`, `distill`) save `last.pt` every
`checkpoint_minutes` (default 15) and at the end of each epoch, with the optimiser,
loss-scaler and RNG state, and the position in the epoch for the streaming jobs.
`best.pt` is written after each validation pass that improves the score; the first one is
followed at once by a `last.pt`, so a run evicted before its first timed save can still
resume.

**Resume.** Rerunning a job with the same `out` continues from `last.pt`, but only when
its run fingerprint matches: the job, the manifest hash, the config and the inputs
(shards, init checkpoint, teacher or features directory, encoder weights). Inputs are
compared by content hash, not by path, so the same init mounted under another path still
resumes. Config keys that say where and how a session runs are exempt and may change
between sessions: `out`, `device`, `num_workers`, `multi_gpu`, `time_limit_minutes`,
`checkpoint_minutes`, `log_every`, `eval_every`, and the input paths `manifest`,
`shards`, `features`, `init` and `teacher`. Config keys added to the code after a
`last.pt` was written count as their defaults.

A job refuses to:

- train on a manifest that breaks the split policy (`check_disjoint`: a subject in two
  splits, a holdout dataset outside holdout, a holdout view in train, val or test, an
  excluded dataset outside `excluded`, a train-only dataset in val, test or holdout, a
  removed subject); rebuild it with `bulwark split`;
- resume a `last.pt` whose fingerprint differs; the error lists what changed;
- resume a run that already reached its last step and wrote `final.pt`;
- start in a directory that holds `best.pt` or `final.pt` from an earlier run but no
  `last.pt`.

Two overrides exist for `heads`, `lora` and `distill`. `--set fresh=true` (or `--fresh`)
moves the old run's files to `out/stale-<UTC time>/` and starts over.
`--set allow_mismatch=true` (or `--allow-mismatch`) resumes a mismatched `last.pt` anyway
and logs what differed; it never applies across a `commercial_only` or `release_only`
change, and a run with either flag never resumes a `last.pt` that does not itself record
that flag.

**Feature caches.** `bulwark train cache` resumes shard by shard: it skips shards whose
features file exists and whose fingerprint matches. A directory written by a different
encoder, init or precision is refused rather than topped up. `--overwrite` (or `--set
overwrite=true`) clears such a directory, or one without `meta.json`; it leaves a
directory that belongs to the same run alone, so passing it again to the next session of
a long cache job does not undo the first. The cache job writes no `last.pt`, so it
refuses `fresh` and `allow_mismatch` rather than ignoring them.

**Provenance.** Every checkpoint, and `provenance.json` in the run directory, records:
the resolved config, `commercial_only` and `release_only`, the datasets used with their
licence, `commercial_ok` and `weights_release_ok` flags and clips per split, the
manifest's sha256, the split salt and taxonomy version, the inputs (shards, init
checkpoint, teacher or features directory, encoder weights, with their hashes and the
`commercial_only` and `release_only` flags, datasets and commit they record), the label
plan (masked pairs, missing classes, supervised labels), the augmentation policy of
`lora` and `distill`, the package version, the code commit and whether code paths had
local changes, and the Python, PyTorch, CUDA, NumPy, timm and onnx versions and the GPU.
Each model input (init, teacher, features) also records `upstream_datasets`: the datasets
of every model up the chain behind it, so an E1 names the heads run's datasets behind its
B1 teacher, and the `commercial_only` and `release_only` checks cover them too. It also
records `unrecorded_inputs`: the inputs up that chain whose training data is not recorded
(a checkpoint without provenance, say), so `bulwark release-check` on an E1 still reports
an unrecorded init two steps up ("teacher -> init (records no provenance)"). Inputs
written before these keys existed contribute their own datasets and status only.
`sessions` lists the start time and code version of every session, so a run whose code
changed between Kaggle sessions says so (the job also warns). Each features directory has
a `meta.json` with the encoder weights and their hash, the init checkpoint (with its
datasets, `upstream_datasets` and `unrecorded_inputs`), the precision, the package version
(`bulwark_version`), `taxonomy_version`, `split_salt`, the code version and a fingerprint
of every shard it covers. Remote jobs run from a packed archive without `.git`;
`scripts/pack_code.sh` records its commit in the archive. The archive holds what a job
installs and runs (the package, configs, infra and scripts, `pyproject.toml`, `uv.lock`),
not the tests, docs or dataset covers: a Kaggle job carries it inside its script, and
`push_job.sh` stops before uploading a script over Kaggle's 1 MiB limit.

**Logs.** Each run appends to `log.jsonl`: a `start` event with the provenance, a
`resume` event with the step it resumed at, `allow_mismatch` and `code_changed` events
when they apply, and every `log_every` steps the loss and its parts; for `lora` and
`distill` also `s_per_step`, `skipped_steps`, `loss_scale` (fp16) and `max_memory_gb`
(CUDA). Every validation pass adds a `val` event with the selection score, the score and
support per label and the full metrics; for `lora` and `distill` it also gives the
pass's duration in `seconds`.

**Time limits.** With `time_limit_minutes`, the `cache`, `lora` and `distill` jobs stop
cleanly (the Kaggle configs stop at 660 minutes, 11 hours of a 12-hour session), and
running the job again resumes. `lora` and `distill` stop with `last.pt` saved; `cache`
stops after the shard in progress, and the next run continues with the shards left. A
validation pass that starts just before the limit still runs to its end. The `heads` job
does not read the limit.

**No silent empty runs.** A job stops with an error if its filters leave no training
clips or the shards contain none of them.

**Data mixing.** Each dataset's share of an epoch grows with the square root of its size
(`dataset_power: 0.5`), both on cached features and when streaming shards. The streaming
jobs plan every epoch up front, interleave datasets, and resume mid-epoch at the batch
where they stopped (`bulwark/train/stream.py`).

**Loss weights.** The learned per-label loss weights have their own fixed learning rate,
`loss_weight_lr` (default 0.001), with no weight decay and no schedule, so a small
fine-tuning learning rate does not freeze them.

**Multiple GPUs.** When several GPUs are visible (Kaggle's T4 x2 machine), `lora` and
`cache` split each batch across them with `DataParallel` (`--set multi_gpu=false` to use
one); `evaluate` uses one GPU. How much the second T4 speeds up `lora` and `cache` has not
been measured yet. `distill.yaml` and `distill-clean.yaml` set `multi_gpu: false`, because
`DataParallel` over Kaggle's two T4s made E1's steps about ten times slower: in a timing
script with E1's frame encoder (EfficientViT-B1) alone, on random frames, at the
distillation batch (8 clips of 16 frames, 4 accumulation steps, fp16), a step took
11.75 s on two T4s against 1.24 s on one, and the distillation probe logged `s_per_step`
13.86 over its first 100 steps on two T4s against 1.25 on one.
The two-GPU probe then ran out of host memory and was killed before step 150
([Measured for v0](#measured-for-v0)). `DataParallel` runs every forward in new threads,
and torch 2.11 keeps cuDNN convolution plans per thread without freeing them, so E1's
convolutions are most likely planned afresh on every forward; B1 has one convolution and
does not show it.

Overrides: any config key can be set on the command line, e.g. `--set batch_size=8`.

## Before a long run

Run times depend on the data and the machine. The v0 measurements, a 600-step
distillation probe on one T4 among them, are in [Measured for v0](#measured-for-v0).
Before a fine-tune or distillation run on other data or hardware, run the same job for
200 steps into a scratch directory:

```bash
bulwark train lora --config configs/train/lora-kaggle.yaml \
  --set max_steps=200 --set out=checkpoints/probe-lora
```

Then read `s_per_step`, `max_memory_gb` and `loss_scale` from
`checkpoints/probe-lora/log.jsonl`; `s_per_step` is the mean over the session's steps so
far, validation excluded, so it includes the slower first steps (the v0 `distill` probe's
was 1.91 s after 25 steps, 1.15 s after 200 and 1.07 s after 600). At start the job
prints its optimizer steps per epoch (the total it prints is 200 here, because of
`max_steps`). `s_per_step` x steps per epoch x `epochs` (3 in `lora-kaggle.yaml`, 4 in
`distill.yaml`) estimates the full run in seconds; divided by 39,600 (each session's
`time_limit_minutes: 660`), it gives the number of Kaggle sessions the run needs. The
limit counts from the first step, so what a session does before it (installing the code,
loading the manifest, reading the shards) comes out of the 60 minutes a 12-hour session
has left after `time_limit_minutes`. On Kaggle, the time from loading the manifest to a
job's `train:` line (the cache job's `Caching` line) varied by session, from 2.7 to 55.7
minutes for the 170 shards without AUC Distracted Driver v2, and was 32.7 minutes for the
185 v0 shards in the B1 v0 run, and 3.9, 8.2 and 7.5 minutes in the three sessions so
far that listed the shards in threads (the cache of B1 v0's outputs and the two sessions
of E1 v0's distillation; [Measured for v0](#measured-for-v0)). Counted from the start of
the session, as those 60 minutes are, the first step of the seven `lora` and `distill`
sessions measured for v0 came 4.0 to 58.5 minutes in, so in the slowest sessions the
setup and a validation pass that starts at the limit (211 to 216 s for `distill`, 353 to
396 s for `lora`) would not fit in the 60 minutes. Both full `lora` runs ended in
one session without reaching the limit, and the `distill` probe projects the same for
distillation; a run that needs more than one session should set a lower limit, for example
`--set time_limit_minutes=640`. `max_memory_gb`, the peak memory allocated on the first
GPU, close to 16 means the batch size is too large for a T4, and a `loss_scale` that keeps
falling means fp16 overflows.

The probe ends with one validation pass over every val clip; its duration is the
`seconds` field of the `val` event in the same `log.jsonl`. Each epoch end of the real
run costs one such pass, and a pass that starts just before the limit must fit in those
60 minutes as well. Raising `eval_every` from 0 in the `lora*` and `distill*` configs
adds passes within each epoch, each costing as much again.

### Measured for v0

Measured on Kaggle from 2026-10-29 to 2026-10-31. The v0 data is the manifest with AUC
Distracted Driver v2 (sha256 starting `1d6175123358`): 183,355 clips in 185 shards, of
which 131,478 train, 18,525 val, 23,902 test, 2,032 holdout and 7,418 excluded
(Drive&Act) clips. The first six rows ran on the manifest without AUC v2
(`5ecfcd74c224`): 168,877 clips in 170 shards, with 118,840 train and 22,062 test clips
and the same val, holdout and excluded clips. The first five ran before AUC v2 was added
and include the first full B1 run, without AUC v2, which is not the release; the sixth,
the AUC v2 feature cache, loaded that manifest but read only the 15 AUC v2 shards (14,478
still images). The last six ran on the v0 manifest: the `distill` probe, then B1 v0
(the fine-tune with AUC v2), its evaluation and the cache of its outputs, and the two
sessions of E1 v0's distillation from those outputs, released as a research preview that
scores well below B1 ([E1: first distillation](evaluation.md#e1-first-distillation)).
B1 v0's heads,
fine-tune and outputs cache wrote to `checkpoints/heads-auc`, `checkpoints/b1-lora-auc`
and `data/features/b1-lora-auc` only because the first full run already held the
configs' `*-v0` paths: on the v0 manifest, the configs as shipped
(`heads.yaml`, `lora-kaggle.yaml`, `cache-b1.yaml`, `distill.yaml`) reproduce the B1 v0
chain. Durations come from each job's own logs (`log.jsonl`) and the kernel log's
timestamps; the code commit is the one recorded in each kernel's `PROVENANCE.json`.
The table's commits, and the others this section names, are from Bulwark's development
history before its first public release; that history is not published (see the model
card's [Versioning and provenance](../MODEL_CARD.md#versioning-and-provenance)). Setup,
the time from loading the manifest to the job's `train:` or `Caching` line, is given
apart from the job's own time because it varied by session (see below the table).

| Job | Hardware | Config | Code commit | Measured |
|---|---|---|---|---|
| `cache` (frozen features), without AUC v2 | Kaggle, 2x T4, fp16, `DataParallel` | `cache.yaml` | `bcd6aac` | all 170 shards in one session of about 70 min (kernel log timestamps), about 51 min of it encoding |
| `lora`, 200-step probe, without AUC v2 | Kaggle, 2x T4, fp16, `DataParallel` | `lora-kaggle.yaml` with `--set max_steps=200`, and `out` and `init` overridden (`init`: a heads checkpoint trained on the cached features); batch 16 x `grad_accum` 4, 64 clips per optimizer step | `94822e3` | `s_per_step` 2.16 over the 200 steps; 1,856 optimizer steps per epoch; one val pass (18,525 clips) 356 s; `max_memory_gb` 3.2; `loss_scale` 1024, with 6 skipped steps, all in the first 50; about 35 min from the start of the session to the first step |
| `lora`, first full run, without AUC v2 (not the release) | Kaggle, 2x T4, fp16, `DataParallel` | `lora-kaggle.yaml`, `init` `checkpoints/heads-v0/best.pt` | `0845aef` | 5,568 optimizer steps (3 epochs) at a median `s_per_step` of 2.30; three val passes of about 396 s each; `max_memory_gb` 3.2; 10 skipped steps, 7 of them in the first 450, ending at `loss_scale` 128; first step about 4 min into the session, which ended 3 h 57 min in (kernel log timestamps) |
| `cache` (B1 outputs for distillation), first full run | Kaggle, 2x T4, fp16, `DataParallel` | `cache-b1.yaml`, `init` the first full run's `best.pt` | `8cf5083` | all 170 shards (168,877 clips) in one session: 3,812 s (about 64 min) from the `Caching` line to the last shard; no clip left out as non-finite; 6.6 GB of kernel output; setup 36.6 min; the session ended 102 min in |
| `evaluate`, first full run's B1 | Kaggle, one T4 (`evaluate` does not use `DataParallel`), fp16 | `--model b1 --precision fp16`, checkpoint `checkpoints/b1-lora-v0/best.pt`, batch 16 | `8cf5083` | 42,619 val, test and holdout clips: the 28,710 video clips at about 15.6 clips per second (about 31 min) and the 13,909 MRL Eye images in about 80 s; 1,958 s (about 33 min) from loading the manifest to the written `report.html`, with no separate setup, since it reads the shards in order without listing them first; the session ended 34 min in |
| `cache` (frozen features), AUC v2 shards | Kaggle, 2x T4, fp16, `DataParallel` | `cache.yaml`, with only the AUC v2 shards attached and the features of the 170 shards carried over from the frozen-features kernel | `9f8e7bf` | 15 shards (14,478 one-frame clips) in 98 s; setup 1.3 min; the session ended 4 min in |
| `distill`, 600-step probe | Kaggle, one T4 (`--set multi_gpu=false`), fp16 | `distill.yaml` with `--set max_steps=600`, `--set log_every=25` and `out` overridden, on the v0 manifest but over the 170 shards without AUC v2 (`bulwark-auc` not attached); teacher the first full run's B1 outputs above, which have no logits for AUC v2's clips; batch 8 x `grad_accum` 4, 32 clips per optimizer step | `f157c05` | `s_per_step` 1.07 over the 600 steps (1.03 over steps 101 to 600); 3,713 optimizer steps per epoch; one val pass (18,525 clips) 211 s; `max_memory_gb` 5.2; `loss_scale` 1024, with 6 skipped steps, all in the first 25; setup 55.2 min; first step about 59 min into the session |
| `lora`, B1 v0 (with AUC v2) | Kaggle, 2x T4, fp16, `DataParallel` | `lora-kaggle.yaml` with `--set init=checkpoints/heads-auc/best.pt` (`heads.yaml` on the frozen features, AUC v2's included, commit `9f8e7bf`) and `--set out=checkpoints/b1-lora-auc`; batch 16 x `grad_accum` 4 | `f157c05` | 6,162 optimizer steps (3 epochs of 2,054) at a median `s_per_step` of 1.96; three val passes of 355.8, 353.2 and 355.4 s; `max_memory_gb` 3.2; 10 skipped steps, 7 of them in the first 150, ending at `loss_scale` 256; setup 32.7 min; first step about 34 min into the session, which ended 4 h 13 min in (15,165 s, kernel log timestamps) |
| `evaluate`, B1 v0 | Kaggle, one T4, fp16 | `--model b1 --precision fp16`, checkpoint `checkpoints/b1-lora-auc/best.pt`, batch 16 | `efe4bab` | 44,459 val, test and holdout clips (the 42,619 above and AUC v2's 1,840 test images): the 28,710 video clips in about 43 min, about 11 clips per second and uneven (800-clip stretches took 44 to 261 s), and the 15,749 images in about 3 min; 2,857 s (about 48 min) from loading the manifest to the written `report.html`; the session ended 49 min in |
| `cache` (B1 outputs for distillation), B1 v0 | Kaggle, 2x T4, fp16, `DataParallel` | `cache-b1.yaml` with `--set init=checkpoints/b1-lora-auc/best.pt` and `--set features=data/features/b1-lora-auc` | `7674655` | all 185 shards (183,355 clips) in one session: 3,323 s (about 55 min) from the `Caching` line to the last shard; no clip left out as non-finite; 7.1 GB of kernel output; setup 3.9 min, of which 202 s listing the 185 shards 16 at a time (`Indexed 185 shards in 202 s`); the session ended 61 min in |
| `distill`, E1 v0, session 1 | Kaggle, one T4, fp16 | `distill.yaml` with `--set teacher=data/features/b1-lora-auc` (B1 v0's outputs above) and `--set time_limit_minutes=210`, the GPU time left that week; batch 8 x `grad_accum` 4 | `5338f65` | 4,108 optimizer steps per epoch, 16,432 for the 4 epochs; setup 8.2 min, of which 297 s listing the 185 shards (`Indexed 185 shards in 297 s`), and training from 583 s into the session; val passes at the ends of epochs 1 to 3 (steps 4,108, 8,216 and 12,324) of 214, 158 and 160 s; stopped at the time limit after step 13,557 with `last.pt` saved; the session ended 13,194 s (3 h 40 min) in |
| `distill`, E1 v0, session 2 | Kaggle, one T4, fp16 | as session 1, without the time limit override (`time_limit_minutes: 660`), resuming from session 1's `last.pt` | `5338f65` | setup 7.5 min, of which 288 s listing the 185 shards (`Indexed 185 shards in 288 s`), and resumed at step 13,557 of 16,432 535 s into the session; the last val pass (step 16,432) 216 s; the session ended 3,450 s (58 min) in. Over both sessions: median `s_per_step` 0.91, `max_memory_gb` 5.2 |

The `lora` probe projected about 67 minutes per epoch (1,856 steps x 2.16 s) and about
3.6 hours for the 3 epochs and 3 validation passes. The first full run, without AUC v2,
stepped a little slower, 2.30 s per step, or about 71 minutes per epoch, and its training
and validation took 3.9 hours: one Kaggle session with `eval_every: 0`. The B1 v0 run,
whose epochs hold 131,478 clips, 12,638 more than the first run's (AUC v2's train
images; with dataset balancing an epoch draws 21,866 AUC v2 clips and fewer from each
other dataset), took 2,054 steps per epoch at 1.96 s, about 67 minutes per epoch, and its
training and validation took 3.6 hours, also in one session.

E1 v0's distillation, from B1 v0's outputs over all 185 shards, ran its 16,432 optimizer
steps at a median `s_per_step` of 0.91, faster than the probe's 1.03 to 1.07 s, with four
validation passes of 158 to 216 s. Training and validation took about 4.3 hours in all:
12,601 s in the first session (from its first training line to the time-limit stop) and
2,905 s in the second (from the resume to the last val line), by the kernel logs'
timestamps. It needed two sessions only because the first was limited to 210 minutes;
with the config's 660-minute limit, the run fits one session.

The `distill` probe projects a distillation from the first full run's B1 outputs at 4.25
to 4.4 hours for its 4 epochs (14,852 optimizer steps at 1.03 to 1.07 s) and about 14
minutes more for four epoch-end validation passes, which fits one Kaggle session; this is
a projection from the probe, not a measured run. Its manifest was the v0 one, but it
attached only the 170 shards without AUC Distracted Driver v2, and that teacher has no
logits for AUC v2's 14,478 rows, so it trained on the same 118,840 clips as the first
full run; it does not show how a job treats AUC v2 shards without teacher logits. With B1
v0's outputs as the teacher, AUC v2's 12,638 train images have logits too, so an epoch
holds 131,478 clips instead of 118,840; the probe did not measure that mix. The same
probe on two T4s with `DataParallel` (`max_steps=200`, commit `f157c05`, on the manifest
without AUC v2) logged `s_per_step` 14.02 after 50 steps and 13.86 after 100, with
`max_memory_gb` 2.7, and was killed (SIGKILL) for host memory about 116 min into the
session, before step 150 (see [Multiple GPUs](#how-runs-behave)).

Setup varied by session. From loading the manifest to the `train:` or `Caching` line, the
five sessions over the 170 shards without AUC v2 took 2.7 min (`lora` first full run),
32.7 (`lora` probe), 36.6 (first full run's B1 outputs), 55.2 (`distill` probe, one T4)
and 55.7 (`distill` probe, two T4s), and the B1 v0 `lora` session, over all 185 shards,
32.7.
Most of it is listing the shards, which reads one tar header per frame from Kaggle's
dataset mount; how fast those reads were differed between sessions. The AUC cache, with
15 small shards, took 1.3 min. On a CPU session (4 CPUs) listing Drive&Act shards (about
16,000 headers each), 16 threads at once took 2.35 s per shard against 3.79 s one after
another, 1.6 times faster; jobs now list 16 shards at a time (`index_shards`, from commit
`acb99ad`, later than every `cache`, `lora` and `distill` run above but the cache of B1
v0's outputs and E1 v0's distillation) and print `Indexed <n> shards in <s> s`. That
cache was the first Kaggle session measured with it: it listed all 185 shards in 202 s,
and its setup took 3.9 min. The two sessions of E1 v0's distillation listed the same
shards in 297 and 288 s, with setups of 8.2 and 7.5 min. Three sessions do not show how
much the threads shorten a slow session's setup, since setup varies by session.

`heads` and the speed-up from the second T4 for `lora` and `cache` are not measured yet.

## Commercial-licence track

`commercial_only: true` keeps only clips whose dataset licence permits commercial use,
in every Bulwark training stage (heads, fine-tune and distillation). The `*-clean.yaml`
configs run the whole chain this way, with their own output directories:

```bash
bulwark train cache --config configs/train/cache.yaml              # frozen encoder; no Bulwark data in it
bulwark train heads --config configs/train/heads-clean.yaml        # -> checkpoints/heads-clean
bulwark train lora --config configs/train/lora-kaggle-clean.yaml   # or lora-clean.yaml on an A100
bulwark train cache --config configs/train/cache-b1-clean.yaml     # reads b1-lora-clean/best.pt -> data/features/b1-lora-clean
bulwark train distill --config configs/train/distill-clean.yaml    # -> checkpoints/e1-clean
bulwark export --checkpoint checkpoints/e1-clean/best.pt --out checkpoints/bulwark-e1-clean.onnx
```

The frozen features are clean by construction: no Bulwark training is behind the
encoder. Every later training step refuses an init checkpoint, features directory or
teacher that does not itself record `commercial_only: true`, or that lists a dataset
without `commercial_ok`; this check has no override. The clean configs use the same
label mask as the v0 configs. `bulwark export` converts whichever checkpoint it is given
under the research licence and copies that checkpoint's provenance, `commercial_only`
included, into the ONNX metadata (`read_contract`). A licence that does not restrict
use to non-commercial research (`--weights-licence`, also in `bulwark release`) is stored
only for a checkpoint that `bulwark release-check --commercial` clears: a
`commercial_only` run on `commercial_ok` datasets cleared for publishing weights.

`release_only: true` is the matching filter for publishing: it keeps only datasets whose
registry entry has `weights_release_ok=True`, and every stage refuses an init, teacher or
features directory not itself produced with `release_only: true` (frozen features pass);
this check has no override either. Drive&Act, DMD, MRL Eye and AUC Distracted Driver v2
have `weights_release_ok=True` by the maintainers' decision of 2026-10-30, which is not a
statement by the datasets' owners (its `weights_release_basis`; see the
[model card](../MODEL_CARD.md#licence)), so a `release_only` run keeps the v0 training
sources.

Three things to know before relying on the commercial track:

- **No commercially licensed data is ingested yet.** All data used for v0 is licensed
  for research or non-commercial use only, so a clean run currently has nothing to train
  on and stops with an error. The registry ([datasets.md](datasets.md)) lists the sources
  marked commercial and which of them have adapters.
- **The filter covers Bulwark's data only, not upstream pretraining.** The V-JEPA 2.1
  ViT-B/16 encoder was pretrained by Meta; the V-JEPA 2 repository's configs for it list
  Kinetics-710, Something-Something v2, HowTo100M and ImageNet-1k. E1's backbone starts
  from timm's ImageNet-1k EfficientViT-B1 weights. Teacher labels (MediaPipe, RF-DETR,
  Qwen3-VL) and synthetic clips (Wan 2.2) come from models trained on their own data
  under their own terms (see [NOTICE](../NOTICE)); `commercial_only` does not filter
  those either. Their terms, including those of the ImageNet-1k weights, need checking
  before a commercial release; this note is a disclosure, not legal advice.
- **Publishing weights is a separate question.** Even for commercially licensed data,
  each dataset must be cleared for releasing a model trained on it (`weights_release_ok`
  in the registry, with the `weights_release_basis` it rests on; see
  [RELEASING.md](../RELEASING.md)).

## Running on Kaggle

The scripts in `infra/kaggle/` run any `bulwark train` job, or `bulwark evaluate`, as a
private background notebook on Kaggle's GPU machine (T4 x2), with sessions of up to 12
hours.

**Prerequisites.** A Kaggle account with an API token set up (see Kaggle's API
documentation); the Kaggle CLI from the `ops` extra (add `--extra ops` to the
`uv sync` command in the README, since `uv sync` removes extras it is not given); and
uv 0.8.17, the version that wrote `uv.lock`, because each job packs the code with
`scripts/pack_code.sh`, which refuses a `uv.lock` that does not match `pyproject.toml`.
Datasets and kernels are created private under your own account.

**Which data may be uploaded.** Uploading shards keeps a copy of a dataset on a
third-party cloud, which not every dataset's terms allow, so the registry clears each
provider separately. `push_data.sh` (through `infra/kaggle/stage_data.py --storage
kaggle`) uploads only datasets whose registry entry has `kaggle_storage_ok=True`: today
Drive&Act, DMD, MRL Eye and AUC Distracted Driver v2, approved by the maintainers on
2026-10-29 and recorded in their registry terms (private Kaggle datasets, for training on
Kaggle's GPUs only; for AUC, without the MI-AUC group's authorization, as its terms
record). The Zenodo set is marked
`False` and must never be uploaded. Any other dataset must be named
in `ALLOW_UNCLEARED`, a deliberate decision for that one dataset made after reading its
terms; otherwise the upload stops with an error that names it. The same applies to the
manifest: rows of uncleared datasets stop its upload, unless `DROP_UNCLEARED=1` leaves
them out (jobs then train without those datasets, and record the hash of that smaller
manifest). Azure Blob has its own flag, `azure_storage_ok`, which no dataset has yet (see
[Running on Azure](#running-on-azure-optional)).

**Uploading data.** Each call creates or replaces one private Kaggle dataset:

```bash
export KAGGLE_USER=<your-kaggle-username>
# Shards. A new version replaces the dataset's whole contents, so SLUG has no default:
# use one slug per source. SHARD_GLOB is required too (one or more globs, or empty).
SLUG=bulwark-drive-and-act SHARD_GLOB='drive_and_act-*.tar' infra/kaggle/push_data.sh
SLUG=bulwark-dmd SHARD_GLOB='dmd-*.tar' infra/kaggle/push_data.sh
SLUG=bulwark-mrl SHARD_GLOB='mrl_eye-*.tar' infra/kaggle/push_data.sh
SLUG=bulwark-auc SHARD_GLOB='auc_v2-*.tar' infra/kaggle/push_data.sh
# The manifest and starting checkpoints go only into bulwark-meta, which every job reads.
# INCLUDE takes paths under checkpoints/ and stages only best.pt, final.pt,
# provenance.json and exported *.onnx graphs from them, never last.pt.
DROP_UNCLEARED=1 SLUG=bulwark-meta SHARD_GLOB= INCLUDE=checkpoints/heads-v0 infra/kaggle/push_data.sh
```

`DROP_UNCLEARED=1` is needed on the `bulwark-meta` call while the manifest holds rows of
a dataset that is not cleared: the Zenodo set (split `excluded`) whenever it was
ingested, and UTA-RLDD (the holdout) once it is. Kaggle jobs then neither train on nor
score those datasets (score the UTA-RLDD holdout locally). The staged manifest records
the rows it left out per dataset in its parquet metadata, so a remote evaluation report
lists UTA-RLDD as left out of the upload rather than as not ingested. Without it,
`push_data.sh` refuses the manifest. Add `ALLOW_UNCLEARED=<dataset>` to upload a dataset
that is not cleared for Kaggle in the registry, on both its shard upload and the
manifest upload.

**Dataset cards.** For the five v0 slugs, `bulwark-auc` among them, `push_data.sh` writes
the card from `infra/kaggle/dataset_cards.py` (title, description with citation and terms, contact page,
sources, update frequency, a description of every staged file and of every manifest
column) and the cover from `infra/kaggle/covers/` (rendered by `covers.py`; no dataset
frames), then runs `dataset_cards.py --apply`, because Kaggle's create and version calls
ignore the sources, update frequency and cover. To refresh the cards without uploading
files, run `KAGGLE_USER=<you> python infra/kaggle/dataset_cards.py --apply [--slug <slug>]`;
it describes the files each dataset lists on Kaggle, and stops on a file it cannot describe.

**Running jobs.**

```bash
D="bulwark-drive-and-act bulwark-dmd bulwark-mrl bulwark-auc bulwark-meta"
DATASETS="$D" infra/kaggle/push_job.sh lora configs/train/lora-kaggle.yaml
kaggle kernels status "$KAGGLE_USER/bulwark-lora"
kaggle kernels output "$KAGGLE_USER/bulwark-lora" -p runs/kaggle-lora
```

- `DATASETS` names the Kaggle datasets to attach. Its default is the five v0 slugs, the
  same as `$D` above; set it when you pushed other slugs.
- A job trains on and scores only the clips whose shards it attaches; manifest clips
  without one are not an error (the training jobs log how many there are, and evaluation
  reports them per dataset). Without `bulwark-auc`, a job on the v0 manifest leaves out
  AUC Distracted Driver v2's clips.
- The code travels inside each job, so a code change needs no data upload.
- Jobs read the manifest only from `MANIFEST_DATASET` (default `bulwark-meta`), and
  `push_job.sh` refuses a job whose `DATASETS` leave it out. Label changes need only that
  small dataset re-uploaded.
- A job brings in only the checkpoint and feature directories it reads or writes
  (`infra/job_inputs.py`, from the config and any `--set key=value` or `--set=key=value`
  overrides), and stops before it starts when an input it reads is not attached.
- Each kernel is named `NAME` (default `bulwark-<job>`). Extra arguments are passed to
  the job, for example the 200-step probe from [Before a long run](#before-a-long-run):
  `NAME=bulwark-probe-lora ... push_job.sh lora configs/train/lora-kaggle.yaml --set max_steps=200 --set out=checkpoints/probe-lora`.
- A kernel's output holds only the directory its job writes; inputs it read from other
  kernels are not copied. `PREVIOUS_KERNELS` attaches the outputs of earlier kernels, so a
  job longer than one session continues as a new kernel that attaches the previous
  session's output (and resumes from its `last.pt`) plus any kernel its inputs came from:

  ```bash
  NAME=bulwark-lora-2 PREVIOUS_KERNELS=bulwark-lora DATASETS="$D" \
    infra/kaggle/push_job.sh lora configs/train/lora-kaggle.yaml
  ```

- For B1 outputs, give the cache job its own kernel name and attach the kernel that
  finished the fine-tune, which carries `checkpoints/b1-lora-v0` (`cache-b1.yaml` reads
  its `best.pt`). Distillation then reads the cache kernel's output, on one GPU; the
  probe with the first full run's B1 outputs projects the run within one session. A
  second distillation session, if one is needed, attaches both:

  ```bash
  NAME=bulwark-cache-b1 PREVIOUS_KERNELS=bulwark-lora-2 DATASETS="$D" \
    infra/kaggle/push_job.sh cache configs/train/cache-b1.yaml
  NAME=bulwark-distill PREVIOUS_KERNELS=bulwark-cache-b1 DATASETS="$D" \
    infra/kaggle/push_job.sh distill configs/train/distill.yaml
  NAME=bulwark-distill-2 PREVIOUS_KERNELS="bulwark-distill bulwark-cache-b1" DATASETS="$D" \
    infra/kaggle/push_job.sh distill configs/train/distill.yaml
  ```

- `evaluate` runs `bulwark evaluate --checkpoint <checkpoint> [options]` on the GPU. The
  checkpoint must be under `checkpoints/` in an attached input. Name the model kind with
  `--model`: `b1`, `e1` and `onnx` read the shards, and `heads` and `logits` read
  `--features` (default `data/features/frozen`) from an attached input. `--model auto` (the
  default) is refused for a `.pt` checkpoint unless `--features` is given (an `.onnx`
  checkpoint counts as `--model onnx`), because the paths to bring in are worked out
  before the checkpoint can be opened. `--reference` is refused, and so are `--shards`
  and `--manifest` other than the defaults; compare models locally with
  `bulwark report --predictions <out>/predictions --reference <dir>`. The report goes to
  `runs/eval` in the kernel's output unless `--out` says otherwise:

  ```bash
  NAME=bulwark-eval-b1 PREVIOUS_KERNELS=bulwark-lora-2 DATASETS="$D" \
    infra/kaggle/push_job.sh evaluate checkpoints/b1-lora-v0/best.pt --model b1 --precision fp16
  ```

  An exported graph is evaluated the same way once it is in an attached input: stage it
  into `bulwark-meta` with `INCLUDE=checkpoints/bulwark-e1.onnx` (next to the other
  `INCLUDE` paths, since a new version replaces the whole dataset), then run
  `push_job.sh evaluate checkpoints/bulwark-e1.onnx --model onnx`.

- Each kernel's output includes `run-inputs.json` and `environment.txt`, which record
  the inputs it used and its library versions.

## Running on Azure (optional)

`infra/azure/` runs a job on a single on-demand A100 VM (NC24ads_A100_v4) that syncs its
outputs to Blob storage and deletes itself when the job ends, fails or runs out of time.

**Prerequisites.** An Azure subscription with 24 vCPUs of NCADS_A100_v4 quota
(`infra/azure/quota.sh` shows it); the Azure CLI, signed in with `az login`; and azcopy
v10. Launching a job assigns roles to the VM's managed identity (so it can delete
itself), which needs Owner or User Access Administrator on the resource group.
`BULWARK_CAP_USD`, the total spending cap in US dollars, has no default: every job is
refused until it is set in the environment or in `infra/azure/env.local.sh`.

```bash
export BULWARK_CAP_USD=<your-cap>                             # or set it in infra/azure/env.local.sh
BULWARK_ALERT_EMAIL=<your-address> infra/azure/setup.sh       # once: resource group, storage, budget alert
DROP_UNCLEARED=1 infra/azure/upload.sh data/shards data/manifest.parquet checkpoints/heads-v0
infra/azure/run_job.sh lora configs/train/lora.yaml --hours 4
infra/azure/run_job.sh evaluate checkpoints/b1-lora-v0/best.pt --hours 2 --model b1 --precision bf16
infra/azure/status.sh <vm-name>
infra/azure/spend.sh status
```

`setup.sh` needs `BULWARK_ALERT_EMAIL`, the address that receives the budget alerts, and
`BULWARK_CAP_USD`, and records the date the cap starts counting from in
`infra/azure/.spend_start`. The other settings are described in `infra/azure/env.sh`;
values for one machine can go in `infra/azure/env.local.sh`, and `BULWARK_ENV_LOCAL`
names another such file. Git ignores `.spend_start` and `env.local.sh`.

`upload.sh` applies the registry's Azure Blob clearance, `azure_storage_ok`, as
`push_data.sh` applies the Kaggle one: the storage account is private and in your own
subscription, but it is still a third party's cloud, and clearance for Kaggle does not
carry over. No dataset is cleared for Azure Blob yet; the maintainers decide it when an
Azure job is needed, and record the decision in the registry (`azure_storage_ok=True`,
dated in `terms`). Until then the example above stops before it uploads anything.
Shards and feature files of datasets that are not `azure_storage_ok` stay local, and a
directory upload such as `data/shards` leaves them out, unless `ALLOW_UNCLEARED` names
the dataset. A manifest with rows of such a dataset is refused unless `ALLOW_UNCLEARED`
names it or `DROP_UNCLEARED=1` uploads it without those rows, and a manifest that would
keep no rows is never uploaded. Even with datasets cleared for Blob, a manifest that
holds the Zenodo clips (split `excluded`, when that set was ingested) or UTA-RLDD
still needs one of the two, which is why the example above sets `DROP_UNCLEARED=1`.
Azure jobs, like Kaggle jobs, then neither train on nor score the datasets left out: a
remote `evaluate` has no UTA-RLDD clips, and its report lists the holdout as left out of
the upload (the uploaded manifest records the rows it left out). Score the UTA-RLDD
holdout locally. `upload.sh` never uploads unfinished files (`*.partial`, `*.tmp`,
`*.tmp.npz`) or the `*.shards.json` records, and takes only `data/manifest.parquet`,
`data/shards`, `data/features` and `checkpoints/<dir>` (or files under them).
`data/labels` is refused: teacher labels reach jobs through the manifest.

The VM fetches the manifest, the shards and only the checkpoint and feature paths the job
reads or writes (`infra/job_inputs.py`). `evaluate` takes the same options and
restrictions as on Kaggle; its checkpoint must be in Blob under `checkpoints/`, and the
report goes to `runs/<vm-name>` in Blob unless `--out` says otherwise. `--hours=N` works
as well as `--hours N`.

`infra/azure/spend.py` is a hard cap on top of Azure's budget alerts, which cannot stop
spending:

- Each job reserves its worst case before the VM is created: (`--hours` plus a 30-minute
  finish window) x (the hourly VM price plus disk and IP). A job that would take
  reservations, plus a reserve kept back for storage and billing lag
  (`BULWARK_RESERVE_USD`), past `BULWARK_CAP_USD` is refused.
- The VM deletes itself when the job ends, fails or reaches its deadline, and an Azure
  auto-shutdown 30 minutes after the deadline backs that up.
- `spend.sh reconcile` closes finished jobs at their measured runtime, records any
  overrun, and deletes public IPs, NICs and disks left behind once no job is running.
- Running job VMs carry their reservation as tags, and any VM without a ledger entry is
  counted at that reservation. The cap therefore holds across clones of the repository
  only while those VMs exist; the ledger itself (`infra/azure/ledger.jsonl`) is per
  machine.

Before relying on this path, run one short job (`--hours 0.25`) and check that its
checkpoints reach Blob storage, that the VM deletes itself, and that `spend.sh reconcile`
closes its reservation. Each job also uploads `jobs/<name>.environment.txt`, its library
versions.

## Removing data from cloud copies

A subject removal is applied locally with `bulwark purge` and then carried over to the
cloud copies with `scripts/purge_cloud_copies.sh <dataset>`. In Azure Blob, for a dataset
cleared for Blob (`azure_storage_ok`), it uploads the purged local shards over their
copies before deleting the dataset's shard blobs that no local shard replaces; for a
dataset that is not cleared for Blob it uploads no shard and deletes every shard blob of
it. It deletes the dataset's feature blobs for the cache job to recompute, replaces the
manifest after the same clearance check as `upload.sh`, and deletes the teacher-label
blobs under `data/labels/` (jobs read labels through the manifest). When no dataset in
the local manifest is cleared for Blob, as with the registry today, it refuses to touch
a manifest blob unless `DROP_UNCLEARED=1`, which deletes that blob instead.
`KAGGLE_SLUGS="<slug> ..."` deletes the Kaggle datasets that held the dataset's shards or
the manifest, to be created again with `push_data.sh` (with `SHARD_GLOB` listing every
source the slug held); the script prints those steps either way. Kernel outputs that
hold copies (a cache job's features) are deleted with `kaggle kernels delete`.
`DRY_RUN=1` shows what would happen. Each purge appends a row to `docs/purge-log.md` with
the date, the dataset, the number of subjects and what was done, without subject ids. The
[data card](data-card.md#personal-data-and-removal-requests) gives the whole procedure.

## Speed and latency

The run times measured for v0, with their hardware, settings and code commit, are in
[Measured for v0](#measured-for-v0): the frozen feature cache, the B1 fine-tunes with and
without AUC Distracted Driver v2 and their evaluations, the caches of both runs' B1
outputs, a 600-step distillation probe on one T4, and the full distillation of E1 v0 on
one T4 in two sessions. The heads run and the speed-up from a second T4 for `lora` and
`cache` are not measured yet;
estimate a run on other data or hardware with the probe in
[Before a long run](#before-a-long-run).

Before a long GPU run, `scripts/overfit_batch.py --video <video>` checks the pipeline end
to end: it cuts 8 clips from one video, labels them with the face teacher and trains B1
on that single batch for 60 steps (`--clips`, `--steps`). This is a sanity test, not a
quality result: if the loss on a single batch does not fall sharply, something between
pixels and loss is broken.

**E1 latency.** `bulwark export` measures the exported graph with onnxruntime on the CPU:
100 timed calls after 10 warm-up calls, on one thread and on all cores, each call one
window at the graph's input shape (16 frames of 256 x 256 by default), with the median
(p50) and 95th percentile (p95) in milliseconds. It writes the results, with the
onnxruntime version, the CPU name, the input shape and the graph's sha256, to
`<graph>.bench.json`. These are model-only figures: decoding and letterboxing frames are
not included, and each call encodes every frame of the window, so they hold for
non-overlapping windows; a sliding window costs one call per hop. Quote latency only
from a `.bench.json` measured with no other heavy job running, with that file's hardware
and versions and the machine's load just before the measurement.

E1 v0's latency is quoted, in its card and in the model card, from the
`<graph>.bench.json` of each released graph: p50 and p95 on one thread and on all cores,
measured with the benchmark of `bulwark export` (`bulwark.export.onnx.benchmark`) on
macOS 26.5.1 on mains power, with no other Bulwark job running and the usual desktop apps
open. Just before the measurement, the 1-minute load average was 4.26 and the CPU 88%
idle; just after it, the 1-minute load average was 24.41
([RELEASING.md](../RELEASING.md#launch-order)).
