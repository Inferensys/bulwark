# Evaluation

How Bulwark models are evaluated, and what a reported number does and does not mean.
The evaluation code is in `bulwark/eval/`; `bulwark evaluate --help` lists its options.

## Results

B1 v0 and E1 v0 are evaluated and released. E1 v0, distilled from B1 v0, is released as a
research preview that scores well below B1: its test results are under
[E1 v0 test results](#e1-v0-test-results), and its validation scores and how it was trained
under [E1: first distillation](#e1-first-distillation). B1 v0 is the
`best.pt` of the LoRA fine-tune on the four v0 datasets (Drive&Act, the DMD face camera,
MRL Eye and AUC Distracted Driver v2; checkpoint sha256 starting `de184992f7ed`), scored
by `bulwark evaluate --model b1 --precision fp16` (commit `efe4bab`) on the v0 manifest
(sha256 starting `1d6175123358`). That commit, like `5338f65` and `fafa16d` below, is from
Bulwark's development history before its first public release; that history is not published (see
the model card's [Versioning and provenance](../MODEL_CARD.md#versioning-and-provenance)).
The numbers below are copied from that run's `metrics.json`, and those of the first full
run from the `metrics.json` of its re-scored predictions, rounded to 3 decimals, except the
per-class and per-subject F1 and the confusion counts under
[Choosing E1's teacher](#choosing-e1s-teacher), which are computed from the two models'
saved predictions. The generated `report.html` and `metrics.json`
are published with the weights, in `evaluation/b1/` of the
[B1 repository on Hugging Face](https://huggingface.co/Inferensys/bulwark-b1-driver-monitoring),
whose files open once access is granted. Runs with `--limit`, which score a few clips
per split, are pipeline checks: the report marks them as such, and their numbers are
never results.

**Every result here is indicative.** Each dataset's test split holds 3 subjects (AUC
Distracted Driver v2's, one driver group): fewer than the 5 below which the report marks
a result indicative, and than the 10 a subject-level bootstrap interval needs, so there
are no intervals. The numbers describe these few people and cameras, not general
performance.

Each row gives the report's headline metric: AUROC for binary labels, macro-F1 for
activity, and mean absolute error (MAE, lower is better, in the label's normalised range)
for eye openness and PERCLOS. The second score is the first full run, without AUC v2,
which is not the release (see [Choosing E1's teacher](#choosing-e1s-teacher)). The
baseline is the report's constant predictor fitted on the dataset's training labels
(AUROC 0.5 for every binary label). n counts clips, except for Drive&Act on test, where it
counts events (the moments its cameras record together, scored once with the views'
probabilities averaged); n and subjects are the same for both models.

Test:

| Dataset | Label | Metric | B1 v0 | Without AUC v2 | Baseline | n | Subjects |
|---|---|---|---:|---:|---:|---:|---:|
| Drive&Act | activity | macro-F1 | 0.772 | 0.781 | 0.042 | 2,071 | 3 |
| Drive&Act | phone | AUROC | 0.996 | 0.997 | 0.5 | 2,036 | 3 |
| Drive&Act | phone_at_ear | AUROC | 0.999 | 0.998 | 0.5 | 2,036 | 3 |
| Drive&Act | food | AUROC | 0.981 | 0.983 | 0.5 | 2,029 | 3 |
| Drive&Act | drink | AUROC | 0.996 | 0.998 | 0.5 | 2,040 | 3 |
| DMD | activity | macro-F1 | 0.586 | 0.701 | 0.072 | 2,769 | 3 |
| DMD | eyes_closed | AUROC | 0.902 | 0.894 | 0.5 | 721 | 3 |
| DMD | perclos | MAE | 0.089 | 0.091 | 0.157 | 721 | 3 |
| DMD | blink | AUROC | 0.748 | 0.795 | 0.5 | 1,541 | 3 |
| DMD | yawn | AUROC | 0.904 | 0.912 | 0.5 | 724 | 3 |
| DMD | phone | AUROC | 0.981 | 0.987 | 0.5 | 2,520 | 3 |
| DMD | phone_at_ear | AUROC | 1.000 | 1.000 | 0.5 | 2,520 | 3 |
| DMD | drink | AUROC | 0.998 | 0.997 | 0.5 | 2,602 | 3 |
| MRL Eye | eyes_closed | AUROC | 0.998 | 0.997 | 0.5 | 5,514 | 3 |
| MRL Eye | eye_openness | MAE | 0.033 | 0.029 | 0.489 | 5,514 | 3 |
| MRL Eye | eyeglasses | AUROC | 1.000 | 1.000 | 0.5 | 5,514 | 3 |
| AUC Distracted Driver v2 | activity | macro-F1 | 0.794 | not scored | 0.031 | 1,840 | 1 group |
| AUC Distracted Driver v2 | phone_at_ear | AUROC | 0.999 | not scored | 0.5 | 757 | 1 group |

AUC v2's test images give phone and drink positives only, so those two have no AUROC;
B1 v0 was trained with both masked on images, and the report warns that their image
scores are of untrained outputs. AUC v2 has no val clips, so its images are scored without
a val-fitted threshold or temperature (the report warns); neither changes AUROC or
macro-F1. The first full run had no AUC v2 rows to score.

Holdout, Drive&Act's inner-mirror camera (a view never trained on, of the 3 test
drivers; per clip). The UTA-RLDD holdout is not ingested, so it has no scores.

| Label | Metric | B1 v0 | Without AUC v2 | Baseline | n | Subjects |
|---|---|---:|---:|---:|---:|---:|
| activity | macro-F1 | 0.290 | 0.286 | 0.043 | 2,032 | 3 |
| phone | AUROC | 0.808 | 0.773 | 0.5 | 1,997 | 3 |
| phone_at_ear | AUROC | 0.845 | 0.716 | 0.5 | 1,997 | 3 |
| food | AUROC | 0.911 | 0.851 | 0.5 | 1,992 | 3 |
| drink | AUROC | 0.911 | 0.929 | 0.5 | 2,001 | 3 |

### Choosing E1's teacher

Two B1 fine-tunes finished with the same configs (`heads.yaml`, then
`lora-kaggle.yaml`): the first full run, on Drive&Act, the DMD face camera and MRL Eye
(the manifest without AUC Distracted Driver v2, 168,877 clips), and B1 v0, on those and
AUC v2 (183,355 clips). The first run's predictions were scored again with the code
above (commit `efe4bab`, thresholds and temperatures per media type) against its own
manifest, which gives the column "Without AUC v2". The maintainers chose B1 v0, the model
with AUC v2, as the v0 B1 and as the teacher for E1's distillation:

- **The unseen camera.** On the cross-view holdout, the generalisation check the project
  fixed in advance, B1 v0 scores higher on `phone_at_ear` (0.845 against 0.716),
  `food` (0.911 against 0.851) and `phone` (0.808 against 0.773), about the same on
  activity (macro-F1 0.290 against 0.286) and a little lower on `drink` (0.911 against
  0.929).
- **The student's data.** E1 trains on the v0 manifest, AUC v2's images included, which
  only B1 v0 has learned (the first run's cached outputs have no logits for them).
- **In-domain test.** Every test AUROC but DMD's `blink` is within 0.01 between the two
  models.

The cost, on DMD's test split:

- Activity macro-F1 falls from 0.701 to 0.586. It falls for each of the 3 DMD test
  subjects (0.748 to 0.592, 0.707 to 0.648 and 0.608 to 0.449), and the classes that lose
  most are `talking` and `grooming`. The most common new errors are `attentive` clips
  predicted as `reaching` (174 clips, against 73 without AUC v2) and `phone_call` clips
  predicted as `grooming` (49, against 5).
- `blink` AUROC falls from 0.795 to 0.748.

DMD activity F1 per class on test, from the argmax of each model's saved activity logits
over the test clips with ground truth (macro-F1 over the classes present in labels or
predictions, as in the report; the macro means match `metrics.json`):

| Class | Test clips | B1 v0 | Without AUC v2 |
|---|---:|---:|---:|
| adjusting_controls | 100 | 0.000 | 0.036 |
| attentive | 1,116 | 0.777 | 0.825 |
| drinking | 94 | 0.886 | 0.933 |
| grooming | 92 | 0.615 | 0.843 |
| phone_call | 670 | 0.939 | 0.977 |
| reaching | 357 | 0.596 | 0.666 |
| talking | 77 | 0.215 | 0.609 |
| texting | 263 | 0.656 | 0.717 |

Two further notes:

- **Checkpoint selection.** On val, B1 v0's `best.pt` scored 0.9328 (`gt_mean_headline`
  over 13 labels) against 0.9394 for the first run's. The val split has no AUC v2 clips
  (AUC v2 has no val split: its drivers form one group per side), so this score does not
  measure what AUC v2 added.
- **DMD `eyes_closed` F1.** Its AUROC is 0.902 against 0.894, but its F1 at the
  val-fitted video threshold is 0.101 against 0.316. That threshold (0.232 for B1 v0,
  0.086 for the first run) rests on 360 video val clips with 30 positives from 2
  subjects, so the F1 says more about that fit than about the model; check the threshold
  on your own data before using it.

All of this rests on 3 test subjects per dataset and no intervals, so none of these
differences is shown to hold beyond these people and cameras. The first run's B1 and its
cached outputs are kept, so an E1 distilled from it remains possible.

### E1: first distillation

E1 v0 is the first distillation. It ran the shipped `distill.yaml` on the v0 manifest with
B1 v0's cached outputs as the teacher (`--set teacher=data/features/b1-lora-auc`), for its
4 epochs of 4,108 optimizer steps, on one T4 in two Kaggle sessions (code commit
`5338f65`; [measured for v0](training.md#measured-for-v0)). Its `best.pt` is the last of
its four epoch-end validation passes, at step 16,432, with a selection score of 0.7846
(`gt_mean_headline` over 13 labels), against 0.9328 for B1 v0's `best.pt` at step 6,162.

The table compares the two per label. These are validation scores, the ones each run used
to choose its `best.pt`; they are not test results (those are under [E1 v0 test
results](#e1-v0-test-results)). They come from the last `val` event of each run's
`log.jsonl` (run directories `checkpoints/e1-v0` and `checkpoints/b1-lora-auc`), rounded
to 2 decimals. Each label is scored as in the selection score: AUROC for yes/no labels,
macro-F1 for activity and camera type (`modality`), and 1 - MAE for eye openness and
PERCLOS, so higher is better everywhere. Both runs scored the same val clips, and the val
split is small: the last two columns give the clips and people each label rests on.

| Label | Metric | E1 v0 | B1 v0 | Val clips | Val people |
|---|---|---:|---:|---:|---:|
| `eye_openness` | 1 - MAE | 0.96 | 0.97 | 8,395 | 3 |
| `eyes_closed` | AUROC | 0.98 | 0.99 | 8,755 | 5 |
| `perclos` | 1 - MAE | 0.86 | 0.86 | 360 | 2 |
| `blink` | AUROC | 0.50 | 0.81 | 713 | 2 |
| `yawn` | AUROC | 0.77 | 0.97 | 360 | 2 |
| `phone` | AUROC | 0.72 | 0.98 | 9,050 | 4 |
| `phone_at_ear` | AUROC | 0.75 | 0.99 | 9,050 | 4 |
| `food` | AUROC | 0.73 | 0.96 | 7,726 | 2 |
| `drink` | AUROC | 0.83 | 0.99 | 9,225 | 4 |
| `eyeglasses` | AUROC | 1.00 | 1.00 | 8,395 | 3 |
| `activity` | macro-F1 | 0.20 | 0.65 | 9,411 | 4 |
| `modality` | macro-F1 | 1.00 | 1.00 | 18,525 | 7 |
| `low_light` | AUROC | 0.89 | 0.96 | 8,395 | 3 |
| **Selection score (mean)** | | **0.78** | **0.93** | | |

E1 v0 is close to B1 v0 on eye openness, eyes closed, PERCLOS, eyeglasses and camera
type, but far below it on blink (0.50, no better than chance), yawn, phone use, eating,
drinking and activity. Checks after training found that it scores much better on the
people it was trained on than on the validation people: it generalises poorly to new
people. Re-estimating its BatchNorm statistics did not help. E1 will be retrained with a
different recipe. After seeing E1 v0's test results (below), the maintainers decided to
release E1 v0 as it is, as a research preview: for trying CPU and ONNX deployment and
Bulwark's output format, not for its accuracy. Its card says so on its first screen.

### E1 v0 test results

E1 v0 is the `best.pt` above (sha256 starting `065b89dc5eda`), scored by `bulwark evaluate
--model e1` (commit `fafa16d`) on the same manifest, split policy and clips as B1 v0, with
B1 v0's predictions as the reference. The numbers are copied from that run's
`metrics.json` and from B1 v0's, rounded to 3 decimals; n and subjects are the same for
both models, and the baseline is that of the tables above. The generated `report.html`
and `metrics.json` are published with the graphs, in `evaluation/e1/` of the
[E1 repository on Hugging Face](https://huggingface.co/Inferensys/bulwark-e1-driver-monitoring-onnx),
with B1 v0's in `evaluation/b1/`. Every result is indicative, for the reasons above.

Test:

| Dataset | Label | Metric | E1 v0 | B1 v0 | Baseline | n | Subjects |
|---|---|---|---:|---:|---:|---:|---:|
| Drive&Act | activity | macro-F1 | 0.227 | 0.772 | 0.042 | 2,071 | 3 |
| Drive&Act | phone | AUROC | 0.850 | 0.996 | 0.5 | 2,036 | 3 |
| Drive&Act | phone_at_ear | AUROC | 0.938 | 0.999 | 0.5 | 2,036 | 3 |
| Drive&Act | food | AUROC | 0.763 | 0.981 | 0.5 | 2,029 | 3 |
| Drive&Act | drink | AUROC | 0.909 | 0.996 | 0.5 | 2,040 | 3 |
| DMD | activity | macro-F1 | 0.283 | 0.586 | 0.072 | 2,769 | 3 |
| DMD | eyes_closed | AUROC | 0.497 | 0.902 | 0.5 | 721 | 3 |
| DMD | perclos | MAE | 0.131 | 0.089 | 0.157 | 721 | 3 |
| DMD | blink | AUROC | 0.589 | 0.748 | 0.5 | 1,541 | 3 |
| DMD | yawn | AUROC | 0.753 | 0.904 | 0.5 | 724 | 3 |
| DMD | phone | AUROC | 0.905 | 0.981 | 0.5 | 2,520 | 3 |
| DMD | phone_at_ear | AUROC | 0.990 | 1.000 | 0.5 | 2,520 | 3 |
| DMD | drink | AUROC | 0.841 | 0.998 | 0.5 | 2,602 | 3 |
| MRL Eye | eyes_closed | AUROC | 0.998 | 0.998 | 0.5 | 5,514 | 3 |
| MRL Eye | eye_openness | MAE | 0.017 | 0.033 | 0.489 | 5,514 | 3 |
| MRL Eye | eyeglasses | AUROC | 1.000 | 1.000 | 0.5 | 5,514 | 3 |
| AUC Distracted Driver v2 | activity | macro-F1 | 0.284 | 0.794 | 0.031 | 1,840 | 1 group |
| AUC Distracted Driver v2 | phone_at_ear | AUROC | 0.798 | 0.999 | 0.5 | 757 | 1 group |

Holdout, Drive&Act's inner-mirror camera (per clip):

| Label | Metric | E1 v0 | B1 v0 | Baseline | n | Subjects |
|---|---|---:|---:|---:|---:|---:|
| activity | macro-F1 | 0.085 | 0.290 | 0.043 | 2,032 | 3 |
| phone | AUROC | 0.547 | 0.808 | 0.5 | 1,997 | 3 |
| phone_at_ear | AUROC | 0.501 | 0.845 | 0.5 | 1,997 | 3 |
| food | AUROC | 0.495 | 0.911 | 0.5 | 1,992 | 3 |
| drink | AUROC | 0.404 | 0.911 | 0.5 | 2,001 | 3 |

What these numbers show:

- **Eyes closed.** E1 v0 does not detect closed eyes in driver video: its DMD AUROC is
  0.497, chance level, against 0.902 for B1 v0. On MRL Eye's close-up still images of
  eyes it matches B1 v0 (0.998), and its eye openness error is lower (MAE 0.017 against
  0.033).
- **Blink, yawn and PERCLOS.** All are below B1 v0 on DMD, and blink is close to chance
  (0.589).
- **Phone, food and drink.** On the camera views it was trained on, phone use scores
  0.850 on Drive&Act and 0.905 on DMD, below B1 v0 (0.996 and 0.981). The Drive&Act
  figure is per event, the mean of several cameras' probabilities; with each clip scored
  on its own, from one camera, it is 0.729 (B1 v0: 0.942), and phone at the ear, food and
  drink are 0.795, 0.665 and 0.789 (`per_clip` in `metrics.json`). On the inner-mirror
  camera, which training never used and which is scored per clip, phone, phone at the
  ear, food and drink fall to chance or below (0.404 to 0.547), where B1 v0 keeps 0.808 to
  0.911.
- **Activity.** Macro-F1 is 0.227 to 0.284 on test, against 0.586 to 0.794 for B1 v0,
  and 0.085 on the inner-mirror camera.

**The ONNX graphs.** The two released graphs were scored on the same splits by `bulwark
evaluate --model onnx` on CPU, with the checkpoint's predictions above as the reference
(`--reference`), into `runs/eval/e1-v0-onnx-video` and `runs/eval/e1-v0-onnx-image`. A
graph scores only the clips of its own frame count. The 16-frame video graph scored the
28,710 video clips (DMD and Drive&Act: 10,130 val, 16,548 test and 2,032 holdout) and
skipped the 15,749 image clips. The 1-frame image graph scored those 15,749 image clips
(MRL Eye and AUC Distracted Driver v2: 8,395 val and 7,354 test) and skipped the video
clips. Both graphs agree with the checkpoint on every clip they scored:

| Graph | Clips scored | Largest logit difference | Largest probability difference, yes/no outputs |
|---|---:|---:|---:|
| `bulwark-e1-video.onnx` | 28,710 | 2.3e-03 | 2.9e-04 |
| `bulwark-e1-image.onnx` | 15,749 | 2.6e-04 | 5.4e-05 |

The differences are the largest over every clip and every label's logits, from the two
runs' `predictions/`. In the `reference` entries of the graphs' `metrics.json` (test and
holdout), each yes/no output scored there is on the same side of 0.5 as the checkpoint's
on every clip, and the top activity class is the same on every clip. Every score in the two
tables above differs from the checkpoint's by at most 6.2e-06, so the graphs' scores are
the checkpoint's, with the same n and subjects.

E1's CPU latency is in its card, from each graph's `<graph>.bench.json`
([training](training.md#speed-and-latency)).

## Splits

Splits are assigned once, in `bulwark split` (`bulwark/data/splits.py`), and stored in
the manifest.

- **Subject-disjoint.** A person never appears in more than one of train, val and test.
  For AUC Distracted Driver v2, which publishes no driver ids, this rests on a check by
  image similarity with known gaps (below).
  Drive&Act follows the dataset's official subject split `split_0`: 10 training drivers,
  2 validation drivers and 3 test drivers. Every other dataset is split by a stable hash
  of the subject id (salt `bulwark-v1`; 10% val, 15% test), with at least 2 val and 3
  test subjects for datasets of 10 or more subjects. Adding clips or other datasets never
  moves a person to another split, as long as a dataset's own list of subjects does not
  change (the val and test floors are filled in hash order, so new subjects in a dataset
  can move existing ones).
- **Driver groups: AUC Distracted Driver v2.** Its authors split train and test by
  driver but publish no per-image driver id, so each side is one subject group
  (`SplitPolicy.official_splits` and `subject_group_datasets`). One camera-1 driver is on
  both sides of the authors' split; her 83 test images are moved to train
  ([data card](data-card.md#splits)). No other shared driver was found, but the check
  compared images within each camera: a driver recorded again in other clothes, at
  another time or in another car (the dataset page says some drivers took part in more
  than one session), and a camera-2 test driver who is also among camera 1's train
  drivers, would not have been found. It has no val split, k-fold selection keeps its
  train group in training, and its test results rest on one group: no subject-level
  interval, and every AUC result is marked indicative.
- **Train-only datasets.** Sets that publish no person ids (CEW, SH17) or whose labels
  are prompted rather than annotated (synthetic clips) are used for training only. The
  Zenodo driver-behaviour images also publish no person ids and are listed as train-only
  in `SplitPolicy.train_only_datasets`, but v0 excludes them (below).
- **Excluded datasets.** The Zenodo driver-behaviour set is excluded from v0 by a
  maintainer decision (`SplitPolicy.excluded_datasets`): its data use agreement forbids
  sharing the data with third parties and asks to review publications 30 days before
  submission. Its clips, when ingested, are in split `excluded`, neither trained on nor
  scored; the leak check below refuses a manifest with its clips in any other split.
- **Cross-dataset holdout: UTA-RLDD.** The whole dataset is kept out of training to
  measure transfer to an unseen dataset. It is not ingested yet, so no cross-dataset
  number exists until it is; the report lists it as "not available (not ingested)".
  It is cleared for neither Kaggle nor Azure Blob storage (`kaggle_storage_ok` and
  `azure_storage_ok` are undetermined), so a manifest uploaded to Kaggle or Azure with
  `DROP_UNCLEARED=1` leaves it out; a report scored on such a manifest says so ("left out
  of the upload"), and the holdout is scored locally.
- **Cross-view holdout: Drive&Act inner-mirror camera.** This view records the same
  moments as the other cameras, so only the clips of the 3 `split_0` test drivers form the
  holdout: a new camera and new drivers. The rest of the view is excluded, neither trained
  on nor scored.

The split salt is frozen: changing it would reassign every person to a different split.

**Leak check.** `bulwark evaluate` and `bulwark report` refuse a manifest that breaks the
split policy (`bulwark.data.splits.check_disjoint`: a subject in two of train, val and
test, a clip of a holdout dataset outside holdout, a clip of a holdout view in train, val
or test, a holdout-view clip of a training or val subject, a train-only dataset in val,
test or holdout, a clip of an excluded dataset outside `excluded`, or a removed
subject); training refuses such a manifest too. The check uses the split policy the
manifest records (`bulwark split` writes it into the file's metadata, and `bulwark label
merge` keeps it), else the current code's; subjects removed since the split are refused
either way. The report header's "split policy" row says which: `manifest`, or `code
default` when the manifest records no split policy (re-run `bulwark split`, then
`bulwark label merge`, to record it; clip ids do not change).

### Comparability with published results

Bulwark's numbers are not comparable with the benchmarks published for each dataset.
They score Bulwark's own labels on Bulwark's clips: 16-frame clips (see
[architecture.md](architecture.md#inputs)) and the taxonomy in
`bulwark/data/taxonomy.yaml`, whose activity label has 14 classes. For Drive&Act, the
drivers follow `split_0`, but its 34 mid-level activities are mapped onto 12 of those 14
classes, the camera views are pooled, clips are shorter than the benchmark's 3-second
chunks, and activities shorter than one clip are dropped. Drive&Act is scored on
`split_0` only: the test drivers of its splits 1 and 2 are training drivers of the model.
Other datasets use the subject hash split above rather than their own protocols.

To compare with a dataset's own protocol, `bulwark probe` fits linear probes on frozen
encoder features (see [Running it](#running-it)): `drive_and_act_midlevel` follows
Drive&Act's mid-level protocol (34 classes, official splits 0 to 2, balanced accuracy on
test, per view), and `uta_rldd_folds` leaves out one of UTA-RLDD's published folds at a
time. The probe (`bulwark.eval.probe.LinearProbe`) is not the same code as
`bulwark.adapters.fit_adapter`, so a probe score is not what an adapter with default
settings would get: the probe fits binary labels with one logit and regression labels
with a squared error, picks its L2 strength from a grid on a validation set, and does not
balance classes.

## What is scored

- **Ground-truth labels only.** Teacher labels are training signal, not evaluation
  targets, and are never scored.
- **Only labels with ground-truth training data that the checkpoint trained.** The
  report's training-support table counts, per label, the training clips with ground truth
  and with teacher labels in the model's training scope (its recorded `commercial_only`
  and dataset filters), and its "trained" column says whether the checkpoint lists the
  label in its `supervised_labels` and on which media types training masked it. Labels
  without ground-truth training data are listed under "Not scored" with their
  teacher-label counts; in v0 these include alertness, whose only ground truth (UTA-RLDD)
  is the holdout, so it is not scored on UTA-RLDD either. Labels with training ground
  truth that the checkpoint did not train are listed there too, and neither kind gets a
  threshold or temperature in `calibration.json`. Labels that are constant per dataset or
  camera (context, modality, low light, face visible) go to a separate sanity table,
  since scoring them measures which dataset a clip came from.
- **Only clips of the split.** A split scores only the predicted clips that the manifest
  puts in it. Predictions saved against an earlier split are re-scored without the clips
  that are now in another split or `excluded`; those are counted in the report instead.
  A coverage table gives, per split and dataset, the clips scored next to the clips in
  the manifest, so clips that were never predicted (not sharded or cached, or `--limit`)
  are visible.
- **Untrained classes.** Where an evaluated categorical label includes classes that have
  no ground-truth training labels, the report names those classes and adds a macro-F1
  over the trained classes only.
- **Splits:** val for model selection, thresholds and calibration temperatures (fitted
  per media type, see [Metrics](#metrics)); test and the holdouts for reporting. Each
  training stage's `best.pt` is chosen on ground-truth labels only: on the validation
  split, or, for a heads run with `select_folds` (the shipped heads configs use 5), by
  k-fold cross-validation over the train and val subjects, and the report header then
  gives its selection score as a mean over the held-out folds (see
  [training.md](training.md#2-heads-on-frozen-features)).

The same protocol applies to heads trained on frozen features, to B1 and E1 checkpoints,
and to the exported E1 ONNX file, which is also checked for parity with the PyTorch E1
model it came from. An ONNX graph takes the frame count it was exported with, so a
16-frame graph scores only video clips: it skips the image clips (MRL Eye and AUC
Distracted Driver v2), which the report header counts as skipped, and a `--reference`
comparison with the E1 checkpoint covers the video datasets only. Score image clips with a graph exported with
`--frames 1`. For the same reason the `calibration.json` of a 16-frame graph's
evaluation has video entries only, so it cannot calibrate a `--frames 1` graph: take
that graph's calibration from the evaluation of the E1 checkpoint or of the
`--frames 1` graph.

## Metrics

| Label kind | Headline | Also reported |
|---|---|---|
| binary | AUROC | F1 at the threshold that maximised F1 on the val clips of the clip's media type, Brier score, expected calibration error |
| categorical | macro-F1 | accuracy, balanced accuracy, top-label calibration error |
| regression | mean absolute error | MAE in natural units where the label has one (degrees, KSS points) |

Each number is reported with:

- **a baseline**: a constant predictor fitted on the training split's ground truth (the
  dataset's own where it has training data), so a score can be read against what the
  label distribution alone gives;
- **n**: clips, subjects and independent events. Results on fewer than 5 subjects are
  marked indicative only;
- **a 95% confidence interval** from a subject-level bootstrap (1000 draws by default),
  where there are at least 10 subjects;
- **one column per dataset** and, for test, an unweighted mean over the datasets where
  the metric is defined. Pooling clips across datasets would let a model score by telling
  the datasets apart, so a pooled column is shown only where every contributing dataset
  defines the metric on its own. Holdout groups are scored each on its own, with no mean
  or pooled column: a new camera with new drivers and a dataset never trained on answer
  different questions;
- **slices** by camera view, capture condition (modality, low light), eyewear, face
  occlusion and gender where a dataset records them, and per-video scores for datasets
  labelled per video (outputs averaged over the video's clips). YawDD names its
  "Yawning" videos in the file name without marking when the yawn happens, so its yawn
  label is also scored per video, on the highest clip logit of each video;
- **calibration**: expected calibration error over 15 equal-width bins and the Brier
  score, before and after a temperature per label and media type fitted on val, because
  downstream rules threshold probabilities.

**Events.** Drive&Act records each moment with several cameras, so its clips are not
independent. For datasets whose manifest records event ids, the per-dataset headline and
the pooled column count each event once, scoring one prediction per event with the
views' probabilities averaged; a separate per-clip table counts every view. Condition
slices are per clip.

**Thresholds and temperatures.** Clips are 16-frame video clips or still images (1-frame
clips), and a clip model's confidence differs between the two, so one threshold fitted on
all val clips of a label can suit neither. Each binary label's F1 threshold (on
sigmoid(logit)) and each binary and categorical label's temperature are therefore fitted
per media type, on the val clips of that media type alone, and every calibrated table
(per dataset, per event, per clip, per video, slices and the pooled column) scores each
clip with the values of its own media type; the pooled column computes one F1 with each
clip at its own threshold. The fit on all val clips is kept as well, for comparison and
for readers that predate the per-media-type entries. A threshold and a temperature are
fitted only on val clips that hold at least 10 clips of each of two classes (for a
binary label, 10 positives and 10 negatives); with fewer, the threshold stays 0.5, there
is no temperature, and the entry's `not_fitted` says why. Where val has no clips of a media
type for a label (AUC v2's images are never val clips, for example), nothing is fitted
for it either. Either way its clips of that media type are scored at 0.5 without a
temperature, never with the fit on the other media type, and the report header warns,
also for a label with no val ground truth at all. The report's "Thresholds and
temperatures" table lists every fit with the val clips, positives and subjects it rests
on; a fit on few of them rests on little even above that floor.
Video maxima, the sanity checks and the reference comparison are not calibrated.

Small test sets give wide intervals. A difference between two models whose intervals
overlap is not evidence that one is better.

## Running it

```bash
bulwark evaluate --checkpoint checkpoints/heads-v0/best.pt --features data/features/frozen
bulwark evaluate --checkpoint checkpoints/e1-v0/best.pt --shards data/shards --out runs/eval-e1
bulwark evaluate --checkpoint checkpoints/bulwark-e1.onnx --out runs/eval-onnx --reference runs/eval-e1/predictions
bulwark report --predictions runs/eval-e1/predictions --out runs/eval-e1-rescored
bulwark probe cv --features data/features/frozen --datasets drive_and_act,mrl_eye
bulwark probe drive_and_act_midlevel --features data/features/frozen
```

- `bulwark evaluate` predicts val, test and holdout with one model and scores them.
  Heads checkpoints read cached summary tokens (`--features`), a B1 features directory
  can be scored from its cached logits (`--model logits`), and B1, E1 and ONNX models
  stream the shards. `--model auto` (the default) reads the kind from the checkpoint.
  `--reference` adds a comparison with another model's predictions on the same clips,
  for example the ONNX graph against the E1 checkpoint it came from, or E1 against B1.
  `--limit N` scores at most N clips per split, for a quick pipeline check. The same
  command runs as a remote job on Kaggle or Azure, with `--model` naming the kind
  ([training.md](training.md#running-on-kaggle)).
- `bulwark report` scores saved predictions again, for example against a rebuilt
  manifest, without running the model. It also refits the thresholds and temperatures
  from the saved val predictions, so it brings a `calibration.json` written before the
  per-media-type entries up to date. Pass the manifest the predictions were made with
  (`--manifest`; the default is `data/manifest.parquet`) and a new `--out` (the default
  is `runs/eval`), so the original report stays as it was.
- `bulwark probe` fits linear probes on frozen-encoder features (cached without `init`,
  since a fine-tuned encoder has seen Bulwark's training subjects): `cv` runs a
  leave-subjects-out k-fold per dataset, for labels whose test split holds only a few
  people; `drive_and_act_midlevel` and `uta_rldd_folds` follow those datasets' own
  protocols. `cv` skips the datasets that the manifest's split policy excludes
  (`zenodo_driver`) or keeps for training only, and lists them as skipped. It writes
  `probe-<protocol>.json` and `.html` to `runs/probe/`.

`bulwark evaluate` writes to `--out` (default `runs/eval/`):

| File | Contents |
|---|---|
| `predictions/` | `<split>.npz` with clip ids and raw logits, and `meta.json` with the model, code and manifest behind them and each file's sha256 |
| `metrics.json` | every number in the report (a binary entry's `threshold` and a binary or categorical entry's `temperature` map each media type among its clips to the value applied, null for no temperature; the video maxima and sanity tables, which are not calibrated, give `threshold` as the number 0.5), the definitions of each metric, per-bin reliability data for calibration, the training support, the checkpoint's `supervised_labels` and `masked_labels` (`checkpoint_labels`; null when it records neither), the labels not scored and why (`not_scored`), the clip coverage per split, every holdout group of the split policy with its clip and subject counts (a group without clips, such as UTA-RLDD before it is ingested, has a row with zero counts and the status "not available"), and a `meta` block: the model, its checkpoint and sha256, the training scope, the manifest's sha256 and split policy, the clips per split left out of a features directory as non-finite (`nonfinite_clips`, when there are any), and the code version and time of the prediction (`predicted_with`) and of the scoring (`scored_with`), and its `warnings`, among them the labels whose clips of a media type were scored without a val fit |
| `calibration.json` | per-label thresholds and temperatures fitted on val, with a `meta` block naming the model, the checkpoint's sha256 (for an ONNX graph, also that of the checkpoint it was exported from), the manifest's sha256 and the code versions. Each label's entry (`threshold`, `threshold_from`, `temperature`, `source`, and the val clips `n_val`, positives `n_val_positive` and subjects `n_val_subjects` behind it; `not_fitted`, the reason, when they were too few to fit) is its fit on all val clips; binary and categorical labels also have `media`, one entry of the same form per media type with val ground truth (`image`, `video`). A reader takes the entry of the media type it runs on; with `media` but no entry for that media type, nothing applies there, and an entry without `media` (a file written before media entries) applies to every media type. `bulwark.inference.load(..., temperature=...)`, `bulwark predict`, `bulwark release` and `bulwark export --calibration` refuse a file fitted on other weights. |
| `report.html` | the tables above, with a header that states the model, data and code behind them and any warning (for example predictions made against a different manifest) |

The report's "Definitions" section states each metric's exact definition, and
`metrics.json` repeats them.

**Latency** is not part of the evaluation report. `bulwark export` writes it to
`<graph>.bench.json` next to the exported graph; quote it from that file, with its
hardware (see [training.md](training.md#speed-and-latency)).

## Known limits of the protocol

- Most subjects come from a few datasets with few people each (Drive&Act has 15
  drivers, 3 of them in test), so test sets are small and confidence intervals are wide.
- Apart from MRL Eye's gender code (from its file names, on eye crops), the manifest
  has no demographic fields, so accuracy per demographic group cannot be measured for
  the video tasks. Deployers must evaluate on their own population and cameras.
- Several clips in a split often come from the same recording, so clips are not
  independent samples. This is why n counts subjects and events as well as clips.
- Labels such as alertness and activity are partly subjective or self-reported in the
  source datasets; scores measure agreement with those labels, not ground truth about a
  person's state.
- Which labels are scored follows the checkpoint's record of what it trained: a label
  outside its `supervised_labels` (such as `attention_direction` in v0, which every v0
  config masks) is listed as not scored, and a label masked on one media type only is
  scored with a warning. A checkpoint that records no `supervised_labels` (written
  before training recorded them) has every label with training ground truth scored,
  masked or not.
