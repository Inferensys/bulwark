# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package follows
[Semantic Versioning](https://semver.org/). Model releases (v0, v1, ...) and the label
taxonomy are versioned separately; see the README.

## [Unreleased]

## [0.1.0] - 2026-11-13

First release of the code, with the Bulwark-B1 v0 and Bulwark-E1 v0 weights on Hugging
Face (Inferensys/bulwark-b1-driver-monitoring and
Inferensys/bulwark-e1-driver-monitoring-onnx), trained on Drive&Act, the DMD face camera,
MRL Eye and AUC Distracted Driver v2. E1 v0, the first distillation of E1 from B1 v0, is a
research preview that scores well below B1: it is released for trying CPU and ONNX
deployment, and it will be retrained.

This is the first public release. Development before it happened in a private repository
whose history is not published, and the training, evaluation and export records of B1 v0
and E1 v0 name commits from that history (see the model card's
[Versioning and provenance](MODEL_CARD.md#versioning-and-provenance)).

### Added

- **Dataset registry** (`bulwark/data/registry.py`, `bulwark datasets`): every known
  dataset with its licence, access route, the terms relied on, its citation and four
  flags, `commercial_ok`, `weights_release_ok`, `kaggle_storage_ok` and
  `azure_storage_ok`, with `weights_release_basis` saying what a set
  `weights_release_ok` rests on; rendered to `docs/datasets.md`.
- **Dataset adapters** for DMD, Drive&Act, UTA-RLDD, NTHU-DDD, YawDD, State Farm, the
  Zenodo driver-behaviour images, MRL Eye, CEW, SH17 and generated synthetic clips. Every
  dataset video adapter cuts clips the same way: 16 frames at a stride set by a 2-second
  window (a generated synthetic clip uses its whole video as one window). DMD and
  Drive&Act give phone, drink and food 0s from their own object annotations (DMD gives
  no food label).
- **AUC Distracted Driver v2 adapter** (`auc_v2`): its authors' train/test split by
  driver, one subject group per side, with the 83 camera-1 test images of a driver who
  is also in train moved to train; activity from the class, and phone, phone at the ear
  and drink only where the class shows them in use (no 0s: drivers in other classes hold
  phones). Ingested (14,478 images in 15 shards) and used in v0, which trains on four
  datasets.
- **Data pipeline** (`bulwark ingest`, `split`, `shard`): a clip manifest that records
  each clip's dataset, licence and `commercial_ok` flag, with ground-truth and teacher
  labels kept apart; subject-disjoint splits (Drive&Act's official `split_0`, a stable
  hash of the subject id elsewhere) under a frozen salt, a cross-dataset holdout
  (UTA-RLDD), a cross-view holdout (Drive&Act's inner-mirror camera, test drivers only)
  and datasets excluded by policy; cross-dataset de-duplication of images (an image is
  matched only against other datasets' images, and a copy is kept in a dataset with
  person ids first); JPEG tar shards with a record of how their clips were cut, checked
  by `scripts/check_shards.py`.
- **Teacher labels** (`bulwark label`): MediaPipe Face Landmarker, RF-DETR (negatives
  only) and Qwen3-VL, merged below ground truth at half its loss weight; a label coverage
  report (`bulwark coverage`) and a teacher-versus-ground-truth agreement report
  (`bulwark label agreement`: precision and npv for binary labels, `--split` to score
  some splits only, and an `--out` JSON that records the splits it scored).
- **Face-teacher label filter**: of the face teacher's `eyes_closed`, `blink` and `yawn`,
  `bulwark label merge` keeps only `eyes_closed` 0 (eyes open), and only on cameras that
  face the person. A value is kept when ground truth confirmed it at least 90% of the
  time on the train split and the class that keeping only that value leaves out is rare
  in the clips it labels; the face teacher's `blink` and `yawn` keep no value. The merge
  drops the teacher's `eye_openness` and `perclos` where it says the eyes are closed,
  unless the clip's ground truth confirms that, and where ground truth contradicts its
  `eyes_closed`.
- **Models**: Bulwark-B1 (V-JEPA 2.1 ViT-B/16 with LoRA, 25 summary tokens and
  multi-task heads) and Bulwark-E1 (an EfficientViT-B1 student), over label taxonomy
  version 1 (34 outputs in 4 heads), with a masked, uncertainty-weighted multi-task loss
  and soft-target distillation.
- **Training** (`bulwark train cache`, `heads`, `lora`, `distill`): periodic
  checkpoints that resume only a matching run, session time limits, a shard stream that
  mixes datasets and resumes mid-epoch, and `best.pt` selected on ground-truth labels
  (for the heads, by 5-fold cross-validation over the train and val subjects). Each run
  masks labels whose training targets have one class and records the outputs it trained
  (`supervised_labels`). Every checkpoint and features directory records its provenance:
  config, datasets, manifest hash, upstream models and their datasets, code commit and
  library versions.
- **Licence filters**: a commercial-licence chain (`commercial_only`, the
  `configs/train/*-clean.yaml` configs) and a `release_only` filter, each checked through
  every upstream model.
- **Remote jobs**: scripts for private Kaggle notebooks (`infra/kaggle/`) and for Azure
  A100 VMs (`infra/azure/`, with a hard spending cap). Each refuses datasets the registry
  does not clear for that provider unless the operator names them (`ALLOW_UNCLEARED`).
- **Removal requests**: subjects listed in `bulwark/data/removed_subjects.yaml` are left
  out of ingest and splits, `bulwark purge` deletes their local data, and
  `scripts/purge_cloud_copies.sh` carries the purge over to the cloud copies.
- **Evaluation** (`bulwark evaluate`, `report`, `probe`): ground-truth labels only, with
  a split leak check, baselines, clip, subject and event counts, subject-level bootstrap
  intervals, per-dataset and per-condition slices, calibration, and linear probes on
  frozen features for the datasets' own protocols. F1 thresholds and temperatures are
  fitted on val per media type (image or video) as well as on all val clips, and each
  clip is scored with its own media type's; `calibration.json` keeps the fit on all val
  clips at each label's top level and adds the media entries under `media`, with the val
  clips, positives and subjects behind each fit. A fit needs at least 10 val clips of
  each of two classes, else the entry says why (`not_fitted`). Clips of a media type
  without a fit (too few val clips, or none) are scored at 0.5 without a temperature, with
  a warning, and `bulwark report` re-scores saved predictions to write the media entries.
- **Export and release** (`bulwark export`, `release`, `release-check`): E1 to ONNX with
  its input contract, a parity check, a CPU benchmark, the weights licence, the training
  record and calibration stored in the graph (one threshold and temperature per label,
  those of the graph's media type, recorded in the contract's `calibration` entry);
  release directories in safetensors, whose config.json keeps the calibration of each
  media type (`media_calibration`); and a check that every dataset a model learned from,
  upstream models included, is cleared for publishing the weights, which prints what each
  clearance rests on (`weights_release_basis`, also in the check that release configs and
  ONNX metadata carry).
- **v0 weights release decision** (2026-10-30): Inferensys decided to release the v0
  weights for non-commercial research only, under CC BY-NC 4.0, on the research terms of
  DMD (CC BY-NC-ND 4.0, academic use), Drive&Act (research use only), MRL Eye (no licence
  stated) and AUC Distracted Driver v2 (by request, non-commercial). This is
  Inferensys's decision, not a statement by the datasets' owners; users must also follow
  each dataset's terms, and if a dataset's owners object, the weights trained on it are
  withdrawn and the models retrained without it. The registry records it as
  `weights_release_ok=True` with the decision as `weights_release_basis`.
- **B1 v0** (2026-10-30): the LoRA fine-tune on the four v0 datasets, AUC Distracted
  Driver v2 included, is trained and evaluated on test and on the Drive&Act inner-mirror
  holdout. Its results, all indicative (3 test subjects per dataset, one driver group for
  AUC v2), and the comparison with the first full run, trained without AUC v2, are in
  `docs/evaluation.md`: higher holdout AUROCs for phone at the ear, food and phone, a
  lower holdout drink AUROC (0.911 against 0.929), and on DMD's test split a lower
  activity macro-F1 (0.586 against 0.701) and blink AUROC (0.748 against 0.795). B1 v0 is
  the v0 B1 and the teacher of E1 v0's distillation; the first full run and its B1
  outputs are kept. `docs/training.md` gives both runs' measured runtimes.
- **E1 v0, a research preview** (2026-10-31): the first distillation of E1 from B1 v0's
  cached outputs ran its 4 epochs (16,432 optimizer steps on one T4, in two Kaggle
  sessions). Its validation score, 0.78 against B1 v0's 0.93, fell short of B1, most on
  blink, yawn, phone use, eating, drinking and activity, and it generalises poorly to new
  people. On test it scores well below B1 v0: chance on DMD's eyes closed (AUROC 0.497
  against 0.902), chance or below on the Drive&Act inner-mirror holdout for phone, food
  and drink, and weak activity scores; it matches B1 v0 on MRL Eye's eyes closed (0.998).
  After these results the maintainers decided to release it as it is, as two ONNX graphs
  (16 frames for video, 1 frame for still images), for trying CPU and ONNX deployment and
  the output format, with its weakness on the first screen of its card. E1 will be
  retrained with a different recipe. `docs/evaluation.md` gives its test results and
  compares it with B1 per label on val, and `docs/training.md` gives the run's measured
  times.
- **Vendored encoder code** (`bulwark/model/vjepa21`): Meta's V-JEPA 2.1 ViT-B encoder
  definition from facebookresearch/vjepa2 at the pinned commit, trimmed to what the
  encoder needs, with Meta's headers and MIT licence, so an installed package builds B1
  and loads a B1 release without a clone of that repository. A test compares the vendored
  definitions, parameters and outputs (fp32, official weights) with the clone that
  `scripts/fetch_pretrained.sh` makes for that test. The package's licence is declared as
  `Apache-2.0 AND MIT`, with the MIT text among its licence files, and it does not depend
  on einops, which only Meta's audio patch embedding (not vendored) uses.
- **Weights hosting** (`scripts/hf_release.py`): each model has its own Hugging Face
  model repository, B1 (a release directory, PyTorch safetensors) in
  Inferensys/bulwark-b1-driver-monitoring and E1 (a 16-frame and a 1-frame ONNX graph) in
  Inferensys/bulwark-e1-driver-monitoring-onnx; release config.json
  and ONNX metadata link to their own model's repository
  (`bulwark.export.release.model_card_url`). `stage
  --model` builds one repository folder: config.json at the root (E1: the graphs'
  contracts), the weights, the CC BY-NC 4.0 legal code as LICENSE, the upstream licence
  (LICENSE-vjepa2, or LICENSE-timm for E1), NOTICE, the full card with its links pointed
  at GitHub at the code's release tag, the evaluation of the weights (and of E1's B1
  teacher), the checkpoint's provenance, each artifact's release check and SHA-256
  checksums. README.md is the model's card: Hub metadata (licence, library, tags, E1's
  base model, and an access-request form with an opt-in to be contacted about commercially
  licensed models) over a card rendered from the templates in `release/hf/`, whose shared
  sections both cards hold word for word and whose results, speed, parameter and output
  tables are filled from the staged files. Each card opens with what the model detects,
  its licence and the route to commercially licensed models, its indicative results and
  its regulatory status, and has a tested quick start: B1 through `bulwark predict` or
  `bulwark.inference`, E1 with ONNX Runtime alone (`tests/test_docs_e1_quick_start.py`
  checks that it reads a video exactly as Bulwark does). `--validate` checks the metadata with the
  Hub's validate-yaml endpoint. Staging refuses dataset material, pickles, local paths,
  credentials, dataset subject ids, links that reach neither a staged file nor this
  repository at the release tag, unknown template placeholders, and evaluations of a
  `--limit` run, of part of a split or of other weights; a public stage also refuses B1
  weights not built from the released B1 checkpoint and E1 graphs not distilled from it
  (`bulwark.export.release.B1_CHECKPOINT_SHA256`), and E1's and its teacher's
  evaluations must have scored the same clips. Upload and open check
  the folder again, and that it is what `stage` builds now. Upload goes to the model's
  private repository (a public one only when told) until the release is complete and
  every artifact clears `bulwark release-check`, and a separate `open` step, run by a
  maintainer once Inferensys has reviewed the private pages, makes the repositories
  public, gated with automatic approval; one run opens several repositories, all or
  none, since the B1 and E1 cards link each other. `collection` keeps the collection that
  groups the models private until launch, and `org-card` writes the organisation card locally;
  RELEASING.md gives the launch order. One release set, `released` in
  `release/hf/launch.yaml`, says which models a launch releases (B1 and E1 together): the
  cards' release blocks, the collection's deferred items, `open` and a public stage or
  upload follow it, and no card names the private repository of a model that is not
  released. Evaluation reports give the holdout groups' subject
  counts without the dataset's subject ids.
- **Inference** (`bulwark.inference`, `bulwark predict`): B1, E1, ONNX and release
  models on clips, images and videos, with calibration files checked against the weights
  they were fitted on, the temperatures and val-tuned thresholds of the input's media type
  (`temperature_for`, `thresholds_for`; `model.thresholds` warns where they differ by
  media type) and a rolling PERCLOS; small task adapters on
  Bulwark embeddings (`bulwark.adapters`).
- **Project**: locked dependencies (`uv.lock`), pretrained downloads checked against
  pinned SHA-256 hashes, CI that lints, type-checks and tests on Python 3.10 and 3.11,
  the Apache-2.0 licence and NOTICE, a model card, data card, citation file, contributing
  guide, security policy, code of conduct and release checklist.

### Known issues

- **`head_nod` in the v0 models is wrong.** Until this release the face teacher described
  head pitch as positive when looking up, and its nod rule looked for the pitch falling
  and coming back. MediaPipe's pitch is positive when the head tips down, so the rule
  fired on the head rising and coming back. B1 v0 and E1 v0 learned `head_nod` from those
  labels: ignore that output. This release fixes the sign error in the nod rule and in
  the teacher's `attention_direction` rule, which had up and down swapped, so labels made
  with 0.1.0 differ from v0's for `head_nod`, and for `attention_direction`, which v0
  masks. The fixed nod rule still cannot tell nodding off from looking down, and no
  ground truth has checked it. See the model card's
  [Limitations](MODEL_CARD.md#limitations).
