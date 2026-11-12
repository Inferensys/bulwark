# Releasing

How to release a version of the code and how to release model weights. Each item needs a
maintainer to do it and confirm it; none of it is automated.

## Releasing the code

1. In CHANGELOG.md, replace `Unreleased` in the heading of the version being released
   with the release date (YYYY-MM-DD), and add an `[Unreleased]` section above it for
   later changes.
2. Set `version` in `pyproject.toml` (`bulwark.__version__` reads it from the installed
   distribution), and add `version` and `date-released` (the actual release date) to
   CITATION.cff.
3. Make sure CI passes on the release commit (lint, type check, `uv lock --locked` and the
   tests, which also check that `docs/datasets.md` matches the registry and that the
   relative links in the public docs resolve), then tag it `vX.Y.Z`.
4. Build and check the distribution (`uv build`, then `twine check dist/*`), and publish
   to PyPI as `inferensys-bulwark` if the release is meant for PyPI.
5. Create the GitHub release from the tag with the changelog entry as notes.

## Releasing model weights

**Dataset terms.** Before training the release, decide for every candidate dataset
whether a model trained on it may be publicly released. Record the outcome in the
registry (`weights_release_ok`, `weights_release_basis` and `terms` in
`bulwark/data/registry.py`; `uncleared_for_release()` lists the datasets still unchecked
or not allowed) and in the model card. `weights_release_basis` says, with its date, on
what the outcome rests: the dataset's terms, its owners' written permission, or the
maintainers' own decision (`MAINTAINERS_RELEASE_DECISION`), which is never presented as
the owners' consent. `bulwark release-check <model>` applies this to a checkpoint, ONNX
graph or release directory: it lists every dataset the weights learned from, including
those of upstream models (`via`), prints the basis of each clearance, and exits 1 unless
all are cleared and the code is a recorded commit without local changes; add
`--commercial` for any licence that does not restrict use to non-commercial research.
The entity that releases the weights must be the licensee named in any signed dataset
agreement. Pay attention to:

- [ ] Drive&Act ("Usage for research only."), DMD (CC BY-NC-ND 4.0, "academic purposes"
  only; whether a trained model counts as adapted material under ND), MRL Eye (no licence
  stated) and AUC Distracted Driver v2 (signed agreement: non-commercial; trained models
  not mentioned). Their terms do not settle whether a company's non-commercial research
  use, and a research-only release of weights trained on them, are allowed. On
  2026-10-30 the maintainers decided to release the v0 weights for non-commercial research
  only, under CC BY-NC 4.0, on each dataset's research terms: the registry records
  `weights_release_ok=True` with `MAINTAINERS_RELEASE_DECISION` as the basis. That is
  Inferensys's decision, not a statement by the datasets' owners, and the model card, data
  card and access-request form say so; users must also follow each dataset's terms. If a
  dataset's owners object, the weights trained on it are withdrawn and the models
  retrained without it (see [If a dataset's owners object](#if-a-datasets-owners-object));
- [ ] SH17 (CC BY-NC-SA 4.0): if it is included, check whether its ShareAlike clause
  requires the weights to be released under CC BY-NC-SA rather than CC BY-NC, or leave it
  out;
- [ ] NTHU-DDD and 100-Driver (signed agreements);
- [ ] State Farm (Kaggle competition rules, not yet reviewed);
- [ ] the Zenodo driver-behaviour set stays out (`SplitPolicy.excluded_datasets`;
  registry `weights_release_ok=False`): its data use agreement forbids sharing it with
  third parties and asks to review publications 30 days before submission, and part of it
  appears to be re-encoded State Farm images. Including it needs the provider's written
  permission first;
- [ ] UTA-RLDD (no licence stated);
- [ ] any other dataset in the release manifest.

Leave out any dataset that is not cleared (by terms that clearly allow it, its owners'
written permission, or a dated maintainers' decision recorded as above), and train with
the `datasets` filter set accordingly.

**Subject removals.**

- [ ] Compare each dataset's subject list with its provider's current release, and add
  subjects the provider removed to `bulwark/data/removed_subjects.yaml`.
- [ ] Apply every pending removal (the steps are in `docs/data-card.md`): `bulwark purge`
  locally (logged in `data/purges.jsonl`), then `scripts/purge_cloud_copies.sh <dataset>`
  for Azure Blob and, with `KAGGLE_SLUGS`, the Kaggle datasets with their old versions
  (logged in `docs/purge-log.md`).

**Training and evaluation.**

- [ ] Train B1 and E1 from a clean checkout of a tagged commit; the checkpoints record
  the config, data filters, manifest hash, split salt, taxonomy and package versions and
  the code commit of every session. Check that no session recorded local changes
  (`dirty` in `sessions`).
- [ ] Evaluate B1, E1 and the exported ONNX file on val, test and the holdouts
  (`bulwark evaluate`; `--reference` compares the ONNX graph with the E1 checkpoint it
  came from). A 16-frame graph scores only video clips; score the image clips (MRL Eye
  and AUC Distracted Driver v2) with a graph exported with `--frames 1`. `bulwark export`
  checks parity with PyTorch on real clips when given `--shards`.
- [ ] Measure E1 latency with `bulwark export` with no other heavy job running, record the
  machine's load and CPU idle just before, and keep the `<graph>.bench.json` it writes.
- [ ] Check that the checkpoints were trained on a manifest merged with the current
  teacher-label rules (`FACE_KEEP` and the others in `bulwark/label/run.py`). A merge
  rebuilds `labels_teacher` from scratch, so re-merging a manifest that is already
  merged with the same code, teacher files and library versions rewrites it byte for
  byte. Take the manifest whose SHA-256 is the checkpoints' `manifest_sha256` (in their
  provenance; for a Kaggle or Azure job, the staged copy it read), re-merge a copy, and
  compare:

  ```bash
  cp data/manifest.parquet /tmp/remerged.parquet
  bulwark label merge --manifest /tmp/remerged.parquet --labels data/labels
  shasum -a 256 data/manifest.parquet /tmp/remerged.parquet   # must match each other and manifest_sha256
  ```

  A different digest means the checkpoints learned from teacher labels other than the
  ones the current rules give: re-merge and retrain.
- [ ] Run the label coverage report and the teacher agreement report on the release
  manifest:

  ```bash
  bulwark coverage --markdown runs/release/coverage.md    # clips per label, dataset, split, media type and source
  bulwark label agreement --out runs/release/agreement.json   # leaves out excluded datasets (zenodo_driver); the JSON records the splits scored
  ```

**Model card.** Complete every *at release* item in MODEL_CARD.md:

- [ ] the final list of training datasets with licences and citations, and clip and
  subject counts per split (also in the data card);
- [ ] label counts per split from the label coverage report (`bulwark coverage`), each
  output's status (trained and evaluated, teacher-only, or untrained) from the released
  checkpoints' `supervised_labels` and the evaluation report, and teacher agreement with
  ground truth (`bulwark label agreement`);
- [ ] that the released checkpoints' manifest was merged with the current teacher-label
  rules (Training data; done by the re-merge check under *Training and evaluation*);
- [ ] links to the evaluation report and its numbers, with the protocol, and for each
  stage whether it started from, and released, the previous run's `best.pt` or
  `final.pt` (B1 from the heads run, the B1 outputs for distillation, and E1);
- [ ] E1 latency from `<graph>.bench.json`, with its hardware and onnxruntime version;
- [ ] the exact library versions of each checkpoint, from its provenance, `uv.lock` and
  the job's `environment.txt`;
- [ ] any change to intended use, limitations or licence.

**Bundle.** Each model is published in its own Hugging Face model repository (see
[Publishing the weights on Hugging Face](#publishing-the-weights-on-hugging-face)):
B1 as a release directory written by `bulwark release <checkpoint> <dir> --calibration
<calibration.json>` or `examples/make_release.py --temperature <calibration.json>`, which
write the same config.json with the default research licence; E1 as two ONNX graphs from
`bulwark export` of the released E1 checkpoint, one with 16 frames for video and one with
`--frames 1` for still images. `scripts/hf_release.py stage --model <b1|e1> --public`
assembles each repository's folder and refuses an incomplete one. A release stores the
thresholds and temperatures of both media types (`media_calibration` in config.json), and
a graph those of its own media type: a 16-frame graph takes the video entries, and a
graph for still images (`--frames 1`) the image entries, which only an evaluation that
scored image clips has (the E1 checkpoint's or the `--frames 1` graph's, not a 16-frame
graph's). A `calibration.json` written before the per-media-type entries gives every media
type its fit on all val clips: re-score that evaluation's predictions with `bulwark report`
(see [docs/evaluation.md](docs/evaluation.md#running-it)), then export the graphs and
build the releases again from the new file. In each graph, check the contract's
`calibration` entry (`bulwark.export.onnx.read_contract`): `pooled` must be empty, and
`no_val_clips` lists the labels that carry no threshold or temperature because val had no
clips of that media type for them. A graph's thresholds can then differ widely from those
of a graph exported before the per-media-type entries, so tell whoever runs rules on them.
Run `bulwark release-check` on the release directory and on each graph; all must be
cleared. A B1 release directory contains the V-JEPA 2.1 encoder weights. Training
checkpoints (`.pt`) are pickles and are never published: the B1 repository holds a release
directory (safetensors), the E1 repository ONNX graphs.

Each repository holds (`LAYOUT` in `scripts/hf_release.py`):

- [ ] README.md, the model's card: Hub metadata over the card rendered from
  `release/hf/` (`b1.md` or `e1.md`, with the shared sections in `release/hf/shared/` that
  both cards hold word for word; [release/hf/README.md](release/hf/README.md) lists the
  values the stage fills in). Its tables of results, speed, parameters and outputs come
  from the staged files, never from this document;
- [ ] MODEL_CARD.md, the full card, with its links pointed at this repository on GitHub at
  the code's release tag, and NOTICE (which states that the E1 weights are modified from
  timm's `efficientvit_b1.r224_in1k`);
- [ ] LICENSE, the CC BY-NC 4.0 legal code (`LICENSES/CC-BY-NC-4.0.txt`, the text of
  https://creativecommons.org/licenses/by-nc/4.0/legalcode), which on a gated repository
  is one of the two files every visitor can read, and the licence identifier
  `cc-by-nc-4.0` (the `license` of the card's Hub metadata);
- [ ] for B1, `LICENSE-vjepa2` (written into the release directory), Meta's MIT licence
  notice for the V-JEPA 2.1 encoder weights the release contains;
- [ ] for E1, `LICENSE-timm`, the Apache-2.0 text, which section 4(a) of that licence
  requires when the timm-derived E1 weights are redistributed;
- [ ] config.json at the root, which the Hub counts downloads by: B1's release config,
  which `bulwark.inference.load` reads; for E1 both graphs' contracts (inputs, outputs and
  their decoding, trained and masked labels, temperatures, thresholds, licence) by file
  name;
- [ ] for E1, the licence (CC BY-NC 4.0 plus the training datasets' terms), a link to the
  model card (E1's Hugging Face model repository, `bulwark.export.release.model_card_url`)
  and the model release in each ONNX file's own metadata, so the terms travel with the
  file: export with `bulwark export --checkpoint <E1> --calibration <calibration.json>
  --shards data/shards`, and check its `metadata_props` (`license`, `model_card`,
  `model_release`, `supervised_labels`, `masked_labels`, `temperature`, `thresholds`,
  `release_cleared`). `supervised_labels` and `masked_labels` are the E1 checkpoint's own
  record. With `--manifest <training manifest>` the graph also lists which outputs had
  ground truth, only teacher labels or none, and the outputs with training targets that
  the checkpoint did not train (`not_trained_by_checkpoint`). Each graph's
  `<graph>.bench.json` comes along;
- [ ] the evaluation of the released weights (`evaluation/<model>/`: `report.html`,
  `metrics.json`); for E1 also the evaluation of its B1 teacher (`evaluation/b1/`), which
  its card compares with;
- [ ] the provenance of the released checkpoint (`provenance.json`: config, manifest
  hash, code commits, library versions; a release directory's `config.json` carries part
  of it without local paths, and this copy has its paths scrubbed) and the release check
  of each artifact (`release-check.json`);
- [ ] SHA-256 checksums of every file (`SHA256SUMS`).

Never include dataset frames, manifests or any other part of a dataset.

## Publishing the weights on Hugging Face

Released weights go to one Hugging Face model repository per model
(`bulwark.export.release.MODEL_REPOS`): B1 to
https://huggingface.co/Inferensys/bulwark-b1-driver-monitoring and E1 to
https://huggingface.co/Inferensys/bulwark-e1-driver-monitoring-onnx; both exist and stay
private until the launch. The README of each is that model's card, and release
directories and ONNX graphs link to their model's repository (`model_card_url`); the
code, its documentation, issues and security reports stay on GitHub, and the cards link
to the documentation there at the code's release tag (`v0.1.0` for this version). The
collection "Bulwark: driver drowsiness and distraction detection models"
(`release/hf/collection.yaml`) groups the released models' repositories with the papers
they build on, and the organisation page (`release/hf/org-card.md` and
`release/hf/org.yaml`) introduces them.

**Release set.** `release/hf/launch.yaml` lists the models a launch releases
(`released`), and every step below follows it
([release/hf/README.md](release/hf/README.md#release-set)). The first launch releases
Bulwark-B1 v0 and Bulwark-E1 v0 together (`released: [b1, e1]`). E1 v0, the first
distillation from B1 v0, scores well below B1 on test
([docs/evaluation.md](docs/evaluation.md#e1-v0-test-results)). After seeing those
results, the maintainers decided to release it as it is, as a research preview, with its
weakness stated on the first screen of its card and wherever a reader chooses a model
(B1's card, the organisation card, the collection, README.md and MODEL_CARD.md); E1 will
be retrained. While a model is not listed, `open`, `stage --public` and `upload --public`
refuse it, its uploads stay untagged private previews, no card or organisation card names
its repository (the card templates' `{{#released e1}}` blocks drop the lines that link
it), and `collection` defers its items. Listing a model is not opening its repository: a
card that links another model's repository goes public (an upload to a public
repository, or `open`) only once the Hub reports that repository public, or in the same
`open` run as that repository. The B1 and E1 cards link each other, so one `open` run
opens both ([Launch order](#launch-order)). Each repository stays private until its release
is uploaded there, every artifact in it clears `bulwark release-check` and Inferensys has
reviewed the private repository page. It is then made public as a gated repository with
automatic approval (`GATED` in `scripts/hf_release.py`): the card is public, and downloads
need an access request, which the Hub approves as soon as it is sent. The form
(`GATE_FIELDS`) asks for the requester's name, company or affiliation and intended use, a
checkbox accepting CC BY-NC 4.0 and the training datasets' terms, and whether Inferensys
may contact them about commercially licensed models (Yes or No); its description says
what Inferensys receives and that it contacts only those who answer Yes. Only `open` gates a
repository, so the private page does not show the form: it is reviewed as the
`extra_gated_*` fields in the front matter of the staged README.md (step 4).
`scripts/hf_release.py` does each step below and refuses what the step may not do; only
its `open` step changes a repository's visibility, and it is always run as a step of its
own, by a maintainer. The collection and the organisation page are made public by
hand, at launch.

In a gated repository the Hub shows the card (README.md) and LICENSE to everyone, and
every other file only to users whose access request was approved. With automatic
approval that is anyone who sends the form, so what a requester needs before requesting
access stays in the card itself or in the form: the licence (with its creativecommons.org
link), the training datasets and their terms, the intended and out-of-scope uses, and
whom to contact for a removal request. The cards link to documentation on GitHub, not to
the repository's own copies, which visitors cannot open. The root LICENSE that everyone
sees is the CC BY-NC 4.0 legal code, the licence of the weights; the code's Apache-2.0
licence covers the code on GitHub, and E1's repository holds the Apache-2.0 text as
`LICENSE-timm`.

1. **Once, before any release directory or graph for publication is built** (they record
   their repository's URL): the two repositories exist in the Inferensys organisation on
   huggingface.co, private. The maintainer who uploads installs the tools and logs in with
   a token that can write to them:

   ```bash
   uv sync --frozen --extra dev --extra export --extra ops
   hf auth login
   ```

2. **Stage** each model into a new folder (default `runs/hf-release/<model>`, ignored by
   git):

   ```bash
   python scripts/hf_release.py stage --model b1 --release releases/bulwark-b1-v0 \
     --eval runs/eval/b1-lora-auc --provenance checkpoints/b1-lora-auc/provenance.json --validate
   python scripts/hf_release.py stage --model e1 \
     --onnx runs/export/e1-v0/bulwark-e1-video.onnx --onnx runs/export/e1-v0/bulwark-e1-image.onnx \
     --eval runs/eval/e1-v0 --teacher-eval runs/eval/b1-lora-auc \
     --provenance checkpoints/e1-v0/provenance.json --validate
   ```

   B1 v0 is the checkpoint whose sha256 `bulwark.export.release.B1_CHECKPOINT_SHA256`
   records (starting `de184992f7ed`). Its run wrote to `checkpoints/b1-lora-auc`, its
   evaluation to `runs/eval/b1-lora-auc` and its cached outputs to
   `data/features/b1-lora-auc`, because the first full run, trained without AUC Distracted
   Driver v2 and not the release, already held the configs' `*-v0` paths
   (docs/training.md, "Measured for v0"). E1 v0 was distilled with
   `--set teacher=data/features/b1-lora-auc`. It is the `best.pt` of its run
   (`checkpoints/e1-v0`, sha256 starting `065b89dc5eda`, in the output of its Kaggle job),
   evaluated into `runs/eval/e1-v0` and exported from that file with that evaluation's
   `calibration.json`, `--shards` and `--manifest` (the training manifest), once with 16
   frames and once with `--frames 1`, into `runs/export/e1-v0`. `stage --public` refuses a
   B1 release directory built from another checkpoint, and E1 graphs whose
   `teacher_init_sha256` is not B1 v0's.

   It copies the release directory or the graphs unmodified (the graphs as
   `bulwark-e1-video.onnx` and `bulwark-e1-image.onnx`, each with its `bench.json`), the
   `report.html` and `metrics.json` of each evaluation (not its predictions or
   `calibration.json`), the checkpoint's provenance without local paths, the licences,
   NOTICE and MODEL_CARD.md with its links pointed at GitHub, and writes README.md (the
   card rendered from `release/hf/` under the Hub metadata: licence `cc-by-nc-4.0`, library,
   tags, E1's base model and the access-request form), E1's config.json,
   `release-check.json` (the verdict of `bulwark release-check` for each artifact, under
   this checkout's registry) and last `SHA256SUMS` (`shasum -a 256 -c SHA256SUMS` checks
   it). `--validate` sends the front matter, and only that, to the Hub's validate-yaml
   endpoint, which checks it and creates nothing, and refuses the stage on any error or
   warning. It refuses dataset material (frames, video, tables, shards, archives,
   anything named `manifest`), pickles, local paths, credentials, email addresses other
   than Inferensys's public contact address, dataset subject ids (in any staged file, the
   cards included; the data card's two AUC driver groups are not a person's), a link that
   reaches neither a staged file nor a file or folder git tracks here (at the release tag,
   for links to GitHub), an unknown template placeholder, a card that breaks the heading
   rules of `release/hf/README.md`, an evaluation of a `--limit` run or of other weights,
   graphs from different checkpoints, and inputs under `data/`, `ops/` or `docs/local/`.
   An evaluation belongs to the weights only if it scored the checkpoint they were built
   from or a graph itself (same SHA-256), and E1's teacher evaluation only if it scored
   the B1 checkpoint the graphs record as their teacher. Each must have scored every test
   and holdout clip of its manifest (metrics.json `coverage`), and E1's and its teacher's
   the same manifest under the same split policy, since E1's card sets their scores side
   by side. A private stage may hold artifacts
   that are not cleared and need not be complete: a value it cannot fill yet renders as an
   *at release* item, and `stage` lists what keeps the folder from a public repository.

3. **Upload** it with `python scripts/hf_release.py upload --model <b1|e1>`. It checks the
   stage again (checksums, the scan, the input checks of step 2 on the staged copies, the
   release checks, which must equal those made at staging, and every file `stage`
   derives, which must be what it derives now from this checkout's templates and docs),
   creates the model's repository private if it does not exist, and uploads the folder so
   that main mirrors it: files that are no longer staged are removed from main. It refuses
   a repository that is already public unless given `--public`, and then takes only a
   stage that `stage --public` would take (every artifact cleared, the release complete,
   no *at release* items); it tags the commit `v0` only for such a stage, and never moves
   a tag. In a public repository, new or changed weights need a new `--tag`; `--no-tag`
   uploads a later card fix whose weights main already holds. Until a release is complete
   and every artifact is cleared, uploads are untagged private previews, and upload lists
   what keeps each from a public release.

4. **When a model's v0 release is complete:**

   - [ ] check that the model is in the release set (`released` in
     `release/hf/launch.yaml`), that the registry clears every dataset of the release
     (`weights_release_ok` with its `weights_release_basis`; for the v0 sources, the
     maintainers' decision of 2026-10-30), complete the *at release* items of the card
     templates, the full card and the data card, and commit;
   - [ ] review the wording of the access-request form (`GATE_HEADING`, `gate_prompt`,
     `GATE_DESCRIPTION` and `GATE_FIELDS` in `scripts/hf_release.py`): with automatic
     approval, the prompt and the checkbox are all that a requester sees before getting
     the files. Access requests give Inferensys each requester's username, email address
     and answers; the privacy notice the form links (https://inferensys.com/privacy) must
     cover them, with a retention rule;
   - [ ] from that commit, build the release directory or export the graphs again (their
     own records must say cleared and carry their model's `model_card_url`), and evaluate
     what you export;
   - [ ] stage and upload; the repository stays private:

     ```bash
     python scripts/hf_release.py stage --model b1 --public --validate ...   # the inputs of step 2
     python scripts/hf_release.py upload --model b1
     ```

   - [ ] an Inferensys owner reviews the private repository page on the Hub (the card as
     the Hub renders it and the file list) and the access-request form as it was staged:
     the `extra_gated_heading`, `extra_gated_prompt`, `extra_gated_description` and
     `extra_gated_fields` in the front matter of the staged README.md (also in the raw
     README.md on the Hub). The private page does not show the form, since only `open`
     gates the repository, so the first time the Hub renders it, it is live;
   - [ ] then, at launch (see [Launch order](#launch-order)), as a separate step, a
     maintainer opens the repositories, both in one run, since each card links the other:

     ```bash
     python scripts/hf_release.py open --model b1 --model e1
     ```

   `stage --public` also requires, for B1, its release directory and, for E1, a 16-frame
   and a 1-frame graph from one checkpoint and its B1 teacher's evaluation; for both, the
   evaluation of the weights, the checkpoint's provenance, and a card, full card and data
   card without *at release* items. `stage --public`, a cleared upload and `open` refuse a
   checkout with uncommitted changes, since its registry decides the release check. `open`
   checks each stage again, compares main with it file by file and scans every revision of
   the repository, for every model it is given, before it changes anything; one failure
   leaves every repository as it was. It refuses a card that links a repository that is
   still private, unless the same run opens it. It then turns on gating with automatic
   approval and makes each repository public in one settings change, in turn, and checks
   the result. If a change did not take, or the change or that check failed (the Hub may
   have applied it before failing), it makes that repository private again, and also every
   repository the run opened before it, since a run opens all its repositories or none.
   It then says which repositories are public; if making one private again fails, it says
   so, and that repository's settings must be checked on the Hub at once.

5. **The collection.** `python scripts/hf_release.py collection` creates the collection
   private, or updates it: title, description, the items in order (at launch the demo
   Space, then the released models, then the papers) and their notes, from
   `release/hf/collection.yaml`. It never makes it public and refuses one that is public.
   The items of a model that is not in the release set, its model item and the papers
   marked `for` it (for E1, the EfficientViT paper), are deferred: it leaves them out, and
   removes them from the collection if they are there, printing each removal. It deletes
   no other item, and lists items that are not in the file. Record the address it prints
   as `url` in `collection.yaml`, so the cards and the organisation card link to it, and
   stage again. `collection --public` is for launch: it refuses until the repository of
   every released model and every Space it lists (the demo Space, once its item is in
   the file) are public, then adds the items marked `at_launch` (the AUC paper, whose Hub
   page appears only once a public card cites it) and makes the collection public.

6. **The organisation page.** `python scripts/hf_release.py org-card` writes the
   organisation card into `runs/hf-org-card/README.md` (the README.md of a Space named
   `Inferensys/README`, which the Hub shows on the organisation page only when that Space
   is public) and prints the organisation's settings, the short description and the
   website, from `release/hf/org.yaml`. Nothing is sent to the Hub; at launch, an
   organisation admin creates or updates the Space with that file, makes it public, and
   enters the settings on the organisation's settings page.

7. **Access requests** are approved by the Hub as they come in. The requests list in each
   repository's settings on the Hub shows each requester's answers, and an approved
   request can be rejected there later, which removes that user's access. Inferensys
   contacts about commercially licensed models only requesters who answered Yes. The
   tool approves and rejects none.

Approved users can read every revision, not only main, so `open` refuses while any
branch, tag or pull request reaches a file other than the release's: other weights,
dataset material, a manifest, a dotfile or any other path a stage of that model does not
hold (an annotation table, for example). Graphs uploaded as private previews before the
release was cleared are always other files (their metadata records the release check), so
the usual path is to delete the repository in its settings on the Hub, run `upload` again
and then `open`. The tool never deletes or squashes anything. Changes made on the Hub
website disappear from main at the next upload; make every change in this repository and
stage again.

### Launch order

Bulwark-B1 v0 and Bulwark-E1 v0 launch together (`released: [b1, e1]` in
`release/hf/launch.yaml`). The cards link to the code and its documentation on GitHub at
the release tag, each card links the other model's repository, and the collection and
organisation page point at both repositories, so they go public in this order, each step
checked before the next:

1. **E1's measurements.** The cards link MODEL_CARD.md at the release tag, so finish what
   it still marks *at release* for E1 before the release commit:
   - Wait for the evaluation of the two graphs (`bulwark evaluate --model onnx` with
     `--reference runs/eval/e1-v0/predictions`, into `runs/eval/e1-v0-onnx-video` and
     `runs/eval/e1-v0-onnx-image`) to finish, and check in the `reference` entries of
     their `metrics.json` how closely each graph agrees with the checkpoint.
   - Then, with no other heavy job running, measure each graph's CPU latency as `bulwark
     export` does, on the evaluated graphs themselves. Do not export them again for this:
     a graph records the commit it was exported at (`exporter` in its contract), so an
     export at another commit is another file.

     ```bash
     python - <<'EOF'
     import os
     from pathlib import Path
     from bulwark.export.onnx import benchmark, write_benchmark
     for name in ("bulwark-e1-video.onnx", "bulwark-e1-image.onnx"):
         graph = Path("runs/export/e1-v0") / name
         threads = sorted({1, os.cpu_count() or 1})  # one thread and all cores, as `bulwark export`
         print(write_benchmark(graph, [benchmark(graph, threads=n) for n in threads]))
     EOF
     ```

     It writes `<graph>.bench.json` next to each graph, with the graph's sha256; `stage`
     copies it with the graph and refuses one measured on another graph.
   - Run `bulwark release-check` on each graph; both must be cleared.
   - Complete the *at release* items of MODEL_CARD.md (the graphs' agreement with the
     checkpoint, and their latency, with the load conditions of the measurement) and
     commit.

   Done for E1 v0 on 2026-11-10: both graphs cleared, and MODEL_CARD.md, E1's card and
   [docs/evaluation.md](docs/evaluation.md#e1-v0-test-results) give their agreement
   with the checkpoint and their latency.
2. **GitHub.** Publish the code at https://github.com/Inferensys/bulwark with the `v0.1.0`
   tag ([Releasing the code](#releasing-the-code)). Its history starts with one commit,
   "Bulwark 0.1.0", which holds this tree and carries the tag. The development history
   before it stays in a private archive repository, so the Bulwark commits that the v0
   records name are not public ([model card](MODEL_CARD.md#versioning-and-provenance)).
   The private repository at that address already holds development commits. Do not
   force-push the new commit over them and then make that repository public: GitHub's
   guide to removing sensitive data says that rewritten commits can stay reachable by
   their hash in cached views on GitHub. Instead:
   1. Rename the existing private repository to the archive's name; it stays private. In
      every development checkout, point `origin` at the new name. Until a new repository
      takes the old name, GitHub redirects the old address to the renamed one; after that,
      a push to the old address would send the whole history to the public repository.
   2. Create a new, empty, private repository named Inferensys/bulwark.
   3. Export the release commit's tree into a new directory
      (`git archive <commit> | tar -x -C <dir>`), run `git init` there and commit it as
      "Bulwark 0.1.0", with the maintainer's name and GitHub no-reply address
      (`<id>+<username>@users.noreply.github.com`) as author and committer
      (`GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME`, `GIT_COMMITTER_EMAIL`).
      Check that `git rev-parse HEAD^{tree}` there prints the same hash as
      `git rev-parse <commit>^{tree}` in the development checkout. Tag the commit `v0.1.0`
      with the same identity.
   4. Push only `main` and `v0.1.0` to the new repository, check on GitHub that it holds
      one commit, and make it public.

   Check that the documentation links of the staged cards open (the cards link
   `https://github.com/Inferensys/bulwark/blob/v0.1.0/...`), and that every other link of
   the cards answers. The MRL Eye page (https://mrl.cs.vsb.cz/eyedataset.html) timed out
   on 2026-10-31, so the B1 card links the Internet Archive's copy next to it (`archive`
   in `release/hf/names.yaml`, and the Training data table of `release/hf/b1.md`); the
   cards also cite the dataset's paper by its DOI.
3. **Cards.** Stage and upload both models while their repositories are private. B1's
   repository holds private previews of a card staged while B1 was to launch alone, so
   stage B1 again: its card now links E1. Then run each card's quick start as a reader
   would, in a new virtual environment: for B1, `pip install` of the tag's archive and
   `hf download` of the repository; for E1, the card's Python, which downloads the graphs
   from the repository. Each card says when its quick start was tested, with which
   versions, under an *at release* item that keeps it private: update those lines in
   `release/hf/b1.md` and `release/hf/e1.md` and drop the items. `stage
   --public`, `upload` and `open` refuse a checkout with uncommitted changes, so commit
   the edits to `main` of the public repository, with the identity of step 2, as a second
   commit after "Bulwark 0.1.0" (untagged: the cards' links stay at `v0.1.0`), and push
   it. Then stage both again with `--public --validate`, upload both, and later `open`
   (step 4), all from a clean checkout at that commit, so that the commit
   `release-check.json` records is one readers can find. Run `collection` once more: it
   adds E1's items (its model and the EfficientViT paper) to the private collection, whose
   address `release/hf/collection.yaml` already records. An Inferensys owner reviews both
   private pages and access forms (step 4 of the steps above).
4. **Repositories.** `open --model b1 --model e1`. It takes both in one run: each card
   links the other model's repository, and `open` refuses a card that links a private
   repository unless the same run opens it. It checks both repositories before it changes
   either, and opens both or neither. Check, logged out, that both pages and their LICENSE
   open and that the cards' links work, the links between the two cards included.
5. **Collection.** `collection --public`: it adds the AUC paper (`at_launch`) and makes
   the collection public. The demo Space's item is commented out in
   `release/hf/collection.yaml` until the Space exists; once it is public, uncomment the
   item with the Space's real id and run `collection --public` again, which adds it as
   the first item.
6. **Organisation page.** Make the `Inferensys/README` Space public with the card from
   `org-card`, and enter the organisation's description and website. `org-card` works
   offline and lists the repositories its card links: both must be public first.

### If a dataset's owners object

If the owners of a training dataset object to the release of weights trained on it, the
weights trained on that dataset are withdrawn and the models retrained without it:

- [ ] make every repository that holds such weights private in its settings on the Hub at
  once (both B1's and E1's: E1 learned from B1's outputs), and the collection too: approved
  users can read every revision, so removing files from main does not withdraw them;
- [ ] set the dataset's `weights_release_ok` to `False`, with the date and the objection
  in `weights_release_basis`, and commit; `bulwark release-check` and `open` then refuse
  every artifact trained on it, upstream models included, and `upload` takes such a stage
  only as a private preview;
- [ ] retrain without the dataset (the `datasets` filter), and release the new weights as
  above; since `open` refuses a repository whose history holds other weights, delete each
  repository in its settings on the Hub and upload again first.
