# Hugging Face cards: templates and data

The model cards, the collection and the organisation card of the Hugging Face release are
built from the files in this folder by `scripts/hf_release.py` (see
[RELEASING.md](../../RELEASING.md#publishing-the-weights-on-hugging-face)). Nothing here is
published as it is: `stage` renders it into each model repository's README.md.

| File | What it is |
|---|---|
| `b1.md`, `e1.md` | The card of each model, under the Hub metadata that `front_matter` writes |
| `shared/<name>.md` | A section both cards include, word for word: `licence`, `metrics` (how to read the results), `product` (Commercial models), `out-of-scope`, `regulatory`, `contact` |
| `names.yaml` | Plain names and groups of the model outputs, and a note for an output a reader must not use; the datasets' short names, first-mention names, notes, citations and archived copies of official pages that do not answer; plain names of held-out camera views |
| `launch.yaml` | The release set: the models this launch releases (`released`), which everything below follows |
| `collection.yaml` | Title, description, items (the demo Space, the models and papers) and notes of the collection (`collection`), and its address once it exists |
| `org-card.md`, `org.yaml` | The organisation card (`org-card`) and the organisation's settings |

## Release set

`launch.yaml` lists the models this launch releases. It is the one place that says so, and
`scripts/hf_release.py` follows it everywhere:

- **Cards.** A release block (below) keeps its lines only when its model is released, or
  only while it is not. Neither a staged card nor the organisation card may name the
  repository of a model that is not released, since that repository is private and its
  link dead to visitors: the card is refused, and a full card (MODEL_CARD.md) that names
  it keeps the stage private.
- **Collection.** The items of a model that is not released (its model item, and papers
  marked `for` it in `collection.yaml`) are deferred: `collection` leaves them out and
  removes them from the collection if they are there. `collection --public` needs the
  repository of every released model to be public.
- **Publishing.** `open`, `stage --public` and `upload --public` refuse a model that is not
  released, and its upload stays an untagged private preview. Being listed is not being
  public: `upload` to a public repository and `open` also refuse a card that links another
  model's repository until the Hub reports that repository public, unless the same `open`
  run opens it.

Bulwark-B1 v0 and Bulwark-E1 v0 launch together: `released: [b1, e1]`. E1 v0 is a research
preview that scores well below B1; its card says so on its first screen, and B1's card,
the organisation card and the collection say so where they name it. The two cards link
each other, so one `open` run opens both repositories (the launch order in
[RELEASING.md](../../RELEASING.md#launch-order)). The public docs follow the same list:
`tests/test_docs_project_files.py` requires README.md, MODEL_CARD.md, SECURITY.md and
RELEASING.md to link the repository of every released model, and every public doc and card
template (`PUBLIC_FILES`) but RELEASING.md to leave out that of a model that is not
released. That test does not read the code, which names E1's repository
(`bulwark.export.release.MODEL_REPOS`, the URL `bulwark export` writes into a graph).

## Syntax

- `{{> name }}` on a line of `b1.md` or `e1.md` includes `shared/name.md`. A shared section
  cannot include another.
- `{{#released e1}}` on a line of its own, then lines, then `{{/released}}` on a line of
  its own, keeps those lines only when `launch.yaml` releases the model (`b1` or `e1`).
  `{{#unreleased e1}}` ... `{{/unreleased}}` keeps them only while it does not. Blocks do
  not nest. Link a model's repository (`{{ e1_url }}`) only inside a `released` block of
  that model.
- `{{ name }}` is replaced by a value from the table below. An unknown name, or any other
  `{{` or `}}`, refuses the stage.
- A shared section may use only the values marked "both" below, which are the same for the
  two models, so the shared sections of the two cards are identical.
- A value the stage cannot fill yet (no evaluation staged, no collection address) renders
  as an *At release* item, which keeps the stage from a public repository. Write the same
  marker for text that is still to be written.

## Links

- A relative link must reach a file of the model's repository, such as `LICENSE` or
  `evaluation/b1/report.html`. Name in the link text that such a file needs access: on a
  gated repository, visitors can open only README.md and LICENSE.
- Documentation links go to this repository on GitHub at the code's release tag:
  `{{ code_url }}/docs/data-card.md` for a file, `{{ code_tree_url }}/examples` for a
  folder. The stage refuses a link into the repository at another ref, or to a path git
  does not track, or to an anchor without a heading.
- The access-request form links the card's `#licence-and-dataset-terms` heading, so each
  card needs a `## Licence and dataset terms` heading.

## Structure

The stage refuses a card without exactly one H1, with a skipped heading level, with a
heading below H1 that is not plain ASCII or holds `?` or parentheses, or with a code fence
that has no language.

## Values

| Placeholder | Model | Value |
|---|---|---|
| `b1_repo`, `e1_repo` | both | the repository ids |
| `b1_url`, `e1_url` | both | the repositories' pages (link a model's only inside its `released` block) |
| `github` | both | the code repository on GitHub |
| `code_url`, `code_tree_url` | both | the code repository at the release tag (`blob` for files, `tree` for folders) |
| `code_raw_url` | both | the code repository's raw files at the release tag, for images in cards (files in a gated repository don't show to visitors without access) |
| `code_tag`, `code_version` | both | the release tag (`v0.1.0`) and version (`0.1.0`) of the code |
| `model_release` | both | the model release (`v0`) |
| `contact_url`, `contact_email`, `privacy_url` | both | Inferensys's contact page, address and privacy notice |
| `licence_url`, `legal_code_url` | both | the CC BY-NC 4.0 deed and legal code |
| `collection_link` | both | a link to the collection, once `collection.yaml` has its `url` |
| `model_name`, `repo_id`, `card_url` | each | Bulwark-B1 or Bulwark-E1, its repository id and page |
| `weights_licence` | each | the licence the weights record |
| `checkpoint_sha256` | each | the first 12 hex digits of the released checkpoint's sha256 |
| `parameters`, `parameters_exact` | each | values stored in the weights (safetensors header; the video graph's initializers), as `94.4M` and `94,372,153` |
| `files_table` | each | the weights and config.json, with what they are and their size |
| `labels_table` | each | the trained outputs, each row with its group in the first column, then name, key, value range (B1) and whether ground truth checked it (an output's names.yaml `note` in place of the reason); E1 the media types its graphs were trained for in place of the range, with "copies B1 only" for a media type whose training clips have no label for the output (names.yaml `labelled`) |
| `untrained_outputs` | each | the outputs that were not trained, as a list in a sentence |
| `untrained_classes` | each | a sentence naming the classes of trained outputs that had no training clip, or nothing when there are none |
| `not_trained_on_images`, `not_trained_on_video` | each | a sentence naming the trained outputs masked on that media type, or nothing when there are none |
| `copied_on_images` | each | E1: a sentence naming the outputs its image graph only learned to copy from B1, since no training image has a label for them (names.yaml `labelled`); nothing for B1 or when there are none |
| `image_calibration` | each | a sentence naming the outputs with a temperature or threshold fitted on still images (B1's config.json; E1's image graph) |
| `results_table` | each | the test split's summary: metric, score (baseline), E1 also B1's score ("not comparable" where B1's is a mean over other datasets), and the test people; one row per dataset for an output whose datasets' scores differ by more than 0.25; under E1's table, a sentence on the B1 column (and on its "not comparable" or "not scored" cells, when it has any) |
| `holdout_summary` | each | for each held-out camera view, a sentence with its scores against the same dataset's test cameras, in the same unit: per clip, from the test split's `per_clip` where the test side is scored per event |
| `test_split_table` | each | test people and clips per dataset |
| `test_people`, `test_people_short` | each | who the test split holds, in sentences for the Results caveat and as "3 test people per dataset" for the opening |
| `outputs_as_b1` | each | E1: "the same" when it trained the outputs its B1 teacher did (the teacher's evaluation), else "most of the same"; B1: "the same" |
| `results_by_dataset` | each | one table per test dataset, counted in clips, or in events where one moment was filmed by several cameras |
| `holdout_table` | each | the holdout groups that were scored, each on its own |
| `evaluated` | each | the date of the evaluation |
| `datasets_table` | each | the training datasets with their licence and official page (with names.yaml's `archive` copy of it, if any), and names.yaml's notes under it |
| `dataset_citations` | each | a list item citing each training dataset's paper (names.yaml `cite`) |
| `release_basis` | each | on what each dataset's clearance rests (the registry's `weights_release_basis`) |
| `speed_table` | each | E1: device, threads, onnxruntime version, input and p50/p95 per call, from each graph's bench.json |
| `speed_scope`, `speed_measured` | each | E1: what the measurement covers, and its date |
| `parity` | each | E1: what `bulwark export` checked the graphs against PyTorch on, and the largest logit difference |

Scores have 2 decimals and come only from the staged evaluation of these weights
(`evaluation/<model>/metrics.json`; for E1's B1 column, `evaluation/b1/`), never from a
`--limit` run or one that predicted part of a split. E1's evaluation and its teacher's
must have scored the same manifest under the same split policy. Outputs are listed in
taxonomy order, by group, never by score.
