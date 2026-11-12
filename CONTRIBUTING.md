# Contributing to Bulwark

Thanks for your interest. Bulwark is research code maintained by Inferensys. Bug reports,
fixes, dataset adapters and evaluation improvements are welcome. For larger changes
(new model components, label changes, split changes), please open an issue first so we
can agree on the approach.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security problems privately as described in [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/Inferensys/bulwark.git && cd bulwark
uv sync --frozen --extra dev --extra export --extra label   # the versions in uv.lock, into .venv
source .venv/bin/activate
scripts/fetch_pretrained.sh     # V-JEPA 2.1 weights, MediaPipe face model, V-JEPA 2 code for a parity test
```

`uv.lock` was written by uv 0.8.17. Add `--extra ops` for the Kaggle and UTA-RLDD
scripts and the Hugging Face release script, and `--extra detect` or `--extra vlm` for
the optional teachers; `uv sync` removes extras it is not given, so list them all in one
command.

Before opening a pull request, run what CI runs:

```bash
ruff check .
mypy bulwark scripts infra   # settings in pyproject.toml
uv lock --locked             # uv.lock matches pyproject.toml; after changing dependencies, run `uv lock`
pytest -m "not vjepa"        # as in CI; plain `pytest` also runs the tests that need V-JEPA
```

CI runs these on Python 3.10 and 3.11 with the locked versions and CPU-only PyTorch
(`.github/workflows/ci.yml`). Tests use small synthetic fixtures and must not need any
real dataset, network access or GPU. A test that needs the V-JEPA weights fetched by
`scripts/fetch_pretrained.sh` must be marked `@pytest.mark.vjepa`; CI deselects those, so
run them locally. The encoder code itself is vendored (`bulwark/model/vjepa21`); the tests
that compare it with Meta's clone in `third_party/vjepa2` skip when the clone is absent.

## What must never be committed

- **Data.** Raw datasets, manifests, shards, labels, features and frames from any
  dataset. Most source datasets forbid redistribution, and their videos show real people.
  `data/` is gitignored; keep test fixtures synthetic.
- **Weights and run outputs** (`checkpoints/`, `runs/`), and third-party code fetched by
  scripts (`third_party/`).
- **Credentials and account details**: tokens, keys, storage URLs with signatures,
  personal email addresses, account names or support-ticket numbers. Use environment
  variables and placeholders such as `<your-kaggle-username>`.

## Adding a dataset

1. **Registry entry** in `bulwark/data/registry.py`: name, title, official URL, licence,
   access route, size, the labels the adapter writes (`provides`), the terms relied on
   (`terms`, quoted, with where and when they were checked) and the citation copied from
   the dataset's page or paper. Set `commercial_ok=True` only if the terms clearly permit
   commercial use, including training models that are deployed commercially. Leave
   `weights_release_ok` as `None` until the maintainers have decided whether trained
   weights may be published; when it is set, `weights_release_basis` records what the
   decision rests on (the terms, the owners' written permission, or the maintainers' own
   dated decision, which is not a statement by the dataset's owners). Leave
   `kaggle_storage_ok` and `azure_storage_ok` as `None` until the maintainers have read
   the terms and decided, for each provider, whether a private copy may be kept there (a
   private Kaggle dataset, or the Azure Blob container); record the dated decision in
   `terms`. `infra/kaggle/push_data.sh` (Kaggle) and
   `infra/azure/upload.sh` (Azure Blob) refuse to upload the shards, features and
   manifest rows of a dataset whose flag for that provider is not `True` unless the
   operator names it in `ALLOW_UNCLEARED`. When in doubt, choose the restrictive value
   and say why in the notes.
2. **Adapter** in `bulwark/data/sources/<name>.py`, a `SourceAdapter` subclass registered
   in `bulwark/data/sources/__init__.py`, with `adapter=True` in the registry. Adapters
   turn the native layout into `ClipRecord`s:
   - media paths relative to the dataset root, so manifests work on any machine;
   - subject ids prefixed with the dataset name (`<dataset>:<id>`); if the dataset has no
     person ids, add it to `train_only_datasets` in `bulwark/data/splits.py` (datasets
     the maintainers decide to keep out of training and scoring go in
     `excluded_datasets`);
   - only labels the dataset actually annotates, mapped onto the taxonomy in
     `bulwark/data/taxonomy.yaml`; never put teacher-model output into `labels_gt`.
3. **Local root** in `configs/data.yaml`.
4. **Tests** in `tests/test_data_source_adapters.py` that build a tiny synthetic copy of
   the native layout and check the records and labels.
5. **Check against the real data**: `bulwark ingest <name> --dry-run` prints the clip count
   and a sample record without writing anything; after `bulwark split`, `bulwark coverage`
   shows labels the new data can teach only one class of. The header of
   `scripts/prepare_local.sh` lists the steps that add one dataset to prepared data.
6. **Regenerate the dataset table**: `bulwark datasets --markdown docs/datasets.md`, and
   commit it with the registry change. Do not edit `docs/datasets.md` by hand.

## Changing labels or splits

- Labels are defined in `bulwark/data/taxonomy.yaml`. Adding, removing or reordering a
  label or a class changes the model's outputs: bump `version` in the taxonomy, say so in
  the changelog, and expect existing checkpoints not to load.
- The split salt in `bulwark/data/splits.py` is frozen. Changing it reassigns every
  person to a different split and makes all earlier results incomparable.

## Code style

- Type hints in `bulwark/`, `scripts/` and `infra/`, which CI type-checks: `mypy bulwark
  scripts infra` must pass, and a `# type: ignore[code]` there names its error code and is
  only for a third-party library without type information.
- Short docstrings, and comments that explain why rather than what.
- `ruff` settings are in `pyproject.toml` (line length 100).
- Keep public names stable; if a change breaks them, note it in [CHANGELOG.md](CHANGELOG.md).
- Prefer small, focused commits with an imperative subject line, e.g.
  `fix: skip annotation rows shorter than one clip`.

## Reporting numbers

Quote only numbers that the evaluation report generates, with the command that produced
them, the split, and the number of clips and subjects. Never edit a figure by hand.

## Licence of contributions

Bulwark's code is licensed under Apache-2.0. Unless you state otherwise, any contribution
you submit is licensed under the same terms (section 5 of the [licence](LICENSE)).
