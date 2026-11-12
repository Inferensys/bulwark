## What and why

<!-- What this change does and the problem it solves. Link the issue if there is one. -->

## How it was checked

<!-- Commands run, with results. Quote only numbers produced by the evaluation report. -->

- [ ] `ruff check .`, `mypy bulwark scripts infra` and `uv lock --locked` pass
- [ ] `pytest` passes (CI runs `pytest -m "not vjepa"`), and new behaviour has tests that
      use synthetic fixtures only

## Data and licences

- [ ] No dataset files, frames, manifests, weights, credentials or personal data are added
- [ ] A new or changed dataset has a registry entry with its licence, access route,
      citation and a justified `commercial_ok` value taken from the official terms;
      `weights_release_ok`, `kaggle_storage_ok` and `azure_storage_ok` stay unset until
      the maintainers have decided, and a set `weights_release_ok` has a
      `weights_release_basis`
- [ ] `docs/datasets.md` was regenerated with `bulwark datasets --markdown docs/datasets.md`
      if the registry changed
- [ ] Label or class changes bump the taxonomy `version`; the split salt is unchanged

## Docs

- [ ] README, `docs/` (including the data card), MODEL_CARD.md and CHANGELOG.md are
      updated where behaviour changed
