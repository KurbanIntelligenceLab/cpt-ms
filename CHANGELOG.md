# Changelog

## Unreleased

### Fixed

- `pyproject.toml` was unparseable by `uv`: the `authors` table sat after
  `[project.optional-dependencies]`, so TOML read it as an extra. The documented
  `uv sync` install could never have worked.
- Four CIF tests reached `pymatgen` with no optional-dependency guard, so a clone without
  the `crystal` extra saw 15 errors where the README promised none.
- Seven artifacts carried a `script` provenance field naming a module that no longer
  existed, making their stated rerun instruction impossible to follow.
- The README's quoted pass/skip counts described a suite several hundred tests smaller than
  the one in the repo.

### Changed

- The package moved from `scripts/` to `src/cpt/`; experiments now run as
  `python -m cpt.<area>.<name>`. Repo-root discovery is defined once in
  `cpt.common.artifacts.REPO_ROOT` rather than recomputed in seven modules.
- The cited short run left the ignored `*_smoke.json` namespace: it is now
  `results/crystal_conditional_shortrun.json`, and `.gitignore` no longer needs a negation
  to keep one file out of its own rule.
- Tests read artifacts through `tests/_artifacts.payload()`, which verifies `payload_sha256`.
  Thirty sites previously read payloads raw, so only one test checked any stamp at all.
- The run ledger moved to `development/ledger/`. No experiment ever called it.

### Added

- `uv.lock`, committed, with CI running `uv sync --locked`.
- A ruff configuration whose every exception carries its reason, and CI that runs it.
- `test_artifact_script_exists`, which would have caught the stale `script` fields.

### Refactored

- The projected-arm constraint block was copy-pasted at four call sites, byte-identical
  apart from which budget function and channel tensor it used. It is now
  `channels.budget_constraint`, parameterised on exactly those two.
- `mass_channel` was `weight_channel` with `ATOMIC_MASS` inlined, and `_mass_budget` was
  `_additive_budget` with the density target and unit factor inlined; both now delegate,
  and the g/cm3 conversion factor has a name instead of being a bare literal in two places.
- The structural-validity token check appeared verbatim in three modules; it is now
  `metrics.is_structurally_valid`.
- `train_crystal.py` was a 680-line trainer that eighteen modules imported as a library,
  eight of them lazily to avoid paying for it. Model, channels, sampling and metrics are
  their own modules; the trainer keeps `main` and its CLI, and re-exports the moved names.
  `surrogate_phi.py` was two experiments in one file and is split at the seam.
- `consolidate_retargeting.py` ran its whole body at import time, so importing it emitted
  an artifact and wrote it relative to the caller's working directory.
- `ELECTRONEG` sat in `surrogate_phi`; it is an element table and now lives beside
  `ATOMIC_MASS` and `VALENCE` in `mp20`.
