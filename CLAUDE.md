# CLAUDE.md

Project bindings for `cpt`, the artifact repo for *Steering Crystal Generation at Every Token*.
The lab's general rules live in `development/.claude/`; this file records what is specific here,
including where this repo deliberately departs from them.

## What this repo is

One study, one package. `src/cpt/` holds every importable module. Each experiment is a module
that emits exactly one provenance-stamped JSON into `results/`, run as `python -m cpt.<area>.<name>`
and accepting `--smoke`. `results/` is the evidence: every number in the manuscript is read from
an artifact at build time, never transcribed, and the suite recomputes them and fails on
disagreement.

## Environment

`uv sync --extra crystal --extra baselines --extra dev`, then `uv run pytest tests/ -q`. `uv.lock`
is committed and CI runs `uv sync --locked`, so pyproject and the lock cannot drift apart.
`PYTHONPATH` is not needed; pytest gets `src` from `[tool.pytest.ini_options] pythonpath`.

## Deliberate deviations from CODING_RULES.md

**No `scripts/` directory** (§A reserves it for wrappers that call the package). There is nothing
to wrap: each of the 55 modules *is* an experiment, and a wrapper layer would be the "convenience
scripts that duplicate package logic" §L forbids. `scripts/` was this package's old name; it now
lives at `src/cpt/`.

**No `configs/`** (§D wants scientific config in versioned YAML). Config here lives in frozen
dataclasses — `CrystalTrainConfig` and its siblings — whose values are hashed into every
artifact's `config_hash`. That meets §D's intent (no hidden science; config traceable from every
output) by a different mechanism. Converting 55 experiments to YAML would be a rewrite, not a
cleanup.

**No `mypy`** yet (§C.2). The numerical interior sits on pymatgen, ase, and chgnet, none of which
ship stubs, and seven modules still lack `from __future__ import annotations`. Worth doing as its
own change, not as a side effect of one.

**Lint exceptions are notated, not blanket.** `pyproject.toml`'s `[tool.ruff.lint] ignore` list
gives a reason per rule. `N803`/`N806` are off because `A`, `b`, `L`, `V` are the paper's
notation; `E501` is off because long lines are provenance strings emitted verbatim into payloads,
so rewrapping one changes its checksum.

## Working on artifacts

Never hand-edit a payload. `payload_sha256` certifies it, and `test_artifact_checksums` will
catch you. Re-emit by running the module named in the artifact's `script` field, and do it from a
clean tree — `_git_commit()` stamps `-dirty` otherwise, and a `-dirty` stamp on a reported result
is not provenance.

Tests read artifacts through `tests/_artifacts.payload()`, which verifies the checksum. Do not
reintroduce raw `json.loads(path.read_text())["payload"]`.
