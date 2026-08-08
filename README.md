# cpt-ms

Constraint-Projected Transformer: training-time expectation-constraint projection for
autoregressive sequence models, evaluated on a synthetic task with exact targets and on
crystal structure generation.

## Layout

    scripts/common/      artifact emission, run ledger, pre-flight harness
    scripts/synthetic/   synthetic constrained-sequence task and exact targets
    scripts/crystal/     incremental constraint parser, per-step determinability
    scripts/layer/       batched KL projection layer and gradient verification
    tests/               verification suite
    results/             emitted artifacts (JSON, provenance-stamped)
    reports/             standing notes, rebuilt on change
    preregistration/     append-only threshold definitions
    figures/             figures referenced by reports

## Conventions

**Nothing quantitative is typed by hand.** Every number in a report is read from an
artifact in `results/`, and each report's numbers are checked against its artifact before
the report is saved. `tests/test_report_consistency.py` enforces this for test counts,
which are the claims most prone to silent drift.

**Artifacts carry their own provenance.** `scripts/common/artifacts.py` stamps each emission
with the script, config hash, seeds, commit, environment, and a payload checksum. When git
is unavailable the commit field records the sentinel `UNVERSIONED` rather than a fabricated
hash.

**Verdicts are derived, not asserted.** Narrative fields inside artifacts are generated from
the computed values they describe, so a change to the computation cannot leave a stale claim
beside it.

**Never suppress stderr on a measurement run.** A measurement script that raises must be
seen to raise. Redirecting stderr — `2>/dev/null`, `2>&1 >/dev/null`, or a bare `try/except`
around a measurement — converts a hard failure into a stale artifact that still exists on
disk from the previous run and looks like a fresh result. This happened once here: a run
appeared to succeed while silently emitting no rows for a new backbone variant, because the
script was raising on an unrecognized backbone name and the redirection hid it. Suppress
stdout if the volume is inconvenient; never stderr.

**The ledger is append-only.** A run writes a `started` record and later a terminal one; the
started record is never rewritten, so a crashed run stays visible as exactly that. The
ledger stores no results — results are artifacts — so the two cannot disagree.

## Running

    export PYTHONPATH="$PWD"
    python -m pytest tests/ -q

    python -m scripts.common.preflight            # before any launch
    python -m scripts.synthetic.compute_ground_truth
    python -m scripts.synthetic.alignment_sweep
    python -m scripts.crystal.determinability
    python -m scripts.layer.verify_gradients

Every script accepts `--smoke` for a seconds-long shard used in pre-flight.

The projection layer and its verification require `torch`, `cvxpy`, and `cvxpylayers`; the
synthetic and crystal tracks need only `numpy` and `scipy`.
