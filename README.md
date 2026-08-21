# cpt

Code, artifacts and verification for:

**Steering Crystal Generation at Every Token: A Differentiable Constraint Projection for
Autoregressive Models**

Can Polat<sup>1</sup> ([0000-0002-1458-302X](https://orcid.org/0000-0002-1458-302X)) ·
Mustafa Kurban<sup>2,3</sup> ([0000-0002-7263-0234](https://orcid.org/0000-0002-7263-0234)) ·
Erchin Serpedin<sup>1</sup> ([0000-0001-9069-770X](https://orcid.org/0000-0001-9069-770X)) ·
Hasan Kurban<sup>4</sup> ([0000-0003-3142-2866](https://orcid.org/0000-0003-3142-2866))

<sup>1</sup> Department of Computer and Electrical Engineering, Texas A&M University, College Station, TX, USA
<sup>2</sup> Department of Electrical and Computer Engineering, Texas A&M University at Qatar, Doha, Qatar
<sup>3</sup> Department of Prosthetics and Orthotics, Ankara University, Ankara, Turkey
<sup>4</sup> College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar

Corresponding authors: kurbanm@ankara.edu.tr, hkurban@hbku.edu.qa

## Abstract

Inverse design asks a generative model for a crystal meeting a numerical property target.
The established route conditions the model on the property, tying one trained model to one
target family and concentrating output where labels are dense. We take a different route: at
every decoding step of a text serialization of crystal structures we solve a constrained
projection of the next-token distribution, and we train through it. The projection is convex
with an analytic derivative, a layer rather than a decoding heuristic, and the target enters
as a value rather than a training condition. Two findings follow, on crystal generation from
a public materials database and on an exactly enumerable synthetic task. First, one trained
network serves a family of ten mass-density targets it never trained for, statistically
equivalent to separately trained models at one seventh of their cost, and parseable at every
requested target. Second, the per-step route is not a worse approximation of the
sequence-level optimum but an approximation of a different distribution, and the difference
favours it: at equal satisfaction of the average, the constrained property is 6.76 times
less variable and samples violate the requirement 1.42 times less often. The reachable range
and the steerable property class are set by the backbone and the serialization, design
choices a practitioner controls.

## Installation

    uv sync --extra crystal --extra baselines --extra dev

The extras are separate because they are separately expensive. `crystal` pulls the materials
stack (pymatgen, smact, ase, chgnet) needed to screen generated structures; `baselines`
pulls the convex-optimization stack (cvxpy, cvxpylayers) used to verify the projection
layer's gradients against a reference solver. The synthetic track needs neither — numpy and
scipy carry it.

## Verifying the claims

    export PYTHONPATH="$PWD"
    python -m pytest tests/ -q

On a fresh clone with no MP-20 data and no optional extras, expect **250 passed, 120
skipped, 0 failed**. Nothing should fail. Anything skipped names its own reason: the
manuscript-consistency checks need the paper sources under `development/`, which are not
distributed; the crystal tests need the MP-20 CSVs in `data/`; four plausibility tests need
`smact` from the `crystal` extra.

With MP-20 in place and the extras installed, the same command gives **391 passed, 11
skipped**. A failure means a real disagreement between code and artifact -- that is the
point of the suite, and it is why missing inputs skip rather than fail.

Before launching any measurement run:

    python -m scripts.common.preflight

## Running the experiments

Each experiment is a module that emits one artifact into `results/`:

    python -m scripts.synthetic.compute_ground_truth
    python -m scripts.synthetic.alignment_sweep
    python -m scripts.crystal.determinability
    python -m scripts.layer.verify_gradients

Training takes an arm by name:

    python -m scripts.train.train_crystal --arm projected --steps 6000
    python -m scripts.train.train_synthetic --arm mask_only

Every script accepts `--smoke` for a seconds-long shard, which is what pre-flight runs.

## Terminology

The arms are named for what they do. `scripts/common/arms.py` is the definition, and it also
resolves the older `A1`-`A7` codes that some committed artifacts are still stamped with.

| arm | what it is |
|---|---|
| `unconstrained` | backbone decoder: no grammar mask, no projection |
| `mask_only` | grammar mask only, no expectation projection |
| `projected` | grammar mask plus trained-through projection, one active constraint |
| `projected_backbone` | the projection-trained model read *without* projection at inference |
| `projected_cell_first` | `projected`, cell-first static-channel variant |
| `conditional` | conditional-training baseline, retrained on target-filtered MP-20 |
| `comparator` | sequence-level comparator trained by MLE on exact tilt samples |
| `joint` | projection under two simultaneous constraints |

Every arm is reported at two readings: **backbone** (mask only) and **deployed** (mask plus
projection).

Serializations are `typed_fields` (`<COMP> el:Li:2 ...`), `cif` (the same content spelled
character-by-character), and `cif_standard` (`cif` in CrystaLLM's ordering).

Artifact filenames encode their configuration: `d<density>` and `v<valence>` are constraint
targets, `t<target>_s<seed>` is a target/seed pair, and `_smoke` marks a pre-flight shard.

## Data

MP-20, the standard 20-atom subset of the Materials Project used by CDVAE and its
successors. The CSVs are ~58 MB and re-downloadable from the CDVAE repository, so they are
not tracked; `scripts/crystal/mp20.py` loads them from `data/`.

## Results

`results/` holds 83 provenance-stamped JSON artifacts. They are the evidence: every number
in the manuscript is read from one of them at build time, never transcribed, and the test
suite recomputes them and fails if the prose disagrees.

## Layout

```text
scripts/common/      artifact emission, run ledger, pre-flight harness, arm vocabulary
scripts/synthetic/   synthetic constrained-sequence task with exact enumerable targets
scripts/crystal/     serialization, incremental constraint parser, per-step determinability
scripts/layer/       the batched KL projection layer and its gradient verification
scripts/train/       training entry points for the crystal and synthetic tracks
scripts/plotting/    figure renderers, one per figure
tests/               the verification suite
results/             emitted artifacts, provenance-stamped JSON
development/         not distributed: manuscript sources, run notes, preregistration,
                     checkpoints, logs. Nothing here is needed to install or verify.
```

## Conventions

**Nothing quantitative is typed by hand.** Every number in a report is read from an artifact
in `results/`, and each report's numbers are checked against its artifact before the report
is saved.

**Artifacts carry their own provenance.** `scripts/common/artifacts.py` stamps each emission
with the script, config hash, seeds, commit, environment, and a payload checksum. When git is
unavailable the commit field records the sentinel `UNVERSIONED` rather than a fabricated hash.
Because the checksum covers the payload's own key strings, artifacts are never edited in
place — readers go through `scripts/common/arms.py` instead.

**Verdicts are derived, not asserted.** Narrative fields inside artifacts are generated from
the computed values they describe, so a change to the computation cannot leave a stale claim
beside it.

**Never suppress stderr on a measurement run.** A measurement script that raises must be seen
to raise. Redirecting stderr converts a hard failure into a stale artifact that still exists
on disk from the previous run and looks like a fresh result. This happened once here: a run
appeared to succeed while silently emitting no rows for a new backbone variant, because the
script was raising on an unrecognized backbone name and the redirection hid it. Suppress
stdout if the volume is inconvenient; never stderr.

**The ledger is append-only.** A run writes a `started` record and later a terminal one; the
started record is never rewritten, so a crashed run stays visible as exactly that. The ledger
stores no results — results are artifacts — so the two cannot disagree.

## Known limitations

- **Checkpoints are not distributed.** They are build products of the training commands
  above. The SI's plausibility analysis samples from a `projected` checkpoint at density
  target 5.0 g/cm³; reproducing it means rerunning that training command first.
- **The manuscript sources are not distributed.** They live under `development/`, so the
  five tests that check prose against artifacts skip on a public clone.
- **Wall-clock comparisons cross hardware.** The conditional-baseline record was written on
  a different host from the arms it is compared against; the caveat is stated wherever the
  ratio is used, but it is not a controlled timing experiment.
- **The reachable target range is set by the backbone and serialization**, not by the
  projection. A target outside what the vocabulary can express is infeasible by construction.

## Citing this work

See `CITATION.cff`. The DOI and version fields are placeholders until the archival deposit
is made.

## License

MIT. See `LICENSE`.
