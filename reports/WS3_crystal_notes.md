# WS-3 — Crystal arms on MP-20, and a scoping finding on the constraint form

Standing note, rebuilt on change. Feeds CP4, CP5. Numbers are read from the twelve
per-run artifacts in `results/WS3_crystal_*_seed*.json`.

## Summary

The masking claim transfers to real crystals: a masked model generates structurally valid
output at exactly 1.0 across three seeds, against 0.9789 for the unconstrained arm, and it
does so **untrained** as well as trained, because validity is a property of the mask rather
than of the model.

The density steering claim does not transfer, and the reason is a scoping finding rather
than a bug: **the per-step expectation form applies cleanly to an extensive quantity over a
fixed-length sequence, and density is neither.** Three successive formulations were built
and measured; none is a correct instantiation. That boundary should be stated in the paper
rather than engineered around.

## Setup

MP-20 from the CDVAE split: 27,136 training structures, 26,910 parsed (99.2%). Model is a
5.0M-parameter causal decoder over a 500-token vocabulary, 6,000 steps, three seeds per arm.

Discretization was chosen by measurement, not convention. Median relative error in decoded
density against the source structure: 4.4% at 64 length x 32 angle buckets, 2.4% at 128/64,
1.1% at 256/128, 0.6% at 512/256. The 256/128 setting is used — the error sits well below
the steering effects being resolved, and doubling again buys 0.5 points for a 77% larger
vocabulary.

## Results

Mean and standard deviation over seeds 0, 1, 2. MP-20 data density is 6.679 g/cm^3 mean,
6.317 median.

| Arm | Structural validity | Density computable | Mean density |
|---|---|---|---|
| A1 unconstrained | 0.9789 +/- 0.0041 | 0.9750 +/- 0.0038 | 6.865 +/- 0.095 |
| A2 masked | **1.0000 +/- 0.0000** | 0.9952 +/- 0.0008 | 6.783 +/- 0.041 |
| A3 backbone | 1.0000 +/- 0.0000 | 0.9488 +/- 0.0198 | 5.567 +/- 0.204 |

**Validity by construction reproduces on real data.** A2 is exactly 1.0 on every seed. The
claim is not vacuous: A1 under identical training reaches only 0.9789, so an unconstrained
model of this capacity does emit malformed structures. And an *untrained* masked model is
also at 1.0, which is the sharpest form of the claim — validity comes from the mask, not
from learning.

**The A3 backbone reading moves density toward the target** (5.567 against A2's 6.783,
target 5.0), so training through the layer does change the network. But this figure cannot
be reported as a steering result, because the constraint it trained against is not a correct
expression of density. See below.

## Scoping finding: what the per-step expectation form can and cannot express

The synthetic task's constraint is a total charge that is a **sum of per-symbol
contributions over a fixed-length sequence**. Both properties are load-bearing. The running
budget "remaining charge divided by remaining steps" is exact because the denominator is
known in advance, and the constraint channel is a constant per token because each symbol
contributes its own charge.

Density satisfies neither property:

1. It is **intensive** — a ratio of mass to volume — so a target density fixes a total mass
   budget only once the cell volume is known.
2. The crystal sequence has **model-determined length**: the number of element slots is
   chosen during generation, so "remaining mass divided by remaining slots" has an estimated
   denominator.
3. Mass is committed at the **count** token rather than the element token, so a constant
   per-token channel is zero exactly where the mass is decided.

Three formulations were built and measured against a 5.0 g/cm^3 target:

| Formulation | Deployed mean density |
|---|---|
| Fixed mass-per-token target on a static channel | 0.648 |
| Running mass budget, cell-first ordering, static channel | 6.540 |
| Running mass budget, cell-first ordering, prefix-dependent channel | 17.190 |

The first asks for 5 u of atomic mass per token — lighter than lithium — and drives the
composition to the lightest available elements. The second fixes the target but leaves the
channel zero at every count position, so 52% of steps are infeasible. The third makes the
channel prefix-dependent, which fixes the infeasibility but leaves the estimated slot
denominator, and overshoots in the opposite direction.

Each attempt moved further from the target rather than closer. Continuing to patch would be
fitting the formulation to the measurement. The correct statement is the scoping one: **the
method's constraint form covers extensive quantities over sequences whose constraint-bearing
length is known, and an intensive quantity over a variable-length serialization needs a
different treatment.** Charge on a fixed composition template would satisfy the conditions;
density as serialized here does not.

This does not weaken the synthetic result, which is exact and holds. It bounds the claim.

## Cell-first ordering

The cell-first variant was built because density needs the volume determined before the
composition, and it is retained: both orderings accept every real transition with zero
violations, decode identically, and cost nothing. It is the right serialization for any
future volume-dependent constraint even though the density constraint itself is now out of
scope.

## Corrections made during this work

**The 1.35e-30 residual was not float underflow.** float64 denormals reach ~5e-324, so
1.35e-30 is a representable value, and its seed-to-seed variance of +/-3e-32 confirms it —
an underflowed exact zero would be bit-identical. The actual source is the deployed scoring
path, which used `log(q.clamp_min(1e-30))`. The projection returns exactly 0.0 at masked
entries, but the clamp gives every illegal token a probability of the floor, so a sequence
with one illegal token carries 1e-30 times an O(1) factor — the observed value is 1.35 times
the clamp. Removing the clamp from scoring gives identically 0.0 on all three seeds with
mean charge unchanged at +0.0170 +/- 0.0014, confirming it was an evaluation artifact rather
than a property of the layer. The training path keeps its clamp, where the log feeds a loss
whose gradient must stay finite and no illegal token is ever scored.

**Two off-by-one defects in the crystal grammar mask**, each producing a plausible number
rather than a crash. The mask was written before consuming the current token, so it
described what may follow the previous position and rejected every real transition — 16 of
21 on the first structure — while training loss still fell. And the sampler indexed the mask
*at* the last position rather than for the token about to be appended, so every token was
emitted twice. Both are pinned by tests that fail on the reverted code.

## Verification

Six tests in `tests/test_crystal_mask.py`: the mask accepts every real transition across 300
structures; it is strictly narrower than the vocabulary; an untrained masked model is
structurally valid at 1.0; the sampler does not duplicate tokens; an untrained unmasked model
is *not* valid, so the arms are distinguishable; and decoded density round-trips within the
discretization error. Four tests in `tests/test_feasibility_exactness.py` pin the exactness
claim, including one asserting that a clamped scorer leaks at the clamp scale — so the
diagnosis is itself a regression test.
