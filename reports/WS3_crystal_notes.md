# WS-3 — Crystal arms on MP-20, and a scoping finding on the constraint form

Standing note, rebuilt on change. Feeds CP4, CP5. Numbers are read from the twelve
per-run artifacts in `results/WS3_crystal_*_seed*.json`.

## Summary

The masking claim transfers to real crystals: a masked model generates structurally valid
output at exactly 1.0 across three seeds, against 0.9789 for the unconstrained arm, and it
does so **untrained** as well as trained, because validity is a property of the mask rather
than of the model.

The density steering claim also transfers, once the constraint is formulated correctly.
Trained through for the full run, the deployed system reaches 4.810 g/cm^3 against a 5.0
target — 9.4x closer than the unsteered baseline — and the probability-weighted
infeasible-step fraction falls 6.9x through training, reproducing on real crystals the
complementarity mechanism the synthetic task established. An earlier version of this note
retired the dial on the strength of a claim about the data that the data did not support;
that conclusion and its correction are recorded below.

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

The A3 rows in that table come from the first, incorrect density formulation and are
superseded by the results in the next section. They are retained only as the record of what
the mis-scaled constraint produced.

## The density dial survives: complementarity confirmed on crystals

The previous version of this note concluded that the per-step expectation form cannot express
density, and retired the dial. **That conclusion was wrong**, and the review that caught it was
right about why: it rested on a claim that each formulation moved further from the target, which
is not what the numbers showed. Against a 5.0 g/cm^3 target the three errors were 4.352, 1.540
and 12.190 — the second formulation was the best of the three and already beat the unsteered A2
baseline at 1.783. The series was bad, better-than-baseline, catastrophic; not monotone
deterioration. The justification for stopping did not hold.

Trained through for the full 6,000 steps across three seeds, with two defects in that
formulation fixed, the dial works.

| Quantity | At initialization | At convergence |
|---|---|---|
| Probability-weighted infeasible-step fraction | 0.3430 +/- 0.0598 | **0.0496 +/- 0.0067** |
| Deployed mean density (target 5.0) | 5.710 +/- 0.582 | **4.810 +/- 0.153** |
| Structural validity | 1.0000 | 1.0000 +/- 0.0000 |

**The complementarity mechanism reproduces on real crystals.** The infeasible-step fraction falls
6.9x through training, against 19.9x on the synthetic task. Training moves the backbone to where
the running mass budget rarely goes unattainable, the fallback rarely fires, and residual error
does not accumulate — measured now on MP-20 with a real backbone and approximate targets, not
only on an exactly enumerable task.

**The steering effect is large relative to every floor it must clear.** Deployed density lands
0.190 from target against the A2 baseline's 1.783, i.e. 9.4x closer. In relative terms the shift
from 6.783 to 4.810 is 29.1%, against a 1.1% discretization error floor — 26x the floor.

Separation from the baseline is **Cohen's d = 14.4** pooled across arms with sample standard
deviations, n = 3 seeds per arm, with Welch's t = 17.66, p = 0.0017. Welch rather than Student
because the arm variances differ by 13.9x. Two figures should not be quoted as separations: an
earlier draft reported 12.9, which used the deployed arm's standard deviation alone; and 17.66
is the t-statistic, which carries a factor of sqrt(n/2) from the standard error and is not in
units of standard deviations.

**The 9.4x decomposes into layer and training, and both parts are published.**

| Stage | Density error vs target | Gap closed | Share |
|---|---|---|---|
| A2 baseline, no layer | 1.783 | — | — |
| Layer on the untrained backbone | 0.710 | 1.072 | 67% |
| Trained through the layer | 0.190 | 0.520 | 33% |

The layer alone accounts for two thirds of the gap closed and training through it for the
remaining third. The complementarity claim attaches specifically to that third, and the
infeasibility drop is its independent mechanism figure: quoting 9.4x without the split would
invite a reviewer to compute it and find it unstated.

### Two defects in formulation 2, both found by testing the constraint against the data

**The budget denominator was in the wrong units.** The static channel gives an element token its
ATOMIC mass, a per-atom quantity, but the budget was divided by remaining element TOKENS. Since
each element carries a count, this demanded about three times the mass the data's own next
element carries — median 177u against 59u actual — and 32.7% of actionable positions demanded
more than the heaviest element in the periodic table. Dividing by remaining ATOMS puts the target
on the channel's footing: the demand falls to 0.85x the data's own next element and unattainable
demands drop to 5.6%.

**The constraint was applied at non-actionable positions.** At a count or structural position
every legal token has channel value zero, so the attainable range is the single point {0} and any
nonzero budget is infeasible by construction. Counting those inflated the reported infeasible
fraction and gave the projection no decision to make. Restricting to element positions is the
same scoping already pre-registered in `CP0_actionable_window.md`, applied here.

Neither defect crashed. Both trained to completion while producing a constraint the training data
itself could not satisfy, which is why the diagnostic that found them was to ask whether real
structures satisfy the demanded target — not to inspect the implementation.

### Backbone versus deployed runs the other way here

On the synthetic task the backbone moved partway toward the target and the layer supplied the
rest. On crystals the backbone reading is 9.200 while the deployed system is 4.810: **the
backbone alone moves away from the target**. The network has not internalized the constraint; it
has adapted to a training signal that already includes the projection, and it is only correct in
combination with the layer it trained through. This is reported as measured rather than smoothed
into the synthetic story, and it strengthens the case for the projection as a training component
rather than a decoder: removing it at inference does not merely lose accuracy, it inverts the
result.

## Scope of the constraint form, retained

The analysis that motivated the earlier retirement is a real contribution and stands, with the
empirical claim corrected. The per-step expectation form is most natural for an **extensive**
quantity whose constraint-bearing length is **known in advance** — the synthetic task's total
charge over a fixed-length sequence, where the running budget's denominator is exact and the
channel is a constant per token.

Density satisfies neither condition cleanly. It is intensive, so a target fixes a total mass
budget only once the cell volume is known — which is why cell-first serialization is required.
And the crystal sequence has model-determined length, so the budget denominator is an estimate
from the data mean rather than a known count.

The finding is that **neither condition is necessary in practice**: an estimated denominator,
corrected at every step from the mass actually emitted, is sufficient for the mechanism to work,
because the re-derivation prevents error from accumulating. What the conditions govern is how
much care the formulation needs, not whether the method applies. That is a more useful scope
statement than the one it replaces, and it is now earned by measurement in both directions.

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
