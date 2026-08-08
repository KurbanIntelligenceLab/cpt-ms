# WS-2 — Data-constraint alignment sweep: a controlled probe of multiplier decay

Standing note, rebuilt on change. Feeds CP0 and CP1. Every number is read from
`results/WS2_alignment_sweep.json`, emitted by `scripts/synthetic/alignment_sweep.py`.

## Summary

Sweeping the data distribution's alignment with the constraint from strongly violating to
nearly satisfying traces multiplier decay exactly, with the answer known in closed form at
every point. The multiplier scales linearly in the data-constraint mean gap (fitted
exponent 1.03) and the constraint's KL work scales as the square of it (fitted exponent
2.04), so constraint activity vanishes quadratically as data alignment improves — faster
than the alignment gap itself closes. The consequence for measurement is that the gap in
per-sample violation rate between the constrained and unconstrained distributions shrinks
with alignment too, from 11.73 percentage points at the most misaligned setting to 0.04
points at the best-aligned one — a difference that would need roughly 10 million samples
per arm to resolve at 80% power, against 67 at the misaligned end. Since CP4 makes
"A3 beats A2 on Regime-E violation rate" a pass criterion and MP-20 is a well-aligned
dataset, that criterion needs a pre-registered effect size and a power calculation, not
just a direction. No gate outcome; this is pre-checkpoint work sizing CP0's and CP4's
pre-registered thresholds.

## Why this experiment

Two sections of the research plan (v2) bear on this. The risk register's item 1 states the
mechanism: constraints internalized at convergence make the projection an identity, so
A3's endpoint equals A2's and the internalization curve doubles as the method's
disappearance. The "Abstract register" section states the expectation — that multiplier
decay is the modal outcome rather than a tail risk, because training data is real
materials and so satisfies the constraints, and any constraint the data satisfies will see
multipliers shrink as the model learns. On MP-20 both are claims available only after
training. Here `markov_bias` is a dial on
exactly that alignment, and the enumerated task makes the exact multiplier computable at
every setting.

## Results

![Alignment sweep](figures/alignment_sweep.png)

Fifteen alignment levels, all S-feasible, spanning a data mean charge from -5.66 to +5.12
against a constraint target of 0.

| Quantity | At worst alignment (gap 5.66) | At best alignment (gap 0.26) | Fold change |
|---|---|---|---|
| Multiplier magnitude | 0.892 | 0.0253 | 35x |
| KL(tilt ‖ base), nats | 3.083 | 0.0033 | 933x |
| TV(tilt, base) | 0.751 | 0.033 | 23x |
| Per-sample violation rate | 0.8771 | 0.8759 | 1.001x |

**Scaling.** On the closest-aligned half of the sweep, log-log fits give an exponent of
1.026 for the multiplier and 2.038 for the KL. These match the exponential-family
prediction of exactly 1 and 2: the multiplier is first-order in the moment gap, and the KL
of a small tilt is second-order. The agreement is a correctness check on the solver as
much as a result.

**Sign and monotonicity.** The multiplier strictly opposes the gap — negative when the
data mean sits above the target, positive when below — and decreases monotonically in the
signed gap across all fifteen levels. Monotonicity holds in the signed gap, not its
magnitude: the attainable charge distribution is skewed, so two points at nearly equal
|gap| on opposite sides of alignment can order either way without contradiction. An
earlier version of this check tested |lambda| against |gap| and reported a spurious
non-monotonicity of 4e-4; the check is now sign-aware.

## Three consequences for the project

**CP0's pre-registered threshold needs a scale, and this provides one.** The converged-model
active-multiplier fraction is a first-class reported number with its own threshold. This
sweep says what multiplier magnitudes to expect as a function of how well MP-20 satisfies
each constraint family. For a constraint the data satisfies to within a few percent, the
expected multiplier is small not because the layer is inactive but because the quadratic
KL scaling makes it so. The threshold should be set against this curve rather than against
an intuition about what counts as "active."

**The violation-rate criterion loses power exactly where the project operates.** Two
distinct facts, which an earlier draft of this note conflated.

First, the constrained distribution's *absolute* violation rate is nearly invariant across
the sweep: 0.8755 to 0.8786, a range of 0.31 percentage points, while the KL moves
933-fold. The tilt shifts the charge distribution's mean while barely changing its spread,
and per-sample violation is dominated by the spread. So an absolute violation rate read on
its own says almost nothing about how hard the constraint is working.

Second — and this is what matters for CP4 — the *paired* difference between the
unconstrained base and the constrained tilt at the same alignment does track constraint
work closely, with a correlation of 0.94 against KL. Violation rate is therefore a valid
discriminator in an A3-versus-A2 comparison. But the size of the difference collapses as
data-constraint alignment improves:

| Absolute mean gap | KL (nats) | Base violation | Tilt violation | Difference | n/arm for 80% power |
|---|---|---|---|---|---|
| 5.66 | 3.083 | 0.994368152307 | 0.877109120908 | 11.7259 pp | 68 |
| 3.75 | 0.862 | 0.948080486065 | 0.877014462953 | 7.1066 pp | 247 |
| 2.03 | 0.214 | 0.900310649160 | 0.876506434241 | 2.3804 pp | 2,746 |
| 1.13 | 0.064 | 0.883834284917 | 0.876188011584 | 0.7646 pp | 28,350 |
| 0.26 | 0.003 | 0.876298046391 | 0.875888625086 | 0.0409 pp | 10,165,806 |

Test: two-proportion comparison, pooled-variance form, two-sided alpha = 0.05, power 0.80,
unpaired (the two arms are independently sampled generations, so McNemar does not apply).
**The base-rate pair (p1, p2) for each row is the row's own "Base violation" and "Tilt
violation" columns** — these are the enumerated exact rates of the S-conditioned base and
the S+E tilt at that alignment, not assumed values, and they are the two inputs the n
column is computed from. The difference column is p1 - p2. A row is reconstructible from
those three numbers alone. Note that the base rates here sit near 0.88-0.99 rather than
near 0.5; a calculation that assumes rates centered on 0.5 for the same difference gives n
in the 100-290 range, because binomial variance falls as p approaches 1.

**Accuracy of the approximation.** The normal-approximation formula is conservative at the
top row and accurate below it. Checked against direct binomial simulation at the tabulated
n: 400,000 replicates give empirical power 0.892 at n = 68, 0.811 at n = 247, and 0.803 at
n = 2,746, against the nominal 0.80. The first row's true 80% crossing is near n = 55. The
approximation degrades as p1 approaches 1, where the normal model overstates the variance
of a near-degenerate binomial. The tabulated values are retained because erring toward
more samples is the safe direction for a power floor, but any row whose base rate exceeds
about 0.95 should be simulation-checked before it sizes a real run. Sample sizes are the ceiling of the exact requirement,
since a fractional n is not attainable — the first row needs 67.46 and so is reported as
68, and the third needs 2745.28 and so is 2,746.

Rates are printed to twelve decimals deliberately. At four decimals the last two rows do
not reproduce at all (28,699 and 10,649,829), because rounding a sub-1-pp difference to
four decimals discards two of its significant figures and n scales as the inverse square
of that difference. Any table of this
kind must carry enough precision in its rates to regenerate its own n column, and the
precision required grows as the effect shrinks.

CP4 makes "A3 beats A2 on distribution fidelity and Regime-E violation rate" a pass
criterion, and MP-20 is a well-aligned dataset — 91.13% of its ground-truth compositions
already pass the SMACT charge-balance check. Its alignment is better than any point in
this sweep, so the expected effect sits at or below the bottom row's regime. The direction
of the criterion is sound; what it lacks is a pre-registered minimum detectable effect and
a sample-size calculation. Without one, a null result at CP4 is uninterpretable: it cannot
distinguish "the projection does nothing" from "the comparison had no power."

The synthetic task can supply the missing number, because the effect size is computable
in closed form at any alignment. The direct MP-20 quantity — the paired violation
difference at MP-20's own alignment — is not readable off this task, since the constraint
form and base rates differ; that requires the crystal parser and is a WS-2 deliverable.
What transfers is the shape of the problem and the requirement for a power calculation.

**The degenerate case is reachable and should be run deliberately.** At zero mean gap the
multiplier is exactly zero, the constrained target equals the base, and A3's converged
endpoint provably equals A2's. That is the collapse outcome CP0 is built to detect,
available here as a controlled setting rather than a risk. Running the synthetic bake-off
at both a satisfying and a violating alignment gives a positive and a negative control for
the collapse test before any crystal training starts.

## Verification

Seventeen tests pass across `tests/test_synthetic_targets.py` and
`tests/test_tilt_lemmas.py`. The sweep's own internal checks — multiplier sign opposition,
signed monotonicity, and per-side KL monotonicity — are recomputed and emitted with the
artifact rather than asserted here.
