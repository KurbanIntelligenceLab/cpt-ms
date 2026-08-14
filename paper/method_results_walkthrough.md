# Constraint-Projected Transformer — method and results, step by step

A self-contained walkthrough for an outside reader judging whether the contribution is novel
enough to carry a paper. Each section states what was built, what was measured, and what the
measurement returned. Every number is read from an artifact in `results/`; the file is named at
the head of each section so any figure can be checked at its source.

## The method in one paragraph

A differentiable projection layer is inserted into the decoding step of an autoregressive
generative model and trained through, rather than applied at inference. At each step it maps the
model's next-token distribution onto the set satisfying a constraint expressed as a linear
functional of the next token, solved as a dual Newton iteration whose gradient is available in
closed form through the implicit function theorem. Two constraint regimes are kept separate
throughout. Regime S is a support restriction: a mask making some tokens unreachable. Regime E is
an expectation constraint: a running budget requiring a statistic of the completed sequence to
meet a target.

Five arms recur below. A1 is unconstrained training. A2 adds the Regime-S mask. A3 adds the
Regime-E projection and trains through it, and is read two ways — the backbone reading applies the
mask only, and the deployed reading applies the same projection the model trained with. A4 is the
conditional-training baseline: a masked model trained only on the data slice near the target, with
no decode-time constraint. Rejection sampling from the data supplies an exact comparator.

## Step 1 — An exactly enumerable synthetic task

`results/WS2_synthetic_ground_truth.json`

Method. A four-symbol vocabulary at length eight gives 65,536 sequences, so every distribution
below is computed by enumeration rather than sampled. Symbols carry integer charges; the Regime-E
target is a total charge of zero, and the Regime-S mask restricts which symbols may appear.

Results. The mask leaves 36,120 of 65,536 sequences feasible, holding 36.8% of the data mass. The
constraint binds: the S-conditioned base has mean charge 2.516 against a target of 0. Four exact
reference distributions were computed — the base, the base conditioned on hard per-sequence
satisfaction, the exponential tilt that matches the target in expectation, and the tilt restricted
to the mask.

The result that shapes every later claim: the exact expectation-constrained optimum places 87.55%
of its mass on individually violating sequences. Bounding an expected increment bounds the
expected total, not its variance. The hard-conditioned distribution and the expectation-matched
tilt are 2.083 nats apart and separated in total variation by 0.875 — they are different targets,
not two roughly equivalent notions of constraint satisfaction.

## Step 2 — The per-step surrogate is not the sequence-level optimum

`results/WS1_induced_vs_tilt.json`

Method. The per-step projection is applied exactly, by enumeration, to the base distribution, and
the induced sequence distribution is compared against the exponential tilt that solves the same
constraint at the sequence level.

Results. The induced distribution reaches mean charge 0.1028 against the tilt's 0. Divergence from
the tilt is 1.052 nats forward and 35.50 nats reverse, with total variation 0.555. The surrogate
induces a materially different distribution, not a slightly worse one, and it pays 1.025 nats of
excess divergence over the optimum.

The mechanism is localized. Across 16,537 enumerated nodes the running budget is unsatisfiable at
37.13% of steps, and the first infeasible position is step 5 of 8 — infeasibility is absent early
and reaches 46% of nodes at the final position. Where the budget becomes unsatisfiable partway
through a sequence, the surrogate misses the constraint it was built to enforce.

## Step 3 — When the surrogate and the optimum coincide

`results/WS2_alignment_sweep.json`

Method. The data distribution is swept through fifteen Markov-bias settings, moving the data's own
mean charge relative to the fixed target, and the dual multiplier and divergence are traced in
closed form.

Results. At the best-aligned setting the mean gap is 0.261 and the multiplier is 0.0253: as
data-constraint alignment improves the multiplier vanishes and the constrained target collapses
onto the base, so A3's converged endpoint provably equals A2's. The approach is governed by clean
exponents — the multiplier scales with the gap at log-log slope 1.026 and the divergence at 2.038,
the quadratic-versus-linear relationship expected of a KL-projection.

A decoupling worth reporting: across the sweep the multiplier spans 35.2-fold and the divergence
933-fold, while the tilt's per-sequence violation rate moves only from 0.8755 to 0.8786. Violation
rate is nearly invariant to how hard the constraint is pulling, which is why it is not used as the
headline metric anywhere below.

## Step 4 — The projection layer, verified against two independent oracles

`results/CP2_gradient_verification.json`

Method. The layer solves the dual by Newton iteration and returns gradients through the implicit
function theorem. Gradients are checked against `cvxpylayers`, which differentiates the same
program by a different route, and against a finite-difference oracle whose step size is swept
rather than fixed.

Results. Across five constraint configurations, forward residuals reach at most 7.8e-11 in four to
five Newton iterations, all converged. Gradients agree with the step-swept finite-difference
oracle to 4.6e-10 relative and with `cvxpylayers` to 2.0e-6 relative. Two hundred randomized
Jacobian rank checks found zero rank-deficient cases.

A defect found here is recorded with its mechanism rather than as a changelog line: the dual
line search failed on flat-dual cases near the optimum. Its regression test is confirmed
load-bearing — it fails against the reverted fix at offsets 1e-8 and 1e-9 and passes when restored.

## Step 5 — What the layer costs

`results/CP2_overhead_benchmark.json`, `results/CP2_hessian_speedup_sweep.json`

Method. Wall-clock overhead is measured on the training GPU, an RTX 5090 under torch 2.8.0 with
CUDA 12.8, against two denominators: the bare masked softmax the layer replaces, and a full
training step of a 29.7M-parameter causal decoder at the project's own model scale.

Results. Against the masked softmax the forward pass is up to 72.6x and forward-plus-backward
13.85x — the pre-registered 30% budget fails by a wide margin against that denominator. Against a
full training step the same measurement reads 52.1% at every position for one constraint, and
13.6% to 19.9% when scoped to the actionable positions where the projection has anything to
decide. Two constraints at every position reach 167.6%.

The denominator, not the measurement, was the substantive issue, and it was resolved by team
amendment (`preregistration/CP2_amendment_denominator.md`). One recorded defect is
methodological: the first scoping table combined a per-token figure measured at batch 256 with
shares measured at up to 8,192 rows, and its two-constraint rows silently used an opt-in
fast-Hessian path while being presented as the default. Corrected, the two-constraint share moves
from 70.1% reported to 167.6% measured. The float64 Hessian is 43x slower than float32 in raw
throughput but at most 1.70x slower in this solve — and at the smallest batch the fast path is
0.98x, i.e. no faster at all — while the two agree to exactly 0.0, so the accurate path is the one
that trains.

## Step 6 — Training the three arms on the synthetic task

`results/A2_training_summary.json`

Method. Three arms, three seeds each, 4,000 steps, identical initialization and data order. A3 is
read both ways.

Results. Regime S is exact by construction: A2 places 0.0 mass outside the feasible set against
A1's 8.2e-4, at a cost of 0.0176 nats from the data. On Regime E the deployed system reaches mean
charge 0.0170 against a target of 0 — six times closer than the exact per-step projection applied
to raw data at 0.1028 — while the backbone alone reaches 0.6287. The layer and the training are
therefore complementary rather than redundant.

The mechanism is measured independently of the outcome. The probability-weighted infeasible-step
fraction falls from 0.3713 under the exact projection on raw data to 0.0187 in the trained
deployed system, a factor of 19.8. Training moves the backbone into a region where the running
budget is attainable, so the fallback rarely fires and residual error does not accumulate.

Training through the layer costs 6.0x wall clock against the masked arm at this scale. A
train-inference precision question was closed rather than assumed: both deployed distributions
come from the same weights, so any difference is precision alone, and the total variation between
the float64 and float32 Hessian routes is 2.1e-10 over 10^6 samples, with the two-proportion test
on violation rate giving z below 0.97 on every seed.

## Step 7 — Where a constraint is computable, and where it can still be moved

`results/WS2_family_determinability.json`

Method. Two constraint families are instrumented per step on three crystal serializations. Family
(i) is charge balance, family (ii) is density. Determinability is when the constraint becomes
computable from the prefix; actionability is the fraction of positions where a format-legal
substitution still changes its final value. Only the second bounds what a per-step projection can
do.

Results. The two differ by up to a factor of 23, and the ordering is serialization-dependent.

| Serialization | Family | Usable fraction | Actionable fraction |
|---|---|---|---|
| B1, naive CIF layout | charge | 0.0021 | 0.0570 |
| B1, naive CIF layout | density | 0.6691 | 0.2831 |
| B1ref, reference field order | charge | 0.9493 | 0.0410 |
| B1ref, reference field order | density | 0.4897 | 0.3849 |
| B2, typed fields | charge | 0.7848 | 0.1280 |
| B2, typed fields | density | 0.4795 | 0.3896 |

On the naive CIF layout, charge balance is determinable only over the last 0.2% of a 300-800
character sequence, because CIF spells out composition in the atom-site loop at the very end.
Density has three to nine times more actionable room on every serialization, which is why density
was pre-registered as the primary steering dial on all three
(`preregistration/WS2_dial_assignment.md`).

A defect found here is instructive about the measurement rather than the method: the first pass
reported B2's two families as identical at 0.1280, which is structurally impossible, because the
fixture emitted placeholder element symbols carrying no charge and no mass. Two tests now cover
it, one of them asserting the strict inequality whose violation was the defect's signature.

## Step 8 — Regime S on real crystals

`results/WS2_parse_validity.json`

Method. 500 samples per arm from a CIF character model with and without grammar masking, judged by
`pymatgen`'s CIF parser, which shares no code with the emitter. Both arms use the same character
proposal distribution, so the comparison isolates the grammar.

Results. 500 of 500 masked samples parse; 0 of 500 unmasked samples do, all failing with no data
block at all. The mask is not merely forcing the output — it leaves a mean of 4.14 allowed
characters per step across 80,228 measured positions, with 29.4% of steps offering a real choice.

The same artifact bounds the claim. Of the masked samples that parse, 47% build into a structure;
the remaining 53% are physically incoherent cell geometries the grammar never claimed to prevent.
Reporting only the second figure would charge the grammar for failures outside its scope;
reporting only the first would overstate what the method delivers.

## Step 9 — The three arms on MP-20

`results/WS3_crystal_A{1,2,3}_seed{0,1,2}.json`, `results/WS3_crystal_A3_f2_seed{0,1,2}.json`

Method. A 5.0M-parameter model on 26,910 MP-20 training structures, 6,000 steps, three seeds,
density target 5.0 g/cm^3 against a data mean of 6.679 and median 6.317.

Results. Regime S transfers exactly: A2 reaches structural validity of 1.0000 on every seed
against A1's 0.9789 under identical training, so the claim is not vacuous, and an untrained masked
model also reaches 1.0000, which is the sharpest form of it.

On Regime E the deployed system reaches 4.810 +/- 0.153 against the unsteered A2 baseline's 6.783,
landing 0.190 from target against the baseline's 1.783 — 9.4x closer, a 29.1% relative shift
against a 1.1% discretization floor. Separation is Cohen's d = 14.4 with three seeds per arm,
Welch's t = 17.66, p = 0.0017; Welch rather than Student because the arm variances differ by 13.9x.

The gain decomposes, and both parts are published: the layer applied to an untrained backbone
closes 1.072 of the 1.592 gap (67%) and training through it closes the remaining 0.520 (33%). The
complementarity claim attaches to that third, and its mechanism figure is independent — the
probability-weighted infeasible-step fraction falls from 0.3430 at initialization to 0.0496 at
convergence, a factor of 6.9 against 19.8 on the synthetic task.

Backbone against deployed runs the other way here, and this is reported as measured rather than
smoothed into the synthetic story. The backbone reading is 9.200 while the deployed system is
4.810: under a constraint pulling density down, the backbone's unaided distribution sits 2.5 above
the data mean. It has not internalized the constraint; it has adapted to a training signal that
already contains the projection. Removing the layer at inference does not degrade the result by a
margin, it inverts it — from 4.810 to 9.200, an error of 4.200 against the baseline's 1.783.

Two defects in the density formulation were found by asking whether real structures satisfy the
demanded target, not by inspecting the implementation. The budget denominator was in the wrong
units, dividing a per-atom mass channel by remaining element tokens rather than remaining atoms,
which demanded a median 177u against the data's actual 59u and made 32.7% of actionable positions
demand more mass than the heaviest element in the periodic table. And the constraint was applied
at non-actionable positions, where the attainable range is the single point zero and any nonzero
budget is infeasible by construction. Neither defect crashed; both trained to completion while
enforcing a constraint the training data itself could not satisfy.

## Step 10 — The steering frontier and its ceiling

`results/WS3_steering_frontier_rows.json`

Method. Ten density targets from 1.5 to 20.0 g/cm^3, two seeds each, 18 completed runs. Backbone
and deployed readings at every target, with infeasibility measured at initialization and at
convergence.

Results. Steering is asymmetric and has a ceiling. Mean absolute error is 0.345 g/cm^3 for targets
at or below the data mean and 4.577 above it, and achieved density saturates at 10.405 regardless
of the request.

| Target | Deployed | Error | Backbone | Infeasibility at init | At convergence | Repair factor |
|---|---|---|---|---|---|---|
| 1.5 | 1.848 | 0.348 | 8.087 | 0.2315 | 0.0079 | 29.2x |
| 2.0 | 2.266 | 0.266 | 8.191 | 0.2602 | 0.0117 | 22.2x |
| 3.0 | 3.240 | 0.240 | 9.034 | 0.2989 | 0.0220 | 13.6x |
| 5.0 | 4.849 | 0.151 | 9.196 | 0.3379 | 0.0483 | 7.0x |
| 6.317 | 5.598 | 0.719 | 10.422 | 0.3528 | 0.0649 | 5.4x |
| 8.0 | 6.760 | 1.240 | 10.261 | 0.3666 | 0.0875 | 4.2x |
| 10.0 | 8.330 | 1.670 | 15.591 | 0.3777 | 0.1224 | 3.1x |
| 13.0 | 8.744 | 4.256 | 9.798 | 0.3948 | 0.2068 | 1.9x |
| 16.0 | 9.875 | 6.125 | 8.458 | 0.3843 | 0.3051 | 1.3x |
| 20.0 | 10.405 | 9.595 | 7.803 | 0.4002 | 0.4133 | 0.97x |

Targets 13.0 and 16.0 carry one seed each; both sit in the failed region.

The mechanism is measured rather than inferred. Steering down always has light elements available;
steering up eventually demands more mass per atom than the heaviest legal token supplies, at which
point the running budget becomes infeasible, the fallback fires, and the constraint stops binding.
Infeasibility at convergence rises monotonically with target from 0.0079 to 0.4133, and training's
ability to repair it falls from 29.2x to 0.97x — at the highest target, training no longer reduces
infeasibility at all.

The backbone's unaided distribution is not a well-behaved object, and pretending otherwise would
have produced a false claim. It varies non-monotonically with steering difficulty, drifting
furthest above the data mean where the target sits near the achievable ceiling and returning
toward the data at both extremes. Regressing the offset on distance to the 10.405 ceiling gives
R^2 = 0.810 over the nine rows outside the single-seed excursion, with a better adjusted R^2 than
the quadratic in target at 0.828 and one parameter fewer. At target 10.0 one seed produced 22.894
against the other's 8.288, so the seed-averaged 15.591 is a single-seed excursion and is reported
as such. A linear test for compensation on this relationship would be the same class of error as
the invalid regression described next.

The compensation hypothesis — that the backbone learns an offset calibrated to the projection's
pull — was tested and rejected, and the first test of it was invalid. The identity
offset minus pull equals deployed minus data mean holds exactly on the table values, so that
regression was a quantity against itself. It was caught by two checks contradicting each other
rather than by inspecting the fit, and is pinned by a test showing random inputs pass the invalid
framing. The rejection stands on the pull-sign argument, which does not depend on the regression.

## Step 11 — The efficiency comparator that does not weaken

`results/WS1_steering_frontier.json`, `results/WS3_fidelity_baseline.json`

Method. Rejection sampling from the base distribution, accepting only samples that meet the target
exactly on the synthetic task and within 0.5 g/cm^3 on MP-20, gives an exact comparator whose cost
is one over its acceptance rate. The comparison is on efficiency, divergence, and mass on target
— never efficiency alone.

Results. On the synthetic task, expected draws per accepted sample range from 7.1 to 362.2 across
seven targets, and target-mass enrichment from 2.6x to 183.3x. Excess divergence over the
sequence-level optimum is approximately fixed overhead at 1.070 +/- 0.160 nats, while the
divergence the optimum itself must pay spans 129-fold. The wasted fraction therefore falls from
97.8% at the weakest steering to 32.4% at the strongest: per-step projection is nearly all
overhead where the base model already satisfies the constraint, and increasingly worth its cost as
the constraint pulls harder.

On MP-20 all ten targets are reachable by rejection from the empirical data, at expected costs
from 7.4 draws at target 5.0 to 4,485 at target 20.0 where only 6 of 26,910 structures qualify.
The comparator is stated as generous to itself: it draws from the data, so it assumes a generator
that reproduces the data distribution perfectly.

## Step 12 — The conditional-training baseline, which wins on accuracy

`results/WS3_conditional_baseline.json`, `results/WS3_conditional_sparse.json`

Method. A4 is a masked model trained only on the MP-20 subset within 0.5 g/cm^3 of the target,
with no decode-time constraint — the "just retrain on filtered data" objection, instantiated.
Sixteen runs across four targets and two seeds, plus sparse-target runs at 13.0 and 16.0. Both
arms share initialization, data order, optimizer settings and step budget, and both are scored
against the full split's chemistry rather than A4's filtered slice. A4's budget is not reduced,
which is generous to it: its training set is 5-14% of the split, so it sees each structure roughly
7-19 times more often.

Results. On the target metric alone, the objection wins.

| Target | Arm | Train set | Density | Error | Validity | Novel | Unique |
|---|---|---|---|---|---|---|---|
| 3.0 | A4 | 2,392 | 3.138 | 0.138 | 1.0000 | 0.073 | 0.422 |
| 3.0 | A3 | 26,910 | 3.240 | 0.240 | 1.0000 | 0.927 | 0.640 |
| 5.0 | A4 | 3,630 | 5.014 | 0.033 | 1.0000 | 0.059 | 0.519 |
| 5.0 | A3 | 26,910 | 4.849 | 0.151 | 1.0000 | 0.946 | 0.782 |
| 8.0 | A4 | 3,034 | 8.027 | 0.047 | 1.0000 | 0.073 | 0.500 |
| 8.0 | A3 | 26,910 | 6.760 | 1.240 | 1.0000 | 0.941 | 0.845 |
| 10.0 | A4 | 1,411 | 10.008 | 0.008 | 1.0000 | 0.029 | 0.318 |
| 10.0 | A3 | 26,910 | 8.330 | 1.670 | 1.0000 | 0.948 | 0.876 |

Mean absolute error at or below the data mean is 0.085 for A4 against 0.196 for A3, a factor of
2.3. Above it, 0.027 against 1.455, a factor of 53. The high-target gap is the steering ceiling
already reported: A4 has no ceiling because it never steers, so its advantage above the mean is
evidence that not steering avoids the failure mode of steering, which is a different claim from
steering better.

What A4 gives up is coverage, by more than an order of magnitude: 5.9% novel formulas against
94.1%, and 44.0% unique against 78.6%. Roughly 94% of its compositions are copied from its own
training data. Structural validity is exactly 1.0000 for both arms at every target and seed,
including the arm that loses on accuracy and the targets where steering fails.

A claim about A4 starving at extreme targets was drafted and then measured, and the measurement
refuted it.

| Target | A4 train set | A4 error | A3 error | A4 novel | A4 unique | A4 distinct elements |
|---|---|---|---|---|---|---|
| 5.0 | 3,630 | 0.033 | 0.151 | 0.059 | 0.519 | 84.0 |
| 10.0 | 1,411 | 0.008 | 1.670 | 0.029 | 0.318 | 79.5 |
| 13.0 | 388 | 0.031 | 4.256 | 0.016 | 0.108 | 73.5 |
| 16.0 | 119 | 0.071 | 6.125 | 0.004 | 0.031 | 56.5 |
| 20.0 | 6 | refused | 9.595 | — | — | — |

A4 stays accurate on 119 structures. What degrades is coverage, and it tracks slice size rather
than target distance: correlation with log training-set size is 0.981 for unique formulas, 0.908
for novel, and 0.962 for distinct elements. Unique formulas fall 17-fold and the element inventory
contracts from 84 to 56. At target 16.0, A4 emits 3.1% unique formulas — roughly four distinct
compositions recited from a slice seen thousands of times. It hits the target because those four
compositions are the target. At 20.0 the filter leaves 6 structures and A4 refuses to train, the
one genuine starvation case among the ten targets, and the refusal is the measurement.

A metric was nearly reported wrongly here and the diagnosis is part of the result. Element-frequency
total variation against the full data does not separate the arms — it favours A3 at target 3.0 and
A4 at 5.0, so either target alone would have supported either conclusion. A4's TV tracks the TV
between its own filtered slice and the full data to within 0.004, which is what reproducing a
slice looks like. TV measures the shift any arm must make to reach a non-mean target; only the
unique- and novel-formula fractions separate reproduction from reach.

## Step 13 — What can and cannot be said about sample quality

`results/WS3_energy_calibration.json`, `results/WS3_plausibility_reference.json`

Method. An ML-potential energy screen was built and calibrated against MP-20's own DFT labels
before being used to judge anything: CHGNet v0.3.0 predicts total energy, so formation energy
required fitting 79 elemental references by least squares, with the train-test split taken before
the reference fit.

Results. Held-out agreement on 1,456 structures is 0.0210 eV/atom MAE, 0.0298 RMSE, +0.0015 bias,
Pearson r = 0.99958 against a label spread of 1.0265 eV/atom. MAE is 2.0% of the spread and below
the 30 meV/atom window conventionally used for near-hull stability.

The screen cannot be applied to the generated samples, and this is a structural limit rather than
a tooling gap. The serialization is composition plus cell with no atomic positions, and an energy
method requires positions. Three routes were considered and each fails for a recorded reason:
solving for positions is crystal structure prediction, which the method does not attempt; relaxing
from random positions measures the initialization; and matching to a known prototype imports
plausibility from the match. A working DFT engine is available and verified — GPAW 25.7.0 with libxc 7.1.2, PAW datasets and
MPI on the compute box, completing a PBE calculation on two-atom silicon in the diamond structure
at a 300 eV cutoff and 3x3x3 k-points in 3.2 s — and is not used for the same reason:
generated samples have no positions, which blocks DFT and the surrogate identically. Nothing in
this work is labelled DFT; the artifact carries `is_dft: false` as an asserted field.

What is measured instead is position-free plausibility — charge neutrality, electronegativity
ordering, and packing — each reported against a reference obtained by scoring 800 real MP-20
structures through the identical path.

| Screen | Real MP-20 |
|---|---|
| Charge balanced | 0.7150 |
| Electronegativity ordered | 0.6400 |
| Packing plausible | 0.9775 |

The reference matters more than the screens. Real materials fail the charge screen 28.5% of the
time, because oxidation-state enumeration covers only common states, so an arm scored against an
assumed ceiling of 1.0 would be penalized for a property of the screen. An arm reaching 0.70 would
be indistinguishable from the data, which is the outcome that matters and is invisible without the
reference in hand.

## What the results support as claims

Regime S is exact and holds by construction, on synthetic data and on real crystals, trained and
untrained, at every target including those where steering fails outright. This is a property of
the mask, not a statistical property of a trained model, and it is independent of every steering
result.

Regime E steers within and below the bulk of the training distribution, and the projection and the
training are complementary rather than redundant, with the split published as 67/33 and the
mechanism measured independently as the infeasibility drop.

The projection is a training component rather than a decoder. Removing it at inference inverts the
crystal result rather than degrading it by a margin, which is the sharpest available evidence for
that framing and the case where the decoder reading of the method fails outright.

The per-step surrogate is not the sequence-level optimum, and the gap is characterized rather than
asserted: a fixed overhead of about 1.07 nats against an optimum whose own cost spans 129-fold, so
the surrogate's relative waste falls from 97.8% to 32.4% as the constraint pulls harder.

Against the strongest baseline, the honest claim is a trade rather than a win. Conditional
training is more accurate everywhere it can run and buys that accuracy by reproducing a slice,
with the price rising as the slice thins — precisely where a generative model is wanted. The
steered model is less accurate and degrades past its ceiling, but produces novel compositions at
every target, including the one where the conditional baseline has no training data at all.

## What is not claimed

No claim is made that the layer improves in-distribution sample quality. No claim covers
constraints the projection's linear-functional form cannot express, nor quantities not determinable
from a prefix. Thermodynamic stability of generated structures is not assessed. Expectation
constraints are satisfied in expectation, not per sample. The compensation hypothesis was tested
and rejected.

## What has not been run

The steering results rest on one crystal serialization; the training entry point exposes an arm
selector but no backbone selector, so the two-backbone comparison the plan describes would need
both a flag and more runs. Crystal arms carry three seeds at the fixed target and two per target
across the frontier, against the five-seed matrix the plan sizes. The two extreme frontier targets
carry one seed each. The A6 comparator that fine-tunes toward the sequence-level tilt directly is
specified and scoped in code but not trained, so the surrogate-versus-optimum comparison is exact
on the synthetic task and absent on crystals.

## Verification posture

Nothing quantitative in the reports is typed by hand: every number is read from an artifact in
`results/`, each artifact carries its script, config hash, seeds, commit, environment and payload
checksum, and narrative verdict fields inside artifacts are generated from the values they
describe. The suite is 209 tests, all passing. Several of those tests exist because a claim in a
report was found to be wrong — a stale test count that survived a green suite through a regex
gap, a Regime-S exactness figure that verified because the field was absent, a per-file count that
was corrected in the wrong direction — and each is now pinned by a check that fails against the
reverted text.
