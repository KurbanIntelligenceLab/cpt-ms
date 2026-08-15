# Results

Draft. Each subsection names the artifact it reads from; every figure was verified against
that artifact by recomputation rather than transcription.

## The two regimes are not interchangeable

`results/WS2_synthetic_ground_truth.json`, `results/WS1_induced_vs_tilt.json`

On a four-symbol vocabulary at length eight, all 65,536 sequences are enumerable, so every
distribution here is computed exactly rather than sampled. The support restriction leaves
36,120 sequences feasible holding 36.8% of the data mass, and the expectation constraint
binds: the restricted base has mean charge 2.516 against a target of 0.

The two notions of satisfaction are far apart. The distribution conditioned on
per-sequence satisfaction and the tilt matching the target in expectation differ by
2.083 nats and 0.875 in total variation, and the expectation-constrained optimum places
87.55% of its mass on individually violating sequences. Bounding an expected
increment bounds an expected total, not its variance.

The per-step surrogate is also not the sequence-level optimum. Applied exactly by
enumeration it reaches mean charge 0.1028 against the tilt's 0, at 1.052 nats forward
divergence and 1.025 nats of excess cost over the optimum. The mechanism is localized:
across 16,537 enumerated nodes the running budget is unsatisfiable at 37.13% of steps,
first at step 5 of 8. Where the budget becomes unsatisfiable partway through a
sequence, the surrogate misses the constraint it was built to enforce.

## The layer is correct and its cost is bounded

`results/CP2_gradient_verification.json`, `results/CP2_overhead_benchmark.json`

The forward solve reaches a worst residual of 7.8e-11 and its gradient agrees with finite
differences to 4.6e-10 and with an independent differentiable-optimization implementation to
2.0e-06 across 5 constraint configurations. Two hundred random systems produced no
rank-deficient case.

In isolation the layer is expensive — 72.6x forward and 13.85x forward-and-backward
against an unconstrained step. Measured as a fraction of a real training step on a
29.7M-parameter model the figure that matters is 13.6%-19.9% at the positions where the
constraint is actionable. The published budget was exceeded on one measurement path and
the artifact records the superseded figure alongside the measured one rather than
replacing it silently.

## Steering works, asymmetrically, with a measured ceiling

`results/WS3_steering_frontier_rows.json`, `results/WS3_crystal_A3_f2_seed*.json`

At a density target of 5.0 g/cm^3 the deployed system reaches 4.810 against an unsteered
baseline of 6.783, an error of 0.190 against 1.783. Removing the projection at inference
inverts rather than degrades the result: the same model read without its layer gives 9.200.

The complementarity of the layer and the training is measured rather than assumed. The
total improvement of 1.593 g/cm^3 over the unsteered baseline decomposes into 1.072 from
applying the layer to an untrained backbone and 0.520 from training through it, a 67/33
split. The second part's mechanism is measured independently: the probability-weighted
infeasible-step fraction falls from 0.3430 at initialization to 0.0496 at convergence, so
training moves the backbone into a region where the running budget is attainable rather
than teaching it to satisfy the constraint directly. Removing the projection at inference
gives an error of 4.200 against the unsteered baseline's 1.783 — worse than not steering at
all, which is the sharpest evidence that the layer is a training component rather than a
decoder.

On the synthetic task, where the target is computed by enumeration rather than estimated,
the same arm reaches mean charge 0.0170 against a target of 0, against 0.1028 for the
exact projection applied to raw data — 6.0 times closer, with the infeasible-step
fraction lower by a factor of 19.8.

Across ten targets the asymmetry is systematic. Mean absolute error is 0.345 g/cm^3 at or
below the data mean of 6.679 and 4.577 above it, and achieved density saturates at
10.405 regardless of the request. The cause is measured rather than inferred: the
infeasible-step fraction rises from 0.0079 to 0.4133 with the target, and training's
ability to repair it falls from 29.2x to 0.97x — at the highest target training no longer
reduces infeasibility at all. Steering downward always has light elements available;
steering upward eventually demands more mass per atom than the heaviest legal token
supplies.

## One training run covers a family of targets

`results/WS4_zero_shot_retargeting.json`, `results/WS4_retargeting_seed_replication.json`

The target is an argument to the dual solve, not a training-time label, so a checkpoint can
be deployed at targets it never saw. Three checkpoints deployed across ten targets give 30
cells with structural validity exactly 1.0000 in every one and novel-formula fraction
between 0.928 and 0.982 throughout.

Held against the frontier's separately trained models at the seven sub-ceiling targets —
the comparison that holds training compute fixed — one checkpoint reaches mean error
0.650 ± 0.151 g/cm^3 against 0.662 from 7 runs. It is better at 3 of 7 targets,
the mean paired difference is -0.012 g/cm^3, and a Wilcoxon signed-rank test over the
paired targets gives p = 1.0 across three seeds. One training run is statistically
indistinguishable from seven.

A stronger reading was available and did not survive replication. The first checkpoint
trained at 10.0 reached 15.495, well past the 10.405 ceiling, with a backbone at
22.769. Two further seeds gave backbones of 8.381 and 8.692 and reached 10.439 and
10.596. Every seed exceeds the ceiling, so it is soft rather than hard, but the size
of the excursion belongs to the checkpoint rather than to the method.

## Simultaneous constraints: exact, but not composable

`results/CP6_multiconstraint.json`, `results/CP7_second_channel.json`,
`results/WS5_two_constraint_grid.json`

Exactness extends to several constraints at once. At [1, 2, 3, 4] simultaneous constraints all
solves converge, with worst residual 8.71e-13 and worst multiplier-Hessian condition
number 1.739. Reporting a residual without the conditioning would hide a converged
solve on a near-singular system, so both are recorded.

Whether constraints compose is a separate question with a different answer. Joint
attainability over pairs of targets depends on vocabulary size, rising from 53.1% at
vocabulary 8 to 100.0% at 256; at the smallest vocabulary 46.9% of grid points are
attainable for each constraint separately but not for both together.

On real crystals that regime binds. A second additive family — valence electron density,
correlated at 0.725 with mass density — solves jointly with density on 64 crystal
prefixes at worst residual 7.67e-13, but only at element positions: of the positions with
channel variation, 260 carry two independent constraints and 196 carry only one, because at
a count position both channels are proportional to the count.

Trained jointly, the second dial does not steer. Over a requested range of 0.404 the
achieved values span 0.0250 — pass-through 6.2% — and the response is not monotone
(0.2447, 0.2603, 0.2353 for increasing targets), with the largest deviation from an
unconstrained control being 0.0179, or 8.9% of one data standard deviation.

The mechanism is competition for one token budget. Fallback to the single constraint rises
0.102 → 0.518 → 0.826 with the second target (correlation 0.9963) while jointly solvable
positions fall 230 → 136 → 49 (correlation -0.9998). Of exactly 518 actionable positions per
cell, 236 to 262 are excluded because one running budget has reached zero while the other
is still positive. Both quantities are additive in element identity and count, so the atoms
satisfying one budget are the atoms the other needs. Density steering is unaffected
throughout (3.281, 3.041, 3.071 against target 3.0, against a control at 3.175) and
structural validity stays exactly 1.0000.

## The expressible class has a boundary

`results/CP7_second_channel.json`

The constraint a materials reader asks for first is charge neutrality, and it is not a
linear functional. As the plausibility screen here measures it — at a real-data rate of
0.7150 — it is existential: a composition is balanced if there exists an assignment of
formal charges from each element's allowed oxidation states summing to zero. Of 87
elements in the training split, 85 have tabulated states and 71 are multivalent, so
81.5% of element-occurrences involve an element whose charge is not fixed by its
identity.

Collapsing the quantifier to one charge per element makes the constraint linear and
destroys it. On 4,000 structures the screen calls 71.08% balanced and the linear proxy
7.95%; they agree on 36.78%, and the proxy recovers 11.12% of the balanced set with
2,527 false negatives against 2 false positives. A target built on that proxy would
steer a quantity nearly unrelated to the screen the paper reports.

## Constraint satisfaction is not sample quality

`results/WS2_parse_validity.json`, `results/WS3_plausibility_reference.json`,
`results/WS3_energy_calibration.json`

Syntactic and semantic validity are measured separately, because reporting either alone
misleads in a different direction. Of masked samples 500 of 500 parse, while 47% build into
a structure given that they parse; the shortfall is physically incoherent cell geometry,
which the grammar never claimed to prevent. Charging the grammar for that would understate
it, and quoting only the parse rate would overstate what the method delivers.

Plausibility is scored against a reference obtained by pushing 800 real MP-20 structures
through the identical path, and that reference is not 1.0: real materials satisfy the charge
screen at 0.7150, electronegativity ordering at 0.6400, and packing at 0.9775. An arm scored
against an assumed ceiling of 1.0 would be penalized for a property of the screen rather
than of the model.

An ML-potential energy screen was calibrated against MP-20's DFT labels and agrees to
21.0 meV/atom on 1,456 held-out structures, with correlation 0.99958 — 2.0% of the
label spread. It is used for reference structures only, is labelled an ML-potential screen
rather than DFT throughout, and no DFT calculation is reported in this work.

## The backbone's unaided distribution is not well behaved

`results/WS3_steering_frontier_rows.json`

Backbone and deployed readings are reported at every cell because either alone misleads.
The backbone drifts furthest above the data mean when the target sits near the achievable
ceiling, fitting distance-to-ceiling at R^2 = 0.810 over nine rows, and returns toward the
data at both extremes. In one run it produced a mean density exceeding the maximum in MP-20
itself. The compensation hypothesis — that the backbone learns an offset calibrated to the
projection's pull — was tested and rejected.

## Against the strongest baseline the result is a trade

`results/WS3_conditional_baseline.json`, `results/WS3_conditional_sparse.json`,
`results/WS3_fidelity_baseline.json`

Conditional training — a masked model trained only on the data slice near the target — is
more accurate wherever it can run: mean absolute error 0.085 against 0.196 below the data
mean and 0.027 against 1.455 above it. It buys that accuracy by reproducing a slice:
5.9% novel formulas against 94.1%, and 44.0% unique against 78.6%.

The price rises as the slice thins, which is where a generative model is wanted. Unique
formulas fall 0.519 to 0.031 and distinct elements 84.5 to 56.5 as the training slice shrinks,
correlating with log slice size at 0.981. At the extreme the slice holds 6 structures and the
baseline cannot be trained at all, while the steered model still produces novel
compositions there.

Rejection sampling from the data supplies the exact comparator and shows the same shape
from the other direction: expected draws to hit a target within 0.5 g/cm^3 range from
7.4 to 4485, and at the hardest target only 6 of 26,910 structures qualify.
