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

![Steering frontier and its mechanism]({{artifact:art_5ede2afb-a542-4260-9833-f46b3c033c13}})

**Figure 1.** Steering is asymmetric and the asymmetry has a measured cause. (a) Achieved against
requested density; the dotted line is unity, the dashed line the achieved ceiling of 10.40 g cm^-3.
Deployed points carry the projection, backbone points the mask only, and open markers denote the
two targets measured at one seed rather than two. (b) The infeasible-step fraction at
initialization and at convergence, with the repair factor (init / converged, log scale) on the
right axis. Training reduces infeasibility by a factor of 29 at the lowest target and by
0.97 at the highest — at the top of the range training no longer repairs infeasibility at all,
which is why the achieved density saturates.

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
paired targets gives p = 1.0 across three seeds.

A large p-value is not evidence of equivalence, so the claim is made as a bounded one rather
than as an unrejected null. The 95% confidence interval on the mean paired difference is
[-0.111, 0.086] g/cm^3, and the design could have detected 0.135 g/cm^3 at 80% power —
20% of the per-target arm's own mean error. A two-one-sided-tests procedure
establishes equivalence at a margin of 0.189 g/cm^3 (p = 0.0023), that margin being the
retargeted arm's own mean within-target seed standard deviation: a difference smaller than
the method's run-to-run reproducibility is not one a user could act on. The conclusion is not
margin-sensitive — equivalence holds down to a margin of 0.10 g/cm^3 (p = 0.0362).

A stronger reading was available and did not survive replication. The first checkpoint
trained at 10.0 reached 15.495, well past the 10.405 ceiling, with a backbone at
22.769. Two further seeds gave backbones of 8.381 and 8.692 and reached 10.439 and
10.596. Every seed exceeds the ceiling, so it is soft rather than hard, but the size
of the excursion belongs to the checkpoint rather than to the method.

## Simultaneous constraints: exact, and composable only where targets agree

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

Trained jointly, how much the second dial moves depends on whether the two targets pull the same
way. Where they conflict it is inert: at density target 3.0, a requested range of 0.404 yields an
achieved range of 0.0250 — pass-through 6.2% — and the response is not monotone
(0.2447, 0.2603, 0.2353 for increasing targets), deviating from an unconstrained control by
0.0179, or 8.9% of one data standard deviation. Where they agree — density target 8.0, where a
high density and a high valence electron density both call for heavier or more numerous atoms per
volume — pass-through rises to 11.3%, the response is monotone
(0.3251, 0.3482, 0.3707), and the deviation from control is 0.0705, 3.9 times the
conflicting row and 34.9% of one standard deviation. Neither row makes the dial controllable.

The mechanism separates the rows more cleanly than the pass-through does. Fallback to the
single-constraint solve rises with the second target in both, correlating at 0.9963 and 0.9997.
But jointly solvable positions move in opposite directions: they fall with the target where the
constraints conflict (correlation -0.9998, from 230 to 49 of 518 actionable positions) and rise
where they agree (correlation +0.7842, from 265 to 319), while budget-exhausted exclusions fall
from 212 to 53 across the compatible row as the two demands align. Both quantities are additive in
element identity and count, so the atoms satisfying one budget are the atoms the other needs; how
badly they compete is set by whether the targets ask for the same atoms.

Density steering is unaffected in both rows and structural validity is exactly 1.0000 in all six
cells, so the second constraint disturbs neither the first nor the support restriction.

## The surrogate approximates a different target, and the difference favours it

`results/WS7_a6_tilt_comparator.json`

The per-step projection approximates the sequence-level tilt, and the tilt is not a construct of
this work: it is the KL-minimal distribution satisfying the moment constraint, the target
distributional policy gradient trains toward. Measuring what the surrogate loses therefore requires
a model trained toward the tilt, not just the tilt's closed form. That arm is A6, trained by maximum
likelihood on exact tilt samples — distributional policy gradient's fixed point with unit importance
weights, so the comparator gets the best case of the method it stands for and no gradient-estimator
artefact can be blamed for the outcome. Both arms carry the mask; both induced distributions are
enumerated over all 65,536 sequences, so every divergence is exact.

A6 reaches its target: 0.0167 ± 0.0048 nats from the tilt, mean charge -0.0619 ± 0.1220
against a target of zero, no mass outside the mask. It is not a failed run, which is what makes the
comparison informative.

| arm | mean charge | per-sequence violation | charge variance |
|---|---|---|---|
| A2, mask only | +2.4694 ± 0.0807 | 0.9083 | 8.762 |
| A3 deployed, projection | +0.0170 ± 0.0014 | 0.6155 | 1.547 |
| A6, trained toward the tilt | -0.0619 ± 0.1220 | 0.8751 | — |
| tilt, exact optimum | +0.0000 | 0.8755 | — |

Both arms satisfy the expectation constraint, and A6's seed spread is wider than the gap between
them, so no ordering on mean charge is established. The difference that is not seed noise is
per-sequence violation: A6 inherits the tilt's 0.8751, as an approximation of that distribution
should, while A3 deployed violates 0.6155 — a factor of 1.42 fewer than the optimum it is said to
approximate.

The reading that follows is not that the surrogate approximates the tilt better. A6 is far closer to
the tilt than the induced distribution is. It is that the surrogate approximates a different
distribution, and on this task the different distribution is closer to what a request for a
constraint usually means. Bounding an expected increment at every step bounds the total in
expectation and also concentrates it: A3's charge variance is 1.547 against A2's 8.762, a factor of
5.7. The tilt has no such mechanism — it is the KL-closest distribution with the right mean, and
the cheapest way to buy a mean of zero from a distribution centred at +2.469 is to move mass to
both extremes, which is why it violates more per sequence than the model it started from.

The surrogate's 1.025-nat excess divergence is therefore not a pure loss. It purchases a variance
reduction the sequence-level optimum does not offer, and a per-sequence violation rate 1.42 times
lower than that optimum. Whether that trade is wanted depends on whether "most samples satisfy the
constraint" or "the average satisfies it" is the requirement.

## Best-of-n is the strongest amortizer, and it wins on accuracy

`results/WS6_best_of_n.json`

Conditional training is not the strongest baseline available. Best-of-n needs no training at
all: draw n samples from the unsteered model and keep whichever lands closest to the target.
One model serves every target, so it amortizes perfectly by construction.

Measured against the projection's published frontier error, best-of-n reaches it at
8 of 10 targets within n = 64, and at 5 targets within n = 4. Drawn from a pool of
1024 samples from the same checkpoint's backbone reading, with expected error estimated by
repeated subsampling rather than by the pool's best single draw, the errors at target 5.0 run
4.303, 1.006, 0.250 and 0.073 for n of 1, 4, 16 and 64, against the projection's 0.151.
On accuracy alone the projection loses.

Accuracy alone is the wrong comparison, and the reason is measurable rather than rhetorical.
Best-of-n buys accuracy by discarding samples, so the accepted set concentrates on the pool's
tail near the target: unique-formula fraction among accepted outputs falls from 0.805 at
n = 1 to 0.205 at n = 64, consistently across all ten targets. The projection arm's unique
fraction at the same target is 0.782 — above best-of-16 and more than three times best-of-64.
Selection filters a distribution and its diversity degrades as the target gets harder to hit;
shaping moves the distribution and its diversity does not.

The costs also fall in different places. Best-of-n pays n forward passes per accepted output,
at every target, indefinitely; the projection pays one training run and one forward pass per
output. At n = 64 that is a standing 64x inference multiplier against a one-time training cost.

The claim this supports is therefore narrower than accuracy parity. Among the arms measured
here, the projection is the only one that reaches target accuracy without either a per-target
training run or a per-output sampling multiplier, and the only one whose output diversity is
independent of how hard the target is to hit.

## Amortized control by finite automaton is a different constraint class, not a smaller one

`results/CP8_automaton_size.json`

Methods that compile a constraint into a deterministic automaton over tokens amortize across
constraints without retraining, which is the word this paper uses for its own claim, so the
distinction needs stating. The intuitive one — that a real-valued running budget needs an
intractable automaton — is arithmetic, and it does not hold. The budget spans 4397 amu and
the serialization's own discretization floor is 0.0020 relative, since cell parameters are
emitted as bin indices; at the finest expressible tolerance the budget needs 100 states and
88,400 once multiplied by parse position, which is 0.02x the controlled model's
parameter count and entirely buildable.

The real distinction is per-instance construction. The budget bound is compiled into the
transition structure, so each target is a separate automaton — about 884,000 states across the
ten frontier targets — while the dual solve takes the target as a runtime argument. Automaton
methods amortize across constraints at a fixed model; this work amortizes across targets of one
constraint at a fixed model. Neither subsumes the other.

![Composition failure and its mechanism]({{artifact:art_63f81e90-de87-4aef-9788-e9055f96c6dc}})

**Figure 2.** Composition depends on target compatibility. (a) Achieved against requested valence
electron density. Where the two constraints conflict (density target 3.0) pass-through is 6.2%, the
response is non-monotone — the largest request gives the smallest achieved value — and the row sits
essentially on the unconstrained control of 0.242. Where they agree (density target 8.0)
pass-through is 11.3% and the response is monotone, though still far from controllable.
(b) The mechanism. Fallback to the single-constraint solve rises with the second target in both rows,
but jointly solvable positions fall where the constraints conflict and rise where they agree, of 518
actionable positions per cell.

## The direction the second constraint is asked to move

The pass-through measure needs two valence cells in a row, so the partial density-5.0 row
contributes nothing to it. The cells at a shared valence target across density rows are directly
comparable, though, and each is paired with its own single-constraint control — which turns out to
be the more informative comparison, because the achieved valence is dragged by the density
constraint alone and an absolute level hides that.

| valence target | density target | achieved | control | deviation | asked | moved correctly |
|---|---|---|---|---|---|---|
| 0.282 | 3.0 | 0.2447 | 0.2424 | +0.0023 | up | yes |
| 0.282 | 5.0 | 0.3062 | 0.3141 | -0.0079 | down | yes |
| 0.282 | 8.0 | 0.3251 | 0.3956 | -0.0705 | down | yes |
| 0.484 | 3.0 | 0.2603 | 0.2424 | +0.0179 | up | yes |
| 0.484 | 8.0 | 0.3482 | 0.3956 | -0.0474 | up | no |
| 0.686 | 3.0 | 0.2353 | 0.2424 | -0.0071 | up | no |
| 0.686 | 8.0 | 0.3707 | 0.3956 | -0.0248 | up | no |

At valence target 0.282 the constraint moves in the requested direction at all three density
targets, and the deviation from control grows with the density target (correlation -0.9614). At
0.484 and 0.686 — both above every control, so both asking the constraint to raise valence — it
moves the wrong way at density 8.0 in each case.

Aggregating by direction rather than by row: asked to lower valence, the constraint moves correctly
2 of 2 times; asked to raise it, 2 of 5. The asymmetry follows from what the budget
is. The running budget is a remaining quantity, so lowering a per-volume density means spending less
of it, which is always available — choose lighter or fewer atoms. Raising it means spending more than
the prefix has left, which is not available once the budget is exhausted, and exhaustion is the
largest exclusion class in every cell (53 to 262 of 518 actionable positions).

This refines the compatibility reading rather than replacing it. Compatibility governs how many
positions admit a joint solution; direction governs whether the constraint can act at the positions
that do. Both point at the same underlying fact: the second constraint spends a budget the first has
already committed.

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

![Accuracy against inference compute]({{artifact:art_8cef8e99-a939-4381-bf2b-dffc9b18a250}})

**Figure 3.** The comparison is a point against a curve. (a) Mean absolute density error over the
seven sub-ceiling targets against forward passes per accepted output, both axes logarithmic. The
projection is a single point at one forward pass (0.662); best-of-n and SMC are curves that reach it
by spending compute, best-of-n crossing at n = 16. Above the ceiling the projection saturates, so those
three targets are excluded as uninformative. (b) The axis the inference-time family pays on.
Unique-formula fraction among accepted outputs falls from 0.830 at n = 1 to 0.220 at n = 64, while the
projection's 0.782 does not depend on how hard the target is to hit. Accuracy bought by selection is
diversity spent.

## Sequential Monte Carlo is the principled version, and does not close the gap here

`results/WS6_smc_steering.json`

Best-of-n discards whole sequences; SMC carries K partial sequences and reallocates them as
information arrives, spending compute where it changes the outcome. If the projection's advantage
were only that naive selection is wasteful, SMC would remove it. The potential is a Gaussian on each
particle's expected final density, evaluated at the same element positions where the projection is
actionable, and the density estimate reuses the published decoder — verified bit-identical on
complete sequences — so both methods are scored on the same quantity.

This implementation reaches the projection at 6 of 10 targets within K = 64, all of them above the
data mean where the projection saturates against its ceiling. Below the mean the projection is better
at every K tested. Against best-of-n at equal compute SMC is worse at all ten targets, by a mean
factor of 4.2 at K = 64. Structural validity is exact at every particle count, since the
mask is active throughout.

That comparison is a bound rather than a result, and the reason is worth stating. The potential's
width is a free parameter; a probe at one target suggested a narrower width was much better, and
sweeping both widths across all ten targets refuted the probe — the narrower width wins at 4 of
10 targets, loses at the rest, and has a worse mean because one target degrades badly. There is no
single good width, and the best width varies with the target. Selecting it per target from the same
runs that report the error would be selection on the evaluation set, so the fixed-width rows are what
is reported; the best-per-target mean is 0.855 and is declined. The proposal, the resampling
threshold, and per-target width chosen on held-out draws remain untuned, so a stronger SMC is
available and would not change the claim: the inference-time family buys accuracy with compute per
output at every target indefinitely, while the projection buys it once at training.

## The constraint class and the serialization are coupled

`results/CP9_serialization_expressibility.json`

The steering results rest on one token ordering, and the natural way to separate properties of
the method from properties of that ordering is to retrain under a second one. That experiment
cannot be run as posed, for a reason prior to engineering: under composition-first ordering the
density constraint is not expressible as a per-step linear expectation at all.

Density is a ratio, so a target density fixes a total mass budget only once the cell volume is
known. Cell-first ordering closes the volume before any element token, making the running budget
a determined real number at every element position; composition-first leaves the volume
undetermined while the element tokens are emitted, so there is no budget to project onto.
Measured on 393 real structures, the budget is determined at 100.0% of element positions
under cell-first and 0.0% under comp-first. The second family inherits the same
dependence, being also a per-volume density: 100.0% against 0.0%.

Training the density arm under the second ordering would therefore train with no actionable
positions and measure the unconstrained model rather than a serialization effect. Substituting a
fixed mass per token instead is a different constraint, and it drives the composition toward the
lightest available elements.

The honest statement of the limitation is not that a serialization comparison was skipped. It is
that the expressible constraint class depends on the serialization: a quantity is constrainable
only where the prefix determines its budget, which makes token ordering a design variable of the
method rather than an independent axis to vary. A serialization comparison for this method needs
two orderings that both determine the budget before the constrained tokens.

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
