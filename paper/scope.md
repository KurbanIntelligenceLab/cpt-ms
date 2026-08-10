# Scope

Draft. Every quantitative claim below is sourced to an artifact in `results/` and
verified against it; the verification scripts live alongside the reports in `reports/`.

## What the method is

A differentiable projection layer inserted into the decoding step of an autoregressive
generative model, trained through rather than applied at inference. At every step it maps
the model's distribution over the vocabulary onto the set satisfying a constraint
specified as a linear functional of the next token, solved as a dual Newton iteration
whose gradient is available in closed form through the implicit function theorem.

Two constraint regimes are distinguished throughout. **Regime S** is a support
restriction: a mask that makes some tokens unreachable. **Regime E** is an expectation
constraint: a running budget requiring a statistic of the completed sequence to meet a
target. They behave differently and the paper does not conflate them.

## What is established

**Regime S is exact and holds by construction.** Grammar masking yields 500 of 500
syntactically valid CIF files against 0 of 500 unconstrained, judged by a third-party
parser sharing no code with the emitter. On MP-20 the masked model reaches structural
validity of exactly 1.0000 across every target and seed of a ten-target steering sweep,
and an *untrained* masked model does the same. This is a property of the mask, not a
statistical property of a trained model, and it is independent of every steering result
below — including at targets where steering fails outright.

**Regime E steers, and the projection and the training are complementary rather than
redundant.** On the synthetic task, where targets are computed exactly by enumeration, the
deployed system reaches a mean charge of +0.0170 against a target of 0, against the exact
projection applied to raw data at +0.1028. On MP-20 at a density target of 5.0 g/cm^3 the
deployed system reaches 4.810 against an unsteered baseline of 6.783, a separation of
d = 14.4 with three seeds per arm. The gain decomposes into 67% from the layer applied to
an untrained backbone and 33% from training through it; the complementarity claim attaches
to the second part, and its mechanism is measured independently — the probability-weighted
infeasible-step fraction falls from 0.3430 to 0.0496 as training moves the backbone into a
region where the running budget is attainable.

**The projection is a training component, not a decoder.** Removing it at inference does
not degrade the crystal result by a margin, it inverts it: from 4.810 to 9.200, an error
of 4.200 against the unsteered baseline's 1.783.

## What the method does not do

**Steering is asymmetric and has a ceiling.** On MP-20 density, mean absolute error is
0.345 g/cm^3 for targets at or below the data mean of 6.679 and 4.577 above it. Achieved
density saturates at 10.405 g/cm^3 regardless of the request. The mechanism is measured
rather than inferred: steering downward always has light elements available, while
steering upward eventually demands more mass per atom than the heaviest legal token
supplies, at which point the running budget becomes infeasible, the fallback fires, and
the constraint stops binding. Infeasibility rises monotonically with target from 0.0079 to
0.4133, and training's ability to repair it falls from 29.2x to 0.97x — at the highest
target, training no longer reduces infeasibility at all.

The claim is therefore scoped to targets within and below the bulk of the training
distribution, and the ceiling is reported as a property of the method on this dial rather
than omitted.

**Expectation constraints are satisfied in expectation, not per sample.** On the synthetic
task the exact expectation-constrained optimum still places 87.55% of its mass on
individually violating sequences. Bounding an expected increment bounds the expected total
but not its variance, and the per-step surrogate additionally suppresses the compensating
spread the sequence-level optimum uses.

**The per-step surrogate is not the sequence-level optimum.** Per-step projection induces
a materially different distribution from the exponential tilt, not a slightly worse one.
Where the running budget becomes unsatisfiable partway through a sequence the surrogate
misses the constraint it was built to enforce.

**Constraint satisfaction is not sample quality.** Syntactic validity and semantic
plausibility are measured separately: 100% of masked CIF samples parse, while 47% build
into a structure given syntactic validity, the shortfall being physically incoherent cell
geometries the grammar never claimed to prevent. Reporting only the second figure would
charge the grammar for failures outside its scope; reporting only the first would overstate
what the method delivers.

**The backbone's unaided distribution is not a well-behaved object.** It varies
non-monotonically with steering difficulty, drifting furthest above the data mean when the
target sits near the achievable ceiling (R^2 = 0.810 against distance-to-ceiling) and
returning toward the data at both extremes. In one run it generated a mean density
exceeding the maximum in MP-20 itself. Backbone and deployed readings are therefore both
reported everywhere, since either alone misleads, in opposite directions.

**Thermodynamic stability of generated structures is not assessed.** The serialization is
composition plus cell with no atomic positions, and an energy model requires positions;
obtaining them is crystal structure prediction, which the method does not attempt. Relaxing
from random positions would measure the initialization rather than the generated crystal.
What is measured instead is composition-level and cell-level plausibility — charge
neutrality, electronegativity ordering, and packing — each reported against a reference
obtained by scoring real MP-20 structures through the identical path. That reference is not
1.0: real materials satisfy the charge screen at a rate of 0.715, because oxidation-state
enumeration covers only common states, so an arm reported against an assumed ceiling would
be penalized for a property of the screen rather than of the model.

An ML-potential energy screen (CHGNet) was calibrated against MP-20's DFT labels and agrees
to 21 meV/atom on held-out structures, which is below the 30 meV/atom near-hull window. It
is used for reference structures only, is labelled an ML-potential screen rather than DFT
throughout, and no DFT calculation is reported in this work.

## What is not claimed

The layer is not claimed to improve in-distribution sample quality; an external benchmark
reports that constraints do not, and nothing here contradicts that. No claim is made about
constraints the projection's linear-functional form cannot express, nor about quantities
that are not determinable from a prefix. The compensation hypothesis — that the backbone
learns an offset calibrated to the projection's pull — was tested and rejected.
