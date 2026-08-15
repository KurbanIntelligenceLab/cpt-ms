# Introduction

Draft. Every quantitative claim is sourced to an artifact in `results/` and verified
against it by recomputation.

Generative models of structured objects are usually asked to satisfy requirements their
training distribution does not enforce. A materials generator should emit parseable
crystallographic files; a designer wants samples at a specified density rather than at the
density the corpus happens to favour. The standard responses are to filter after the fact,
to condition on the requirement during training, or to intervene at decode time. Each has a
cost that shows up in the measurements below: rejection sampling becomes intractable
precisely where the request is unusual, conditional training reproduces a data slice and
needs one model per target, and decode-time intervention leaves the model unaware of the
constraint it will face.

This work inserts a differentiable projection into the decoding step and trains through it.
At each step the model's next-token distribution is mapped onto the set satisfying a
constraint expressed as a linear functional of the next token, solved as a dual Newton
iteration whose gradient follows in closed form from the implicit function theorem. The
layer is therefore a component of the model rather than a decoder wrapped around it, and
the distinction is empirical rather than terminological: removing the projection at
inference does not degrade the crystal result by a margin, it inverts it.

Two regimes are separated throughout, because they behave differently and conflating them
overstates what the method delivers. A support restriction — a mask making some tokens
unreachable — is exact by construction and requires no training. An expectation constraint
— a running budget requiring a statistic of the finished sequence to meet a target — is
satisfied in expectation only, and the gap between those two notions is measured rather
than asserted: on a task small enough to enumerate exactly, the optimal
expectation-constrained distribution places 87.55% of its mass on sequences that
individually violate the constraint.

Four claims are established, each with a stated boundary.

The support restriction is exact. Grammar masking yields 500 of 500 parseable files
against 0 of 500 unmasked, judged by a parser sharing no code with the emitter, and
structural validity is exactly 1.0000 across every target and seed — including targets
where expectation steering fails outright, and including an untrained model.

Expectation steering works below and within the bulk of the training distribution, and the
projection and the training are complementary rather than redundant. The gain decomposes
into two thirds from the layer applied to an untrained backbone and one third from training
through it, with the second part's mechanism measured independently as a fall in the
infeasible-step fraction.

One training run amortizes across a family of targets. Because the target enters the dual
solve as an argument rather than as a training-time label, a single checkpoint deployed at
seven targets it never saw is statistically indistinguishable from seven separately trained
models: mean error 0.650 ± 0.151 g/cm^3 against 0.662, Wilcoxon signed-rank
p = 1.0 across three seeds, from 1 training run against 7. This is the axis on which
conditional training otherwise wins, since that baseline requires one model per target by
construction.

Several constraints are solved simultaneously and exactly, but they do not necessarily
compose. Exactness holds at up to four simultaneous constraints against two independent
oracles. Whether a joint solution exists at a given position, however, is set by the
constraints and the prefix rather than by the solver, and on real crystals two additive
per-element quantities correlated at 0.725 fail to steer jointly because they compete for
the same token budget. Reporting the exactness without the composition limit would be the
more flattering and less useful result.

The method's boundary is also stated positively. The layer constrains additive functionals
of the emitted symbols. Charge neutrality, the constraint a materials reader asks for first,
is an existential predicate over oxidation-state assignments and is measurably not
expressible in that form: a linear proxy recovers 11.1% of the balanced set. That is a
property of the constraint class rather than an unimplemented feature, and it bounds where
the method applies.
