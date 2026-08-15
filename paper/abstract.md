# Constraint-Projected Transformers

Training through a differentiable constraint projection: one model, many targets,
and where composition fails

## Abstract

Generative models of structured objects are routinely asked to satisfy requirements their
training distribution does not enforce, and the usual remedies each carry a cost: rejection
sampling becomes intractable exactly where the request is unusual, conditional training
reproduces a data slice and needs one model per target, and decode-time intervention leaves
the model unaware of the constraint it will meet. We insert a differentiable projection into
the decoding step and train through it. At each step the next-token distribution is mapped
onto the set satisfying a constraint expressed as a linear functional of the next token,
solved as a dual Newton iteration whose gradient follows in closed form from the implicit
function theorem, so the solve is $m$-dimensional in the number of constraints rather than
in the vocabulary.

We separate two regimes that are easily conflated. A support restriction is exact by
construction: grammar masking yields 500 of 500 parseable crystallographic files against 0
of 500 unmasked, and structural validity is exactly 1.0000 across every target and seed,
including in an untrained model and at targets where expectation steering fails outright. An
expectation constraint is satisfied only in expectation, and we quantify the gap on a task
small enough to enumerate exactly, where the optimal expectation-constrained distribution
places 87.55% of its mass on individually violating sequences.

The layer is a training component rather than a decoder, and the evidence is that removing
it at inference inverts rather than degrades the result. Our central claim is amortization:
because the constraint target enters the dual solve as an argument rather than as a
training-time label, one checkpoint deployed at seven density targets it never trained on is
statistically indistinguishable from seven separately trained models
(0.650 ± 0.151 against 0.662 g cm$^{-3}$, Wilcoxon signed-rank $p$ = 1.0, three seeds).
Conditional training remains more accurate per target, so the contribution is a different
cost structure rather than a better number — and it is strongest where that baseline is
weakest, since the data slice a conditional model needs thins or vanishes at unusual targets.

Two boundaries are reported as results rather than as future work. Several constraints are
solved simultaneously and exactly, verified at up to 4 constraints against two independent
oracles, but exact solving is not steering: whether a joint solution exists at a position is
determined by the constraints and the prefix, not by the solver. Two additive per-element
quantities correlated at 0.725 fail to steer jointly, with pass-through of 6.2% and a
non-monotone response, because they compete for the same token budget. Separately, the layer
constrains additive functionals of the emitted symbols, so charge neutrality — the
constraint a materials reader asks for first — is outside its class: as standardly defined it
is an existential predicate over oxidation-state assignments, and the linear proxy that would
make it fit recovers 11.1% of the balanced set. Stating both boundaries positively tells a
reader which of their own constraints will fit.


## Abstract (short, 287 words)

Generative models are routinely asked to satisfy requirements their training distribution
does not enforce. We insert a differentiable projection into the decoding step and train
through it: at each step the next-token distribution is mapped onto the set satisfying a
linear functional of the next token, solved as a dual Newton iteration whose gradient
follows from the implicit function theorem.

Two regimes behave differently and we keep them separate. A support restriction is exact by
construction — grammar masking gives 500 of 500 parseable crystallographic files against 0 of
500 unmasked, with structural validity exactly 1.0000 across every target and seed, in an
untrained model included. An expectation constraint holds only in expectation, and on an
exactly enumerable task the optimal such distribution places 87.55% of its mass on
individually violating sequences.

The layer is a training component, not a decoder: removing it at inference inverts rather
than degrades the result. Our central claim is amortization. Because the target enters the
dual solve as an argument rather than a training-time label, one checkpoint deployed at
seven density targets it never saw matches seven separately trained models
(0.650 ± 0.151 against 0.662 g cm$^{-3}$, Wilcoxon $p$ = 1.0). Conditional training stays
more accurate per target, so this is a different cost structure rather than a better number.

We report two boundaries as results. Exact multi-constraint solving is not multi-constraint
steering: two additive per-element quantities correlated at 0.725 fail to steer jointly
(6.2% pass-through, non-monotone) because they compete for one token budget. And the
layer constrains additive functionals of emitted symbols, so charge neutrality — an
existential predicate over oxidation-state assignments — is outside its class, with a linear
proxy recovering 11.1% of the balanced set.


## Title alternatives

The title above is the recommendation. Alternatives, all describing the same content:

- **A (recommended)** — Training through a differentiable constraint projection: one model, many targets, and where composition fails
- **B (claim-first)** — One training run, many constraint targets: amortized expectation constraints for autoregressive generation
- **C (mechanism-first)** — Constraint projection as a training component: exact support restriction, amortized expectation steering, and the limits of composition
- **D (short)** — Amortized constraint steering in autoregressive generative models

A names the amortization result and the composition boundary in one line, which is what
distinguishes this from the constrained-decoding literature; B leads with the claim and is
the better fit if the venue prefers a result-forward title; C is the most precise and the
least readable; D is the shortest and loses the boundary.
