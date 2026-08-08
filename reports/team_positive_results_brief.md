# Where the positive results are, and the two that come next

Raised because the recent run of findings reads as negative. Audited against the artifacts
rather than impression: the ledger is not negative, but it is currently weighted toward
*enabling* results and *quantified limits*, and neither reads as a headline. Only one genuine
negative exists, and it is external to this project.

## What is already established, and positive

**The projection layer is exact.** Worst-case agreement with the cvxpylayers oracle is 2.00e-6
relative across five constraint configurations; against a step-swept finite-difference oracle,
4.56e-10; forward residuals at most 7.84e-11 in 4-5 Newton iterations, with 200 randomized
Jacobian rank checks and zero deficiencies. This is the CP2 correctness gate, cleared. It is a
positive result that happens not to be surprising, which is why it does not feel like one.

**Both constraint families are instantiable per step on both backbones.** The plan's stated
risk was that the density family would have no sound per-step form on a character stream, which
would have left the ionic-scoping risk unhedged. Measured, every backbone/family pair is usable:

| Backbone / family | Determinable | Actionable |
|---|---|---|
| B1ref charge | 0.949 | 0.0410 |
| B1ref density | 0.490 | 0.3849 |
| B2 charge | 0.785 | 0.1280 |
| B2 density | 0.480 | 0.1280 |

**Per-step projection demonstrably steers.** This is the point most at risk of being lost in
the A3-vs-tilt comparison. Applying the layer moves mean charge from 2.516 to 0.103 against a
target of 0, and cuts the per-sample violation rate from 91.1% to 62.0%. Regime S is enforced
exactly — zero mass outside the feasible set. The mechanism works; the finding was that it is
sub-optimal relative to the sequence-level ideal, which is a different statement from "it does
not work."

## What is a quantified limit, not a negative result

Three results have been read as negative and are better described as design constraints with
numbers attached:

- Per-step projection carries 1.0254 nats of excess divergence over the sequence-level tilt and
  lands 0.1028 off target. That bounds what A3 can claim and tells A6 what it must beat.
- The running budget goes infeasible from step 5 of 8. That identifies the fallback policy as a
  pre-registration item rather than an implementation detail.
- Multiplier activity decays with exponents 1.026 and 2.038 against theoretical 1 and 2. That
  is a *confirmation* of exponential-family theory to three decimals, and it is what converted
  decay from a threat into a control condition.

A paper reporting these as measurements of a mechanism reads as thorough. The same content
reported as a series of things that did not work reads as a failed project, and the difference
is framing rather than data.

## The one genuine negative, and it is not ours

Constraints do not improve in-distribution generation quality. At matched MP-20 training data
the symmetry-constrained model is second-worst of six in dynamical stability, 11.57% against
27.15% for unconstrained DiffCSP. This is a third party's measurement about a third party's
model, retired as a claim this project will not make, and it is the empirical motivation for
the steering framing. It costs us nothing.

## Two positive results available next, both cheap

**1. Parse validity, A1 versus A2.** The reference model discards up to 87% of generations
before evaluation, attributed primarily to parse failures. A grammar-constrained decoder
eliminates the syntactic failure mode by construction, so this is a large positive delta that
requires no Regime E, no steering sweep, no DFT, and no GPU beyond generation. It is the
single highest-visibility-per-cost result on the board, and it is the natural Figure 1.

**2. Steering beyond the data distribution.** The alignment sweep already measures constraint
activity as a function of how far the target sits from the data. The positive claim available
is that A3 reaches targets the base model reaches only by rejection sampling, with the
efficiency ratio as the headline number. This is the reframe's central experiment, it is
positive by construction because it measures capability rather than fidelity, and the synthetic
task can deliver it exactly before any crystal run.

Both are unblocked. Experiment 2 needs no GPU at all; experiment 1 needs generation from a
trained A2, so it follows the training launch and therefore the compute decision.

## Recommendation

Run experiment 2 on the synthetic task now, since it converts the existing sweep infrastructure
into a positive capability claim at no additional cost. Hold experiment 1 for the first A2 run
and make it Figure 1. The compute decision remains the only blocker and is unchanged.
