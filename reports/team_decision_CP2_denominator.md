# Decision needed — CP2's overhead threshold fails as pre-registered, and the fix changes the denominator

Compute is unblocked and the benchmark has run on the training hardware. The result needs a
decision I should not make alone, because the honest version of it involves changing a
pre-registered quantity after seeing the data.

## The pre-registered test fails

`preregistration/CP2_overhead_budget.md` sets a 30% ceiling on per-token wall-clock overhead of
the projection layer measured "against the identical A2 masked-softmax forward pass". Measured
exactly that way on the RTX 5090 the project now owns:

| Active constraints | Masked softmax | Projection | Overhead |
|---|---|---|---|
| 1 | 0.203 us/token | 7.66 us/token | 3672% |
| 2 | 0.197 us/token | 14.22 us/token | 7119% |
| 4 | 0.194 us/token | 14.18 us/token | 7220% |
| 8 | 0.194 us/token | 14.30 us/token | 7260% |

That is 122x to 242x over budget. All entries converge; the residual is at machine epsilon. The
layer is correct and slow.

## The table did not reproduce, and there were two defects, not one

The review is correct that the published table does not follow from its own inputs, and the
diagnosis is candidate (b): the projection batches across positions, so per-token cost is not
constant in batch size. Measured directly:

| Rows in the batch | Per-row projection cost, m=1 |
|---|---|
| 256 | 7.66 us |
| 335 | 5.76 us |
| 1,048 | 2.52 us |
| 3,194 | 2.50 us |
| 8,192 | 2.55 us |

At 256 rows the solve is launch-overhead dominated; by ~1,000 rows it has amortized and stays flat.
Extrapolating the 256-row figure to a full 8,192-token step overstates cost by **3.0x**, which is
exactly the factor the review identified. The per-token table (batch 256) and the scoping table
(up to 8,192 rows) are different measurements and must never be combined, and publishing them
adjacent to each other without saying so was the defect.

The share figures themselves were computed from directly measured times at each token count, so
the m=1 rows do reproduce. **But re-measuring surfaced a second and worse defect: the published
m=2 rows silently used the opt-in fast Hessian while being presented as the accurate default.**
Every row below is re-measured at its own token count, with both precisions labelled.

| Scope | rows | m=1 (fp64 Hess) | m=2 (fp64 Hess) | m=2 (fp32 Hess, opt-in) |
|---|---|---|---|---|
| Every position | 8,192 | 52.1% | 167.6% | 70.1% |
| Actionable, density 0.390 | 3,194 | 19.9% | 64.9% | 26.8% |
| Actionable, density 0.283 | 2,318 | 13.6% | 46.9% | 18.8% |
| Actionable, charge 0.128 | 1,048 | 6.6% | 22.8% | 8.7% |
| Actionable, charge 0.041 | 335 | 4.8% | 9.0% | 5.3% |

Denominator is a 29.7M-parameter causal decoder at the plan's own A2 scale (V=4096, 8 layers,
batch 16 x 512), measured at 40.115 ms per forward-backward-optimizer step on the same card.

Both corrections cut against the recommendation. The honest position is now weaker than the one
first presented: at two active constraints with the accurate solver the projection costs 167.6% of
a training step if applied everywhere, and 46.9% to 64.9% even scoped to the density dial's
actionable window. Only the single-constraint case is comfortably affordable at 13.6% to 19.9%.

## The amendment, on the review's grounds rather than mine

The review's framing is better than the one I used and I accept it. This is not a threshold that
proved too tight; the pre-registered denominator cannot answer the question it was written for.
Masked softmax is 0.197 us/token against 4.895 us/token of total per-token training cost — about
4% of a step. A ratio against a 4% component is dominated by the smallness of the denominator and
would have failed for essentially any layer doing real work. That is a category error in the
measurement.

Recorded as: denominator replaced because the original quantity does not measure affordability;
the as-registered figure retained as hardware characterization, not a gate. Both numbers stay in
`results/CP2_overhead_benchmark.json` permanently.

Actionable-position scoping stands on the grounds CP0 already uses: at a non-actionable position
no format-legal substitution changes the constraint, so the projection provably cannot affect the
outcome. That is wasted compute, not skipped compute.

## What the corrected numbers mean for the design

Leading with the density dial as instructed: **m=1 at 13.6% to 19.9% is affordable; m=2 at 46.9%
to 64.9% is not comfortably so.** This is a design constraint rather than a blocker, and it points
somewhere specific — the per-constraint cost is dominated by the Hessian contraction, which scales
with the square of the active-constraint count while the residual scales linearly. Running one
active constraint per step, or accepting the fast Hessian for the m >= 2 configurations with
gradients disabled at inference only, are the two routes. Neither needs deciding before A2, which
uses a single dial.

The charge dial's 4.8% is cheap for the reason the review flags: B1ref's 0.041 actionability is
the narrowest window in the matrix, so the low cost reflects rarely running rather than running
cheaply. It should not be quoted as the headline.

## Hardware finding that explains the magnitude

Consumer Blackwell cripples float64: 1.61 TFLOP/s against 69.34 in float32, a 43x penalty. The
projection's dominant cost is the Hessian contraction over the vocabulary, 1.66 ms in float64
against 0.062 ms in float32.

Demoting only that contraction is defensible — the Hessian steers the Newton direction and does
not define the fixed point, which is set by the residual. It is implemented as an opt-in
`fast_hessian_dtype` and gives 1.7x at m >= 2, with the converged distributions agreeing to 0.0
on this batch. It is not the default, because it perturbs the stopping point inside the tolerance
ball by ~1e-13 to 1e-9, and that is enough to break a finite-difference gradient check: the
implicit-function-theorem backward assumes q is the exact solution of the forward problem. Trading
gradient correctness for 1.7x would be a bad trade in a project whose contribution is a
differentiable layer.

The residual contraction stays in float64 at no cost, measured at 1.00x the float32 time, because
it is a two-way reduction that is bandwidth-bound rather than arithmetic-bound.

An earlier defect in the same area is worth recording: the solve originally ran entirely in the
input dtype, where at float32 only 11 of 64 batch entries reached tolerance and half precision
raised outright, since torch has no half-precision LU. The convergence flag could not catch it
because the flag was computed in the same degraded precision as the solve.

## Compute status

RTX 5090 registered as `ssh:cpt-a2-gpu`, contract 47219544, $0.4453/hr. sm_120 kernels verified
by a real device matmul rather than `is_available()`. The box has 128 cores and 125 GB RAM, better
than the offer listing showed. Per the owner's instruction it will not be destroyed without an
explicit command.

A2 training is now unblocked and is the next item.
