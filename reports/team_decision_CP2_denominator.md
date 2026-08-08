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

## Why the denominator is the wrong one, and why saying so now is uncomfortable

A masked softmax over a 4096 vocabulary takes 0.2 us per token. It is close to the cheapest
operation in the network. Any added work compared against it looks catastrophic, which makes the
ratio a statement about the baseline rather than about affordability.

The question the threshold was written to answer is whether the projection is affordable to train
with. Against a full training step of a model at the plan's own A2 scale — a 29.7M-parameter
causal decoder, V=4096, 8 layers, batch 16 x 512, measured on the same card at 40.1 ms per
forward-backward-optimizer step:

| Scope | m=1 | m=2 |
|---|---|---|
| Projection at every position | 52.9% | 70.1% |
| Projection at actionable positions only, density fraction 0.390 | 20.1% | 26.8% |
| Projection at actionable positions only, density fraction 0.283 | 13.8% | 18.8% |
| Projection at actionable positions only, charge fraction 0.128 | 6.9% | 8.7% |
| Projection at actionable positions only, charge fraction 0.041 | 5.0% | 5.4% |

Restricting to actionable positions is not a trick to get under the line. It follows from the
project's own actionability measurement: at a non-actionable position no format-legal substitution
changes the constraint, so the projection provably cannot affect the outcome there and running it
is wasted work. The CP0 threshold is already scoped to actionable positions for exactly this
reason. Applying the same scoping to the cost measurement is consistent, not convenient.

**But I am proposing it after seeing that the pre-registered version fails, and that is precisely
the move pre-registration exists to prevent.** The argument is sound on its own terms and would
have been sound before the measurement; it is still a post-hoc change to a registered threshold,
and the correct handling is a team decision recorded as an amendment with the reason and the date,
not a quiet substitution. The failing number stays in the artifact either way.

## What I recommend

Amend CP2 to state the threshold against a full training step at the A2 model scale, with the
projection applied at actionable positions, and record the as-pre-registered figure alongside as
a hardware characterization rather than a gate. Under that definition the layer passes at 13.8%
to 26.8% for the density dial, which is the primary dial per `WS2_dial_assignment.md`.

If the team prefers to hold the original denominator, then CP2 fails and the honest report is that
training-time projection costs two orders of magnitude more than a masked softmax per token — with
the training-step context given as the reason that ratio does not by itself block the method.

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
