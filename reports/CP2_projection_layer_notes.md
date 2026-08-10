# CP2 — Projection layer: construction and gradient verification

Standing note, rebuilt on change. Feeds CP2. All numbers are read from
`results/CP2_gradient_verification.json`, emitted by
`scripts/layer/verify_gradients.py`.

## Summary

The batched projection layer is implemented and its gradients are verified against two
independent oracles. Worst-case agreement with cvxpylayers is 2.0e-6 relative across five
constraint configurations, and worst-case agreement with a step-swept finite-difference
oracle is 4.6e-10. Forward residuals are at most 7.8e-11 with 4-5 Newton iterations per
solve. 200 randomized Jacobian rank checks show no rank-deficient configurations. Two
defects were found and fixed during verification, one of which would have silently capped
constraint satisfaction at about 1e-9 in training.

## Construction

Per token the layer solves

    minimize_q  KL(q || p)   subject to  A q = b,  1'q = 1,  q >= 0,  q_i = 0 on masked i

The KL-optimal solution has the exponential-tilt form q proportional to p * exp(lambda'a),
so the layer never solves a general convex program. It solves for the multipliers, which
is a smooth strictly convex root-finding problem in as many unknowns as there are active
constraints — one to three here, against a vocabulary of thousands. That asymmetry is the
whole efficiency argument CP2 measures.

The dual is g(lambda) = logsumexp(log p + lambda'a) - lambda'b, with gradient E_q[a] - b
and Hessian Cov_q[a]. Two consequences are worth stating because they are load-bearing:
the dual gradient *is* the constraint residual, so the convergence test and the constraint
check are the same quantity; and the Hessian is a covariance, so it is positive definite
whenever the constraint statistics are not almost surely constant under q.

The backward pass applies the implicit function theorem at the converged multiplier rather
than differentiating through the Newton iterations. The cost is cubic in the number of
constraints and independent of the iteration count, and the result is the exact gradient
of the true optimum rather than the gradient of an unrolled approximation.

## Solver precision — reported before benchmarking, as requested

The dual Newton solve runs in **float64 internally regardless of the input dtype**, casting the
result back at the end. The convergence flag is evaluated on the float64 solve, before the
output cast, so a solver that reached the optimum is not marked failed merely because its answer
cannot be represented in a lower-precision output.

This was a blocking defect, found while preparing the CP2 benchmark rather than during the
layer's own verification. The iteration originally ran in the input dtype:

| Input dtype | Converged (of 64) | Residual, evaluated in float64 |
|---|---|---|
| float64 | 64 | 9.4e-11 |
| float32 | 11 | 3.4e-04 |
| bfloat16 | raised `NotImplementedError` | — |
| float16 | raised `NotImplementedError` | — |

Half precision raises because torch has no half-precision LU on CPU. Under any mixed-precision
training run the layer would therefore have crashed outright or, at float32, silently returned
distributions that are not KL-optimal for 53 of 64 batch entries — precisely the condition the
convergence flag exists to prevent, defeated by the flag being computed in the same degraded
precision as the solve.

With the internal float64 solve, all four dtypes reach 64/64 convergence. The residual after
cast-back is the output dtype's own representation floor (2.6e-08 at float32, 1.7e-04 at
float16, 2.5e-03 at bfloat16), which no solver can improve on.

Consequence for the CP2 benchmark: the projection's cost must be measured with the float64
internal solve in place, since that is what will run in training. Benchmarking the
input-dtype version would measure code that cannot be used.

## Verification results

Five configurations, batch 3-4, vocabulary 10-32, one to three constraints, float64.

| B | V | m | Forward residual | Iterations | Rel. diff vs finite differences | Rel. diff vs cvxpylayers |
|---|---|---|---|---|---|---|
| 4 | 10 | 1 | 3.4e-11 | 5 | 4.6e-10 | 4.6e-7 |
| 4 | 16 | 1 | 2.4e-12 | 4 | 1.1e-10 | 2.0e-7 |
| 4 | 16 | 2 | 7.8e-11 | 5 | 1.1e-10 | 2.5e-7 |
| 3 | 24 | 2 | 6.3e-12 | 5 | 1.0e-10 | 7.3e-7 |
| 3 | 32 | 3 | 6.8e-11 | 5 | 1.6e-10 | 2.0e-6 |

Randomized Jacobian rank check: 200 trials across random constraint counts and vocabulary
sizes, zero rank-deficient.

The two oracles fail in different ways, which is why both are run. Finite differences are
model-free and catch a wrong Jacobian, but are noisy. cvxpylayers differentiates the KKT
system of the general cone program and shares no code path with the closed-form tilt, so
agreement there is genuine cross-validation. Its own solver reports "Solved/Inaccurate" on
some instances, which places a floor of roughly 1e-6 on what that comparison can resolve —
so the 2.0e-6 worst case is at the oracle's resolution limit, not a measurement of our
error.

## Finding: the flat-dual line-search trap

This is a result, not a changelog entry, and it belongs in the paper's appendix. Anyone
implementing a Newton dual solve for a KL projection will hit it, and it fails silently in
the direction that flatters the wrong conclusion.

**Mechanism.** The dual objective g(lambda) = logsumexp(log p + lambda'a) - lambda'b is
minimized at the optimum, so its gradient — which is exactly the constraint residual —
vanishes there and the objective is locally quadratic and flat. A backtracking line search
that accepts a step only on a strict decrease, `g(lambda - t*step) < g(lambda)`, therefore
fails near convergence for a reason that has nothing to do with the step being bad: a step
reducing the residual from 5e-9 to 1e-16 changes g by less than one unit in the last place,
so floating-point rounding decides the comparison. The test reads noise as failure, halves
t repeatedly, reaches an effectively zero step, and the iterate freezes one Newton step
short of the solution.

**Signature.** The residual pins at a fixed value — about 4e-9 here — and stays there
regardless of the iteration budget. Running to 400 iterations produced no improvement, which
is what distinguishes a stall from slow convergence. The solver also reports convergence if
its tolerance is looser than the floor, so the failure is invisible without an independent
check.

**Why it matters here specifically.** A residual floor is a floor on constraint
satisfaction, and it biases the active-multiplier fraction downward: the multipliers stop
short of the magnitude the constraint actually requires. CP0 gates on that fraction at
convergence, and the reading selects the paper's register — active multipliers at
convergence mean the layer shapes the final distribution, decayed multipliers mean it is a
training-dynamics prior whose endpoint equals A2. A numerically induced low reading would
have looked exactly like early internalization. The defect would have argued for the
weaker claim, from an artifact.

**Fix.** Accept any step that does not increase the dual beyond a small relative tolerance,
rather than requiring strict decrease. Iteration counts fell from the 50-iteration cap to
4-5 and residuals from 5e-9 to 1e-11.

**Regression coverage.** `test_no_residual_floor_when_near_optimal_at_initialization`
constructs the trap directly: since the solver initializes at lambda = 0, setting the
target just off the base distribution's own statistic puts the first iterate within a
chosen offset of the optimum, where the dual is flattest. It is parameterized over offsets
from 1e-6 down to 1e-12. A companion test asserts that a tenfold larger iteration budget
does not change the answer, which is the property a stalled solver violates. Both were
confirmed to fail against the reverted fix — at offsets 1e-8 and 1e-9 specifically — and to
pass once restored, so they are known to detect the defect rather than merely to accompany
it. The pre-flight harness runs the same near-optimal case on every launch.

## Second defect: the finite-difference oracle reported its own noise

The
first run showed 8.4e-3 relative disagreement at a fixed step of 1e-6, against 4.6e-7 for
cvxpylayers. Sweeping the step size showed the error scaling as 1/eps — 8.5e-6 at 1e-2
rising to 8.2e-2 at 1e-7 — which is roundoff amplification, not a Jacobian error: the
difference quotient divides the forward solve's own convergence error by the step. The
oracle now sweeps the step and reports the whole curve alongside its minimum, so the
U-shaped trade-off between truncation and roundoff is checkable by a third party rather
than asserted. The headline agreement figure is the sweep minimum, 4.6e-10 after the
line-search fix; the curve is in `fd_sweep_rel_diff` per configuration.

The full sweep for the first configuration, which shows the 1/eps signature directly:

| Step | Relative disagreement |
|---|---|
| 1e-2 | 7.4e-6 |
| 1e-3 | 7.5e-8 |
| 1e-4 | 9.5e-10 |
| 1e-5 | 4.6e-10 |
| 1e-6 | 1.9e-9 |

This is the U in full: truncation error falling as roughly eps^2 down the large-step arm,
a minimum at 1e-5, and roundoff amplification turning the curve back up below it. A wrong
Jacobian would instead show a floor that no step size improves, which is what separates
the two diagnoses. Before the line-search fix the same sweep ran from 8.5e-6 at 1e-2 to
8.4e-3 at 1e-6 — the minimum sat at the largest step tried and the small-step arm rose as
1/eps, because the solver's own residual floor was the noise being amplified.

## Verification tests

22 tests in `tests/test_projection_layer.py`, each checking a property against an
independent characterization rather than against the implementation's own output: the
output is a distribution; the constraint is satisfied; the solution matches the KKT tilt
form as a ratio identity; no feasible perturbation has smaller KL (200 random
constraint-preserving perturbations, the variational definition of the I-projection);
masked tokens receive exactly zero mass; the multiplier is zero and the projection is the
identity when the constraint already holds; gradients pass autograd's gradcheck at one and
two constraints; and a regression test asserting both a tight residual and a low iteration
count, since the stall presented as hitting the iteration cap.

All 157 tests in the suite pass: 22 here, 5 in `tests/test_fidelity_baseline.py`, 4 in `tests/test_tokenizer_coverage.py`, 12 in `tests/test_sweep_analysis.py`, 6 in `tests/test_crystal_mask.py`, 5 in `tests/test_crystal_constraint.py`, 4 in `tests/test_feasibility_exactness.py`, 18 in `tests/test_cif_grammar.py`, 15 in
`tests/test_parser_determinability.py`, 14 in `tests/test_actionability.py`, 9 in
`tests/test_ledger_preflight.py`, 9 in `tests/test_tilt_lemmas.py`, 8 in
`tests/test_synthetic_targets.py`, 8 in `tests/test_provenance.py`, 8 in
`tests/test_induced_distribution.py`, 7 in `tests/test_steering_frontier.py`, and 3 in
`tests/test_report_consistency.py`. Five of the provenance tests build throwaway
repositories and skip where `git init` is unavailable, so they exercise fully on a
developer machine and report as skipped in a restricted sandbox.

That last file exists because of this section. A hand-typed count in a standing note drifts
the moment a test is added, and it drifts silently — nothing fails, the prose simply becomes
wrong. It collects the real per-file and suite totals from pytest and fails on any report
claim that disagrees, in words or digits. The stale counts it was written to prevent were
found by review, not by the suite.

## Run ledger and pre-flight harness

The reporting infrastructure the plan requires from day one is in place.

The ledger (`scripts/common/ledger.py`) is append-only and line-delimited. Every run writes
a `started` record and later a terminal one; the started record is never rewritten, so a run
that crashed mid-way stays visible as exactly that rather than vanishing. A correction is a
new record that supersedes an earlier one by id. Records carry the config hash, seeds,
commit, and environment but deliberately carry no results — results are artifacts, and
keeping numbers out of the ledger prevents the two from disagreeing. A truncated final line
from a hard kill is tolerated on read.

The pre-flight harness (`scripts/common/preflight.py`) runs six checks and reports all of
them rather than stopping at the first failure, since a remote box usually has more than one
thing wrong and each round trip is expensive: imports, seed determinism, ledger round-trip,
artifact checksum round-trip, solver convergence, and any runs left in `started` from a
previous crash. The solver check includes the near-optimal-at-initialization case, so the
flat-dual trap is re-tested on every launch rather than only in CI. Current readings on this
machine: residual 5.8e-11 far from the optimum, 6.9e-17 near it, 1.1e-16 at it, in 5, 2, and
1 iterations.

## Hardware note

Nothing here needs a GPU, and a GPU would not help: cvxpylayers' cone solver runs on CPU
regardless of tensor placement, and the per-token problems are tiny by construction. The
CP2 per-token overhead benchmark is the part that does need one, and it needs a GPU of the
class A2 actually trains on, since the ratio it measures depends on kernel launch overhead
and memory bandwidth. Measuring it on this machine would produce a number that does not
transfer to the pre-registered setting.
