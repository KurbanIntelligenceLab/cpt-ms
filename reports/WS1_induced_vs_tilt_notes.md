# WS-1 — A3's induced sequence distribution vs the exponential tilt

Standing note, rebuilt on change. Feeds CP0, CP4, and CP5. Every number is read from
`results/WS1_induced_vs_tilt.json`, emitted by
`scripts/synthetic/induced_distribution.py`. Figure: `figures/induced_vs_tilt.png`.

## Summary

The A3-vs-A6 comparison is closed exactly on the synthetic task, and it does not resolve the
way the plan's framing implies. Per-step projection is not a slightly worse approximation to
the sequence-level tilt; it is a materially different distribution that misses the constraint
it was built to enforce. Induced mean charge is 0.1028 against a target of 0, where the tilt
hits 0 exactly. Its excess divergence over the tilt is 1.0254 nats — the induced distribution
sits 1.3601 nats from the base where the optimum needs only 0.3347, a factor of 4.06.

The mechanism is identified and measured, not inferred. The per-step surrogate is a running
budget: with target `c`, accumulated `s` after `t` steps, and `T - t` remaining, each step is
constrained to contribute `(c - s) / (T - t)` in expectation. That quantity is small while
many steps remain and grows without bound as the sequence closes, so late in decoding it
exceeds what any single token can contribute. Infeasibility first appears at step 5 of 8, and
by step 7 it affects 46.4% of nodes — 37.1% of all 16,537 tree nodes overall.

## Results

| Distribution | Mean charge | Var | Entropy (nats) | Violation rate | KL from base |
|---|---|---|---|---|---|
| base, S-conditioned | 2.5164 | 8.564 | 9.641 | 91.09% | 0 |
| tilt, sequence-optimal | 0.0000 | 10.452 | 9.896 | 87.55% | 0.3347 |
| induced, per-step A3 | 0.1028 | 3.103 | 9.330 | 62.00% | 1.3601 |

Divergences between the two constrained arms: KL(induced ‖ tilt) = 1.0518 nats,
KL(tilt ‖ induced) = 35.4968 nats, total variation 0.5553. The asymmetry is large because
induced concentrates mass — its charge variance is 3.103 against the tilt's 10.452, a factor
of 3.4 — so the tilt places substantial mass where induced places almost none.

Infeasibility of the running budget, by decoding step:

| Step | Nodes | Infeasible | Rate |
|---|---|---|---|
| 0-4 | 337 | 0 | 0.0% |
| 5 | 960 | 40 | 4.2% |
| 6 | 3,480 | 640 | 18.4% |
| 7 | 11,760 | 5,460 | 46.4% |

## What this changes

**The A3-vs-A6 comparison is not a refinement contest.** The plan treats A6 as an ablation
against A3's per-step route to the same target. These are not two routes to one place: the
per-step surrogate lands 0.1028 off target with a fundamentally different shape. Reporting A3
against the tilt as though it were approximating it will understate what is happening. Both
distributions should be characterized in their own terms, with the tilt as the reference
optimum rather than as A3's intended output.

**Per-sample violation moves in the opposite direction from divergence.** Induced has the
lowest violation rate of the three, 62.00% against the tilt's 87.55%, while being furthest
from the base in KL. It reduces per-sample violations by concentrating mass on near-balanced
sequences, which is not the same as satisfying the constraint in expectation and costs a
factor of 4 in divergence to achieve. This is a second, independent reason not to read
violation rate as a proxy for constraint satisfaction — the first being the near-invariance
recorded in the alignment sweep notes. A reviewer seeing only violation rates would conclude
that A3 outperforms the sequence-level optimum.

**Variance is where the cost lands.** The plan states that bounding expected increments bounds
the expected total but not its variance. The measured direction is the opposite of a warning:
the per-step route does not inflate variance, it collapses it, from 10.452 to 3.103. The tilt
achieves its constraint by spreading mass across compensating high- and low-charge sequences;
the running budget forbids that compensation, because a step that overshoots must be repaid
within the remaining steps rather than anywhere in the sequence. The entropy gap of 0.566 nats
is the same fact in another currency.

**Fallback behavior on infeasibility is now a pre-registered design choice, not an
implementation detail.** When the running budget is unsatisfiable, this implementation falls
back to the masked base conditional. Other choices exist — clamp to the nearest attainable
increment, or redistribute the residual over remaining steps — and they will produce different
induced distributions. Because infeasibility affects 37.1% of nodes here, that choice is
load-bearing rather than an edge case, and any A3 result must state which fallback was used.

## Two defects found and fixed

**The multiplier bracket was fixed-width.** Bisection ran over a hardcoded ±60, so a target
near the edge of the attainable range saturated the bracket and returned its own endpoint —
which looks like a converged multiplier and would bias every downstream number. Same failure
class as the flat-dual line-search stall recorded in the CP2 notes: a solver reporting success
while stopping short. The bracket now expands geometrically until it contains the root.

**The saturation counter added to detect that condition was a tautology.** It compared the
solved multiplier against the bisection bracket *after* bisection had run. Bisection reassigns
both bracket ends onto the root, so `abs(lam) > 0.9 * abs(hi)` was satisfied at essentially
every node with a non-zero multiplier. The counter reported 10,397 of 16,537 nodes — exactly
every node not already flagged infeasible, which is the signature of the bug rather than a
measurement. Comparing against the pre-bisection bracket instead gives 1,585 saturating nodes,
15.2% of the 10,397 feasible ones.

The instructive part is that the tautology's own output contained the evidence:
infeasible + saturated summed to the node count exactly. A diagnostic that cannot report
absence is not measuring anything, and a partition that exhausts its domain should have been
read as a defect signature immediately. The regression test asserts the two counts do not
exhaust the node set, and it fails on the reverted code.

No reported distribution value changes under either fix — the bracket was not distorting these
numbers, though it would on a task with a tighter target, and the saturation count was never
an input to any other quantity.

## Verification

Eight tests in `tests/test_induced_distribution.py`, each checking a property against an
independent characterization: normalization and support-constraint compliance; support
equality with the tilt, so the KL comparison is between reweightings of one support rather
than different objects; KL-optimality of the tilt as a lower bound the greedy route cannot
beat; the infeasibility mechanism, asserted as absent at the first step, present at the last,
and worsening in between; the bracket-saturation regression and the assertion that the saturation counter can report absence; and that per-step projection
moves the mean toward target relative to the unconstrained base even though it does not
arrive. The per-position infeasibility counts were also reproduced by an independent tree walk
that shares no code with the measurement.
