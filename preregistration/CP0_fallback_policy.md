# Pre-registration — infeasible-step fallback policy for the projection layer

Status: draft, not yet committed. Becomes binding at the first A3 training run, which is why
it is written now rather than after.

## Why this needs pre-registering

The per-step surrogate is a running budget: with target `c`, accumulated `s` after `t` steps,
and `T - t` steps remaining, each step must contribute `(c - s) / (T - t)` in expectation.
That requirement grows without bound as the sequence closes, so late in decoding it can exceed
what any single token can contribute. On the synthetic task the budget is unsatisfiable at
37.1% of all tree nodes — first at step 5 of 8, reaching 46.4% by step 7. This is not an edge
case, so the policy applied at those nodes materially determines what A3 produces and must be
fixed in advance rather than chosen after seeing results.

## Measured comparison

Both candidate policies were implemented and enumerated exactly. Numbers are from
`results/WS1_induced_vs_tilt.json` at the default target, with the sequence-level tilt paying
0.3347 nats as the reference optimum.

| Policy | Mean-charge error | KL from base | Excess over tilt | Charge variance |
|---|---|---|---|---|
| `base` — emit the masked base conditional | +0.1028 | 1.3601 | 1.0254 | 3.103 |
| `clamp` — clip the target to the nearest attainable increment | +0.0033 | 1.5950 | 1.2603 | 1.505 |

The trade is clean and runs in the expected direction. Clamping cuts the constraint error by a
factor of 31 and pays 0.2349 additional nats of divergence, collapsing charge variance by a
further factor of two. Falling back to the base conditional preserves fidelity to the model
and accepts a residual constraint error.

A third policy — redistributing the unmet residual over the remaining steps — is not
implemented and is explicitly out of scope. It is not a per-step projection: the redistributed
amount depends on steps not yet taken, so the constraint stops being a function of the prefix
and the layer stops being trainable without a straight-through estimator.

## Pre-registered choice

**`base` is the primary policy for all reported A3 results.** `clamp` is reported as a
secondary arm wherever the primary is reported.

The reasoning is that the project's claim concerns training-time projection, not a decoding
heuristic. The `base` policy has one property `clamp` does not: it never pushes the
distribution toward a target the alphabet cannot reach at that step. `clamp` buys accuracy by
distorting steps whose constraint is already unsatisfiable, which is precisely where the
surrogate has no information — and it pays for that in the variance collapse the plan warns
about, halving an already-collapsed spread.

Reporting both is not hedging. The gap between them is a measurement of how much of A3's
residual error is attributable to the fallback rather than to the per-step surrogate itself,
which is a quantity a reviewer will ask for.

## Binding requirements

1. Every A3 result states its fallback policy. The emitting code records it in the artifact
   (`per_step_diagnostics.fallback_policy`), so a result cannot be reported without it.
2. An unrecognized policy name raises rather than silently defaulting. A pre-registration that
   can be bypassed by a typo is not a pre-registration; a test asserts the rejection.
3. The infeasible-node fraction is reported alongside any A3 result, since the policy's
   influence scales with it. A result at low infeasibility is nearly policy-independent; one at
   37% is not.
4. Changing the primary policy after the first A3 training run requires an append to this file
   stating what was seen first. This directory is append-only.

## Interaction with the CP0 threshold

The fallback affects multiplier activity, which CP0 reads. At an infeasible node under `base`
the projection is inactive by construction — no multiplier is solved — so the active-multiplier
fraction is mechanically depressed wherever the budget fails. On the synthetic task that would
remove 37.1% of nodes from the numerator for reasons unrelated to constraint internalization.

CP0's active fraction is therefore computed over **feasible actionable positions**, excluding
nodes where the budget was unsatisfiable, and the excluded fraction is reported alongside. This
composes with the actionable-window scoping in `CP0_actionable_window.md`: the denominator is
positions that are both actionable and feasible. Without this the gate would read collapse from
budget infeasibility, which is the same failure mode as reading collapse from a residual floor.
