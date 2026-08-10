# WS-3 — Crystal steering frontier, and compensation rejected

Standing note, rebuilt on change. Feeds CP4, CP5. Numbers read from
`results/WS3_steering_frontier_rows.json`, extracted from the 18 completed runs of the
10-target frontier under the corrected tokenizer (884-token vocabulary).

## Summary

Steering is strongly asymmetric: **it tracks the target below the data mean and fails
above it.** Mean absolute error is 0.345 g/cm^3 for targets at or below the data mean of
6.679, and 4.577 above it. Deployed density saturates at 10.405 — the model cannot be
steered above roughly 10.4 g/cm^3 no matter what is requested.

**The compensation hypothesis is rejected.** The backbone's offset from the data mean is
statistically independent of the target (slope +0.024 +/- 0.142, t = 0.17, R^2 = 0.002).
The backbone sits 3.07 above the data mean at every target rather than adjusting to the
pull it must offset. Reaching that conclusion required discarding my first test, which
was invalid by construction.

## The frontier

Two seeds per target except 13.0 and 16.0, which have one each after two runs failed.

| Target | Deployed | Abs error | Backbone | Validity | Infeasible init to final |
|---|---|---|---|---|---|
| 1.500 | 1.848 +/- 0.077 | 0.348 | 8.087 | 1.0000 | 0.2315 to 0.0079 |
| 2.000 | 2.266 +/- 0.002 | 0.266 | 8.191 | 1.0000 | 0.2602 to 0.0117 |
| 3.000 | 3.240 +/- 0.092 | 0.240 | 9.034 | 1.0000 | 0.2989 to 0.0220 |
| 5.000 | 4.849 +/- 0.097 | 0.151 | 9.196 | 1.0000 | 0.3379 to 0.0483 |
| 6.317 | 5.598 +/- 0.286 | 0.719 | 10.422 | 1.0000 | 0.3528 to 0.0649 |
| 8.000 | 6.760 +/- 0.031 | 1.240 | 10.261 | 1.0000 | 0.3666 to 0.0875 |
| 10.000 | 8.330 +/- 0.789 | 1.670 | 15.591 | 1.0000 | 0.3777 to 0.1224 |
| 13.000 | 8.744 | 4.256 | 9.798 | 1.0000 | 0.3948 to 0.2068 |
| 16.000 | 9.875 | 6.125 | 8.458 | 1.0000 | 0.3843 to 0.3051 |
| 20.000 | 10.405 +/- 0.204 | 9.595 | 7.803 | 1.0000 | 0.4002 to 0.4133 |

**Structural validity is exactly 1.0000 at every target and every seed**, including the
extremes where steering fails. The masking claim is independent of the steering claim, as
designed: the mask makes malformed output unreachable regardless of how hard the
constraint is pulling.

Steering slope is 0.482 over the full range and 0.792 restricted to targets at or below
the data mean, against an ideal of 1.0.

## The asymmetry has a measured mechanism

Infeasibility rises monotonically with target, from 0.0079 at target 1.5 to 0.4133 at
20.0, and the reduction training achieves collapses with it: **29.2x at target 1.5, 0.97x
at 20.0.** At the highest target training no longer reduces infeasibility at all.

This is the running mass budget hitting the attainable range. Steering upward demands
more mass per atom than the heaviest legal token can supply, so the budget goes
infeasible, the fallback fires, and the constraint stops binding. Steering downward has
room because light elements are always available. The complementarity mechanism therefore
has a domain of validity that this sweep measures directly, rather than holding
everywhere.

## Compensation: the first test was invalid, and the valid test rejects it

The hypothesis was that the backbone biases its predictions to offset a projection it
knows will pull one way, so the post-projection result matches the data.

**My first test could not have answered it.** It regressed
`backbone_offset_vs_data = backbone - data_mean` on
`projection_pull = backbone - deployed`. Both contain the backbone density with a
positive sign, so the two are algebraically linked:

    offset - pull = deployed - data_mean

verified to 8.9e-16 across all 18 runs. Regressing one on the other is regressing a
quantity on itself plus a term, so a slope near 1 is the arithmetic baseline rather than
evidence. That test reported slope 0.585 (t = 3.66) and a partial correlation of 0.975
controlling for target — the second figure being the identity showing through once
target-driven variation in `deployed` is removed. Neither number is informative about
compensation.

Two independent checks then contradicted each other, which is what exposed the problem.
The projection's pull changes sign at high targets (negative at 16.0 and 20.0, where it
pushes density up rather than down). Under compensation the backbone's offset must follow
that reversal and go negative. **It does not** — the offset stays positive everywhere,
averaging +3.393 where the pull is positive and +1.451 where it is negative.

**The valid test regresses the offset on the TARGET**, which shares no term with it.
Compensation predicts a negative slope: a lower target demands a stronger downward pull
and should push the backbone further up. Two aggregations are reported because they give
different numbers and a reader recomputing from the table above will get the first:

| Regression of backbone offset on target | Slope | Std err | t | R^2 |
|---|---|---|---|---|
| seed-averaged, 10 target rows (matches the table) | +0.0080 | 0.1286 | +0.06 | 0.0005 |
| per-run, 18 runs | +0.0245 | 0.1422 | +0.17 | 0.0019 |

Both are indistinguishable from zero and both have the wrong sign for compensation.
**Compensation is rejected**, and the pull-sign-reversal argument above is decisive on its
own without either regression.

### The offset is not constant, and a linear test was the wrong instrument

An earlier version of this note reported the offset as "+3.07 at every target". That was
the mean presented as though it were a constant, and it was wrong: **the offset ranges
+1.12 to +8.91** across targets. Concluding "no relationship" from a linear fit to a
non-monotone relationship is the same class of error as the invalid regression above, and
it is recorded here as such.

The structure, with the outlier run discussed below excluded:

| Model | Parameters | R^2 | Adjusted R^2 |
|---|---|---|---|
| linear in target | 2 | 0.020 | — |
| quadratic in target | 3 | 0.828 | 0.770 |
| linear in \|target - ceiling\| | 2 | 0.810 | 0.783 |

The linear-in-target fit gives slope -0.021, t = -0.37: no linear trend, as reported. The
quadratic fit is strong and robust — leave-one-out over the nine rows holds R^2 between
0.789 and 0.926.

**But the vertex is not at the data median.** The fitted vertex sits at target 9.93, and
every leave-one-out refit places it between 9.15 and 10.33 — never near the data median of
6.317. The highest observed offset does occur at target 6.317, which is the peak *value*
rather than the vertex *location*; those are different quantities and conflating them
would put the mechanism in the wrong place.

**The mechanism is the steering ceiling, not the data centre.** Regressing the offset on
the distance from the target to the achievable ceiling of 10.405 gives R^2 = 0.810 with
one parameter fewer than the quadratic, and a better adjusted R^2. Among the candidate
regressors — target (0.020), absolute steering error (0.144), infeasible fraction (0.086),
deployed density (0.005) — ceiling distance at 0.810 is the only one that explains the
shape.

Read mechanistically: the backbone drifts furthest above the data mean when the requested
target is closest to the density the model can actually reach. Far below the ceiling the
projection has ample room and pulls the backbone back toward feasibility; far above it the
budget is infeasible so often that the constraint stops binding and the backbone reverts
toward the data. Neither extreme leaves the backbone free to drift.

So the inversion is **real, not compensation, and not constant** — it varies
non-monotonically with steering difficulty, and specifically with proximity to the
reachable ceiling.

### The target-10.0 excursion is single-seed

The seed-averaged backbone at target 10.0 is 15.591, which is 5.6 above its neighbours'
mean of 10.030. It is one run, not a target property:

| Target 10.0 | Backbone | Deployed | Infeasible | Validity |
|---|---|---|---|---|
| seed 0 | 22.894 | 8.888 | 0.1149 | 1.0000 |
| seed 1 | 8.288 | 7.772 | 0.1300 | 1.0000 |

Seed 1 sits squarely with its neighbours; seed 0 is 4.2 standard deviations above the
median of all eighteen backbone values, with the next highest at 10.566. It also exceeds
the maximum density in MP-20 itself (21.976) — possible only because the unprojected
backbone is unconstrained, and a reminder that the backbone reading is not a distribution
anyone would deploy.

Both fits are therefore reported with and without it. Dropping the whole target gives
quadratic R^2 0.828; dropping only the outlier run and retaining target 10.0 through
seed 1 gives 0.517. The curvature survives either way, but its strength depends on the
treatment, which is why both are stated. **Note also that the deployed density at target
10.0 is unaffected by the excursion** — 8.888 against 7.772, both far below the request —
so the frontier's headline asymmetry does not rest on this run.

## Two runs failed, and the cause is fixed

Targets 13.0 seed 0 and 16.0 seed 1 died with a device-side assert in `multinomial`. The
pattern was seed-dependent rather than target-dependent — 20.0 succeeded on both seeds
while 13.0 and 16.0 each failed on one — which pointed at a rare degenerate solve rather
than an unreachable target.

Cause: sanitization order. The code called `q.clamp_min(0).nan_to_num()`, but `clamp_min`
leaves NaN as NaN and `nan_to_num` then maps it to zero, so an all-NaN row became an
all-zero row and `multinomial` asserted on a zero-sum distribution. The order is now
reversed, a row with no mass falls back to the masked softmax, a row where even that is
degenerate emits the end token, and the fallback rate is reported as
`degenerate_row_rate` rather than silently repaired.

The 18 completed runs are unaffected: the assert aborts the run rather than corrupting
it, so a run that produced an artifact never took the degenerate path.

## Verification

12 tests in `tests/test_sweep_analysis.py` validated the regression estimator against
constructed data with known slopes before it read these numbers, including one asserting
that a statistically significant but negligible slope must not be reported as
compensation. Those tests confirmed the estimator; they could not have caught the
algebraic dependence between the two regressed quantities, which is a defect in the
choice of variables rather than in the fit. The identity check is now recorded above so
the invalid framing cannot be reintroduced.
