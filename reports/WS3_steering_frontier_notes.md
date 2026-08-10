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
and should push the backbone further up.

| Regression | Slope | Std err | t | R^2 |
|---|---|---|---|---|
| backbone offset on target | +0.0245 | 0.1422 | +0.17 | 0.0019 |

The slope is indistinguishable from zero and has the wrong sign for compensation. The
backbone sits at 9.75 on average, +3.07 above the data mean, with no dependence on the
target at all (R^2 0.002), while the deployed distribution does track the target.

**So the backbone inversion is real but it is not compensation.** It is a fixed upward
bias, present at every target, not an adaptive offset tuned to the projection's pull.
What causes it remains open; what is now closed is that it does not scale with the
quantity it would need to scale with.

**This retires the replacement risk from the risk register.** That risk was that the
combined system might be correct only at the target it trained for, on the reasoning that
a learned compensation would be target-specific. A target-independent bias does not carry
that failure mode. The risk register entry should be replaced with the narrower and now
measured statement that steering has an upward ceiling near 10.4 g/cm^3.

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

10 tests in `tests/test_sweep_analysis.py` validated the regression estimator against
constructed data with known slopes before it read these numbers, including one asserting
that a statistically significant but negligible slope must not be reported as
compensation. Those tests confirmed the estimator; they could not have caught the
algebraic dependence between the two regressed quantities, which is a defect in the
choice of variables rather than in the fit. The identity check is now recorded above so
the invalid framing cannot be reintroduced.
