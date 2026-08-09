# Response — the 67/33 split is right, and 17.6 is a t-statistic rather than a separation

## The decomposition is confirmed exactly

The correction to 67/33 reproduces to the digit against the artifacts:

| Stage | Density error vs 5.0 target | Gap closed | Share |
|---|---|---|---|
| A2 baseline, no layer | 1.783 | — | — |
| Layer on the untrained backbone | 0.710 | 1.072 | 67% |
| Trained through the layer | 0.190 | 0.520 | 33% |

This will be published as the split rather than as a bare 9.4x, and the note now carries it.
The point that a reviewer would compute it anyway is the right reason.

The complementarity claim attaches specifically to the 33%, and the mechanism figure that
supports it is the infeasibility drop: the probability-weighted infeasible-step fraction falls
6.9x, from 0.3430 +/- 0.0598 at initialization to 0.0496 +/- 0.0067 at convergence. That is
what training buys and it is measured independently of the density figure.

## On 17.6: it is the t-statistic, not a separation in standard deviations

You are right that my 12.9 was wrong — it used the deployed arm's standard deviation alone,
which is not a sensible denominator for a between-arm comparison. But 17.6 is not the pooled
separation either. Recomputed from the twelve artifacts:

| Quantity | Value |
|---|---|
| A2 baseline | 6.783, sample sd 0.0501 |
| A3 deployed | 4.810, sample sd 0.1870 |
| Pooled sd (sample, n = 3 per arm) | 0.1369 |
| **Cohen's d, pooled** | **14.4** |
| t-statistic | 17.66 |
| Welch's t, p | 17.66, p = 0.0017 |

17.66 is exactly the t-statistic, and it reproduces your figure to two decimals. The two
differ by a factor of sqrt(n/2) = 1.2247: t = d * sqrt(n/2), and 14.42 * 1.2247 = 17.66. A
t-statistic carries the sample size through the standard error, so it is not in units of
standard deviations and grows with n at fixed effect size. Pooling population standard
deviations instead of sample ones gives 17.7, which is the same arithmetic reached another
way — at n = 3 dividing by n rather than n - 1 understates the variance enough to matter.

So the figure to publish is **d = 14.4, n = 3 per arm**, with the significance stated
separately as Welch's t = 17.66, p = 0.0017. Welch rather than Student because the arm
variances differ by 13.9x (a 3.7x ratio in standard deviations), which at three seeds per arm
is not a difference to assume away.

This does not change any conclusion. It is a labelling question, and the labelling matters
precisely because a reviewer will recompute: quoting 17.6 as a separation invites the same
objection you raised about 12.9, in the opposite direction.

## Compensation test: folded into the sweep, now running

The hypothesis is worth testing and the instrumentation is cheap, so it is already in the
sweep rather than waiting for a separate run. Each evaluation now records two quantities per
target:

  - `projection_pull` — backbone mean density minus deployed mean density, i.e. how far the
    projection moves the distribution at generation.
  - `backbone_offset_vs_data` — backbone mean density minus the data mean of 6.679.

If compensation is the mechanism, the offset scales with the pull across targets, with a slope
near 1 and a positive intercept only if some of the offset is target-independent. If the
backbone's offset is flat while the pull varies, compensation is not the explanation and the
inversion needs a different account.

The sweep covers six targets — 3.0, 4.0, 5.0, 6.317 (the data median), 8.0, 10.0 — at two
seeds each, spanning roughly one standard deviation either side of the median. The 6.317 cell
is the control: with the target at the data's own centre the projection has almost nothing to
pull against, so both pull and offset should approach zero. A compensation slope that survives
with that cell included is the stronger result.

## Internalization on crystals

Agreed, and the plan sections will be updated rather than left standing beside the result. The
two-domain statement is:

  - Synthetic task, fixed length and exact channel: the backbone internalizes partway and the
    layer supplies the remainder.
  - Crystals, estimated denominator and a real backbone: the backbone moves against the layer.
    Its unaided distribution sits at 9.200, which is 2.522 above the data mean of 6.679, under
    a constraint pulling density down. The system is correct only in combination.

Both support the projection as a training component rather than a decoder, which is the
thesis. The crystal case is the sharper evidence: removing the layer at inference does not
degrade the result, it inverts it.

The multiplier-decay framing in the risk register should be retired on crystals for the same
reason. Decay was framed as the layer becoming unnecessary; what the measurement shows is the
opposite dependency.

## The data-satisfies-the-constraint diagnostic

Agreed that it belongs in methods. Stated as a practice: before training against a constraint
channel, check whether the training data itself satisfies the demanded target at the positions
where the constraint is active. Both formulation defects were found this way and neither would
have been found by reading the code — they trained to completion, loss fell, and the outputs
were structurally valid throughout. The first was visible as a demanded mass 3.00x what the
data's own next element carries; the second as an attainable range collapsed to a single point
at positions where the projection was still being asked for a decision.

## Scope

Agreed on all four drops. Nothing further will be added before the sweep, baselines, and DFT
land.
