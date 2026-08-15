# WS-4: zero-shot retargeting — one checkpoint, ten targets

`results/WS4_zero_shot_retargeting.json`

## The question

The constraint target is an argument to the dual solve, not a training-time label. If
that is more than a description of the code, a checkpoint trained at one density target
should steer to targets it never trained for, without retraining. The conditional
baseline A4 cannot do this by construction: its target is baked into which data it saw.

## What was run

Three A3 checkpoints, trained at density targets 3.0, 5.0 and 10.0 g/cm^3, 6,000 steps
each at seed 0. Each was then deployed at all ten frontier targets with no retraining and
no gradient steps — only `cfg.density_target` changes between cells. Weights, vocabulary,
grammar mask and sampling seed are identical within a row, so a difference in achieved
density is attributable to the deployment target alone. Thirty cells, 30 rows in the
artifact, 241 s of inference in total against roughly 1,660 s for one training run.

Checkpointing had to be added to make this possible: before this work no run in the
project saved weights, so the "inference-only" experiment had no checkpoints to run on.

## Result 1 — Retargeting works, and costs about a factor of two in accuracy

Comparing each off-diagonal cell against the diagonal at the same deployment target
(train-at-target, the existing per-target result), off-diagonal error is roughly 1.9x to
2.7x the diagonal, rising with distance from the training target.

The comparison that matters for the amortization claim is different, because it holds
training compute fixed rather than accuracy. Against the published frontier, which trains
one model per target, mean absolute error over the seven sub-ceiling targets is
0.662 g/cm^3 across seven training runs. A single checkpoint deployed at all seven:

| checkpoint | training runs | MAE over 7 sub-ceiling targets | ratio |
|---|---|---|---|
| trained at 3.0 | 1 | 1.106 | 1.67x |
| trained at 5.0 | 1 | 0.790 | 1.19x |
| trained at 10.0 | 1 | 0.441 | 0.67x |
| frontier, train-per-target | 7 | 0.662 | 1.00x |

On this single seed the checkpoint trained at 10.0 covers the whole sub-ceiling family more accurately
than seven separately trained models, at one seventh of the training compute. That reading does not
survive replication, and the difference matters: the three-seed replication below finds the checkpoint
better at 3 of 7 targets with a paired Wilcoxon p of 1.0, which is statistically
indistinguishable rather than superior. The claim this section supports is therefore parity at one
seventh of the training compute, not an accuracy win, and the 0.67x ratio above is a favourable draw
from one seed rather than a result.

## Result 2 — Regime S survives retargeting exactly

Structural validity is [1.0] at every one of the 30 cells: 1.0000 everywhere,
including targets no checkpoint trained for and targets past the reachable range. This is
the strongest form of the mask claim available — the mask is a property of the grammar,
not of the trained weights or the target — and it is asserted by test rather than
described.

Coverage holds too: novel-formula fraction stays between 0.928 and 0.982, and unique-formula
fraction between 0.503 and 0.977, across every cell.

## Result 3 — The "ceiling" is a property of the checkpoint, not of the method

The frontier reported a reachable ceiling of 10.405 g/cm^3: no target above it was
achievable, and the interpretation on record is that steering saturates near the top of
the data. The retargeting matrix reaches 15.495.

The mechanism is visible in the backbone reading. The three checkpoints sit in very
different places with the mask alone and reach correspondingly far:

| checkpoint | backbone density (mask only) | max achieved with projection |
|---|---|---|
| trained at 3.0 | 8.544 | 9.787 |
| trained at 5.0 | 8.677 | 10.397 |
| trained at 10.0 | 22.769 | 15.495 |

So what the frontier measured as a ceiling of the method is the reach of the particular
backbone those runs happened to train. A backbone sitting at 22.8 reaches half again as
far as one sitting at 8.5.

**This was single-seed when first measured; the replication below settles it.** The 22.8 backbone
reproduces the frontier's own seed-0 excursion at target 10.0 — the frontier's two seeds
there were 22.894 and 8.288, an eightfold spread already disclosed as single-seed in
`WS3_steering_frontier_notes.md`. Deploying the same seed-0 weights across ten targets
confirms the projection is deterministic given the backbone; it does not establish that a
high backbone is reachable generically. Seeds 1 and 2 have since been run; see the replication section
below, which finds the excursion does not reproduce and reduces this to a soft-ceiling
claim.

## Result 4 — Infeasibility rises with the ask, as it should

For the checkpoint trained at 10.0, the probability-weighted infeasible-step fraction
climbs monotonically with the deployment target, from 0.0030 at target 1.5 to
0.3769 at target 20.0. The running budget becomes unsatisfiable more often as the ask
moves further from what the backbone produces, which is the same mechanism the synthetic
task showed by enumeration.

## The full matrix

| train | deploy | achieved | error | backbone | infeasible | novel | unique |
|---|---|---|---|---|---|---|---|
| 3.0 | 1.5 | 1.708 | 0.208 | 8.573 | 0.0039 | 0.960 | 0.503 |
| 3.0 | 2.0 | 2.281 | 0.281 | 8.411 | 0.0096 | 0.950 | 0.585 |
| 3.0 (diag) | 3.0 | 3.193 | 0.193 | 8.646 | 0.0243 | 0.928 | 0.663 |
| 3.0 | 5.0 | 4.384 | 0.616 | 8.538 | 0.0451 | 0.934 | 0.686 |
| 3.0 | 6.317 | 5.096 | 1.221 | 8.614 | 0.0548 | 0.943 | 0.698 |
| 3.0 | 8.0 | 6.005 | 1.995 | 8.502 | 0.0662 | 0.955 | 0.703 |
| 3.0 | 10.0 | 6.770 | 3.230 | 8.532 | 0.1064 | 0.951 | 0.685 |
| 3.0 | 13.0 | 8.193 | 4.807 | 8.448 | 0.1712 | 0.957 | 0.690 |
| 3.0 | 16.0 | 9.059 | 6.941 | 8.660 | 0.2745 | 0.955 | 0.691 |
| 3.0 | 20.0 | 9.787 | 10.213 | 8.512 | 0.3962 | 0.961 | 0.694 |
| 5.0 | 1.5 | 1.713 | 0.213 | 8.509 | 0.0024 | 0.952 | 0.571 |
| 5.0 | 2.0 | 2.312 | 0.312 | 8.809 | 0.0056 | 0.948 | 0.663 |
| 5.0 | 3.0 | 3.353 | 0.353 | 8.719 | 0.0211 | 0.935 | 0.750 |
| 5.0 (diag) | 5.0 | 4.839 | 0.161 | 8.619 | 0.0432 | 0.944 | 0.781 |
| 5.0 | 6.317 | 5.617 | 0.700 | 8.807 | 0.0653 | 0.948 | 0.798 |
| 5.0 | 8.0 | 6.538 | 1.462 | 8.719 | 0.0844 | 0.962 | 0.808 |
| 5.0 | 10.0 | 7.671 | 2.329 | 8.642 | 0.1159 | 0.953 | 0.811 |
| 5.0 | 13.0 | 8.830 | 4.170 | 8.575 | 0.1873 | 0.962 | 0.822 |
| 5.0 | 16.0 | 9.572 | 6.428 | 8.705 | 0.2872 | 0.952 | 0.824 |
| 5.0 | 20.0 | 10.397 | 9.603 | 8.672 | 0.4086 | 0.959 | 0.829 |
| 10.0 | 1.5 | 1.835 | 0.335 | 22.577 | 0.0030 | 0.965 | 0.604 |
| 10.0 | 2.0 | 2.366 | 0.366 | 22.618 | 0.0060 | 0.961 | 0.717 |
| 10.0 | 3.0 | 3.373 | 0.373 | 23.010 | 0.0182 | 0.953 | 0.795 |
| 10.0 | 5.0 | 5.064 | 0.064 | 23.072 | 0.0477 | 0.959 | 0.865 |
| 10.0 | 6.317 | 6.004 | 0.313 | 22.709 | 0.0610 | 0.963 | 0.894 |
| 10.0 | 8.0 | 7.384 | 0.616 | 22.724 | 0.0869 | 0.965 | 0.925 |
| 10.0 (diag) | 10.0 | 8.982 | 1.018 | 22.737 | 0.1209 | 0.975 | 0.945 |
| 10.0 | 13.0 | 11.154 | 1.846 | 22.851 | 0.1844 | 0.977 | 0.957 |
| 10.0 | 16.0 | 13.222 | 2.778 | 22.713 | 0.2428 | 0.981 | 0.960 |
| 10.0 | 20.0 | 15.495 | 4.505 | 22.681 | 0.3769 | 0.982 | 0.977 |

## Seed replication — what survives and what does not

`results/WS4_retargeting_seed_replication.json`

The seed-0 result above raised two claims that a single seed cannot settle. Seeds 1 and 2
of the target-10.0 checkpoint were trained and retargeted identically to test both.

| seed | backbone density (mask only) | max achieved with projection | sub-ceiling MAE |
|---|---|---|---|
| 0 | 22.769 ± 0.153 | 15.495 | 0.441 |
| 1 | 8.381 ± 0.103 | 10.439 | 0.789 |
| 2 | 8.692 ± 0.116 | 10.596 | 0.720 |

**The ceiling claim is weaker than seed 0 suggested, and it is not gone.** Seed 0's
backbone at 22.769 reproduces the frontier's own seed-0 excursion and is not generic: seeds 1
and 2 land at 8.381 and 8.692, close to the data mean. Their reach is correspondingly
lower, 10.439 and 10.596, against seed 0's 15.495. The published ceiling of 10.405 is
exceeded at all three seeds, but by 0.03 and 0.19 g/cm^3 at seeds 1 and 2 rather than by
5.09. So the ceiling is soft rather than absent: it moves with the backbone, and the
dramatic version of that claim rests on an initialization the other seeds do not
reproduce. Across the five checkpoints now available the backbone-to-reach correlation is
0.992, but drop the excursion and the remaining four span a backbone range of only
8.38 to 8.69, where the correlation is 0.217 — the relationship is established by one
extreme point, not by a spread.

**The amortization claim survives, and it strengthens.** Averaged over three seeds, one
checkpoint deployed at seven sub-ceiling targets reaches MAE 0.650 ± 0.151 g/cm^3, against
0.662 for the frontier's seven separately trained models:

| deploy target | retargeted error (3 seeds) | frontier error (train-per-target) | better |
|---|---|---|---|
| 1.5 | 0.292 ± 0.036 | 0.348 | retargeted |
| 2.0 | 0.329 ± 0.027 | 0.266 | frontier |
| 3.0 | 0.301 ± 0.080 | 0.240 | frontier |
| 5.0 | 0.208 ± 0.115 | 0.151 | frontier |
| 6.317 | 0.574 ± 0.194 | 0.719 | retargeted |
| 8.0 | 1.080 ± 0.339 | 1.240 | retargeted |
| 10.0 | 1.764 ± 0.533 | 1.670 | frontier |

The retargeted model is better at 3 of 7 targets, the mean paired difference is
-0.012 g/cm^3, and a Wilcoxon signed-rank test over the seven paired targets gives
p = 1.0. One training run is statistically indistinguishable from seven at one seventh
of the training compute. That is a stronger statement than the seed-0 reading of "0.67x
the error", which was a favourable draw rather than a result.

Structural validity remains exactly 1.0000 across every cell of all three seeds.

## What this does not establish

Three training targets on one crystal serialization; the target-10.0 checkpoint carries
three seeds, the 3.0 and 5.0 checkpoints one each. No claim is made about
retargeting across constraint families — every cell here retargets density to density.
