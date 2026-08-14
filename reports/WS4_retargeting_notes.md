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

The checkpoint trained at 10.0 covers the whole sub-ceiling family more accurately than
seven separately trained models, at one seventh of the training compute. That is not a
result we should lean on hard from one seed, and Result 3 explains why.

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

**This is single-seed and must not be published as it stands.** The 22.8 backbone
reproduces the frontier's own seed-0 excursion at target 10.0 — the frontier's two seeds
there were 22.894 and 8.288, an eightfold spread already disclosed as single-seed in
`WS3_steering_frontier_notes.md`. Deploying the same seed-0 weights across ten targets
confirms the projection is deterministic given the backbone; it does not establish that a
high backbone is reachable generically. Seeds 1 and 2 at target 10.0 are running to
settle it. Until they land, the defensible claim is the conditional one: reach scales with
where the backbone sits, and the published ceiling is not known to be a property of the
method.

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

## What this does not establish

Three checkpoints at one seed each, on one crystal serialization. The ceiling result rests
on a backbone whose value is a known single-seed excursion. No claim is made about
retargeting across constraint families — every cell here retargets density to density.
