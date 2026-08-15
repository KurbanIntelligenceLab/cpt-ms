# WS-7: A6, the sequence-level comparator, and what the surrogate actually buys

`results/WS7_a6_tilt_comparator.json`

## What A6 is and why it is the comparison that matters

Khalifa, Elsahar and Dymetman (ICLR 2021) pose controlled generation as moment constraints and
derive the same KL-minimal exponential-family optimum this project calls the tilt, then train an
autoregressive model toward it by distributional policy gradient. That target is what the per-step
projection approximates, so the question "what does the surrogate lose?" is only answerable against
a model actually trained toward the tilt rather than against the tilt's closed form.

A6 does maximum likelihood on exact samples from the tilt. The tilt is available in closed form on
this task, and MLE on its exact samples is DPG's fixed point with unit importance weights, so A6
gets the best case of the method it stands for and no gradient-estimator artefact can be blamed for
the outcome. A6 carries the Regime-S mask like every arm; the only difference from A3 is that A6 has
tilt supervision and no decode-time projection, and A3 has the projection and no tilt supervision.

The synthetic task is exactly enumerable, so both arms' induced distributions are computed over all
65,536 sequences rather than sampled. Every divergence below is exact.

## A6 reaches its target

| quantity | A6 (3 seeds) | tilt (exact optimum) |
|---|---|---|
| KL from the tilt | 0.0167 ± 0.0048 | 0 |
| total variation from the tilt | 0.0713 ± 0.0099 | 0 |
| mean charge | -0.0619 ± 0.1220 | +0.0000 |
| mass outside the mask | 0.0000 | 0.0000 |

A6's mean charge is within its own seed spread of the target, so it satisfies the expectation
constraint. This matters for reading everything below: A6 is not a failed training run, and the
comparison is not between a working method and a broken one.

## The comparison

| arm | mean charge | per-sequence violation | charge variance |
|---|---|---|---|
| A2, mask only | +2.4694 ± 0.0807 | 0.9083 | 8.762 |
| A3 deployed, projection | +0.0170 ± 0.0014 | 0.6155 | 1.547 |
| A6, trained toward the tilt | -0.0619 ± 0.1220 | 0.8751 | — |
| tilt, exact optimum | +0.0000 | 0.8755 | — |

Both arms satisfy the expectation constraint. A3's mean charge is 3.6 times closer to zero than
A6's, but A6's seed spread (0.1220) is larger than that gap, so the ordering on mean charge is
not established by three and two seeds.

The difference that is not seed noise is per-sequence violation. A6 inherits the tilt's: 0.8751
against the optimum's 0.8755, as it should, since A6 is approximating that distribution and the
distribution itself places 87.55% of its mass on individually violating sequences. A3 deployed
violates 0.6155 — a factor of 1.42 fewer — with a seed spread of 0.0080.

## What this means

The per-step surrogate is not a worse approximation to the tilt. It is an approximation to a
different thing, and on this task the different thing is closer to what someone asking for a
constraint usually wants.

Bounding an expected increment at every step bounds the total in expectation, and it also
concentrates it: A3's charge variance is 1.547 against A2's 8.762, a factor of 5.7 reduction. The
sequence-level tilt has no such mechanism — it is the KL-closest distribution with the right mean,
and the cheapest way to buy a mean of zero from a distribution centred at +2.469 is to move mass to
both extremes. That is why the tilt violates more per sequence than the model it started from.

So the surrogate's 1.025-nat excess divergence from the tilt, measured earlier, is not a pure loss.
It buys a variance reduction the tilt does not offer and a per-sequence violation rate 1.42 times
lower than the optimum it approximates. A practitioner who wants "most samples satisfy the
constraint" rather than "the average satisfies it" should prefer the surrogate on this task, and
that preference is measured rather than argued.

## What this does not establish

One task, one constraint family, one target. The variance reduction follows from per-step bounding
in general, but its size and the resulting violation gap are properties of this task's charge
distribution. A constraint whose feasible set is not symmetric about the target could behave
differently, and nothing here rules that out.

A6 is MLE on exact tilt samples rather than a DPG implementation with sampled importance weights.
That is deliberate — it removes estimator variance as an explanation — but it also means the
comparison is against DPG's fixed point rather than against DPG as run in practice, where the
proposal is imperfect and convergence is partial. A practitioner's DPG would be no better than this.

The crystal counterpart is not run. It would need the tilt over a space that cannot be enumerated,
so the target itself would be estimated and the comparison would be weaker than this one, not
stronger.
