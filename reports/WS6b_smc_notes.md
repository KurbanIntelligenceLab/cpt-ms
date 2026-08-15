# WS-6b: sequential Monte Carlo steering, and what it says about inference-time control

`results/WS6_smc_steering.json`

## Why this baseline

Best-of-n is the cheapest amortized alternative and it beats the projection on accuracy. SMC is the
principled version of the same idea: instead of drawing n complete sequences and discarding all but
one, carry K partial sequences and reallocate them as information arrives, so compute goes where it
changes the outcome. If the projection's advantage were only that naive selection wastes samples,
SMC would erase it.

Implementation. The potential is a Gaussian on each particle's expected final density against the
target, evaluated at element positions — the same positions where the projection is actionable, so
the two methods are compared on the same decision points. The density estimate reuses the published
decoder's parsing and volume arithmetic, verified bit-identical on complete sequences, so an SMC
particle is scored on exactly the quantity the projection targets. Resampling is multinomial,
triggered when effective sample size falls below K/2. All runs deploy the same checkpoint
(A3_d5.0.pt, trained at density 5.0).

## Results, fixed potential width

| target | projection | SMC K=1 | K=4 | K=16 | K=64 | best-of-64 |
|---|---|---|---|---|---|---|
| 1.5 | 0.348 | 10.161 | 3.110 | 0.901 | 0.536 | 0.484 |
| 2.0 | 0.266 | 9.661 | 2.732 | 0.917 | 0.543 | 0.290 |
| 3.0 | 0.240 | 8.661 | 1.424 | 0.344 | 0.525 | 0.138 |
| 5.0 | 0.151 | 6.679 | 0.838 | 0.640 | 0.430 | 0.073 |
| 6.317 | 0.719 | 5.785 | 1.484 | 0.675 | 0.493 | 0.062 |
| 8.0 | 1.240 | 5.081 | 1.248 | 0.876 | 0.488 | 0.097 |
| 10.0 | 1.670 | 4.712 | 1.812 | 1.112 | 0.733 | 0.130 |
| 13.0 | 4.256 | 5.173 | 1.960 | 1.320 | 1.139 | 0.177 |
| 16.0 | 6.125 | 7.024 | 4.803 | 1.086 | 1.317 | 0.448 |
| 20.0 | 9.595 | 9.690 | 5.714 | 2.224 | 1.347 | 0.950 |

SMC reaches the projection at 6 of 10 targets within K = 64 — the high targets, where the
projection saturates against its ceiling and almost anything beats it. Below the data mean the
projection is better at every K tested.

Against best-of-n at equal compute, SMC is worse at all ten targets, by a mean factor of
4.2 at K = 64. Structural validity is 1.0 in every run at every K, since the mask is
active throughout.

## Why that comparison is reported as a bound, not a result

The obvious objection is tuning, and it is correct: the potential's width is a free parameter and the
first pass fixed it at 1.0 g/cm^3. A probe at one target suggested 0.3 was much better — 0.259
against 0.640 at K = 16, a factor of 2.5 — which would have made the fixed-width result a claim
about this implementation rather than about SMC.

Sweeping both widths across all ten targets at K = 16 refuted the probe. Width 0.3 wins at 4 of
10 targets and loses at the other 6, and its mean error over targets is worse
(1.801 against 0.944) because one target degrades badly (9.222 at target 16.0).
There is no single good width, and the best width varies with the target.

The best-per-target mean is 0.855, better than either fixed width. It is not reported as SMC's
performance, because selecting the width per target from the same runs that report the error is
selection on the evaluation set. The fixed-width rows are the honest comparison; the sweep is here to
show tuning sensitivity, not to produce a better number.

What remains untuned: the proposal distribution, the resampling threshold and its trigger points, and
per-target width chosen on held-out draws. Any of these could improve the numbers, so the figures
above are a lower bound on SMC rather than its ceiling. A tuned SMC beating the projection on
accuracy would not change the paper's claim, which is about cost structure rather than accuracy.

## What this adds to the argument

The inference-time family — best-of-n and SMC — occupies one axis: accuracy bought with compute per
output, at every target, indefinitely. The projection occupies another: accuracy bought once at
training, one forward pass thereafter. SMC is the stronger member of that family in principle, and
this untuned implementation does not realize it, which is a fact about the implementation. What does
not depend on tuning is the cost structure: at K = 64 SMC spends 64 forward passes per accepted
output forever, and the diversity cost that selection imposes on best-of-n applies to any method
whose mechanism is reweighting toward a target.
