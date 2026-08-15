# WS-5: two constraints compose only when their targets are compatible

`results/WS5_two_constraint_grid.json`

Two complete rows of the grid, and they disagree in a way that narrows the claim rather than
confirming or refuting it.

## The two rows

| density target | valence target | achieved valence | fallback rate | joint positions | unique formulas |
|---|---|---|---|---|---|
| 3.0 | 0.282 | 0.2447 | 0.102 | 230 | — |
| 3.0 | 0.484 | 0.2603 | 0.518 | 136 | — |
| 3.0 | 0.686 | 0.2353 | 0.826 | 49 | — |
| 8.0 | 0.282 | 0.3251 | 0.134 | 265 | — |
| 8.0 | 0.484 | 0.3482 | 0.228 | 329 | — |
| 8.0 | 0.686 | 0.3707 | 0.314 | 319 | — |

Structural validity is exactly 1.0000 in all six cells, as in every other arm.

## What the second row changes

The published claim rested on the density-3.0 row: pass-through of 6.2%, a non-monotone
response with the largest request producing the smallest achieved value, and a deviation from the
unconstrained control of 0.0179 — 8.9% of one data standard deviation. Read alone it says the
second dial does not move.

The density-8.0 row was run because a high density target and a high valence target pull the same
way, so their token budgets should compete less. They do. Pass-through is 11.3%, the response is
monotone, and the deviation from control is 0.0705 — 34.9% of one standard deviation, 3.9 times
the compatible-target-free row.

The mechanism separates the two rows cleanly, and it is the mechanism rather than the pass-through
that carries the finding. Fallback to the single-constraint solve rises with the second target in
both rows, correlating at 0.9963 and 0.9997. But jointly solvable positions move in opposite
directions: they fall with the target at density 3.0 (correlation -0.9998, from 230 to
49 of 518) and rise at density 8.0 (correlation +0.7842, from 265 to 319). Budget-exhausted
exclusions fall from 212 to 53 across the 8.0 row as the two demands align.

## The claim that survives

Not "two additive constraints on a shared token budget do not compose" — that was the one-row
reading and the second row contradicts it. And not "they compose", because 11.3% pass-through is
not a controllable dial: requesting a 0.404 change in valence electron density yields 0.0457.

What holds is compatibility-dependence. Two constraints reading the same tokens through correlated
weight tables compete for one budget, and how much they compete depends on whether their targets
pull the same way. Where they conflict the second dial is inert and non-monotone; where they agree
it moves monotonically and about 4 times as far, while remaining far short of control. Composition
is neither free nor impossible: it degrades with target conflict, and the degradation is visible in
the fraction of positions at which a joint solution exists at all.

Density steering is unaffected in both rows, and structural validity stays exact throughout, so
nothing about the second constraint disturbs the first or the support restriction.

## Scope

Two of three density rows, three valence targets each, one seed per cell, one serialization. The
density-5.0 row is running and will add a third point between the two extremes. One seed per cell
means the pass-through figures carry no error bar; the mechanism correlations are across three
targets within a row rather than across seeds, and the opposite signs of the joint-position
correlation are large enough that seed noise is unlikely to explain them, but that is an
expectation rather than a measurement.
