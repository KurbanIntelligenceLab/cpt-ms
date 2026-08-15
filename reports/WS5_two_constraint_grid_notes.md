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

## Direction, which the partial row does contribute to

Pass-through needs two valence cells in a row, so the single density-5.0 cell adds nothing there. But
cells at a shared valence target across density rows are comparable, each against its own control,
and that comparison is sharper because the achieved valence is dragged by the density constraint on
its own.

| valence target | density target | achieved | control | deviation | asked | correct |
|---|---|---|---|---|---|---|
| 0.282 | 3.0 | 0.2447 | 0.2424 | +0.0023 | up | yes |
| 0.282 | 5.0 | 0.3062 | 0.3141 | -0.0079 | down | yes |
| 0.282 | 8.0 | 0.3251 | 0.3956 | -0.0705 | down | yes |
| 0.484 | 3.0 | 0.2603 | 0.2424 | +0.0179 | up | yes |
| 0.484 | 8.0 | 0.3482 | 0.3956 | -0.0474 | up | no |
| 0.686 | 3.0 | 0.2353 | 0.2424 | -0.0071 | up | no |
| 0.686 | 8.0 | 0.3707 | 0.3956 | -0.0248 | up | no |

At valence target 0.282 the constraint moves the requested way at all three density targets, with the
deviation growing as the density target rises (correlation -0.9614). At the two higher targets, both
of which ask the constraint to raise valence above every control, it moves the wrong way in three of
the four cells: at density 8.0 under both, and at density 3.0 under 0.686. Only 0.484 at density 3.0
succeeds.

By direction: asked to lower valence, correct 2 of 2; asked to raise it, 2 of 5. The budget is a
remaining quantity, so lowering a per-volume density means spending less of it, which is always
available; raising it means spending more than the prefix has left, which is not available once the
budget is exhausted — the largest exclusion class in every cell, 53 to 262 of 518 positions.

Compatibility governs how many positions admit a joint solution; direction governs whether the
constraint can act at those that do. Both reduce to the second constraint spending a budget the first
has already committed.

## Scope

Two complete density rows plus one cell of a third, three valence targets each, one seed per cell,
one serialization. The job reached its time ceiling with the density-5.0 row one cell in; that cell
is used in the direction analysis above, where a single cell is valid, and excluded from pass-through,
where it is not. One seed per cell
means the pass-through figures carry no error bar; the mechanism correlations are across three
targets within a row rather than across seeds, and the opposite signs of the joint-position
correlation are large enough that seed noise is unlikely to explain them, but that is an
expectation rather than a measurement.
