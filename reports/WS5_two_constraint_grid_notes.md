# WS-5: two constraints do not compose when they share a token budget

`results/WS5_two_constraint_grid.json`

A7 trains through a joint projection of mass density and valence electron density. The
question is whether the second dial steers. It does not, and the mechanism is measured
rather than inferred: the two quantities compete for the same token budget, so the harder
the second target, the more often no distribution satisfies both and the projection
reverts to density alone.

## What ran

Three single-constraint A3 controls at density targets 3.0, 5.0 and 8.0, then the A7 grid
at density 3.0 crossed with valence targets 0.282, 0.484 and 0.686 — the data mean and plus or
minus one standard deviation (0.4839 ± 0.2019 e/A^3). Each run is 6,000 steps, seed 0,
on one RTX 5090.

The density-3.0 row is complete. The density 5.0 and 8.0 rows are not: each A7 cell costs
about two hours against A3's forty minutes, and the job reached its nine-hour ceiling with
the fourth A7 cell in progress. Its partial output was discarded rather than reported.
Every number below therefore describes one complete row, and the conclusion is stated for
that row alone.

## The second dial does not respond

| valence target | achieved valence | density achieved | joint positions | jointly infeasible | fallback rate | validity |
|---|---|---|---|---|---|---|
| 0.282 | 0.2447 | 3.281 | 230 | 26 | 0.102 | 1.0000 |
| 0.484 | 0.2603 | 3.041 | 136 | 146 | 0.518 | 1.0000 |
| 0.686 | 0.2353 | 3.071 | 49 | 233 | 0.826 | 1.0000 |
| none (A3 control) | 0.2424 | 3.175 | — | — | — | 1.0000 |

Requested valence range 0.404; achieved range 0.0250. Pass-through is 6.2%.

The response is **not monotone**: 0.2447, 0.2603, 0.2353 for increasing targets. The largest
target produced the smallest achieved value. Measured against the control, which carries no
valence constraint at all, the biggest deviation any cell achieves is 0.0179, or 8.9% of
one data standard deviation. A non-monotone response of that size is not weak control; it
is the absence of control.

## The mechanism: two dials, one budget

Joint feasibility collapses as the second target moves away from what density already
implies. Fallback rate rises 0.102 → 0.518 → 0.826 with correlation 0.9963 against the
valence target, while joint positions used fall 230 → 136 → 49, a 79% drop, at
correlation -0.9998.

The cause is visible in the position accounting. Of exactly 518 actionable positions in every
cell, 236 to 262 were excluded because one running budget had already reached zero
while the other was still positive. Both quantities are additive functions of element
identity and count, correlated at 0.725 across the training split, so the atoms that
satisfy the density budget are the same atoms the valence budget needs. Satisfying one
consumes the other's remaining freedom. At the hardest corner — low mass density with high
valence electron density, which asks for fewer or lighter atoms per volume but more valence
electrons per volume — 82.6% of eligible positions are jointly infeasible.

This is the "individually attainable only" regime CP-7 mapped on synthetic grids, now
observed on real crystal prefixes as the binding constraint on composition.

## What is unaffected

Density steering does not degrade. Achieved densities are 3.281, 3.041 and 3.071 against
target 3.0, bracketing the control's 3.175. Structural validity is exactly 1.0000 in
every A7 cell and every control, so Regime S is untouched by adding a second Regime E
constraint — as it must be, since the mask is enforced by construction.

At the one density target present in both arms, A7's mean density error is 0.131 against
the control's 0.175. That difference is not a claim: it rests on one target with one seed
each, and it is smaller than the seed spread the frontier work measured. It is reported to
show the second constraint did not cost density accuracy, not to suggest it helped. The
controls at density 5.0 and 8.0 are excluded from that comparison because A7 has no
matching cells; including them would credit A7 with the absence of the 8.0 cell, whose
control error of 1.263 reflects the reachable ceiling rather than anything about
constraint count.

## What this supports as a claim

The layer solves several expectation constraints simultaneously and exactly — CP-7
verified that at solver precision on real prefixes, and it remains true. What does not
follow is that several constraints can be STEERED simultaneously. Exact joint projection at
the positions where a joint solution exists is compatible with no control over the second
quantity, because whether a joint solution exists is itself determined by the constraints
and the prefix, not by the solver.

For two additive per-element quantities on this serialization the honest statement is that
they do not compose: correlation 0.725 makes them near-substitutes for the same token
budget, and the budget is exhausted on one dial before the other can be moved. Constraint
composition is bounded by how independently the constrained quantities can be varied by the
tokens the model actually emits, and that is a property of the serialization and the
constraint pair, not of the projection.

## What would change the conclusion

Two rows of the grid are missing, and the density 8.0 row is the one that could soften
this. A high density target and a high valence target pull in the same direction, so their
budgets should compete less and pass-through should be higher. If it is, the claim narrows
from "these two quantities do not compose" to "they compose only when their targets are
mutually compatible", which is a more useful statement. Those six cells are the immediate
next run.

A constraint pair chosen for independence rather than availability would be the stronger
test. Nothing here shows that expectation constraints in general fail to compose; it shows
that two constraints reading the same tokens through correlated weight tables do not.
