# Pre-registration — steering dial assignment per backbone

Status: draft, not yet committed. Becomes binding at the first steering sweep.

Recorded as an explicit decision rather than inherited from the plan's default, which assigned
filter strictness on charge as the primary dial on both backbones.

## Measured actionable room

Actionability is the fraction of positions where a format-legal substitution still changes the
constraint's final value — the positions where a per-step projection has anything to act on.
Determinability, which is when the constraint becomes computable, is reported alongside because
the two differ by up to a factor of 23 and only the first bounds what a dial can do.

| Backbone | Family (i) charge | Family (ii) density | Density advantage |
|---|---|---|---|
| B1, naive CIF layout | 0.0570 | 0.2831 | 4.96× |
| B1ref, reference ordering | 0.0410 | 0.3849 | 9.38× |
| B2, typed fields | 0.1280 | 0.3896 | 3.04× |

Density has strictly more actionable room on every backbone, by three to nine times. On B1ref
the gap is widest precisely because the reference ordering front-loads the composition: charge
becomes determined at token 13-21 and is immovable thereafter, while density remains movable
across the cell block.

## Decision

**Density (family ii) is the primary steering dial on all three backbones. Charge (family i) is
the secondary dial, reported alongside.**

This reverses the plan's default and departs from the option of assigning charge as primary on
B2. The reasoning is that B2's charge advantage over B1ref is real but small in absolute terms
— 0.1280 against 0.0410 — while density is above 0.28 everywhere. Assigning a different primary
per backbone would also make the cross-backbone comparison confound two variables: the dial and
the serialization. One primary dial across all three keeps that comparison clean.

Charge is not dropped. It is the constraint with the clearest physical interpretation and the
one the ionic-scoping risk concerns, so it is swept as secondary at every steering strength. The
comparison between a dial with wide actionable room and one with narrow room is itself a result:
it predicts how the method behaves on constraints that bind late in a sequence.

## Consequence for the ionic-scoping hedge

The plan introduced family (ii) as a hedge against family (i) proving unusable. The hedge is now
load-bearing rather than decorative: family (ii) is where the headline experiment runs. That is
a good outcome — the hedge exists because the plan anticipated exactly this, and it was
instantiated per step on both backbones before it was needed.

## Threshold consequence

CP0's active-multiplier fraction is computed over positions that are both actionable and
feasible, per `CP0_actionable_window.md` and `CP0_fallback_policy.md`. With density as primary
the denominator is roughly 0.29-0.39 of positions rather than 0.04-0.13, so the gate has more
signal to read. Charge results, being confined to a narrow window, are the case where a working
layer most easily reads as collapsed, and the pre-registered scoping matters most there.

## Defect that produced this decision

The first measurement reported B2's two families as identical at 0.1280, which is structurally
impossible: density depends on lattice tokens as well as composition, so it must have strictly
more room. The B2 fixture was emitting placeholder element symbols (`el:E0:1`), which carry no
charge and no mass — so the charge statistic was a meaningless zero and density was zero
regardless of the cell, making every lattice token read as non-actionable.

This is the second occurrence of the same defect class; the naive B1 fixture had the same
problem. Two tests now cover it: one asserting every fixture emits elements with defined charge
and mass and a non-zero density, and one asserting density has strictly more actionable room
than charge on every backbone, since equality is the signature of the defect. Both fail on the
reverted code.

Without the fix the recommendation would have been charge as primary on B2, based on a tie that
was an artifact.
