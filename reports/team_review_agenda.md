# What to bring to the team

One decision needs their input. Everything else is a report.

## The decision: CP4's violation-rate criterion has no effect size

CP4 currently reads, in part: "A3 beats A2 on distribution fidelity and Regime-E violation
rate." That names a direction but no magnitude, and the synthetic sweep shows the
magnitude is where the difficulty lives.

The paired difference in per-sample violation rate between a constrained and an
unconstrained distribution does track constraint work — correlation 0.94 against KL — so
the criterion is measuring something real. But the difference collapses as the training
data comes into alignment with the constraint:

| Absolute mean gap | Violation-rate difference | Samples per arm for 80% power |
|---|---|---|
| 5.66 | 11.73 pp | 67 |
| 2.03 | 2.38 pp | 2,745 |
| 1.13 | 0.76 pp | 28,350 |
| 0.26 | 0.04 pp | 10,165,804 |

MP-20 is a well-aligned dataset: 91.13% of its ground-truth compositions already pass the
SMACT charge-balance check. Its alignment is better than any point in this sweep, so the
expected effect sits at or below the bottom row's regime.

The consequence is that a null result at CP4 is currently uninterpretable — it cannot
distinguish "the projection does nothing" from "the comparison had no power." Since CP4 is
a gate that can trigger the pivot rule, that ambiguity is expensive.

**What the team decides.** Three options, and this is a judgment call about what the paper
claims rather than something the synthetic task can settle:

1. Pre-register a minimum detectable effect on violation rate with a sample-size
   calculation, and size the CP4 sampling run to it. Honest, and possibly expensive.
2. Demote violation rate to a reported-but-not-gating metric at CP4, and let distribution
   fidelity carry the gate. Fidelity metrics do not suffer the same collapse.
3. Gate on the multiplier and KL quantities instead, which track constraint activity
   directly and do not lose resolution with alignment.

My recommendation is 2 with 1 as a reported companion: keep violation rate visible because
the plan promises it is "always reported," but do not make a gate depend on a quantity
whose effect size vanishes on the flagship dataset.

**What they should not need to decide.** The direct MP-20 effect size. That requires the
crystal parser and MP-20's own base rates, and it is a WS-2 deliverable, not a meeting
question. Bring the shape of the problem, not a request for a number nobody has yet.

## Report, no input needed

**Both structural claims from their review are confirmed and one generalizes.** The
mass-displacement identity holds at every charge target tested, not only at zero — the
proof needs the statistic to be constant on the feasible set, not zero. The I-projection
ordering holds on 25 random base distributions. Both are now proved in the test docstring
with nine tests behind them.

**Their 1.334/3.417 and my 0.335/2.418 are the same result.** Theirs are measured against
the raw data distribution, mine against the S-conditioned base. The offset is exactly
-log(0.3682) = 0.999 nats, the log-reciprocal of the data mass on the S-feasible set, and
the separation is 2.083 nats either way. A test pins the invariance. Reports should name
the reference base alongside any absolute KL.

**The multiplier and KL scaling exponents come out at 1.026 and 2.038.** Theory predicts
exactly 1 and 2 for an exponential family, so this doubles as a correctness check on the
solver. Constraint activity vanishes quadratically as alignment improves.

**Their three-target amendment is implemented and the estimator caveat is noted.** The
tilt is exactly computable only because the task enumerates; on MP-20 the normalizer is
intractable, which is precisely why GDC needs policy gradient. Exact three-target
reporting is a synthetic-only capability, as their amendment states.

## Corrections to my own prior claims

Two, both caught before this agenda was written.

I reported that per-sample violation rate "tracks neither" the multiplier nor the KL. That
conflated two things. The constrained distribution's *absolute* violation rate is indeed
nearly invariant, moving 0.31 percentage points across a 933-fold change in KL. But the
*paired difference* against the unconstrained base tracks constraint work closely. The
first fact is a caution about reading absolute rates; the second is what makes the CP4
criterion valid in principle and underpowered in practice. The earlier phrasing would have
justified dropping a metric that is in fact informative.

I also flagged a spurious non-monotonicity of 4e-4 in the multiplier by testing magnitudes
rather than signed values. The multiplier is strictly monotone in the signed gap and
strictly opposes it; the attainable charge distribution is skewed, which makes the
magnitude comparison meaningless. Both the check and the artifact are corrected.

## Open item carried forward

Version control still needs to land in the repository before CP1 convenes, and the
ground-truth and sweep scripts re-run afterward so the artifacts carry real commit hashes.
Artifacts currently record UNVERSIONED, which is the designed behavior rather than a
fabricated hash, but the plan requires pre-init artifacts to be re-emitted under version
control before a gate consumes them.
