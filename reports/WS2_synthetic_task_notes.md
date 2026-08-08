# WS-2 — Synthetic task: definition and exact targets

Standing note, rebuilt on change. Feeds CP1. All numbers below are read from
`results/WS2_synthetic_ground_truth.json`, emitted by
`scripts/synthetic/compute_ground_truth.py`.

## Summary

The synthetic constrained-sequence task is defined and both candidate targets are
computed exactly by enumeration over all 65,536 sequences. Computing them surfaced a
distinction the research plan's two-target framing does not name: the Regime-E
sequence-level ideal is an exponential *tilt*, not a *conditioning*, and the two are far
apart — total variation 0.875 on this task. The exact expectation-constrained optimum
still places 87.55% of its mass on individually charge-violating sequences, which makes
the plan's "satisfaction in expectation only" caveat a measured quantity rather than a
stated limitation. A third target definition is therefore required before any distortion
result is reported. No gate outcome: this is pre-checkpoint work feeding CP1.

## Task definition

Vocabulary of four symbols with integer charges (A: +2, B: +1, C: -1, D: -2), sequences
of length 8, giving 4^8 = 65,536 enumerable sequences.

Regime S is a per-symbol multiplicity budget (at most 3 occurrences of any symbol). It
is prefix-closed by construction — a spent budget cannot be recovered — which is the
property that makes the mask emitter well-defined and A2 trainable without
straight-through estimation. 55.11% of sequences are S-feasible, carrying 36.82% of the
data distribution's mass.

The data distribution is a seeded first-order Markov chain with a multiplicative charge
tilt, deliberately biased so it does **not** already satisfy the charge constraint: its
mean total charge is +5.181 against a target of 0. This matters. If the data satisfied
the constraint, every target would collapse onto the data distribution and the
multipliers would be zero from initialization — the degenerate case. Being able to turn
that bias on and off is what makes this task a usable probe of multiplier decay, which
the plan's "Abstract register" section expects to be the modal outcome on MP-20.

## The three targets

The plan specifies two candidate targets: the data distribution restricted to the
feasible set, and the base model's distribution conditioned on the constraints (GAD's
target). Both are conditioning operations. But Regime E is an *expectation* constraint,
and the KL-optimal distribution satisfying an expectation constraint is an exponential
tilt, not a conditioning:

- **Conditioning**: q(x) proportional to p(x) * 1[phi(x) = c]. Hard constraint. Every
  sample satisfies it.
- **Tilting**: q(x) proportional to p(x) * exp(lambda * phi(x)), with lambda solved so
  that E_q[phi] = c. This is GDC's EBM target and what A6 fine-tunes toward.

Measured on this task, with Regime-S masks applied and a charge target of 0:

| Distribution | Mean charge | Var | Per-sample violation rate | Entropy (nats) | Support |
|---|---|---|---|---|---|
| Data | 5.181 | 17.99 | 0.9554 | 9.954 | 65,536 |
| S-conditioned | 2.516 | 8.56 | 0.9109 | 9.641 | 36,120 |
| S+E hard-conditioned | 0.000 | 0.00 | 0.0000 | 7.837 | 4,760 |
| S+E tilted (lambda = -0.2575) | 0.000 | 10.45 | 0.8755 | 9.896 | 36,120 |

Both constrained targets hit the mean exactly. They are otherwise nothing alike:
total variation between them is 0.875, and KL(tilt \|\| hard) is infinite because the
tilt has support on 36,120 sequences while the hard target has support on 4,760.

## Why this matters for the project

**The tilt is the honest Regime-E target, and it violates.** The exact, KL-optimal,
sequence-level solution to the expectation constraint puts 87.55% of its mass on
sequences that individually violate the charge constraint, with variance 10.45 around a
mean of exactly zero. The plan's statement that bounding expected increments bounds the
expected total but not its variance is correct, and this is what it costs. Any A3 result
compared against the hard-conditioned target would be measured against a distribution
that Regime E does not and cannot target.

**Three targets, not two, must be reported.** The plan's requirement that every
distortion result be reported against both defined targets should be amended: the
feasible-restricted data distribution and the constraint-conditioned base model are the
right pair for Regime S, and the tilted target is the correct comparator for Regime E and
for the A3-vs-A6 comparison. Reporting A3 against a conditioning target would penalize it
for failing to do something it does not claim to do.

**The A3-vs-A6 comparison is non-vacuous.** A6 approximates the tilted target by
sequence-level fine-tuning; A3 approximates it by a per-step proxy. Since tilt and
conditioning are 0.875 apart in total variation on this task, the comparison has room to
resolve a real difference rather than measuring two routes to the same place.

## Two structural lemmas (added after team review)

Both properties below were raised as candidate theorems rather than artifacts of the
seed. Both were re-derived on random base distributions and across constraint levels, and
both hold. They are now proved in the docstring of `tests/test_tilt_lemmas.py` and
verified by nine tests there.

**Lemma 1 (mass-displacement identity).** TV(tilt, hard) equals the tilt's mass outside
the hard-feasible set, exactly. On the feasible set F the statistic phi is constant, so
the tilt factor is constant there; the tilt's normalizer also collects mass off F, so the
tilt is pointwise dominated by the hard-conditioned distribution on F. Hard puts no mass
off F, so all the total variation is carried by the tilt's off-F mass. The 0.875 and the
87.55% reported above are therefore the same number by construction, not a coincidence.

The proof nowhere requires a target of zero — phi is constant on F for any target — so
the lemma is stated and tested at non-zero targets too. Verified at charge targets -4, -2,
0, +2, +4, where TV and violating mass agree to 1e-12 at every level.

**Lemma 2 (I-projection ordering).** KL(tilt ‖ base) is at most KL(hard ‖ base). Both
distributions satisfy the moment constraint, so both lie in the same constrained set; the
tilt is the I-projection of the base onto that set and the hard-conditioned distribution
is a feasible but suboptimal member. Verified on 25 random base distributions and at every
tested charge target.

**A note on reading the reported KL values.** Absolute KLs depend on which base they are
measured against; the *gap* between the two targets does not. Measured against the
S-conditioned base the values are 0.335 and 2.418 nats; measured against the raw data
distribution they are 1.334 and 3.417. Both are correct, they differ by exactly
-log(0.3682) = 0.999 nats — the log-reciprocal of the data mass on the S-feasible set —
and the separation is 2.083 nats either way. Reports should name the reference base
alongside any absolute KL. A test pins this invariance.

## Verification

Eight tests in `tests/test_synthetic_targets.py`, each checking a property by a route
independent of the implementation: enumerated probabilities against 400,000 forward
samples (agreement within 0.02 total variation); the prefix mask against brute-force
feasibility on 200 random prefixes; the solved multiplier against its defining moment
condition to 1e-9; KL-optimality of the tilt against 50 random constraint-preserving
perturbations; and the separation between the two targets. All pass.

## Open item

`git init` could not be run from the analysis sandbox (the directory sits under a
coarse-mode git protection grant). The artifact provenance field currently records
`UNVERSIONED`, which is the designed behavior — a missing commit is recorded as such
rather than fabricated. Running `git init && git add -A && git commit` in the repository
makes every subsequent artifact carry a real commit hash. Nothing else depends on it.
