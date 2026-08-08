# WS-1 — Steering frontier: what per-step projection buys and what it costs

Standing note, rebuilt on change. Feeds CP0, CP4, CP5. Every number is read from
`results/WS1_steering_frontier.json`, emitted by `scripts/synthetic/steering_frontier.py`.
Figure: `figures/steering_frontier.png`.

## Summary

The positive capability claim, measured across the whole steering range rather than at one
flattering target. Per-step projection reaches targets that rejection sampling reaches only at
7.1× to 362.2× the sampling cost, and it concentrates mass on the exact-target set by 2.6× to
183.3× over the base model. The price is 1.070 ± 0.160 nats of excess divergence over the
optimal distribution reaching the same place.

The result that makes this a paper rather than a benchmark: **that excess is approximately a
fixed overhead, while the divergence the optimum itself must pay spans 129× across the sweep.**
So the wasted fraction of A3's divergence collapses from 97.8% at the weakest steering to 32.4%
at the strongest, monotonically on both sides of the base mean. Per-step projection is almost
all overhead where the base model already satisfies the constraint, and almost all necessary
work where it does not. That is the reframe's thesis as a measured quantity, and it is the
argument for siting the headline experiment away from matched alignment.

## The three legs

| Target | Signed strength | Rejection draws/sample | Enrichment | KL induced | KL optimal | Excess | Wasted | Mass on target |
|---|---|---|---|---|---|---|---|---|
| −6 | −8.52 | 362.2 | 183.30 | 4.4085 | 2.9781 | 1.4304 | 32.4% | 50.61% |
| −4 | −6.52 | 48.9 | 19.32 | 2.8130 | 1.7499 | 1.0631 | 37.8% | 39.53% |
| −2 | −4.52 | 27.7 | 11.29 | 1.9819 | 0.9087 | 1.0731 | 54.1% | 40.74% |
| 0 | −2.52 | 11.2 | 4.26 | 1.3601 | 0.3088 | 1.0513 | 77.3% | 38.00% |
| +2 | −0.52 | 9.1 | 3.75 | 1.0629 | 0.0230 | 1.0399 | 97.8% | 41.25% |
| +4 | +1.48 | 7.1 | 2.56 | 0.9982 | 0.0688 | 0.9294 | 93.1% | 36.17% |
| +6 | +3.48 | 9.6 | 5.30 | 1.5004 | 0.5988 | 0.9016 | 60.1% | 54.93% |

**(a) Efficiency.** Rejection sampling from the base, accepting only exact-target sequences, is
the exactness baseline: it is correct by construction and costs `1/acceptance` draws per sample.
A3 pays a fixed per-step cost instead. Efficiency and enrichment both rise steeply with
steering strength, which is why a single ratio would be meaningless — it can be made
arbitrarily large by choosing a more distant target.

**(b) Divergence cost.** Reported against the tilt at the mean the induced distribution
actually achieves, so both distributions satisfy the same expectation constraint and the
difference is a genuine efficiency cost. Non-negative at every target, as theory requires.

**(c) Quality at target.** Mass landing exactly on the target statistic ranges from 36.17% to
54.93%. Rejection sampling scores 100% by construction — that is the honest cost of an
expectation constraint against a hard one, and it belongs in the trade rather than in a
footnote.

## Defect found and fixed: the comparator was invalid at the extremes

The first implementation compared the induced distribution against the tilt **at the requested
target**. At extreme targets this produced *negative* excess divergence — the surrogate
appearing to beat a provable optimum, at −0.585 nats in the case that surfaced.

The cause was not a solver error. Where the running budget goes badly infeasible the induced
distribution does not reach the requested target at all: at charge target −6 it achieves −4.96,
and at −4 on a length-6 task it achieves −2.63 while the tilt sits exactly at −4. The two were
not solving the same problem, so their divergence difference was not a cost. A distribution
satisfying a weaker constraint can of course sit closer to the base.

Two changes. The at-target comparison is now gated on the induced distribution actually meeting
the constraint (absolute mean error ≤ 0.5) and reports `None` with an explanatory note
otherwise, rather than a number that invites the wrong reading. And a second comparator was
added that is valid everywhere: the tilt at the mean the induced distribution achieves. The
first covers 4 of 7 targets; the second covers all 7, including the strong-steering end where
the capability claim lives. All divergence-cost statistics use the second.

The regression test exercises the extremes of the attainable range specifically. An earlier
version of it trimmed targets to |c| ≤ 4 and passed on the defective code — the defect only
triggers at ±5 on a length-5 task — so the fixture now always includes the range endpoints.
Confirmed load-bearing: it fails at −0.585 on the reverted implementation.

## Consequence for the fallback policy

The infeasibility that broke the comparator is the same phenomenon the fallback policy governs,
and both candidate policies are now implemented and measured. See
`preregistration/CP0_fallback_policy.md`: falling back to the base conditional leaves +0.1028
of constraint error at 1.0254 nats of excess; clamping to the nearest attainable increment cuts
the error 31× to +0.0033 and pays 0.2349 more nats, halving charge variance again. `base` is
pre-registered as primary with `clamp` reported alongside.

## Verification

Seven tests in `tests/test_steering_frontier.py`: the rejection baseline as an exact
reciprocal rather than an estimate; enrichment strictly above 1 at every reachable target,
without which the steering framing would be invalid; non-negativity of excess divergence under
both comparators; that the achieved comparator genuinely extends coverage beyond the at-target
one rather than inheriting its blind spot; per-side monotonicity of the waste fraction, checked
separately either side of the base mean because the attainable range is not centred there; the
fallback trade in both directions; and rejection of an unrecognized policy name, since a
pre-registration bypassable by a typo is not one.
