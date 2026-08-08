# Response — CP4 decision implemented, one correction to the artifact fix

## Implemented as directed

CP4's violation-rate criterion is now pre-registered in
`preregistration/CP4_violation_rate_MDE.md`: reported not gating, with distribution
fidelity carrying the gate, and an MDE plus sample size registered alongside. A2's MP-20
violation rate is recorded as a measured quantity with confidence interval per filter
setting, not estimated from the 91.13% ground-truth figure. Option 3 is dropped — the
objection is correct and was a real error in the framing: A2 has no Regime-E layer, so
multiplier activity is not a between-arm comparison, and the KL quantity is distribution
fidelity under another name.

## The artifact fix: right conclusion, different cause

Row 1 does reproduce under a standard two-proportion calculation. Checked in both pooled
and unpooled forms, those rates give n = 67.46 at 80% power and alpha 0.05.

The actual defect is rounding direction. A requirement of 67.46 samples per arm means 68,
not 67 — sample sizes round up. The same error appeared in row 3 (2745.28 reported as
2,745, corrected to 2,746). The instinct that row 1 was wrong was right; the cause was the
floor, not the test.

The precision issue underneath turned out sharper than either of us stated. Sample size
scales as the inverse square of the rate difference, so rounding the rates destroys the n
column when the difference is small. At four decimals the bottom two rows do not reproduce
at all — 28,699 and 10,649,829 against the correct 28,350 and 10,165,806. Even ten
decimals leaves the bottom row off by one. The table now carries twelve decimals and
states its test, tails, alpha, power, and unpaired status.

## Effect-size correction verified against the source

Every figure cited from Mastej et al. (arXiv 2607.17910) was checked against the paper
directly and is verbatim: retention falling to 41.8–54.9% against 59.7% for MP-20 at
theta = 0.5; charge balance and redox ordering "broadly captured" under weaker filters with
differences emerging as the prior tightens; 10,000 compositions generated per model across
six models. The argument that effect size is set by model behavior rather than data
alignment is correct, and the sweep's data-alignment axis does not substitute for it.

Working the published rates through the power calculation, 50,000 per arm has more margin
than the target required:

| Filter | MP-20 violation | Model violation range | Largest gap | MDE at n = 50,000 |
|---|---|---|---|---|
| q (charge neutrality) | 3.5% | 2.7–6.9% | 3.4 pp | 0.44 pp |
| q + chi | 5.4% | 4.4–9.2% | 3.8 pp | 0.51 pp |
| C = 3 | 6.8% | 6.9–13.2% | 6.4 pp | 0.59 pp |
| theta = 0.05 | 21.9% | 23.8–40.7% | 18.8 pp | 0.87 pp |
| theta = 0.10 | 26.4% | 28.5–46.6% | 20.2 pp | 0.88 pp |
| theta = 0.50 | 40.3% | 45.1–58.2% | 17.9 pp | 0.88 pp |

MDE is computed at each filter's worst-performing model, which is the conservative choice.
Even a quarter of the observed gap stays detectable at 13,851 per arm in the tightest case
(q + chi). No row requires the full 50,000, so the budget is not the binding constraint.

## Filter configuration, pre-registered

Primary reported setting is **q + chi**. The source study identifies charge neutrality and
the Pauling electronegativity rule as necessary validity criteria, while the consensus and
commonality filters are empirical priors on oxidation-state rarity rather than validity
boundaries. Secondary settings are **q** alone and **theta = 0.10**. Violation rate is
reported at all three, and no claim of chemical invalidity is made from the commonality
filters — lower retention there reflects rare but not forbidden chemistry.

The point about cross-paper comparability is now recorded in the pre-registration: MP-20's
reported compositional validity ranges from 91.13% under the CDVAE-lineage SMACT check to
96.5% under charge neutrality alone to 59.7% at theta = 0.5, so a violation rate quoted
without its filter configuration is not interpretable.

## Next

Tranche item 2 as directed: the WS-2 parser and mask emitter, MP-20 ionic-subset coverage,
the per-step incremental form of Regime-E family (ii) written out for both backbones, and
A2's sampled violation rate per filter setting. If no sound per-step form exists on B1,
that will be reported as a scoping of family (ii) to B2 rather than worked around.

A3's induced sequence distribution remains outstanding and is queued behind the projection
layer, which it requires.

Version control still needs to land in the repository before CP1 convenes; artifacts
currently record UNVERSIONED by design. The ground-truth and sweep scripts are re-run
immediately afterward so every artifact carries a real commit hash.
