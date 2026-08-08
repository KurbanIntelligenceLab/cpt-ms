# Pre-registration — CP4 Regime-E violation rate: role, MDE, and sample size

Status: draft, not yet committed. Becomes binding when the CP4 sampling script first runs.

## Role of the metric

Regime-E per-sample violation rate is **reported, not gating**, at CP4. Distribution
fidelity carries the gate. This resolves the ambiguity that an unpowered violation-rate
criterion would have created: a null result could not have distinguished an inactive
projection from an underpowered comparison.

The metric is still pre-registered with a minimum detectable effect and a sample size, so
that a null is interpretable rather than merely unclaimed. Every reported violation-rate
comparison carries the observed difference, a 95% confidence interval, and the MDE
achieved at the realized n.

## Test and sample size

**Test.** Two-proportion comparison, pooled-variance form, two-sided alpha = 0.05, power
0.80:

    n per arm = [ z_(1-alpha/2) * sqrt(2 * p_bar * (1 - p_bar))
                  + z_(1-beta) * sqrt(p1(1-p1) + p2(1-p2)) ]^2 / (p1 - p2)^2

with p_bar = (p1 + p2)/2. Arms are independently sampled generations from two trained
models, so the comparison is unpaired and McNemar does not apply. If a future comparison
uses matched conditioning (same composition prompt to both arms), it must switch to
McNemar and say so at the point of reporting.

**Sample size.** 50,000 generated structures per arm. Precedent for this scale: the
chemical-filter study of Mastej et al. (arXiv 2607.17910) generated 10,000 compositions
per model across six generative models under SMACT screening. Violation rate costs
sampling plus a SMACT call, not DFT, so 50,000 is a routine budget rather than a
constraint.

## Expected effect size

The effect size is set by the **model's** violation rate under A2, not by MP-20's
data-side rate. Published retention figures for six generative models against the MP-20
baseline (Mastej et al., Table 1; retention is the complement of violation) give the scale
of the model-versus-data gap at each filter strictness:

| Filter | MP-20 violation | Model violation range | Largest model-vs-data gap | MDE at n = 50,000 |
|---|---|---|---|---|
| q (charge neutrality) | 3.5% | 2.7–6.9% | 3.4 pp | 0.44 pp |
| q + chi | 5.4% | 4.4–9.2% | 3.8 pp | 0.51 pp |
| C = 3 | 6.8% | 6.9–13.2% | 6.4 pp | 0.59 pp |
| theta = 0.05 | 21.9% | 23.8–40.7% | 18.8 pp | 0.87 pp |
| theta = 0.10 | 26.4% | 28.5–46.6% | 20.2 pp | 0.88 pp |
| theta = 0.50 | 40.3% | 45.1–58.2% | 17.9 pp | 0.88 pp |

MDE is computed at the worst-performing model's base rate at each filter, which is the
conservative choice. At every setting the MDE sits below 0.9 pp against gaps of 3.4 to
20.2 pp, so 50,000 per arm has ample margin. Even a quarter of the observed gap remains
detectable at 13,851 per arm in the tightest case (q + chi).

The synthetic alignment sweep, which varies **data**-constraint alignment, is not a
substitute for these numbers. It establishes that the effect shrinks with alignment and
that a power calculation is therefore required; the crystal-side effect size is set by
model behavior and is measured, not inferred.

## Filter configuration is pre-registered

Effect size is a function of filter strictness, which is a choice rather than a fixed
property of the dataset. The reported configuration is fixed here:

- Primary reported setting: **q + chi** (charge neutrality plus the Pauling
  electronegativity rule). These are the two criteria the source study identifies as
  necessary validity conditions rather than empirical priors.
- Secondary reported settings: **q** alone and **theta = 0.10**, to show behavior at a
  looser and a stricter prior.

Violation rate is reported at all three. The stricter commonality filters are empirical
priors on oxidation-state rarity, not validity boundaries — lower retention there reflects
sampling of rare but not forbidden chemistry — so no claim of chemical invalidity is made
from them.

## Required measurement, not estimate

A2's sampled violation rate on MP-20 is **measured with a confidence interval at each of
the three filter settings**, and is not estimated from the 91.13% ground-truth composition
validity figure. Ground-truth validity is a property of the dataset; the CP4 comparison
needs the model's rate. This measurement is a WS-2 deliverable.

## Note on dataset variability

Reported compositional validity of MP-20 varies across the literature by filter and
protocol — the CDVAE-lineage SMACT check gives 91.13%, the chemical-filter study gives
96.5% under charge neutrality alone and 59.7% at theta = 0.50, and other recent work
reports far lower rates on different training sets. Comparisons of violation rate across
papers are therefore meaningless without the filter configuration stated. Ours is stated
above.
