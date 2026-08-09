# A2 — Training results: three arms, three seeds, on the synthetic task

Standing note, rebuilt on change. Feeds CP0, CP4, CP5. All numbers are read from
`results/A2_training_summary.json` and the nine per-run artifacts beside it.

## Summary

The first models are trained. All three arms separate cleanly across three seeds, and the
central result is stronger than the design anticipated: the trained-and-deployed A3 system
reaches a mean charge of +0.0170 against a target of 0, which is **6.0x closer to target than
the exact per-step projection applied to the data distribution** (+0.1028, measured in WS-1
with no model involved). The per-step surrogate is not merely inherited — training moves the
backbone into a region where the surrogate works substantially better.

Trained on the synthetic task rather than crystals deliberately: it has exact enumerable
targets, so every distribution below is computed by scoring all 65,536 sequences rather than
sampled, and comparisons against the closed-form optimum carry no sampling error.

## Results

Mean and standard deviation over seeds 0, 1, 2 at 4,000 steps. Data distribution mean charge
+2.5164; sequence-level optimum (tilt) mean +0.0000 at KL 0.3347 from data.

| Arm | Infeasible mass | Mean charge | Charge variance | KL from tilt |
|---|---|---|---|---|
| A1 unconstrained | 8.219e-04 +/- 8e-05 | +2.4639 +/- 0.0415 | 8.7433 +/- 0.127 | infinite |
| A2 masked (Regime S) | **0.000e+00 +/- 0** | +2.4694 +/- 0.0807 | 8.7620 +/- 0.091 | 0.3187 +/- 0.022 |
| A3 backbone reading | 0.000e+00 +/- 0 | +0.6287 +/- 0.3499 | 9.2461 +/- 1.765 | 1.4299 +/- 0.301 |
| A3 deployed system | 1.351e-30 +/- 3e-32 | **+0.0170 +/- 0.0014** | 1.5471 +/- 0.011 | infinite |

## What each row establishes

**The masking claim is exact and non-vacuous.** A2 places identically zero mass outside the
S-feasible set — not "small", zero, because masking makes infeasible sequences unreachable
rather than merely unlikely. A1 under the same training leaks 8.2e-04, so the claim is not
vacuous: an unconstrained model of this capacity does place mass there. A3 deployed shows
1.35e-30, which is float underflow of an exact zero rather than a distinct value.

**Masking alone does not steer.** A2's mean charge is +2.4694 against the data's +2.5164 — a
Regime-S model reproduces the data's constraint violation almost exactly, which is the expected
result and the reason Regime E exists.

**Two readings of A3, and they answer different questions.** The backbone reading applies the
mask only and reports what the network internalized: +0.6287, a substantial move from the data
but far from target and with high seed variance. The deployed reading applies the same
projection the model trained with, which is the system anyone would actually run: +0.0170 with
seed variance of 0.0014. Reporting only the backbone figure would understate the system;
reporting only the deployed figure would hide that the network itself learned only part of the
constraint. The first version of this evaluation reported only the backbone reading, which made
A3 look worse than it is.

**The surprising result, and its mechanism.** The deployed system beats the exact projection
applied to the data distribution by 6.0x on residual error, and its charge variance is 1.5471
against that reference's 3.103. The mechanism is measured rather than argued: the per-step
budget becomes infeasible when the required increment falls outside the attainable range of the
legal tokens, and the probability-weighted infeasible-step fraction is 0.0187 +/- 0.0007 for
the trained system against 0.3713 for the projection applied to raw data — a 19.8x reduction.
Training moves the backbone close enough to the target that the running budget rarely goes
unattainable, so the fallback fires rarely and little residual error accumulates.

This matters for the project's central claim: it means per-step projection and training are
complementary rather than redundant. The surrogate's known weakness — budget infeasibility late
in a sequence, measured in WS-1 — is substantially repaired by training through it.

**A3's infinite KL from the tilt is not a defect.** Both A1 and A3-deployed show infinite
divergence because they place mass on sequences the comparison distribution assigns exactly
zero. For A1 that is the feasibility leak. For A3-deployed it reflects the projection
concentrating mass differently from the tilt, and it is why the residual error and variance,
not the divergence, are the informative comparison in that row.

## Cost

A3 wall-clock is 543.5 s per run against A2's 90.7 s, a 6.0x factor, with the projection
applied at every position and no actionable-position scoping. That is consistent with the CP2
benchmark and is the configuration the scoping decision addresses. These runs carry the extra
precision-mismatch evaluation, which is why the figure is above the 526.5 s of the runs without
it; the training loop is unchanged and every reported distribution is identical.

## Train/inference precision mismatch: measured, not argued

The proposed m >= 2 fallback — fast Hessian with gradients disabled at inference — would deploy a
distribution perturbed inside the solver's tolerance ball relative to the one training used. Since
this project's thesis is that train/inference mismatch is the problem worth solving, shipping one
inside our own solver needs a measurement.

Both deployed distributions are computed from the SAME trained weights, so any difference is the
precision and not training noise.

| Quantity | Result |
|---|---|
| Total variation between deployed fp64 and fp32 distributions | 2.1e-10 +/- 2.9e-10 |
| Violation-rate two-proportion z, 1,000,000 samples per arm | 0.51 mean, 0.96 worst |
| Detectable at alpha = 0.05 | No, on every seed |
| Infeasible samples, either arm | 0 of 1,000,000 |

For scale, a study of 1e6 samples resolves a total variation of order 1e-3. The measured
difference is roughly seven orders of magnitude below that. **The route stays open**, and the
null result is stated with its sample size rather than as a bare assertion.

## Method notes

Three arms share initialization, data order, and optimizer settings, so differences are
attributable to the constraint treatment rather than training noise. Model is a 4-layer causal
decoder, d_model 128, 4 heads, AdamW at 3e-4 with 200 warmup steps and gradient clipping at 1.0,
batch 256, 4,000 steps.

## Corrections made during this work

Two defects were found and fixed before these numbers were reported.

The evaluation initially applied the mask but not the projection for A3, so it measured the
backbone rather than the deployed system and understated A3's residual by a factor of 37.
Both readings are now reported separately.

The infeasible-step statistic was initially an unweighted mean over the enumeration, which is
model-independent — the budget depends only on the sequence and the attainable range only on
the mask — and therefore not comparable to WS-1's probability-weighted figure. The tell was
that it came back bit-identical across all three seeds. It is now weighted by the induced
distribution; the unweighted value (0.1089) is retained in the artifact but never compared.
