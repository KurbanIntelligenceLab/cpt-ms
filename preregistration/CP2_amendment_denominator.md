# Amendment — CP2 overhead denominator

Amends `CP2_overhead_budget.md`. Approved by the team after the benchmark ran on the training
hardware. Recorded here rather than by editing the original, so the pre-registered quantity and
the reason for replacing it both remain readable.

## What changed

The threshold was pre-registered as per-token wall-clock overhead of the projection layer
measured against the identical A2 masked-softmax forward pass, with a 30% ceiling. Measured that
way on the RTX 5090 the layer costs 3672% to 7260%, failing by 122x to 242x.

The denominator is replaced by a full training step at the A2 model scale, with the projection
applied at actionable positions.

## Why, stated as the reason of record

The original quantity does not measure affordability. Masked softmax costs 0.197 us/token
against 4.895 us/token of total per-token training cost — about 4% of a step. A ratio against a
4% component is dominated by the smallness of the denominator and would have failed for
essentially any layer doing real work. This is a category error in the measurement, not a
threshold that proved too tight.

The as-registered figure is retained as hardware characterization, not as a gate. Both numbers
live permanently in `results/CP2_overhead_benchmark.json`.

Actionable-position scoping rests on the grounds CP0 already uses: at a non-actionable position
no format-legal substitution changes the constraint, so the projection provably cannot affect
the outcome. That is wasted compute, not skipped compute.

## Methodological finding, recorded as part of the amendment

**Per-token cost is not constant in batch size, so a per-token figure and a whole-step share are
different measurements and must never be combined.** At 256 rows the solve is launch-overhead
dominated at 7.66 us/row; by roughly 1,000 rows it amortizes to about 2.5 us/row and stays flat
through 16,384. Extrapolating the small-batch figure to a full 8,192-token step overstates cost
by 3.0x.

This is the finding, not merely the fix. The first published table combined the two and was low
by that factor throughout. Any future cost claim states the batch size it was measured at.

The same effect explains a second discrepancy. The fp32-Hessian speedup was first reported as
1.7x, which is a batch-256 measurement. Measured across batch size on the same card, both arms
carry a fixed launch cost that masks the arithmetic difference at small batch:

| Rows | fp64 Hessian | fp32 Hessian | Speedup |
|---|---|---|---|
| 128 | 22.23 us/row | 17.15 us/row | 1.30x |
| 256 | 14.35 us/row | 8.55 us/row | 1.68x |
| 335 | 10.97 us/row | 6.39 us/row | 1.72x |
| 512 | 10.62 us/row | 4.52 us/row | 2.35x |
| 1,024 | 8.85 us/row | 3.38 us/row | 2.62x |
| 8,192 | 8.21 us/row | 3.43 us/row | 2.39x |
| 16,384 | 8.15 us/row | 3.43 us/row | 2.38x |

The ratio plateaus at about 2.4x from 512 rows. The asymptotic per-row costs predict it exactly:
8.15 / 3.43 = 2.38x against a measured 2.38x at 16,384 rows. So the correct figure at deployment
scale is 2.4x, not 1.7x, and the 335-row outlier the review flagged in both ratio columns has the
same cause as the batch-256 reading — small-batch behaviour is qualitatively different because
fixed cost, not arithmetic, dominates there.

Reading (b) from the review — that the fp32 columns were carried over rather than re-measured —
is ruled out: a fresh independent measurement at 8,192 rows gives 2.39x against the scoping
table's 2.39x, agreeing to two decimals.

## Consequence for multi-constraint configurations

The Hessian contraction scales with the square of the active-constraint count while the residual
scales linearly, so m >= 2 is materially more expensive. With the accurate solver, m=2 costs
46.9% to 64.9% of a training step at the density dial's actionable fractions against m=1's 13.6%
to 19.9%. A2 uses a single dial so nothing is blocked now, but the multi-constraint decision
belongs on the Sync-C agenda alongside scale-up and Branch D rather than after Branch C.
