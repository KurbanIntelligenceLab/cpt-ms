# Manuscript figures — data sources, aggregation, verification

Four main-text figures, each a single-column vector PDF (3.4 in wide) in
`template/figures/`, with PNG previews in `figures/`. Every plotted number is
read at render time from the artifact JSON named below; no number is typed
into any plotting script by hand. Each script also runs a render-then-verify
pass that reads the plotted values back off the matplotlib artists and
compares them to the source artifact with `np.allclose`.

## Figure 1 — `fig1_charge_distributions.pdf`

**Gap closed by retraining.** A prior pass could only plot 3 of the 4
requested series (exact tilt and an untrained-backbone proxy for A2/A3),
because `results/A2_training_summary.json` and
`results/WS7_a6_tilt_comparator.json` store only scalar summaries for the
trained arms and no synthetic-task model checkpoint had been saved. This has
been fixed by **retraining A2, A3, and A6 at a single seed (seed 0)**, using
the identical architecture, hyperparameters and training loop as the
published multi-seed runs (`scripts/train/train_a2.py`,
`scripts/synthetic/train_a6.py`), via the new
`scripts/synthetic/charge_marginals.py`. Each trained model's exact induced
distribution over all 65,536 enumerable sequences is computed with the
existing `induced_distribution_from_model` and marginalized onto integer
total-charge values. Results are written to
`results/WS7_charge_marginals.json`.

**Series plotted (4 curves):** trained A2 (mask only), the exact
sequence-level tilt, A6 (trained by MLE on exact tilt samples), and
A3-deployed (the trained projection decoder read through the same projection
it trained with — the system a user would actually run). The S-conditioned
data distribution is **not drawn as a separate curve** — it is nearly
coincident with trained A2 (variance 8.564 vs A2's 8.82) and a fifth series
crowded the panel without adding a visually distinct feature. Its variance is
instead named directly in A2's legend entry ("A2: mask only, trained (var
8.82; unconstrained data var 8.56)"), since the anti-concentration claim (the
tilt's 10.452 sitting *above* the data's 8.564) needs that number visible
somewhere in the figure. Similarly, the tilt and A6 are drawn intentionally
near-coincident — a thick, lightened line for the tilt with A6 dashed on top
in the same hue — so their overlap reads as "A6 reached its target," not as a
missing curve.

**Single-seed caveat and agreement check.** Only seed 0 was retrained (not
the published 3-seed average), because retraining all three arms at 3 seeds
each would cost roughly 30 additional minutes of A3's ≈9-minute-per-seed wall
clock for marginal gain. The scalar quantities derivable from this seed's
marginals were checked against the published multi-seed mean ± sd
(`results/A2_training_summary.json`, `results/WS7_a6_tilt_comparator.json`)
and all eight checks land within 4 standard deviations of the published
mean — most within ~1 sd:

| arm | quantity | this seed | published mean ± sd |
|---|---|---|---|
| A2 mask only | mean charge | 2.3719 | 2.4694 ± 0.0807 |
| A2 mask only | charge variance | 8.8207 | 8.7620 ± 0.0913 |
| A3 deployed | mean charge | 0.0177 | 0.0170 ± 0.0014 |
| A3 deployed | charge variance | 1.5589 | 1.5471 ± 0.0112 |
| A6 | mean charge | −0.1480 | −0.0619 ± 0.1220 |
| A6 | charge variance | 10.4958 | 10.3619 ± 0.2417 |
| exact tilt | charge variance | 10.4520 | 10.4520 (exact, no seed) |
| data (S-conditioned) | charge variance | 8.5636 | 8.5636 (exact, no seed) |

The full check is recorded machine-readably in
`results/WS7_charge_marginals.json` under `payload.agreement_check_vs_published`.

**The headline mechanism is now directly visible.** The exact tilt buys a
mean of zero by spreading mass toward both extremes — its charge variance
(10.45) is *higher* than the unconstrained data's (8.56) — while the deployed
per-step projection instead concentrates mass near zero (variance 1.56),
well below both the data and the tilt/A6. This anti-concentration ordering
(projection < data < tilt ≈ A6) is asserted in the plotting script and in
`tests/test_fig1_charge_marginals.py`.

**Layout.** The legend sits below the axes (frameless, one column) rather
than inside the panel, so it never occludes the A3-deployed peak — the
figure's tallest and most important feature. The constraint-target line at
charge = 0 is drawn thin and lightened (its own annotation, "constraint
target (0)", already names it), and the annotation itself is placed in the
low-mass region at the right edge of the panel, clear of every curve's
rising or falling edge.

**Verification:** 4/4 plotted series matched
`results/WS7_charge_marginals.json` exactly (`np.allclose` on x and y data
read back off the rendered `Line2D` objects).

## Figure 2 — `fig2_steering_frontier.pdf`

**Artifacts read:** `results/FIG2_steering_frontier.json` (new artifact,
emitted by `scripts/crystal/compute_fig2_steering_frontier.py`), aggregated
directly from the 18 per-run rows in `results/WS3_steering_frontier_rows.json`.
`results/WS1_steering_frontier.json` was inspected but is the **synthetic-task
charge** sweep (an enumerable 4-symbol task), not a crystal density
measurement, and does not feed this figure's axes.

**Aggregation convention:** seed-averaged per target — reports/
WS3_steering_frontier_notes.md uses two conventions (seed-averaged for its
headline 10-row frontier table; per-run, 18-row, only for the compensation
regression). This figure reproduces the seed-averaged convention, matching
the notes' "The frontier" table to full precision (e.g. target 20.0: deployed
10.4047 ± 0.2037, matching the note's "10.405 +/- 0.204").

**Steering ceiling:** 10.4047 g/cm³, the achieved density at the highest
requested target (20.0 g/cm³), per the artifact's own `steering_ceiling`
field.

**Verification:** all 10 target points and all 10 error bars matched the
artifact exactly (point x/y read off the `ErrorbarContainer`'s data line;
error-bar half-widths read off its vertical-segment `LineCollection`).

## Figure 3 — `fig3_accuracy_diversity_pareto.pdf`

**Artifacts read:** `results/FIG3_accuracy_diversity_pareto.json` (new
artifact, emitted by `scripts/crystal/compute_fig3_pareto.py`), built from
`results/WS6_best_of_n.json` (best-of-n curve: `rows` for `mean_best_error`,
`diversity` for `unique_formula_fraction`) and
`results/WS3_conditional_baseline.json` (`per_target.<t>.A3` for the
projection's own error/unique-fraction point). `results/WS6_smc_steering.json`
was read but could not be placed on this figure (see gap below).

**Aggregation:** mean over **sub-ceiling targets only** — {1.5, 2.0, 3.0, 5.0,
6.317, 8.0, 10.0}, i.e. every best-of-n/conditional-baseline target at or
below the 10.4047 g/cm³ steering ceiling identified in Figure 2. Above the
ceiling the projection's deployed density saturates regardless of the
request, so a comparison there measures the ceiling rather than the methods,
per the task instruction. The projection's own point averages its four
`WS3_conditional_baseline` targets (3.0, 5.0, 8.0, 10.0), all of which are
already inside the sub-ceiling range.

**Verification:** the 4-point best-of-n curve (n = 1, 4, 16, 64) and the
single projection point matched the artifact exactly (`np.allclose` on the
plotted `Line2D` and scatter-collection offsets).

**Gap — SMC omitted from the diversity axis.**
`results/WS6_smc_steering.json` carries only `mean_error`, `sd_error`,
`wall_clock_s_per_output`, and `forward_passes_per_output` per (target, K)
cell — no unique-formula-fraction or any other diversity field for any cell.
SMC therefore cannot be placed on this figure's y-axis without fabricating a
number. Its accuracy-only comparison against the projection and best-of-n is
reported in `reports/WS6b_smc_notes.md` (SMC is worse than best-of-n at equal
compute by a mean factor of 4.2 at K = 64, and reaches the projection only at
6 of 10 targets, all above the ceiling).

## Figure 4 — `fig4_infeasibility_vs_target.pdf`

**Artifacts read:** `results/FIG4_infeasibility_vs_target.json` (new
artifact, emitted by `scripts/crystal/compute_fig4_infeasibility.py`), from
`results/WS3_steering_frontier_rows.json`'s `infeas_init` / `infeas` fields
per run. `scripts/crystal/analyze_sweep.py` was inspected for a stored repair
factor definition; it computes `infeasible_fraction` / `infeasible_at_init`
per run but does not itself compute a "repair factor" ratio anywhere in the
codebase, so the convention was derived directly from
`reports/WS3_steering_frontier_notes.md`'s stated values.

**Aggregation / repair-factor convention:** ratio of seed-averaged means per
target — `mean(infeas_init) / mean(infeas)`. At the weakest steering target
(1.5), this gives **29.247×**, matching the frontier notes' "29.2x" to the
first decimal. The alternative convention (mean of per-run ratios) gives
29.390×, which is the "29.3x" figure a prior session flagged as a rounding
inconsistency with the notes — confirmed here as the wrong convention, not
plotted, but recorded in the artifact as
`repair_factor_mean_of_ratios_alt` for provenance.

**Verification:** all 3 series (`infeas_init_mean`, `infeas_final_mean`,
`repair_factor_ratio_of_means`, plotted on init/final left axis and repair
factor right axis) matched the artifact exactly across all 10 targets.

**No gap.** All series plotted are fully backed by the source artifact.

## Summary of gaps carried into the manuscript

1. ~~Figure 1 cannot show A6's trained charge marginal or A3-deployed's true
   charge marginal~~ — **resolved**: both were recovered by retraining A2,
   A3, and A6 at a single seed (seed 0), identical hyperparameters to the
   published runs; see the Figure 1 section above. The remaining caveat is
   that this is one seed, not the published 3-seed average — the agreement
   check against the multi-seed summaries is recorded in
   `results/WS7_charge_marginals.json`.
2. Figure 3 cannot place SMC on the diversity axis — no diversity metric was
   ever computed for the SMC sweep.

The Figure 3 gap is recorded machine-readably inside
`results/FIG3_accuracy_diversity_pareto.json` under a `gap` key.
