# Pre-registration — CP2 per-token overhead budget

Status: draft, not yet committed. Becomes binding when CP2's measuring code first runs.

## Threshold

Per-token wall-clock overhead of the Regime-E projection layer, measured against the
identical A2 masked-softmax forward pass, must not exceed **30%** at realistic constraint
loads, reported per active-constraint count.

## Break-even derivation

The budget is derived from an explicit break-even against A5-style inference-time
correction, using primary-source costs rather than an inherited figure.

Inference-time reference costs, from Constrained Discrete Diffusion (NeurIPS 2025),
Appendix D, Table 4, toxicity-mitigation task.

Each method is measured against its own base model, not against a single common
baseline: CDD is a discrete-diffusion method rated against MDLM, while PPLM and FUDGE
are autoregressive guidance methods rated against GPT-2. Mixing the two baselines into
one ratio column would be meaningless, so the base is named on every row.

| System | Base model | Mean runtime (s) | Mean ratio | Median runtime (s) | Median ratio |
|---|---|---|---|---|---|
| MDLM (base) | — | 4.59 | — | 4.49 | — |
| GPT-2 (base) | — | 10.31 | — | 10.32 | — |
| CDD, tau = 0.75 | MDLM | 63.71 | 13.88 | 9.56 | 2.13 |
| CDD, tau = 0.50 | MDLM | 230.89 | 50.30 | 9.62 | 2.14 |
| CDD, tau = 0.25 | MDLM | 401.15 | 87.40 | 9.67 | 2.15 |
| PPLM | GPT-2 | 878.47 | 85.21 | 869.69 | 84.27 |
| FUDGE (lambda = 2) | GPT-2 | 1387.28 | 134.56 | 1387.00 | 134.40 |
| FUDGE (lambda = 9) | GPT-2 | 1481.94 | 143.74 | 1480.27 | 143.44 |

The mean ratios reproduce the paper's own Increase Factor column to within rounding
(published: 13.82, 50.08, 87.02, 85.21, 134.56, 143.74), which confirms that column is a
mean-over-mean ratio against each method's respective base. The median ratios are
computed here from the same table's median columns.

For CDD the mean and median ratios differ by more than an order of magnitude — 13.88 to
87.40 against 2.13 to 2.15 — because the mean is dominated by a heavy tail of slow
generations while the median reflects the typical case. The efficiency claim is therefore
stated against both separately and never against a single unqualified figure. The
autoregressive guidance baselines show no such gap: their mean and median ratios agree to
within 1.2% (largest gap 1.10%, PPLM), so their cost is uniformly high rather than
tail-driven.

Break-even: training-time projection pays its overhead once, over the training run;
inference-time correction pays its overhead on every generation, permanently. With
projection overhead `r` on a training run of `T` token-steps, and inference correction
multiplying per-generation cost by `k` over `G` generations of `L` tokens each,
training-time projection is cheaper end-to-end when `r * T < (k - 1) * G * L`. At the
project's own scale (T on the order of 1e9 training tokens, G unbounded and growing over
the artifact's lifetime), even the least favorable primary-source value of `k` — the
median 2.13 — puts break-even well below a 30% budget. Under the mean ratios it is lower
still by more than an order of magnitude.

Conclusion: 30% is a conservative ceiling, not a tight constraint. It is retained as the
pre-registered threshold because a layer exceeding it would signal an implementation
problem rather than a fundamental cost, but the CP2 artifact reports the computed
break-even point alongside the measured overhead so the margin is visible rather than
implied. The CP8 efficiency claim uses this same accounting.

## Correction history

An earlier draft of the research plan cited CDD's overhead as "5-20x". That range is not
attributable to the paper: it appears nowhere in the text, and it does not correspond to
any row, column, or summary statistic of Table 4. Exactly one of the twelve ratios above —
CDD at tau = 0.75, mean 13.88 — happens to fall inside it; the other two CDD mean ratios
(50.30, 87.40) sit well above it, all three CDD median ratios (2.13, 2.14, 2.15) sit well
below it, and every guidance-baseline ratio (84.27 to 143.74) is far above it. A single
coincidental overlap does not make the range a citation.

The practical consequence is that the earlier figure understated the mean-case inference
cost: the two tighter CDD settings run at 2.5x and 4.4x the top of the cited range, so a
break-even calculation derived from "5-20x" is materially more pessimistic than the
primary source warrants. In the other direction, the median ratios sit below the bottom
of the cited range, which is why the claim is stated against mean and median separately
rather than against any single spanning figure. The values above are read directly from
the paper. This note supersedes any threshold derived from the earlier figure.
