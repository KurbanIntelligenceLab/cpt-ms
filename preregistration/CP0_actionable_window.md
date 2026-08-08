# Pre-registration — CP0 active-multiplier threshold, scoped to the actionable window

Status: draft, not yet committed. Becomes binding when the CP0 collapse-test script first runs.

## The distinction the threshold depends on

Two different questions, previously conflated in this project's measurements:

**Determinability at t** — the constraint's value is computable from the prefix.
**Actionability at t** — some format-legal token at t would change the constraint's final
value.

A per-step projection needs both. A constraint that is determined but frozen admits no
projection: there is nothing left to move. Measured by counterfactual substitution rather
than argued from the format, because the format argument has already been wrong once in this
workstream.

| Backbone | Family | Determinable | Actionable | Ratio |
|---|---|---|---|---|
| B1 naive layout | (i) charge | 0.002 | 0.0570 | 0.04 |
| B1 naive layout | (ii) density | 0.669 | 0.2831 | 2.4 |
| B1ref reference ordering | (i) charge | 0.949 | 0.0410 | 23.1 |
| B1ref reference ordering | (ii) density | 0.490 | 0.3849 | 1.3 |
| B2 typed fields | (i) charge | 0.785 | 0.1280 | 6.1 |
| B2 typed fields | (ii) density | 0.480 | 0.1280 | 3.8 |

The B1ref charge row is the case that forces this pre-registration. The constraint is
computable over 94.9% of positions and movable over 4.1% of them — a factor of 23. Under the
reference ordering the composition is declared in the `data_` tag and then fixed, so all
compositional freedom lives in that opening window and nothing afterward can change the total
charge.

Note the naive layout inverts the relationship: there, charge becomes determinable only at the
very end but is actionable throughout the atom-site loop, since each site line still adds an
atom. A single ratio does not summarize the two backbones; both quantities are reported per
backbone and per family.

## Threshold

CP0 reads the fraction of positions carrying an active multiplier at convergence. That
fraction is computed **within the actionable window for the constraint under test**, not
averaged over the sequence.

Pre-registered thresholds, per backbone and family, on the actionable-window fraction:

- **Active at initialization**: > 0.50 of actionable positions. An untrained model has no
  reason to satisfy the constraint, so most actionable positions should carry a multiplier.
- **Collapse criterion**: the converged active fraction falls below 0.10 of actionable
  positions while training loss is still improving. This is the reading that says the
  constraint has been internalized and the projection has become an identity.
- **Reported alongside, never substituted**: the sequence-averaged fraction, so the two are
  comparable across backbones and the scoping is visible.

## Why the scoping is not a convenience

A correctly functioning layer on B1ref family (i) is active over a mean of 13.2 actionable
positions out of a mean sequence length of 320.3, with every case's window starting at token 0
and closing between tokens 10 and 18. Sequence-averaged that is 3.5-4.6%, which a threshold
calibrated against the 0.949
determinable fraction would read as collapse. The gate would fire on a working layer, and the
project would conclude multiplier decay from a measurement artifact — the same class of error
as the flat-dual residual floor recorded in the CP2 notes, and with the same consequence,
since CP0's reading selects the paper's register.

## Division of labor this implies

Regime E shapes composition choice inside the opening window; Regime S enforces atom-site
consistency with the declared composition thereafter. Charge stays balanced for the rest of
the sequence without E acting again, because S prevents the sites from contradicting the
declaration. This is the S/E split behaving as designed rather than a limitation of it, and
it means E's activity is expected to be concentrated, not diffuse.

## Counter-evidence, pre-registered as a claim we will not make

PhononBench Table 3, dynamical stability as stable-over-relaxed, restricted to models trained
on MP-20 so training data is matched:

| Model | Constraint | Stability |
|---|---|---|
| InvDesFlow-AL | unconstrained | 27.20% (2,176/8,000) |
| DiffCSP | unconstrained | 27.15% (2,488/9,163) |
| MatterGen | unconstrained | 24.55% (2,278/9,279) |
| CrystalFlow | unconstrained | 16.74% (1,428/8,533) |
| CrystalFormer | symmetry-constrained | 11.57% (510/4,408) |
| CrystaLLM | unconstrained (LLM) | 2.97% (58/1,951) |

At matched training data the symmetry-constrained model is the second-worst of six, at 11.57%
against 27.15% for unconstrained DiffCSP — a factor of 2.35. The same ordering holds on Alex20:
CrystalFormer 34.36% against MatterGen 40.99% and InvDesFlow-AL 38.42%, so its widely cited
34.4% reflects the larger training set, not its constraints. The paper attributes the leading
models' performance to pretraining on large high-quality databases.

**Therefore this project will not claim that constraints improve generation quality
in-distribution.** The evidence says the opposite: constraints restrict the generative space
and buy nothing on data the model can already fit. This is stated in the paper as motivation
rather than left for a reviewer to raise, and it is the empirical case for the steering
framing — the value of enforcement has to be demonstrated where the data does not reach.
