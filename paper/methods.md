# Methods

Draft. Numerical values are read from artifacts in `results/`; the verification scripts
that check them are cited by filename at the end of each subsection.

## The projection layer

At decoding step $t$ the backbone emits logits $\ell \in \mathbb{R}^V$ over the
vocabulary. A constraint is supplied as a matrix $A \in \mathbb{R}^{m \times V}$ of
per-token channel values and a target $b \in \mathbb{R}^m$, together with a support mask
$M \subseteq \{1, \dots, V\}$. The layer returns

$$q^\star = \arg\min_{q} \; \mathrm{KL}(q \,\|\, p_M) \quad
\text{s.t.} \quad Aq = b, \; \mathrm{supp}(q) \subseteq M,$$

where $p_M$ is the mask-restricted softmax of $\ell$. The solution is an exponential tilt
$q^\star \propto p_M \odot \exp(A^\top \lambda)$, and the dual variable $\lambda$ solves
$A q^\star(\lambda) = b$. This is $m$-dimensional regardless of vocabulary size, which is
what makes the layer affordable: $m$ is 1 or 2 in every experiment here while $V$ is
$10^2$ to $10^3$.

$\lambda$ is found by damped Newton iteration on the dual, with Hessian
$\mathrm{Cov}_{q}[a]$. Gradients pass through the implicit function theorem rather than by
unrolling the solve, so the backward cost is one linear system in $m$ dimensions and is
independent of the iteration count.

### Solver precision

The dual solve runs in float64 internally regardless of the input dtype, with the result
cast back on exit. This is not conservatism. At float32 only 11 of 64 batch entries
reached tolerance, and at half precision the solve raises outright because torch provides
no half-precision LU factorization — so under mixed-precision training the layer would
either fail or silently return distributions that are not the optimum. Convergence is
judged on the float64 solve before the output cast, so a representation floor in the
output dtype is not misread as solver failure.

The Hessian contraction is the dominant cost and admits a float32 fast path: on consumer
Blackwell it is 27x faster in float32 while affecting only the Newton direction, never the
fixed point. That path is **opt-in and off by default**. It perturbs where Newton stops
inside the tolerance ball by roughly $10^{-9}$, which is enough to make the layer
non-smooth at the scale a finite-difference gradient check probes, and the implicit-function
backward assumes the forward solution is exact. Trading gradient correctness for a
2.4x forward speedup is the wrong trade in a method whose contribution is a differentiable
layer.

The train/inference precision mismatch this introduces was measured rather than assumed,
since a method premised on such mismatches being worth fixing cannot ship one unexamined.
Between the two precisions the deployed distributions differ by total variation
$6.2 \times 10^{-10}$ in the worst of three seeds (mean $2.1 \times 10^{-10}$), and a
direct sampled test on violation rate at $10^6$ samples gives $z \le 0.96$ — undetectable,
with the null stated against its sample size rather than asserted. The worst case is
quoted rather than the mean, since it is the figure that bounds the claim.

Verified by `tests/test_projection_layer.py` (22 tests, including gradient agreement
against `cvxpylayers` to $2.0 \times 10^{-6}$ across five configurations and dtype
robustness across all four dtypes).

## Constraint regimes

**Regime S** restricts the support. The mask must be prefix-closed — a spent budget cannot
be recovered — which makes the emitter well-defined and the layer trainable without
straight-through estimation.

**Regime E** constrains an expectation. A target on a completed sequence is converted to a
per-step budget: with total target $T$, committed amount $c$ after the prefix, and $r$
remaining slots, the per-step demand is $(T - c) / r$. Two properties of this construction
matter and are stated rather than assumed. The demand grows without bound as a sequence
closes, so it can become unattainable partway through; and it bounds the expected total
without bounding its variance.

### Determinability and actionability

A constraint can enter the layer at step $t$ only if it is **determinable** — computable
from the prefix — and it can be steered only if it is **actionable**, meaning some
admissible substitution at $t$ changes it. These differ, and both are required.
Actionability is measured by counterfactual substitution rather than argued from the
serialization format, because the format argument proved wrong once: the charge constraint
is determinable over nearly the whole sequence yet actionable only over a short opening
window. Gate criteria are therefore defined over positions that are both actionable and
feasible, with sequence-averaged figures reported alongside but never substituted.

Verified by `tests/test_parser_determinability.py` and `tests/test_actionability.py`.

## The infeasible-step fallback

When the per-step budget falls outside the attainable range the projection has no
solution, and the policy is pre-registered rather than chosen after seeing results. Two
candidates were implemented and measured. Falling back to the base conditional preserves
fidelity and leaves residual constraint error; clamping to the nearest attainable increment
cuts the target error 31x at a cost of 0.235 nats and a further halving of variance. The
base-conditional policy is the registered default. A third policy — redistributing the
shortfall across later steps — is out of scope because it makes the constraint depend on
steps not yet taken, which breaks per-step trainability.

A row that loses all probability mass falls back to the masked softmax and the rate is
reported as `degenerate_row_rate` rather than silently repaired. This matters: an earlier
sanitization order (`clamp_min` before `nan_to_num`) mapped an all-NaN row to an all-zero
row, and two of twenty sweep runs died on a device-side assert in `multinomial` before it
was corrected.

Pre-registered in `preregistration/CP0_fallback_policy.md`.

## Tasks

**Synthetic.** Four symbols with integer charges, sequences of length 8, giving 65,536
enumerable sequences. Every distribution is therefore scored exactly over the whole space
rather than sampled, and both targets — the constraint-conditioned distribution and the
exponential tilt — are available in closed form. The data distribution is deliberately
biased so it does *not* already satisfy the constraint (mean total charge +5.181 against a
target of 0); if it did, every target would collapse onto the data and the multipliers
would be zero from initialization.

**Crystals.** MP-20, the standard split (26,910 parsed training structures of 27,136 rows),
serialized as element/count pairs and a discretized cell. Serialization is **cell-first**:
density is mass over volume, so the volume must be committed before the composition for the
mass budget to be determinable at the positions where mass is decided. This is not
cosmetic — it is a case where the constraint form dictates the serialization order.

Discretization ranges cover the data rather than spanning its typical values. An earlier
setting clamped 642 cell angles at its floor and 205 lengths at its ceiling; that was
invisible in aggregate (1.2% median decode error) while corrupting 21.4% of structures in
the 20–25 g/cm³ band, precisely where the extreme steering targets sit. Ranges were widened
to $[5, 175]°$ and $[2, 46]\,\text{Å}$ and resolution re-set with them (512 length by 256
angle buckets, 884-token vocabulary), which is better than the original on both the bulk
(0.84% vs 1.17% median) and the tail (3.1% vs 54% worst-case).

Verified by `tests/test_tokenizer_coverage.py` and `tests/test_crystal_mask.py`.

## Arms

All arms share initialization, data order, optimizer settings, and step budget, so
differences are attributable to constraint treatment.

| Arm | Mask | Projection | Training data |
|---|---|---|---|
| A1 | no | no | full |
| A2 | yes | no | full |
| A3 | yes | yes, trained through | full |
| A4 | yes | no | filtered to the target |

A4 is the conditional-training baseline — the "retrain on filtered data" objection. It is
scored against the **full** split's statistics, never the slice it trained on, since
scoring it against its own subset would flatter it by construction. Its step budget is not
reduced, which is generous: the filtered set is roughly 13% of the split, so it sees each
structure about eight times more often.

Every arm is reported at two readings, and neither is a summary of the other. The
**backbone** reading applies the mask but not the projection; the **deployed** reading
applies both. Reporting only the backbone understates the system anyone would run;
reporting only the deployed hides how far the backbone has drifted, which on crystals is
the finding.

## A note on how results were checked

Each reported number is compared against its source artifact by a script that fails on
mismatch, and a repository test parses quantitative claims out of the markdown reports and
checks them against the aggregated artifacts, so a stale figure in prose fails the suite
rather than surviving review. Several results in this paper were corrected by that
mechanism rather than by inspection.
