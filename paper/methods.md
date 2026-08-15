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
what makes the layer affordable: $m$ is 1 or 2 in the training experiments here and up to
4 in the solver verification, while $V$ is $10^2$ to $10^3$. The cost of the solve is
dominated by the number of rows in the batch rather than by $m$: measured on the training
batch, a two-constraint solve is no more expensive than a one-constraint solve, and the
two-constraint arm's overhead comes from solving twice — once jointly and once for the
single-constraint fallback — rather than from the extra constraint.

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
| A3 | yes | yes, trained through, one constraint | full |
| A4 | yes | no | filtered to the target |
| A7 | yes | yes, trained through, two joint constraints | full |

A4 is the conditional-training baseline — the "retrain on filtered data" objection. It is
scored against the **full** split's statistics, never the slice it trained on, since
scoring it against its own subset would flatter it by construction. Its step budget is not
reduced, which is generous: the filtered set is roughly 13% of the split, so it sees each
structure about eight times more often.

A7 is A3 with a second expectation constraint solved jointly rather than sequentially. The
two channels are assembled identically and stacked into a single $A$ of shape $2 \times V$,
so one dual solve satisfies both; solving them in sequence would satisfy whichever was
applied last and would not be the KL optimum for the pair. The second target is a density
of the same form, so the budget construction is shared: `_additive_budget` generalizes the
mass budget over its weight table and is verified to reproduce it exactly for the mass
table at every prefix length, which is what keeps single-constraint results comparable
across the change.

## The second constraint family

The first family is mass density. The second is valence electron density — valence electron
count summed over the formula, divided by cell volume. It was chosen after charge
neutrality was found inexpressible in this layer's form: as the standard screen defines it,
a composition is balanced if there *exists* an assignment of formal charges from each
element's allowed oxidation states summing to zero, which is an existential predicate rather
than a linear functional, and collapsing it to one charge per element recovers 11.1% of the
balanced set. Valence electron density is additive in element counts, defined for every
element in the split, and not directly emitted by the model, which is the property that
makes it a real test rather than a restatement of density.

Its correlation with mass density is 0.725, which is load-bearing in both directions:
much lower would make the pair nearly orthogonal and composition unrealistically easy, much
higher would make them one constraint written twice with a singular dual Hessian.

Whether two channels are independent depends on where the constraint acts. At a count
position with pending element E the tokens are `cnt:k`, so the mass channel contributes
$\mathrm{MASS}[E]\,k$ and the valence channel $\mathrm{VALENCE}[E]\,k$ — both proportional
to $k$, hence exactly collinear. Rank 2 requires an element position, where the two weight
tables differ across elements. Since density steering already restricts actionable positions
to element positions, the second constraint costs no actionable positions.

## Joint feasibility

Two individually attainable targets need not be jointly attainable, and the distinction is
not cheap to predict. The natural test — whether the required ratio of targets lies inside
the range of per-token channel ratios — is necessary but not sufficient: against an exact
linear program on 400 random two-row systems it agrees on 266 and calls 134 infeasible
systems feasible. An exact test is a linear program per position, more expensive than the
projection it would guard.

Feasibility is therefore observed rather than predicted. For a target outside the attainable
set the dual solve does not converge, which is the correct verdict rather than a numerical
failure, and such positions fall back to the single-constraint projection. The fallback rate
is recorded in the artifact rather than hidden, and the ordering matters: discarding a
converged single-constraint projection because the joint solve failed would make the
two-constraint arm worse than the one-constraint arm exactly where the second target was
unattainable.

## Checkpointing and retargeting

Runs before this work trained, evaluated in-process, and discarded their weights. Weight
persistence was added so a trained model can be redeployed, and the checkpoint records the
target it trained at as a field distinct from the target it is later deployed at — the two
are separate quantities and conflating them would make a retargeting result unreadable.
The target, seed, and checkpoint path are command-line arguments rather than edited
configuration, since editing a config between runs is how a run gets mislabelled.

Retargeting deploys one checkpoint at a target it never trained on, changing only the
deployment target passed to the solver. Both the backbone and deployed readings are recorded
at every cell, so retargeting through the layer is distinguishable from a backbone that
happens to sit near the new target.

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
