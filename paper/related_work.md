# Related work

Draft. Claims about cited papers are taken from the paper's own abstract or body rather
than a secondary summary. Sources fetched and matched verbatim so far: HardNet
(arXiv:2410.10807, both the universal-approximation and end-to-end-training claims) and
DAE-HardNet (arXiv:2512.05881, both the convergence magnitude and the droppable-layer
conclusion). Claims attributed to Park et al. and Papay et al. are drawn from their
abstracts and still need the same full-text check before submission; they are flagged here
rather than presented as verified.

## Constrained decoding, and why per-step masking is not conditioning

Grammar-constrained decoding enforces structure at inference by masking, at each step,
the tokens that provably cannot lead to a string in the target language. The technique is
mature and the mask itself is cheap.

The result closest to ours in kind is Park et al., *Grammar-Aligned Decoding* (NeurIPS
2024). They show that grammar-constrained decoding — and, they argue, structured decoding
generally — distorts the model's learned distribution, so outputs are grammatical but
arrive with likelihoods not proportional to the model's own, and therefore low quality.
Their remedy, ASAp, biases each step by an estimate of *expected future grammaticality*,
approximated from prior samples because computing it exactly requires marginalizing over
all continuations.

This is the same mathematical fact that governs our Regime E result, transposed from a
support constraint to an expectation constraint. Greedy per-step enforcement does not
equal conditioning on the constraint: the per-step rule is myopic about what the remainder
of the sequence can still do. We observe the expectation-constraint version directly — the
distribution induced by per-step projection is materially different from the sequence-level
exponential tilt, not a slightly worse approximation of it, and the gap opens precisely
where the running budget becomes unsatisfiable partway through a sequence. Park et al.
identify the problem for support constraints and correct it at inference; we take the
training-time route instead, and quantify what the per-step surrogate costs rather than
attempting to remove it.

Their result also bears on our scope. Because masking redistributes mass rather than
conditioning, syntactic validity by construction does not imply sample quality — which is
why we report syntactic parsing and structure construction as separate numbers rather than
a single validity figure.

## Constrained training versus constrained decoding

The distinction we build on is not new. Papay et al., *Constraining Linear-Chain CRFs to
Regular Languages*, set out both regimes for linear-chain CRFs — train unconstrained and
constrain only at decode time, or constrain during training as well — and prove a
likelihood hierarchy in which constrained training dominates, concluding that the
constrained-training regimen should be preferred.

Our contribution is not that observation but its transfer to a setting where it is harder
to obtain: an autoregressive neural decoder with a constraint that is an *expectation* over
the completed sequence rather than membership in a regular language. Language membership is
prefix-decidable; an expectation over a sequence is not, which is what forces the running
budget, the determinability and actionability analysis, and the infeasible-step fallback.

Grammar-forcing during training has also appeared in sequence-to-sequence translation to
temporal logic, applied alongside grammar-constrained decoding at inference. That work
uses the target sequence's tokens to advance the grammar state during training, which is
teacher-forcing in the grammar rather than a differentiable constraint the gradient passes
through.

## Differentiable projection layers

A substantial literature attaches a differentiable projection to a network's output so
that predictions satisfy constraints while remaining trainable end to end. HardNet
(arXiv:2410.10807) appends a differentiable closed-form enforcement layer to the network's
output and shows the construction retains universal approximation; its abstract states
that unlike approaches which modify outputs only at inference time, HardNet enables
end-to-end training with hard constraint guarantees, leading to improved performance.
Others solve the projection as a differentiable convex program, or restore feasibility by
iterated linearization to a tolerance.

Two things separate our setting from that literature. First, the object being projected is
a *distribution over a discrete vocabulary* at each decoding step, not a real-valued
prediction vector, so the projection is an information projection whose solution is an
exponential tilt with an $m$-dimensional dual — $m$ being the number of constraints, 1 or 2
here — rather than a Euclidean projection onto a polytope. Second, the constraint is not a
property of a single output but of a sequence the model has not finished generating, which
is what makes the per-step reduction necessary and imperfect.

**One prior observation runs against our central result and deserves to be stated
plainly.** DAE-HardNet reports that as training proceeds its network's predictions before
and after projection become similar — a difference on the order of $10^{-6}$ on the
constraint space — and concludes on that basis that the projection layer can be removed at
inference, bringing its cost close to an unconstrained network's. That is the redundancy outcome our own research plan
anticipated as a risk, and on our synthetic task something like it holds in part — the
backbone internalizes much of the constraint, closing the gap between readings to 37x. On
crystals it fails outright: removing the layer at inference does not degrade the result by a
margin, it inverts it, from a density error of 0.190 to 4.200 against an unsteered baseline
of 1.783. Whether the projection becomes redundant is therefore a property of the task and
the constraint, not of the method, and reporting the two readings separately is the only way
to tell which case one is in.

## Constraints in generative models for materials

Generative models of crystals enforce structure by construction (symmetry-aware
representations, Wyckoff-position encodings) or by post-hoc filtering. Charge neutrality
and electronegativity screens are standard, typically applied as an accept/reject step
after generation. An external benchmark reports that constraints do not improve
in-distribution sample quality, and nothing here contradicts that; our claim is about
reaching a specified target, not about improving the unconditional distribution.

The baseline the community reaches for first is filtering — either post-hoc rejection or
retraining on a filtered subset. We measure both. Rejection sampling from the data at our
targets costs between 7.4 and 4,485 draws depending on where the target sits, so the
efficiency argument for a training-time method is weak in the bulk and strong only in the
tails. The argument that does not weaken is fidelity: rejection returns whatever chemistry
happens to occupy that density band, and its composition departs from the full data
monotonically toward both extremes, reaching total variation 0.959 at the highest target,
where its entire acceptance set is six structures dominated by one element. Conditional
retraining reproduces the slice it was shown rather than reaching the target from the full
distribution.

## Positioning

The method is a training-time constraint layer for autoregressive generation with
expectation constraints. Relative to inference-time constrained decoding it moves the cost
to training and changes what the model learns; relative to existing differentiable
projection layers it operates on a per-step distribution under a sequence-level constraint;
and relative to filtering baselines it trades a measured accuracy gap for coverage that
filtering cannot supply.

The sequence-level target this work's per-step surrogate approximates is not a construct of
this paper. Khalifa, Elsahar and Dymetman (ICLR 2021) pose controlled generation as moment
constraints, derive the same KL-minimal exponential-family optimum, and train an autoregressive
model toward it by a distributional policy-gradient variant; their `disco` toolkit (Kruszewski,
Rozen and Dymetman, ACL 2023) implements it. Step 2's measurement — that the per-step projection
induces a distribution roughly one nat from that optimum — is therefore a measurement against a
named method rather than against a hypothetical, and the comparator arm that trains directly
toward the tilt is theirs. Their formulation bakes the moment target into the energy model before
training begins, so like conditional training it needs one run per target; that is where the
amortization claim bites rather than on accuracy.

Ahmed, Chang and Van den Broeck (NeurIPS 2023) are the nearest training-time neighbour: they
relax a logical constraint into a differentiable pseudo-semantic loss over a local
approximation, which is the same shape of move as reducing a sequence-level expectation to a
per-step one. The difference is the constraint class and where exactness lands — their relaxation
is approximate in the constraint, while the projection here is exact in the support restriction
and approximate only in the expectation.

The sharpest distinction is amortization, and it is the one a reader should weigh. Every
baseline that achieves a target by shaping its training distribution — conditional training,
filtered retraining, classifier guidance trained per attribute — needs one model per target
by construction, and needs enough data near that target to train on. Because the constraint
target enters the dual solve as an argument rather than as a training-time label, one
checkpoint here covers a family of targets at parity with separately trained models
(0.650 ± 0.151 g/cm^3 against 0.662 from 7 runs, Wilcoxon p = 1.0). The
comparison is deliberately unflattering on accuracy — conditional training remains more
accurate per target — so the claim is a different cost structure rather than a better
number, and it is strongest exactly where the baseline is weakest: at sparse targets, where
the slice a conditional model needs either thins or does not exist.

The inference-time alternative is measured rather than argued past. Best-of-n from the unsteered
model amortizes perfectly and, given enough samples, is more accurate than the projection at most
targets; what it cannot do is hold diversity while it does so. Sequential Monte Carlo steering
(Loula et al., ICLR 2025) is the principled member of that family, reallocating particles as
information arrives, and it occupies the same axis: accuracy bought with inference compute, per
output, every time. The projection buys accuracy once at training and spends one forward pass
thereafter, so the comparison is a point against a curve and the crossover is the result rather
than a winner.

In materials generation specifically, property-conditioned models set the bar for what
conditioning already achieves. MatterGen (Zeni et al., Nature 2025) fine-tunes toward property
constraints including joint ones, and CrystaLLM (Antunes et al., Nature Communications 2024) and
Gruver et al. (ICLR 2024) establish that autoregressive language modelling of crystal text is a
viable generator at scale. Two things follow. This work's backbone is far smaller than any of
them, so nothing here is a claim about state-of-the-art crystal generation; and MatterGen's joint
conditioning is the natural comparison a materials reader will raise against the composition
result, with the distinction being that fine-tuning per property set is exactly the per-instance
cost the projection avoids and the composition limit measured here is a property of shared token
budgets rather than of conditioning in general.

Two boundaries position the work as much as the capability does. The layer constrains
additive functionals of the emitted symbols, so constraints defined by a search over latent
assignments — charge neutrality being the canonical materials example — are outside its
class rather than unimplemented. The automaton-based methods — GeLaTo (Zhang et al., ICML 2023) and Ctrl-G (Zhang et al., NeurIPS
2024) — amortize across constraints at a fixed model, with exactness guarantees for constraints
expressible as deterministic finite automata over tokens. The measured distinction is not
tractability: at the finest tolerance this serialization can express, an automaton tracking the
running budget is a fraction of the controlled model's size. It is that the bound compiles into
the transition structure, so each target is a separate automaton, while the dual solve takes the
target as a runtime argument. Their amortization and this one are over different axes and
compose rather than compete.

And solving several constraints exactly is not the same as
steering several: whether a joint solution exists at a position is set by the constraints
and the prefix, and two constraints reading the same tokens through correlated weight tables
compete for one budget rather than composing. Neither boundary is visible from the
inference-time constrained-decoding literature, where the constraint is applied to a fixed
model and the question of what training does with it does not arise.


## References

Verified against primary sources: publisher pages, proceedings records, or the venue's own
programme listing. Venue and volume are as the publisher states them, not as secondary citations
report them.

Ahmed, K., Chang, K.-W., and Van den Broeck, G. A pseudo-semantic loss for autoregressive models
with logical constraints. *Advances in Neural Information Processing Systems 36 (NeurIPS)*, Main
Conference Track, 2023. Verified against the NeurIPS 2023 proceedings page and arXiv:2312.03905.

Antunes, L. M., Butler, K. T., and Grau-Crespo, R. Crystal structure generation with
autoregressive large language modeling. *Nature Communications* 15, 10570, 2024.
doi:10.1038/s41467-024-54639-7.

Gruver, N., Sriram, A., Madotto, A., Wilson, A. G., Zitnick, C. L., and Ulissi, Z. Fine-tuned
language models generate stable inorganic materials as text. *Twelfth International Conference on
Learning Representations (ICLR)*, 2024.

Khalifa, M., Elsahar, H., and Dymetman, M. A distributional approach to controlled text
generation. *Ninth International Conference on Learning Representations (ICLR)*, 2021.

Kruszewski, G., Rozen, J., and Dymetman, M. disco: a toolkit for distributional control of
generative models. *61st Annual Meeting of the Association for Computational Linguistics (ACL),
System Demonstrations*, 2023. Outstanding demonstration paper.

Loula, J., LeBrun, B., Du, L., Lipkin, B., Pasti, C., Grand, G., Liu, T., Emara, Y., Freedman, M.,
Eisner, J., Cotterell, R., Mansinghka, V., Lew, A. K., Vieira, T., and O'Donnell, T. J. Syntactic
and semantic control of large language models via sequential Monte Carlo. *Thirteenth
International Conference on Learning Representations (ICLR)*, 2025. Selected for oral
presentation.

Zeni, C., Pinsler, R., Zügner, D., Fowler, A., Horton, M., Fu, X., Wang, Z., Shysheya, A.,
Crabbé, J., Ueda, S., Sordillo, R., Sun, L., Smith, J., Nguyen, B., Schulz, H., Lewis, S., Huang,
C.-W., Lu, Z., Zhou, Y., Yang, H., Hao, H., Li, J., Yang, C., Li, W., Tomioka, R., and Xie, T. A
generative model for inorganic materials design. *Nature* 639(8055), 624-632, 2025.
doi:10.1038/s41586-025-08628-5.

Zhang, H., Dang, M., Peng, N., and Van den Broeck, G. Tractable control for autoregressive
language generation (GeLaTo). *Fortieth International Conference on Machine Learning (ICML)*,
PMLR 202, 40932-40945, 2023. Oral presentation.

Zhang, H., Kung, P.-N., Yoshida, M., Van den Broeck, G., and Peng, N. Adaptable logical control
for large language models (Ctrl-G). *Thirty-Eighth Conference on Neural Information Processing
Systems (NeurIPS)*, 2024.
