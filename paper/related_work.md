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
