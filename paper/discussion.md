# Discussion

Draft. Claims here are restatements of measurements in `results.md`; no new quantity is
introduced without its artifact.

## What the measurements license

The clearest result is the one requiring no training at all. A support restriction is exact
by construction, and the measurements confirm it holds where everything else fails: at
targets past the reachable ceiling, in an untrained model, and in every cell of every sweep.
This is worth separating from the rest because it is the part a practitioner can rely on
unconditionally, and because conflating it with expectation steering would let the exact
result launder the approximate one.

Expectation steering is genuinely approximate, and the paper's contribution on that front is
as much the characterization as the capability. The gap between the per-step surrogate and
the sequence-level optimum is a fixed overhead of roughly one nat against an optimum whose
own cost varies over two orders of magnitude, so the surrogate's relative waste shrinks as
the constraint pulls harder — the method is least wasteful exactly where it is most needed.
The ceiling is likewise a measured property rather than an omission: steering upward fails
because the running budget eventually demands more mass per atom than any legal token
supplies, and training's ability to repair that infeasibility decays to nothing at the
extreme.

Amortization is the result that changes what the method is for, and measuring it against the
strongest available baseline narrowed the claim rather than confirming it. Best-of-n from the
unsteered model needs no training and serves every target from one checkpoint, and given enough
samples it is more accurate than the projection at most targets. Any framing of amortization as
an accuracy advantage is therefore unavailable. What survives is narrower and, being measured
against the harder comparator, more durable: the projection reaches target accuracy without a
per-target training run and without a per-output sampling multiplier, and it does so without the
diversity collapse that selection incurs, because it shapes the distribution rather than
filtering it. Selection's diversity degrades precisely as the target gets harder to hit, which
is where a generative model is wanted. Because the target enters
the dual solve as an argument, one checkpoint covers a family of targets at parity with
separately trained models, and the comparison is a null result in the favourable direction:
one training run is statistically indistinguishable from seven. Conditional training remains
more accurate per target, so the trade is accuracy against a cost structure — the baseline
needs a model per target and a data slice per target, and both requirements bite hardest
where the request is unusual. Read alongside the coverage collapse as the slice thins, the
honest summary is that conditional training wins on a well-populated target and loses on a
sparse one, and that the projection's advantage is amortization rather than accuracy.

## What the negative results are worth

Two negative results carry more weight than they might appear to.

The composition limit separates two claims that are easy to conflate. The solver handles
several constraints simultaneously and exactly; that is verified against two oracles with
the conditioning reported. It does not follow that several constraints can be steered,
because whether a joint solution exists at a position is determined by the constraints and
the prefix rather than by the solver. On real crystals two additive per-element quantities
correlated at 0.725 fail to steer jointly, and the mechanism is competition for one token
budget: satisfying one budget consumes the atoms the other needs, so a majority of
actionable positions are excluded because one budget has already reached zero. Publishing
the exactness without the composition limit would have been the more flattering and less
useful paper, and a reader would have discovered the limit on their own data.

The expressibility boundary is the other. Charge neutrality is the constraint a materials
reader asks for first, and it is not expressible in the class this layer handles: as
defined by the standard screen it is an existential predicate over oxidation-state
assignments, and the linear proxy that would make it fit recovers 11.1% of the balanced
set. Stating that positively — the layer constrains additive functionals of emitted
symbols, not predicates requiring a search over latent assignments — is more useful than
listing charge neutrality as future work, because it tells a reader which of their own
constraints will and will not fit.

## Limitations that shape the claims

The steering results rest on one crystal serialization, and the obvious remedy turns out to be
unavailable rather than merely unbudgeted. Retraining under the alternative ordering would train
with no actionable positions: the density budget is determined at 100% of element positions
under cell-first ordering and 0% under composition-first, because density is a ratio and
a target fixes a mass budget only once the volume is known. The second family inherits the same
dependence.

What that establishes is a coupling rather than a gap. A quantity is constrainable by this layer
only where the prefix determines its budget, so token ordering is a design variable of the method
and not an independent axis to vary — and the determinability measurements already showed the
same constraint family being usable at very different rates depending on how a structure is
written down. A serialization comparison for this method requires two orderings that both close
the budget before the constrained tokens, which is a narrower and more specific experiment than
the one it replaces. Until it is run, nothing here separates properties of the method from
properties of this ordering, and the reason is now measured rather than conceded.

Seed counts are uneven by design of what was worth replicating. The fixed-target arms carry
three seeds, the frontier two per target, and the two extreme targets one each. When a
single-seed reading produced a striking result — a checkpoint reaching well past the
published ceiling — replication reduced it to a soft-ceiling statement, which is the
correct outcome and an argument for the seeds that exist rather than for the ones that do
not.

The composition result rests on one constraint pair chosen for availability rather than for
independence. The two quantities are both additive in element identity and count, which is
what makes them competitors; a pair reading different parts of the sequence might compose
freely. Nothing here shows that expectation constraints in general fail to compose, only
that these two do, and the vocabulary sweep suggests the relevant variable is how much
freedom the support leaves rather than the number of constraints.

Thermodynamic stability is not assessed, and the reason is structural rather than
practical. The serialization carries composition and cell without atomic positions, and
every energy method needs positions; obtaining them is crystal structure prediction, which
this work does not attempt. Composition-level and cell-level plausibility are measured
instead, each against a reference obtained by scoring real structures through the identical
path — a reference that is not 1.0, because the charge screen rejects real materials at a
measurable rate, so an arm scored against an assumed ceiling would be penalized for a
property of the screen.

## What is not claimed

The layer is not claimed to improve in-distribution sample quality; an external benchmark
reports that constraints do not, and nothing here contradicts that. No claim is made about
constraints the linear-functional form cannot express, nor about quantities not determinable
from a prefix. Expectation constraints are satisfied in expectation and not per sample.
Thermodynamic stability of generated structures is not assessed, and no DFT calculation is
reported. The compensation hypothesis was tested and rejected rather than left open.

## What follows

The immediate question is whether the composition limit is a property of the constraint pair
or of the method. Two grid rows at density targets where both constraints pull in the same
direction discriminate between those, and if pass-through is higher there the claim narrows
usefully to compatibility rather than composability. A constraint pair chosen for
independence would be the stronger test and is a larger piece of work.

For the serialization question, the honest position is that three serializations support a
correlational statement and not a predictive one. Fitting a prediction on two points and
testing on one is a residual rather than a held-out test, so that claim is stated
correlationally here and the predictive form waits for enough serializations to hold some
out.
