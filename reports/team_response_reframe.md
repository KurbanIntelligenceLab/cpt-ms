# Response — reframe accepted, one citation corrected, one design consequence

## The reframe is right and I should have raised it

Accepted without reservation. The argument is stronger than a preference about framing: the
current design sizes its headline experiment in the regime where the layer is guaranteed to
look inert, and it does so for a reason visible in our own artifact. The alignment sweep
measured constraint activity vanishing quadratically as data-constraint alignment improves,
and MP-20 sits at high alignment by construction. Designing the primary experiment there
means pre-registering a null.

I flagged multiplier decay as the modal outcome and recommended writing the paper
measurement-first to survive it. That was the wrong conclusion from the right observation.
The correct response to "the layer looks inert where the data already satisfies the
constraint" is to test it where the data does not, not to soften the claim. Steering is that
regime, and it is also what generative materials models are for.

Converting decay from threat to control condition is the part that makes it a paper rather
than a hedge. Multipliers going quiet when the constraint matches the data and staying active
when it does not is the mechanism behaving as specified, and the sweep already measures the
transition.

## Verification of the cited evidence

Our own numbers, checked against the artifacts:

| Claim | Status |
|---|---|
| Scaling exponents 1.026 and 2.038 | Correct as rounded. `WS2_alignment_sweep.json` gives multiplier log-log slope 1.02597 and KL log-log slope 2.03831, i.e. 1.026 and 2.038 to three decimals, against exponential-family predictions of exactly 1 and 2. |
| MP-20 charge-balance rate 91.13% | Correct, and previously traced to CDVAE (arXiv 2110.06197) Table 1, SMACT-based composition validity. |
| MDE table gap 3.4pp (q) to 20.2pp (theta = 0.10) | Correct. `CP4_violation_rate_MDE.md` gives 3.4 pp for charge neutrality and 20.2 pp at theta = 0.10. Note the gap is not monotone in strictness: theta = 0.50 gives 17.9 pp, below theta = 0.10, because model retention falls faster than MP-20 retention at the loosest prior. Cite the range as 3.4-20.2 pp with the maximum at theta = 0.10, not as increasing with strictness. |

External claims:

| Claim | Status |
|---|---|
| MatterGen S.U.N. 83% / 65% / 49% across well-explored, partially explored, unexplored | Correct, verbatim in the MatterGen model card, described there as evaluated on chemical systems in those three categories. |
| MatterGen docs state performance may degrade for extreme property values with few training structures | Correct, verbatim in the model card's Recommendations section. |
| Guidance factor gamma: increasing it produces samples adhering more to target values "at the expense of diversity and realism" | Correct, verbatim in the MatterGen repository README, describing `--diffusion-guidance-factor` as the classifier-free guidance gamma. |
| PhononBench: band-gap-conditioned MatterGen at Eg = 4.5 eV yields 1,037 dynamically stable of 5,340 evaluated (19.4%) | **Incorrect.** The denominator is right, the numerator is not. |

### The PhononBench correction

The paper reports **617 dynamically stable out of 5,340 evaluated at Eg = 4.5 eV, which is
11.6%** — not 1,037 and not 19.4%. This appears twice in the text, in the Figure 3 caption
and in the results table. The string 1,037 does not occur in the paper.

The full band-gap series, for the record: 23.5% at 0.5 eV (1,523 of 6,524), 15.3% at 1.5 eV,
13.3% at both 2.5 and 3.5 eV, and 11.6% at 4.5 eV, with an overall rate of 15.6% across
33,210 structures under phonon analysis.

The error runs against the reframe's own interest, which is why it matters to fix rather than
quietly adopt. The true figure is worse than quoted — dynamical stability at the extreme
band-gap condition is 11.6%, not 19.4% — and the degradation from 23.5% at the mild condition
to 11.6% at the extreme is a cleaner statement of the same point. Roughly halving under
property conditioning beyond the comfortable range is a stronger motivation for
constraint-based steering than the quoted numbers gave.

## One design consequence the reframe implies

Under the old design, the crystal constraint was charge balance on MP-20 — a constraint the
data satisfies. Under steering, family (i) is enforced at strictness levels the data does
**not** satisfy, and the determinability measurement now bears on the primary experiment
rather than a secondary one.

On the CIF character backbone, family (i) has a mean usable fraction of 0.002: active over
roughly the last two tokens of a 300-800 token sequence, because CIF spells composition out
in the atom-site loop at the end. Under matched-alignment that was a weak-signal problem on a
secondary arm. Under steering it sits on the primary experiment's critical path, since the
steering-strength sweep dial (a) is filter strictness on exactly that constraint.

So the B1 format question is promoted from carry-forward to blocking, which the reframe
already anticipates in requiring re-measurement against the reference model's standardized
ordering. If that ordering places composition early, the problem dissolves and no deviation
from convention is needed. That measurement comes first in the next tranche.

## Defect found while verifying the repository state

Three of the four result artifacts recorded their commit as `<hash>-dirty` after the
re-emission, with a clean working tree. The cause was ordering: each script's output dirtied
the tree for the scripts that ran after it, so artifacts two through four were stamped dirty
against a commit whose source was pristine.

The provenance field is supposed to answer "what code produced this", and an artifact cannot
be expected to predate itself. Dirtiness is now judged over `scripts/` and `tests/` only,
excluding `results/`, `reports/`, and `figures/`. Eight tests in `tests/test_provenance.py`
build throwaway repositories to assert the rule in both directions — an emitted artifact must
not dirty provenance, a modified or newly added source file must — plus the sentinel behavior
when no repository exists and a staleness guard rejecting an artifact that predates the run
that was supposed to re-emit it. Five skip where `git init` is restricted and exercise fully
on a developer machine.

The four artifacts need one re-emission under the corrected logic to carry accurate hashes.

## Accepted without further comment

Register flips back to method-first. The new primary crystal experiment and the data-scarcity
companion are the right two experiments and reuse the existing harness. CP4 and CP5 criteria
move to steering-conditional metrics with matched-alignment retained as control. The
validity-versus-stability risk goes to the top of the register, measured in the same
experiment rather than assumed — that mitigation is the right structural choice, since
discovering it late would be the expensive version.
