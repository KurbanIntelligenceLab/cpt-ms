# Internalization and the risk register, revised against the crystal measurement

Standing note, rebuilt on change. Supersedes the internalization framing carried in the
research plan and in the multiplier-decay language of `WS2_alignment_sweep_notes.md`,
`CP0_actionable_window.md`, and `CP0_fallback_policy.md`. Written because the crystal
result contradicts a premise those documents share, and leaving the old framing standing
beside the new measurement would be the error.

## What the plan assumed

The plan treated internalization as the expected outcome and multiplier decay as its
signature: as training proceeds the backbone absorbs the constraint, the projection's
correction shrinks toward identity, and the decay curve doubles as evidence that the
method taught the model something. The associated risk was that decay goes to completion
— the layer becomes redundant, and the contribution reduces to an expensive way of
reaching a distribution ordinary training would have found.

That framing is measured, not speculative, and on the synthetic task it holds.

## What the two domains actually show

**Synthetic task — partial internalization, as expected.** Backbone reading +0.6287
against a target of 0; deployed +0.0170. The network moves substantially toward the
constraint on its own and the layer supplies the remainder. The gap between the two
readings is 37x, and that gap is the internalization measurement — a cleaner statement
than a decaying multiplier curve, because it is a property of the trained model rather
than of the optimizer's trajectory.

**Crystals — anti-internalization.** Backbone 9.200 g/cm^3; deployed 4.810 against a
target of 5.0. The data mean is 6.679. Under a constraint pulling density DOWN, the
backbone's unaided distribution sits 2.522 ABOVE the data mean. It did not partially
learn the constraint; it moved against it.

The two are not in conflict once the mechanism is named. In both cases the backbone is
optimizing loss under a projection that is present at every training step. On the
synthetic task, with a fixed-length sequence and an exact constraint channel, the
cheapest route to low loss overlaps with satisfying the constraint directly. On crystals,
where the budget denominator is an estimate and the projection reliably pulls one way,
the cheapest route is to bias predictions in the opposite direction so the
post-projection result matches the data. The compensation hypothesis is stated formally
in `analyze_sweep.py` and is under test in the steering sweep.

## Consequence for the risk register

**Retire the multiplier-decay-to-redundancy risk on crystals.** It described the layer
becoming unnecessary. The measurement shows the opposite dependency: removing the layer
at inference does not degrade the result by a margin, it inverts it — from 4.810, which
is 1.02 seed-standard-deviations from the target, to 9.200, whose error of 4.200 is 2.4x the
unsteered baseline's 1.783. A layer whose removal makes the system worse
than never having trained with it is not on a path to redundancy.

The risk that replaces it is narrower and testable: **the combined system may be correct
only at the target it trained for.** If the backbone's offset is a learned compensation
for a specific projection pull, a model trained at one target and deployed at another
would be miscalibrated in a way neither reading reveals on its own. The steering sweep
answers this directly — the compensation regression across ten targets is exactly the
measurement that distinguishes a target-specific offset from a general one.

**Retain the decay risk on the synthetic task**, where it is the honest description and
where the alignment sweep already traces it in closed form.

## Consequence for how results are reported

Both readings are mandatory in every arm, in both domains. Reporting only the backbone
understates the system anyone would run; reporting only the deployed hides the
compensation and would have concealed the crystal finding entirely. The two mislead in
opposite directions, which is why neither is a summary of the other.

The framing that survives both domains is that the projection is a **training component
rather than a decoder**. That was the thesis before either measurement; the crystal case
is now its sharpest evidence, because it is the case where the decoder reading of the
method fails outright.

## Documents affected

- `WS2_alignment_sweep_notes.md` — the decay probe stands on the synthetic task. Add a
  pointer noting it does not transfer to crystals.
- `CP0_actionable_window.md` — the gate's reasoning about identity-projection at
  convergence applies to the synthetic arm. On crystals, an active multiplier at
  convergence is the expected state, not a failure to internalize.
- `CP0_fallback_policy.md` — unchanged in substance; its denominator scoping is
  independent of which way the backbone moves.
- The research plan's internalization sections — the GraphAF-style framing of
  validity-under-mask-removal should not be carried over to the density dial, where the
  analogous operation makes the result worse rather than merely unconstrained.
