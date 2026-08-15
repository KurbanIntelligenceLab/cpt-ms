# WS-6: best-of-n, and what selection costs

`results/WS6_best_of_n.json`, `results/CP8_automaton_size.json`,
`results/WS4_equivalence.json`

Three external-comparison gaps, closed. Two of the three produced a result that runs against
the draft, and both are reported as they came out.

## Best-of-n is a real competitor and beats the projection on accuracy

Method. Draw one pool of 1024 samples from the checkpoint's backbone reading — mask on,
projection off — and keep whichever lands closest to the target. One pool serves every target
and every n, so n is not confounded with sampling noise. Expected best-of-n error is estimated
by repeated subsampling of n draws rather than by reporting the single best draw in the pool,
which would overstate it. The projection arm's published frontier error is the comparator.

Results.

| target | projection error | n=1 | n=4 | n=16 | n=64 | n to match | unique @ n=64 |
|---|---|---|---|---|---|---|---|
| 1.5 | 0.348 | 6.905 | 3.045 | 1.438 | 0.484 | >64 | 0.225 |
| 2.0 | 0.266 | 7.127 | 2.348 | 1.008 | 0.290 | >64 | 0.220 |
| 3.0 | 0.240 | 5.090 | 1.910 | 0.575 | 0.138 | 64 | 0.225 |
| 5.0 | 0.151 | 4.303 | 1.006 | 0.250 | 0.073 | 64 | 0.205 |
| 6.317 | 0.719 | 3.910 | 0.843 | 0.252 | 0.062 | 16 | 0.215 |
| 8.0 | 1.240 | 3.729 | 1.139 | 0.391 | 0.097 | 4 | 0.220 |
| 10.0 | 1.670 | 4.112 | 1.571 | 0.516 | 0.130 | 4 | 0.230 |
| 13.0 | 4.256 | 5.844 | 2.342 | 0.683 | 0.177 | 4 | 0.240 |
| 16.0 | 6.125 | 7.906 | 4.269 | 1.480 | 0.448 | 4 | 0.245 |
| 20.0 | 9.595 | 11.854 | 6.972 | 3.149 | 0.950 | 4 | 0.250 |

Best-of-n reaches the projection's accuracy at 8 of 10 targets within n = 64, and at four
targets it needs only n = 4. This is a stronger baseline than conditional training and it
amortizes perfectly across targets, since one unsteered model serves every target. Reporting
it is not optional and dismissing it would be indefensible.

## What selection costs, and why the comparison is not accuracy

Best-of-n buys accuracy by discarding samples, so the accepted set is the pool's tail near the
target. Unique-formula fraction among accepted outputs falls from about 0.81 at n = 1 to
about 0.20 at n = 64, consistently across all ten targets. The projection arm's unique
fraction at target 5.0 is 0.782, which is above best-of-16 and more than three times
best-of-64.

The two approaches differ on the axis of what they do to the distribution rather than on
accuracy. Selection filters and its diversity degrades as the target gets harder to hit;
shaping moves the distribution and its diversity does not depend on how hard the target is.
An accuracy-versus-compute curve alone would hide that, which is why the diversity column is
in the table above rather than in an appendix.

Best-of-n also pays its cost every time. n forward passes per accepted output, at every
target, forever; the projection pays once at training and one forward pass per output
thereafter. At n = 64 that is a 64x standing inference multiplier against a one-time training
cost.

## The DFA-size argument does not hold

Method. Ctrl-G and GeLaTo amortize constraint control across constraints without retraining,
which is the same word this paper uses, so the distinction has to be made rather than assumed.
The intuitive objection is that a real-valued running budget needs an intractable number of
automaton states. That is arithmetic, so it was computed.

Results. It is false. The budget spans 4397 amu and the serialization's own
discretization floor is 0.0020 relative — cell parameters are emitted as bin indices, so a
finer tolerance is not expressible and would be a spurious demand on any controller. At the
finest expressible tolerance the budget needs 100 states, and 88,400 once multiplied by
parse position: 0.02x the controlled model's parameter count, entirely buildable.

The real distinction is per-instance construction. The budget bound is compiled into the
transition structure, so each target is a separate automaton — about 884,000 states for the
ten frontier targets — while the dual solve takes the target as a runtime argument. Their
amortization is across constraints at a fixed model; this paper's is across targets of one
constraint at a fixed model. Neither subsumes the other, and the size argument that would have
been more flattering is not available.

## The equivalence statistics, corrected

The Wilcoxon signed-rank test on seven paired targets cannot return a two-sided p below
0.015625 under any data whatsoever — the floor is 2/2^7 and is a property of the sample
size, not of the result. Quoting p = 1.0 as though it carried evidential weight would
have been caught.

The claim now rests on a two-one-sided-tests procedure against a margin fixed in
`preregistration/WS4_equivalence_margin.md` and read from there by the analysis script, so it
cannot be adjusted after the fact. TOST gives p = 0.0023 at the pre-registered margin of
0.189 g/cm^3, the parametric 95% interval on the mean paired difference is
[-0.111, 0.086] and the bootstrap interval over 10,000 paired resamples is
[-0.090, 0.058]. The parametric interval is the wider of the two and is the one the
prose quotes.

A margin of 0.15 was proposed externally, described as the seed standard deviation at target
5.0. That value is 0.1147, not 0.15, and no single target yields 0.15; the
pre-registration uses the mean across targets and records why.

## What this changes in the paper

The amortization claim survives but its framing must move. Against conditional training the
claim was accuracy-at-parity with a better cost structure. Against best-of-n it cannot be
accuracy at all: best-of-n wins on accuracy at most targets given enough samples. The
defensible claim is that the projection is the only arm measured here that achieves target
accuracy without either a per-target training run or a per-output sampling multiplier, and
that it does so while keeping the generated distribution's diversity.
