# WS-3 — The conditional-training baseline beats us on accuracy

Standing note, rebuilt on change. Feeds CP4, CP5. All numbers are read from
`results/WS3_conditional_baseline.json`, sixteen runs across four targets and two seeds.

## Summary

A4 — a masked model trained only on the MP-20 subset within 0.5 g/cm^3 of the target, with
no decode-time constraint — is **more accurate than A3 at every target tested**, and the gap
widens sharply above the data mean. This is the "just retrain on filtered data" objection
and the honest answer is that on the target metric alone, the objection wins.

What A4 gives up is coverage. It reproduces the slice it was shown: 5.9% of its
compositions are novel against A3's 94.1%. The paper's claim has to be the trade, not the
accuracy.

## The comparison

Both arms share initialization, data order, optimizer settings and step budget, and both are
scored against the **full** split's chemistry rather than A4's filtered slice. A4's budget is
not reduced, which is generous: its training set is 5–14% of the split, so it sees each
structure roughly 7–19 times more often.

| Target | Arm | Train set | Density | \|Error\| | Validity | Novel | Unique |
|---|---|---|---|---|---|---|---|
| 3.0 | A4 | 2,392 | 3.138 | 0.138 | 1.0000 | 0.073 | 0.422 |
| 3.0 | A3 | 26,910 | 3.240 | 0.240 | 1.0000 | 0.927 | 0.640 |
| 5.0 | A4 | 3,630 | 5.014 | 0.033 | 1.0000 | 0.059 | 0.519 |
| 5.0 | A3 | 26,910 | 4.849 | 0.151 | 1.0000 | 0.946 | 0.782 |
| 8.0 | A4 | 3,034 | 8.027 | 0.047 | 1.0000 | 0.073 | 0.500 |
| 8.0 | A3 | 26,910 | 6.760 | 1.240 | 1.0000 | 0.941 | 0.845 |
| 10.0 | A4 | 1,411 | 10.008 | 0.008 | 1.0000 | 0.029 | 0.318 |
| 10.0 | A3 | 26,910 | 8.330 | 1.670 | 1.0000 | 0.948 | 0.876 |

## What the numbers say

**A4 wins accuracy at all four targets.** Mean absolute error at or below the data mean of
6.679: A4 0.085 against A3 0.196, a factor of 2.3. Above it: A4 0.027 against A3 1.455, a
factor of 53.2.

**The high-target gap is the steering ceiling, already reported.** A3 reaches 6.760 when
asked for 8.0 and 8.330 when asked for 10.0, consistent with the frontier's saturation at
10.405. A4 has no ceiling because it never steers — it samples a slice that already sits at
the target, so the constraint machinery is not involved at all. A4's advantage above the mean
is therefore not evidence that conditioning steers better; it is evidence that not steering
avoids the failure mode of steering.

**A3 is 16 times more novel.** Novel-formula fraction averaged over targets: A3 0.941, A4
0.059. Unique-formula fraction: A3 0.786, A4 0.440. Roughly 94% of A4's compositions are
copied from its own training data. A conditional model reproduces its slice; a steered model
reaches the target from the full distribution.

**Structural validity is exactly 1.0000 for both arms at every target and seed.** The
masking claim is independent of everything above, including the arm that loses on accuracy
and the targets where steering fails.

## Total variation is a shift measure and must not be cited as diversity

The element-frequency total variation against the full data does **not** separate the arms,
and reporting it as a diversity metric would have been wrong in a way one target alone would
have concealed. At target 3.0 it favours A3 (0.323 against 0.392); at 5.0 it favours A4
(0.203 against 0.241). The orderings disagree.

The reason is that any arm reaching a target away from the data mean must shift its
chemistry to get there, so a large TV is required of both rather than evidence against
either. A4's TV tracks the TV between its own filtered slice and the full data to within
0.004 — 0.3919 against 0.3836 at target 3.0, 0.2028 against 0.1988 at 5.0. That agreement is
what reproducing a slice looks like, and it is TV's one legitimate use in this comparison.
Diversity is carried by the unique- and novel-formula fractions, which do not confound
shift with collapse.

## What this does not establish

Whether A4 remains competitive where its training data thins is untested here and is being
measured rather than asserted. An earlier draft of this note claimed the conditional
baseline "starves" at extreme targets; at a floor of 100 structures only one of the
frontier's ten targets fails that test, not three as the prose had it. The filtered set does
thin monotonically toward the tails — 3,630 structures at target 5.0, 1,411 at 10.0, 388 at
13.0, 119 at 16.0, and 6 at 20.0 — but sparse is not the same as empty, and the sparse
regime is a measurement rather than an argument.

## Verification

Eight tests in `tests/test_conditional_baseline.py`. They assert that A4's scoring reference
is the passed-in full-data reference rather than the sample itself, that its step budget is
not reduced, that a sample repeating one formula cannot look diverse, that novelty is
relative to what the arm trained on, that the filtered slice's own TV is large at every
compared target, and that A4's measured TV matches its slice's TV — the last two being the
pair that make the shift-versus-diversity distinction checkable rather than assertable.
