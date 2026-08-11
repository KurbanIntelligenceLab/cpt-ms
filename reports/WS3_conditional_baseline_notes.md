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

## Where A4's data thins, accuracy holds and coverage collapses

An earlier draft of this note claimed the conditional baseline "starves" at extreme targets.
That was wrong twice over: at a floor of 100 structures only one of the frontier's ten
targets fails, not three as the prose had it, and running the sparse cases shows A4 does not
lose accuracy there at all. Numbers from `results/WS3_conditional_sparse.json`.

| Target | A4 train set | A4 error | A3 error | A4 novel | A4 unique | A4 elements |
|---|---|---|---|---|---|---|
| 5.0 | 3,630 | 0.033 | 0.151 | 0.059 | 0.519 | 84.0 |
| 10.0 | 1,411 | 0.008 | 1.670 | 0.029 | 0.318 | 79.5 |
| 13.0 | 388 | 0.031 | 4.256 | 0.016 | 0.108 | 73.5 |
| 16.0 | 119 | 0.071 | 6.125 | 0.004 | 0.031 | 56.5 |
| 20.0 | 6 | refused | 9.595 | — | — | — |

A4 stays accurate on 119 structures. What degrades is coverage, and it degrades with slice
size rather than with target: correlation between log training-set size and unique-formula
fraction is 0.981, with novel-formula fraction 0.908, and with distinct elements 0.962.
Unique formulas fall 17-fold from 0.519 to 0.031 and the element inventory contracts from 84
to 56.

At target 16.0 A4 emits 3.1% unique formulas — roughly four distinct compositions recited
from a 119-structure slice seen about 6,500 times each over training. It hits the target
because those four compositions are at the target.

At 20.0 the filter leaves 6 structures and A4 refuses to train. That is the one genuine
starvation case, and the refusal is the measurement rather than a failure of the run.

**The framing both arms' numbers support.** A4 is more accurate everywhere it can run, and
buys that accuracy by memorizing a slice; the price rises as the slice thins, which is
precisely where a generative model is wanted. A3 is less accurate and degrades past its
ceiling, but produces novel compositions at every target, including the one where A4 has no
training data at all.

## Verification

Ten tests in `tests/test_conditional_baseline.py`. They assert that A4's scoring reference
is the passed-in full-data reference rather than the sample itself, that its step budget is
not reduced, that a sample repeating one formula cannot look diverse, that novelty is
relative to what the arm trained on, that the filtered slice's own TV is large at every
compared target, and that A4's measured TV matches its slice's TV — the last two being the
pair that make the shift-versus-diversity distinction checkable rather than assertable. Two
more pin the sparse regime: A4's error must stay small at targets 13.0 and 16.0 (so a future
edit cannot reintroduce the false accuracy framing), and exactly one swept target may fall
below the training floor.
