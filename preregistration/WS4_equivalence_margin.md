# Pre-registration: the equivalence margin for the amortization claim

Status: BINDING. The margin below is the one the analysis script must read; it is recorded here
rather than in the script so that it cannot be adjusted after seeing a result.

## What is being tested

Whether one checkpoint deployed at density targets it never trained on is equivalent to a set of
models each trained at its own target, on absolute density error, paired by deployment target.

## Why a margin is required at all

The comparison is a null result: the difference test returns a large p-value. A large p-value is
the absence of evidence of a difference and does not license the word "equivalent". Two one-sided
tests can license it, but only against a margin — the size of difference declared too small to
matter — and a margin chosen after seeing the data is not a test. Hence this file.

## The margin

    EQUIVALENCE_MARGIN_G_PER_CM3 = 0.189

Rule that produced it: the mean, across the paired deployment targets, of the retargeted arm's
within-target standard deviation across its three seeds.

Rationale. A difference smaller than the method's own run-to-run variation is not a difference a
user could act on: re-running the same configuration with a different seed moves the result by
more than that. Tying the margin to the arm's own reproducibility also means it cannot be widened
to pass without changing the arm itself, which is the property a pre-registered margin needs.

## A margin considered and rejected

An external review proposed 0.15 g/cm^3, described as "the seed standard deviation already
reported at target 5.0". That description is incorrect: the retargeted arm's seed standard
deviation at deployment target 5.0 is 0.1147, not 0.15. The seven per-target values are 0.036,
0.027, 0.080, 0.115, 0.194, 0.339 and 0.533, so no single target yields 0.15 and picking one
target's value would in any case make the margin depend on which target was chosen.

The mean across targets is used instead. It is not the most permissive choice available — the
largest per-target value is 0.533 — and it is reported alongside a sensitivity sweep so a reader
can see the conclusion at tighter margins.

## Decision rule, fixed in advance

Equivalence is claimed if and only if the two-one-sided-tests procedure returns p < 0.05 against
the margin above AND the same procedure still returns p < 0.05 at a margin of 0.10 g/cm^3. The
second condition exists so that a conclusion depending on a generous margin cannot be reported as
if it did not.

If the first condition holds and the second fails, the claim to be written is "equivalent within
0.189 g/cm^3, and margin-sensitive below that", not "equivalent".

If neither holds, the claim is that the comparison is consistent with no difference and that the
design can exclude effects only larger than the reported minimum detectable effect. The word
"equivalent" is not to be used in that case anywhere in the manuscript.

## Statistics to report regardless of outcome

The seven paired differences individually; a bootstrap 95% confidence interval on the mean paired
difference from 10,000 paired resamples; the parametric confidence interval; the minimum
detectable effect at 80% power; and the Wilcoxon signed-rank result stated together with its
minimum attainable two-sided p-value at this sample size, which is 0.015625 and is a property of
n = 7 rather than of the data.
